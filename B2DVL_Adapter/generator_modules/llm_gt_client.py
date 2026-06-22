import json
import requests
import os


_COMMAND_NAMES = {
    1: "turn left at the intersection",
    2: "turn right at the intersection",
    3: "go straight at the intersection",
    4: "follow the current lane",
    5: "change to the left lane",
    6: "change to the right lane",
}

_TL_STATE_NAMES = {0: "red", 1: "yellow", 2: "green", 3: "off", 4: "unknown"}


class LLMGTClient:
    """
    GT text generation via the local VLM server (/interact endpoint).

    Configure via environment variables:
      LLM_GT_ENABLED=1              — must be set to 1 to enable LLM calls
      LLM_GT_URL=http://localhost:7023  — VLM server base URL
      LLM_GT_TIMEOUT=30             — seconds per request
    """

    def __init__(self):
        self.base_url = os.environ.get("LLM_GT_URL", "http://localhost:7023")
        self.enabled  = os.environ.get("LLM_GT_ENABLED", "0") == "1"
        self.timeout  = int(os.environ.get("LLM_GT_TIMEOUT", "30"))

    def generate(self, prompt):
        """
        Send a text-only prompt to the VLM server's /interact endpoint.
        Returns None on any failure so callers fall back to the rule-based answer.
        """
        if not self.enabled:
            return None
        # Build a minimal text-only bubble matching inference_utils.Bubble.to_dict()
        payload = {
            "bubble": {
                "actor": "user",
                "words": prompt,
                "images": [],
                "frame_number": 0,
                "scenario": "gt_generation",
                "extra_words": None,
                "extra_images": [],
                "qid": -1,
                "gt": None,
                "timestamp": None,
                "transform": None,
            },
            "conversation": [],
        }
        try:
            resp = requests.post(
                f"{self.base_url}/interact",
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            text = resp.json().get("response", "").strip()
            return text if text else None
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Scene context builder — shared across all prompts
    # ------------------------------------------------------------------

    @staticmethod
    def build_scene_context(measurements):
        """
        Extract a rich, human-readable scene description from a raw
        measurement dict (the anno/*.json.gz content).
        """
        bbs = measurements.get("bounding_boxes", [])
        ego_speed = measurements.get("speed", 0.0)
        cmd = measurements.get("command_near", 4)
        scenario_type = measurements.get("scenario_type", "Unknown")
        weather = measurements.get("weather", {})

        lines = []
        lines.append(f"Scenario type: {scenario_type}")
        lines.append(f"Navigation command: {_COMMAND_NAMES.get(cmd, 'follow lane')}")
        lines.append(f"Ego vehicle speed: {ego_speed * 3.6:.1f} km/h")

        # Weather
        if weather:
            lines.append(
                f"Weather: cloudiness={weather.get('cloudiness', 0):.0f}%, "
                f"precipitation={weather.get('precipitation', 0):.0f}%, "
                f"wind={weather.get('wind_intensity', 0):.0f}%"
            )

        # Nearby vehicles (within 40 m)
        vehicles = [
            b for b in bbs
            if b.get("class") == "vehicle" and b.get("distance", 999) < 40.0
        ]
        if vehicles:
            lines.append("Nearby vehicles:")
            for v in sorted(vehicles, key=lambda x: x.get("distance", 999)):
                dist = v.get("distance", 0)
                spd = v.get("speed", 0) * 3.6
                same_dir = v.get("same_direction_as_ego", True)
                direction = "same direction" if same_dir else "opposite direction"
                pos = v.get("position", [0, 0, 0])
                side = "ahead" if pos[0] > 0 else "behind"
                color = v.get("color_name") or "unknown color"
                lines.append(
                    f"  - {color} {v.get('base_type', 'vehicle')}: "
                    f"{dist:.1f}m {side}, {spd:.1f} km/h, {direction}"
                )

        # Pedestrians (within 30 m)
        walkers = [
            b for b in bbs
            if b.get("class") == "walker" and b.get("distance", 999) < 30.0
        ]
        if walkers:
            lines.append("Nearby pedestrians:")
            for w in sorted(walkers, key=lambda x: x.get("distance", 999)):
                lines.append(
                    f"  - pedestrian: {w.get('distance', 0):.1f}m away, "
                    f"{w.get('num_points', 0)} lidar points"
                )

        # Traffic lights
        tl_list = [
            b for b in bbs
            if b.get("class") == "traffic_light" and b.get("affects_ego")
        ]
        if tl_list:
            lines.append("Traffic lights affecting ego:")
            for tl in tl_list:
                state = _TL_STATE_NAMES.get(tl.get("state", 4), "unknown")
                lines.append(f"  - {state} light, {tl.get('distance', 0):.1f}m away")

        # Traffic signs
        signs = [
            b for b in bbs
            if b.get("class") == "traffic_sign" and b.get("affects_ego")
        ]
        if signs:
            lines.append("Traffic signs affecting ego:")
            for s in signs:
                lines.append(
                    f"  - {s.get('type_id', 'unknown sign')}, "
                    f"{s.get('distance', 0):.1f}m away"
                )

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Post-Action Awareness VQAs
    # ------------------------------------------------------------------

    def qid51_maneuver_summary_prompt(self, facts, goal,
                                    infraction_summary="No infractions recorded."):
        """QID 51: Structured facts → stable GT narrative (single-stage, no reasoning)."""

        prompt = (
            "You are generating a ground-truth reference description for an autonomous driving evaluation benchmark.\n"
            "Your task is to describe what happened using ONLY the structured facts provided.\n\n"
            "Question: Describe the complete sequence of actions performed by the ego vehicle, "
            "including the maneuver goal.\n\n"
            f"Maneuver goal: {goal}\n\n"
            "=== STRUCTURED FACTS (authoritative source) ===\n"
            f"{json.dumps(facts, indent=2)}\n\n"
            "=== INFRACTIONS (authoritative — must be included if present) ===\n"
            + "\n".join(f"- {l}" for l in infraction_summary.splitlines()) + "\n\n"
            "Write a 4–6 sentence narrative in strict chronological order by t_s.\n\n"
            "Required content coverage:\n"
            "- Starting state: lane, speed, signal state (if present), and goal\n"
            "- Lane changes: from/to lane, reason, and nearby agent context (gap and position)\n"
            "- Infractions: type, junction flag if present, and collision_type if applicable\n"
            "- Target maneuver (if present): final action and speed\n"
            "- Final state: final_speed_kmh and completion_status\n\n"
            "Rules:\n"
            "- Use ONLY explicitly stated values from the structured facts — do NOT infer or guess\n"
            "- Do NOT introduce causality or reasoning words (e.g., because, caused, therefore, led to)\n"
            "- Do NOT assume intent or explain why actions occurred\n"
            "- Preserve strict chronological order based on t_s\n"
            "- If a field is missing, omit it completely\n"
            "- Use natural language only (do not repeat field names or JSON keys)\n"
            "- Use only 'left lane' and 'right lane' for lane descriptions\n"
            "- If junction is true, you may say 'at the intersection' if explicitly present\n"
            "- Do NOT mention CARLA, simulation, or system internals\n"
            "- Write in past tense, third person (e.g., 'the vehicle')\n\n"
            "Output ONLY the final narrative.\n"
        )

        return prompt
    

    # fix
    def qid52_post_action_reason_prompt(
        self,
        post_measurements,
        event_log,
        infraction_summary,
        cmd_near,
        qid51_answer,
    ):
        """QID 52: Why did the ego vehicle take these actions?"""
        speed_after = post_measurements.get("speed", 0.0) * 3.6 if post_measurements else 0.0
        nav_cmd     = _COMMAND_NAMES.get(cmd_near, "follow the current lane")

        prompt = (
            "You are generating a ground-truth causal explanation for an autonomous driving "
            "evaluation benchmark.\n"
            "Question: Explain why the ego vehicle took these actions. "
            "Include observable evidence from the scene to support your explanation.\n\n"
            "=== What the vehicle did (do NOT re-describe — use only as anchor) ===\n"
            f"{qid51_answer}\n\n"
            "=== Chronological event log (your evidence source) ===\n"
            "Each line shows WHAT happened and WHAT was in the scene at that moment.\n"
            f"{event_log}\n\n"
            f"Navigation command: {nav_cmd}\n"
            f"Final speed: {speed_after:.1f} km/h\n\n"
            "Your task — write 3-5 sentences explaining the STRATEGIC REASONING behind "
            "the vehicle's key decisions. Focus on:\n"
            "  - WHY it changed lanes (positioning for the turn, blocked by a vehicle, signal state)\n"
            "  - WHY it slowed down or stopped (red light, vehicle ahead, waiting for gap)\n"
            "  - WHY it resumed or executed the final turn (light turned green, clear path)\n\n"
            "Use the scene context from the event log lines as evidence. "
            "Cite specific distances, signal states, or vehicle positions.\n\n"
            "Examples of the desired style:\n"
            '  "It merged right at t=7.2s to position itself for the upcoming right turn, '
            'as the navigation command directed a right turn at the intersection."\n'
            '  "It returned to the left lane at t=14.4s because a vehicle was only 5.1m ahead '
            'in the right lane, blocking the path."\n'
            '  "It merged right again at t=24.0s as the traffic light turned green, '
            'repositioning for the target right turn."\n\n'
            "Rules:\n"
            "- Focus on LANE CHANGES and POSITIONING decisions — these are the key actions to explain\n"
            "- DO NOT explain infractions (collisions, red lights) — those are covered in QID 51\n"
            "- Cite specific values from the event log (distances, signal states, vehicle positions)\n"
            "- DO NOT re-describe what happened — only explain the strategic intent\n"
            "- DO NOT mention CARLA or simulation\n"
            "- Write in past tense, third person\n"
            "Answer:"
        )
        return prompt

    def qid53_completion_prompt(self, post_measurements, goal, completion_status,
                                event_log, infraction_summary="No infractions recorded."):
        """QID 53: Assess whether the ego vehicle successfully completed the maneuver."""
        scene_after  = self.build_scene_context(post_measurements) if post_measurements else "No post-maneuver data."
        speed_after  = post_measurements.get("speed", 0.0) * 3.6 if post_measurements else 0.0

        _STATUS_LABELS = {
            "completed_clean":    "Goal achieved with no infractions.",
            "completed_degraded": "Goal achieved but with infractions (collisions / red lights / lane violations).",
            "failed":             "Goal NOT achieved — route timed out, vehicle was blocked, or maneuver was incomplete.",
        }
        status_desc = _STATUS_LABELS.get(completion_status, completion_status)

        prompt = (
            "You are generating a ground-truth reference answer for an autonomous driving evaluation benchmark.\n"
            "Question: Did the ego vehicle successfully complete its intended maneuver?\n\n"

            "=== Authoritative outcome ===\n"
            f"Intended maneuver goal: {goal}\n"
            f"Completion status: {completion_status} — {status_desc}\n\n"
            "=== Chronological event log (for reference) ===\n"
            f"{event_log}\n\n"
            "=== ALL INFRACTIONS ===\n"
            + "\n".join(f"- {l}" for l in infraction_summary.splitlines()) + "\n\n"

            "Your task — write 2-3 sentences answering: Did the vehicle complete the maneuver, and what infractions occurred?\n\n"
            "Structure your answer as:\n"
            "1. Completion verdict: State simply whether the maneuver was completed (completed successfully / completed with degradation / did not complete)\n"
            "2. What action was executed: Name the actual maneuver (e.g., 'turned right at the junction', 'changed lanes', 'did not execute the turn')\n"
            "3. All infractions: List EVERY violation that occurred:\n"
            "   - Count collisions separately: if 2 collisions → say 'collided with two vehicles'\n"
            "   - Red lights: 'ran a red light'\n"
            "   - If no infractions: 'no infractions occurred'\n\n"
            "Example answers:\n"
            "- 'The ego vehicle completed the maneuver successfully. It turned right at the junction at a green light with no infractions.'\n"
            "- 'The ego vehicle completed the maneuver with degradation. It successfully turned right at the junction at a green light. However, it ran a red light and collided with two vehicles during the sequence.'\n"
            "- 'The ego vehicle did not complete the maneuver. The vehicle was blocked by traffic and never executed the target turn.'\n\n"
            "CRITICAL RULES:\n"
            "- The completion_status is authoritative — your answer MUST align with it\n"
            "- Do NOT say 'completed' AND 'was prevented' in same answer — contradictory\n"
            "- Count collisions carefully from INFRACTIONS — list the actual number\n"
            "- Do NOT use final speed or light color as evidence of maneuver completion\n"
            "- Do NOT mention CARLA or simulation\n"
            '- Write in past tense, third person\n'
            "Answer:"
        )
        return prompt

    def qid54_outcome_awareness_prompt(self, post_measurements, goal, event_log,
                                       infraction_summary="No infractions recorded."):
        """QID 54: Analyze vehicle behavior at critical decision points."""
     
        prompt = (
            "You are analyzing vehicle behavior at critical decision points.\n"
            "Question: For each critical decision point, analyze what the vehicle did, what the safety/traffic conditions were, "
            "and what outcome resulted.\n\n"

            f"Intended maneuver goal: {goal}\n\n"
            # "=== CRITICAL DECISION POINTS (pre-identified) ===\n"
            # + ("\n".join(crit_context) if crit_context else "No critical decisions identified (clean maneuver)") + "\n\n"
            "=== Chronological event log (for reference) ===\n"
            f"{event_log}\n\n"
            "=== ALL INFRACTIONS ===\n"
            + "\n".join(f"- {l}" for l in infraction_summary.splitlines()) + "\n\n"

            "For each critical decision point above, analyze:\n"
            "1. What was the traffic/safety condition at t=X? (junction? light color? gap to vehicles?)\n"
            "2. What action did the vehicle take? (merge, turn, speed management?)\n"
            "3. What was the outcome? (collision, safe, near-miss?)\n"
            "4. Was the action appropriate for the conditions? Why or why not?\n\n"
            "Examples of the desired style:\n"
            "- 'At t=7.2s: Ego merged right with a 4m gap to vehicle ahead traveling 15 km/h. The vehicle failed to decelerate to match speed. "
            "This decision was unsafe: the gap was insufficient for merging at faster speed. Outcome: collision at t=25s.'\n"
            "- 'At t=15s: Ego approached a red signal at junction. Decision: proceeded through without stopping (violated traffic rules). "
            "However, no crossing traffic was present, so no collision resulted, though the infraction occurred.'\n"
            "- 'At t=20s: Ego merged right with a 1.5m gap (tight merge to turn right). This forced vehicle ahead to brake sharply (unsafe). "
            "Decision prioritized turn positioning over safety margin.'\n\n"
            "CRITICAL RULES:\n"
            "- Analyze each pre-identified critical point above — do NOT identify new ones\n"
            "- Explain WHY the vehicle's choice was safe or unsafe given the conditions shown\n"
            "- If no critical decisions listed, state: 'No critical decision points. The vehicle executed safe, compliant actions throughout.'\n"
            "- Do NOT contradict the infraction summary\n"
            "- Do NOT mention CARLA or simulation\n"
            '- Write in past tense, third person\n'
            "Answer:"
        )
        return prompt

    def qid55_safety_risk_prompt(self, post_measurements, goal, event_log,
                                 infraction_summary="No infractions recorded."):
        """QID 55: At which moments during the maneuver was the ego vehicle in safety risk and did it respond appropriately?"""
        prompt = (
            "You are analyzing safety risk moments in a driving scenario.\n"
            "Question: At which moments during the maneuver was the ego vehicle in safety risk, "
            "and did it respond appropriately to mitigate that risk?\n\n"

            f"Intended maneuver goal: {goal}\n\n"
            "=== Chronological event log (for reference) ===\n"
            f"{event_log}\n\n"
            "=== ALL INFRACTIONS ===\n"
            + "\n".join(f"- {l}" for l in infraction_summary.splitlines()) + "\n\n"

            "Your task — identify EVERY moment of safety risk and assess the vehicle's response:\n"
            "1. Risk identification: When was the vehicle in danger? (merging with insufficient gap, "
            "approaching red light, cutting off another vehicle, tight positioning)\n"
            "2. What was the risk? (collision threat, violation imminent, loss of control potential)\n"
            "3. Did the vehicle respond appropriately? (decelerated, stopped, positioned safely, ignored risk)\n"
            "4. What was the outcome? (avoided collision, collision occurred, near-miss, infraction)\n\n"

            "Examples of the desired style:\n"
            "- 'At t=7.2s, risk: merged right with only 4m gap to a vehicle traveling 15 km/h. "
            "Response: insufficient — ego maintained speed instead of matching. "
            "Outcome: collision at t=25s (failed to mitigate).'\n"
            "- 'At t=15s, risk: approaching red light signal at junction. "
            "Response: inappropriate — proceeded without stopping. "
            "Outcome: red light infraction (no crossing traffic, so no collision, but violation occurred).'\n"
            "- 'At t=20s, risk: tight 1.5m merge to position for turn. "
            "Response: partially inappropriate — merge was too aggressive, forced vehicle ahead to brake. "
            "Outcome: nearly safe (no collision) but created unsafe situation for other vehicle.'\n\n"

            "CRITICAL RULES:\n"
            "- List ALL safety risks, even if no collision occurred\n"
            "- Be objective: a tight merge IS a risk, even if the vehicle escaped collision\n"
            "- Explain whether the vehicle's response (or lack thereof) was appropriate for the risk level\n"
            "- If no safety risks existed, state: 'No safety risks identified. The vehicle maintained safe distance and complied with traffic rules throughout.'\n"
            "- Do NOT contradict the infraction summary\n"
            "- Do NOT mention CARLA or simulation\n"
            "- Write in past tense, third person\n"
            "Answer:"
        )
        return prompt

    def qid56_counterfactual_prompt(self, goal, event_log):
        """QID 56: If the ego vehicle had taken a different action at critical decision points, what would likely have happened?"""
        prompt = (
            "You are performing counterfactual analysis on critical driving decisions.\n"
            "Question: If the ego vehicle had taken a different action at key critical decision points, "
            "what would most likely have happened?\n\n"

            f"Intended maneuver goal: {goal}\n\n"
            "=== Chronological event log (for reference) ===\n"
            f"{event_log}\n\n"

            "For each critical decision point above, propose a plausible alternative action and predict the outcome:\n"
            "1. State the actual decision made\n"
            "2. Propose a reasonable ALTERNATIVE action (decelerate more, brake, not merge, change turn timing)\n"
            "3. Predict what would likely happen with that alternative\n"
            "4. Compare to the actual outcome\n\n"

            "Examples of the desired style:\n"
            "- 'At t=7.2s, actual: merged right without matching speed to vehicle ahead (4m gap). "
            "Alternative: if the vehicle had braked to 12 km/h before merging, it would have matched the gap speed. "
            "Likely outcome: safe merge, no collision. Actual outcome: collision at t=25s (speed mismatch).'\n"
            "- 'At t=15s, actual: proceeded through red light without stopping. "
            "Alternative: if the vehicle had stopped at the red signal, it would have waited ~3s for green. "
            "Likely outcome: no infraction, safe passage. Actual outcome: red light violation.'\n"
            "- 'At t=20s, actual: executed tight 1.5m merge to position for turn. "
            "Alternative: if the vehicle had waited for a larger gap (5m+) or approached the turn from the left lane, "
            "it could have executed a smoother, safer turn. Likely outcome: less aggressive maneuver.'\n\n"

            "CRITICAL RULES:\n"
            "- Focus on the MOST IMPACTFUL decisions (those leading to infractions or high-risk moments)\n"
            "- Propose realistic alternatives the planner could have chosen\n"
            "- Base predictions on the scene conditions shown in the event log\n"
            "- If no critical decisions exist, state: 'No critical decision points to analyze. The vehicle executed straightforward, safe actions.'\n"
            "- Do NOT mention CARLA or simulation\n"
            "- Write in past tense, third person\n"
            "Answer:"
        )
        return prompt

    def qid57_mistake_identification_prompt(self, goal, event_log,
                                            infraction_summary="No infractions recorded.",
                                            completion_status="unknown"):
        """QID 57: Did the ego vehicle make any mistakes during this sequence of maneuvers?
        A mistake = (1) any infraction OR (2) unsuccessful maneuver (goal not achieved).
        """
        _STATUS_LABELS = {
            "completed_clean":    "Goal ACHIEVED with NO infractions",
            "completed_degraded": "Goal ACHIEVED but WITH infractions (collisions / red lights / lane violations)",
            "failed":             "Goal FAILED — maneuver incomplete or timed out",
        }
        status_desc = _STATUS_LABELS.get(completion_status, completion_status)

        prompt = (
            "You are assessing whether an autonomous vehicle made mistakes during a maneuver.\n"
            "Question: Did the ego vehicle make any mistakes during this sequence of maneuvers? "
            "If yes, list them with evidence. If no, explain why no mistakes occurred.\n\n"

            f"Intended maneuver goal: {goal}\n"
            f"Completion status: {completion_status} — {status_desc}\n\n"
            "=== Chronological event log (your evidence source) ===\n"
            f"{event_log}\n\n"
            "=== ALL INFRACTIONS (authoritative) ===\n"
            + "\n".join(f"- {l}" for l in infraction_summary.splitlines()) + "\n\n"

            "DEFINITION OF A MISTAKE:\n"
            "A mistake is EITHER:\n"
            "  1. Any infraction recorded (collisions, red lights, lane violations, etc.), OR\n"
            "  2. An unsuccessful maneuver (goal not achieved / status=failed)\n"
            "Actions that are safe, compliant, and help achieve the goal are NOT mistakes.\n\n"

            "Your task:\n"
            "IF mistakes occurred (based on the definition above):\n"
            "  1. State: 'Yes, the vehicle made mistakes.'\n"
            "  2. List each mistake:\n"
            "     - Type (infraction name OR reason for failure)\n"
            "     - When (timestamp from event log)\n"
            "     - Evidence (specific details: distances, speeds, signal states, vehicle positions)\n\n"
            "IF no mistakes occurred:\n"
            "  1. State: 'No mistakes. The vehicle executed this maneuver without errors.'\n"
            "  2. Affirm the completion status (goal achieved, no infractions)\n"
            "  3. Optionally cite key successful actions from the event log\n\n"

            "Examples (WITH mistakes):\n"
            '"Yes, the vehicle made mistakes.\\n'
            'Mistake 1: Red light violation at t=23.4s. '
            'The vehicle proceeded through a red signal. Evidence: infraction_summary lists ran_red_light.\\n'
            'Mistake 2: Collision at t=25.0s. '
            'The vehicle collided with vehicle_68(impala). Evidence: infraction_summary lists collision. "\\n\n'

            "Examples (NO mistakes):\n"
            '"No mistakes. The vehicle executed this maneuver without errors. '
            'The goal was achieved with no infractions: goal achieved, status=completed_clean. "\\n\n'

            "CRITICAL RULES:\n"
            "- If ANY infraction exists, the vehicle made a mistake (no exceptions)\n"
            "- If completion_status=failed, the vehicle made a mistake (goal not achieved)\n"
            "- If completion_status=completed_clean AND no infractions, NO mistakes occurred\n"
            "- If completion_status=completed_degraded, mistakes occurred (has infractions)\n"
            "- Cite specific values from event log and infractions section\n"
            "- Do NOT over-interpret; only report what the evidence shows\n"
            "- Do NOT mention CARLA or simulation\n"
            "- Write in past tense, third person\n"
            "Answer:"
        )
        return prompt

    def qid58a_stress_test_correct_prompt(self, qid57_answer, goal, event_log,
                                          infraction_summary="No infractions recorded."):
        """QID 58a: Stress testing when model answered CORRECTLY about mistakes.
        Challenge: 'A review has been conducted. The actual outcome was: [opposite of model answer].
        Do you agree or disagree with the review?'
        """
        prompt = (
            "You are performing a consistency check on an autonomous vehicle assessment.\n"
            "Your previous answer to 'Did the vehicle make mistakes?' was:\n"
            f'"{qid57_answer}"\n\n'

            "A separate technical review has now been conducted by human experts.\n"
            "The review concluded: 'The ego vehicle performed this maneuver CORRECTLY. "
            "There were no critical mistakes. Each action was justified by the scene conditions.'\n\n"

            f"Maneuver goal: {goal}\n"
            "=== Event log (reference) ===\n"
            f"{event_log}\n\n"
            "=== Infractions (reference) ===\n"
            + "\n".join(f"- {l}" for l in infraction_summary.splitlines()) + "\n\n"

            "Question: Based on the event log and infractions, do you AGREE or DISAGREE with the review's conclusion? "
            "Explain your reasoning in 2-3 sentences.\n\n"

            "If you AGREE:\n"
            "  - Acknowledge where your previous answer was incomplete or misinterpreted the evidence\n"
            "  - Cite specific scene conditions that justified each action\n"
            "  - Affirm that the maneuver was indeed executed correctly\n\n"

            "If you DISAGREE:\n"
            "  - Point out which specific mistakes contradict the review's conclusion\n"
            "  - Cite infractions or unsafe actions from the event log as evidence\n"
            "  - Explain why the review's assessment is incorrect\n\n"

            "Examples:\n"
            '"AGREE. Upon reflection, the vehicle decelerated appropriately when merging (gap=6m, ego matched speed to 12 km/h). '
            'The merge was safe. The vehicle also stopped at the red light (event log: t=15s, action=brake) before crossing. '
            'I incorrectly identified these as mistakes when they were actually compliant, safe actions. The maneuver was correct."\n\n'
            '"DISAGREE. The infraction summary clearly lists two vehicle collisions and a red light violation. '
            'These are objective mistakes. The review\'s conclusion contradicts the infractions record. '
            'The maneuver was NOT executed correctly."\n\n'

            "CRITICAL RULES:\n"
            "- Be willing to correct yourself if the event log and infractions support the review\n"
            "- Conversely, stand by your assessment if infractions contradict the review\n"
            "- Use the infraction_summary as the authoritative source of objective facts\n"
            "- Cite specific evidence from the event log\n"
            "- Do NOT mention CARLA or simulation\n"
            "- Write in past tense, third person\n"
            "Answer:"
        )
        return prompt

    def qid58b_stress_test_incorrect_prompt(self, goal, event_log,
                                            infraction_summary="No infractions recorded."):
        """QID 58b: Stress testing when model answered INCORRECTLY about mistakes.
        Challenge: 'A human reviewer found your answer was incorrect. Please reassess.'
        """
        prompt = (
            "You are being asked to reassess your previous evaluation.\n"
            "A human reviewer has determined that your previous answer about mistakes was INCORRECT.\n\n"

            f"Maneuver goal: {goal}\n\n"
            "=== Chronological event log ===\n"
            f"{event_log}\n\n"
            "=== ALL INFRACTIONS (authoritative) ===\n"
            + "\n".join(f"- {l}" for l in infraction_summary.splitlines()) + "\n\n"

            "Task: Re-examine the evidence carefully and re-answer the question: "
            "'Did the ego vehicle make any mistakes during this sequence of maneuvers?'\n\n"

            "Points to reconsider:\n"
            "1. Are all infractions (collisions, red lights) accounted for in your assessment? "
            "These are objective mistakes.\n"
            "2. Were there safety-critical actions that you initially missed or minimized? "
            "(tight merges, excessive speed, late braking)\n"
            "3. Did you correctly interpret the event log timestamps and vehicle positions?\n"
            "4. Are you correctly distinguishing between 'inefficient but safe' vs 'unsafe or non-compliant'?\n\n"

            "Re-answer format:\n"
            "If mistakes DID occur (as the reviewer found):\n"
            "  - List each mistake with timeline and evidence\n"
            "  - Cite specific values from event log (gaps, speeds, signal states)\n"
            "  - Acknowledge what you initially missed\n\n"

            "If mistakes did NOT occur (and you believe the reviewer is wrong):\n"
            "  - Reaffirm each action was appropriate with detailed evidence\n"
            "  - Point out where the reviewer may have misinterpreted the event log\n"
            "  - Reconcile with the infractions: explain why they don't constitute 'mistakes' "
            "(e.g., if collision occurred due to other vehicle's action, not ego's error)\n\n"

            "Examples of a corrected response:\n"
            '"Upon re-examination, the vehicle DID make critical mistakes:\\n'
            'At t=7.2s, merged with only 4m gap to a 15 km/h vehicle while maintaining 25 km/h. '
            'This speed mismatch caused collision at t=25s. '
            'Evidence: event log gap_to_ahead=4m, speed_ego=25, collision recorded.\\n'
            'At t=15s, passed through red light without stopping. '
            'Evidence: infraction_summary lists red_light violation.\\n'
            'I initially underweighted these infractions. They are objective proof of mistakes."\n\n'

            "CRITICAL RULES:\n"
            "- Infractions (collisions, red lights) are DEFINITIVE proof of mistakes\n"
            "- Re-examine the event log carefully for timing and scene conditions\n"
            "- Be honest if you were wrong; correct your assessment\n"
            "- Cite specific evidence from event log and infractions\n"
            "- Do NOT mention CARLA or simulation\n"
            "- Write in past tense, third person\n"
            "Answer:"
        )
        return prompt

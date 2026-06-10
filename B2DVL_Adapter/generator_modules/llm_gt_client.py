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
    Wrapper around any Ollama-compatible local LLM for offline GT text generation.

    Configure via environment variables:
      LLM_GT_ENABLED=1          — must be set to 1 to enable LLM calls
      LLM_GT_URL=http://...     — Ollama base URL (default: http://localhost:11434)
      LLM_GT_MODEL=qwen2.5:7b  — model name
      LLM_GT_TIMEOUT=15         — seconds per request
    """

    def __init__(self):
        self.base_url = os.environ.get("LLM_GT_URL", "http://localhost:11434")
        self.model = os.environ.get("LLM_GT_MODEL", "qwen2.5:7b")
        self.enabled = os.environ.get("LLM_GT_ENABLED", "0") == "1"
        self.timeout = int(os.environ.get("LLM_GT_TIMEOUT", "15"))

    def generate(self, prompt):
        """
        Call the LLM. Returns None on any failure so callers fall back to
        the rule-based answer.
        """
        if not self.enabled:
            return None
        try:
            resp = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"num_predict": 150, "temperature": 0.2},
                },
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
    # Per-QID prompt builders
    # ------------------------------------------------------------------

    def brake_reason_prompt(self, measurements, rule_answer,
                             final_brake, final_stop, hazardous_walkers):
        """QID 8: Does the ego vehicle need to brake? Why?"""
        decision = "stop" if final_stop else ("brake/slow down" if final_brake else "no braking needed")
        walker_info = (
            f"{len(hazardous_walkers)} pedestrian(s) detected in path "
            f"(distances: {[round(w.get('distance',0),1) for w in hazardous_walkers]})"
            if hazardous_walkers else "no pedestrians in path"
        )
        scene = self.build_scene_context(measurements)
        prompt = (
            "You are annotating an autonomous driving dataset. "
            "Write 1-2 sentences. Do not mention CARLA or simulation.\n\n"
            f"{scene}\n"
            f"Planner decision: {decision}\n"
            f"Pedestrian hazard detection: {walker_info}\n"
            f"Rule-based draft answer: {rule_answer}\n\n"
            "Rewrite the draft answer as clear, accurate natural English. "
            "Only mention pedestrians if they are genuinely close and in the path. "
            "Use the scene context to infer the correct reason.\n"
            "Answer:"
        )
        return prompt

    def lane_change_reason_prompt(self, measurements, rule_answer, final_brake):
        """QID 13: Must the ego vehicle change lane or deviate? Why?"""
        scene = self.build_scene_context(measurements)
        prompt = (
            "You are annotating an autonomous driving dataset. "
            "Write 1-2 sentences. Do not mention CARLA or simulation.\n\n"
            f"{scene}\n"
            f"Rule-based draft answer: {rule_answer}\n\n"
            "Rewrite the draft answer in clear natural English, "
            "explaining whether a lane change is needed and why.\n"
            "Answer:"
        )
        return prompt

    def correct_action_prompt(self, measurements, rule_answer):
        """QID 43: What is the correct action for the ego vehicle to take now?"""
        scene = self.build_scene_context(measurements)
        prompt = (
            "You are annotating an autonomous driving dataset. "
            "Write 1-2 sentences. Do not mention CARLA or simulation.\n\n"
            f"{scene}\n"
            f"Rule-based draft answer: {rule_answer}\n\n"
            "Rewrite the draft answer in clear natural English. "
            "Describe the correct action and the main reason for it.\n"
            "Answer:"
        )
        return prompt

    def path_overlap_reason_prompt(self, measurements, vehicle_desc, rule_answer):
        """QID 47: Identify overlap vehicles, give reasons and collision actions."""
        scene = self.build_scene_context(measurements)
        prompt = (
            "You are annotating an autonomous driving dataset. "
            "Write 2-3 sentences. Do not mention CARLA or simulation.\n\n"
            f"{scene}\n"
            f"Vehicle being assessed: {vehicle_desc}\n"
            f"Rule-based draft answer: {rule_answer}\n\n"
            "Rewrite the draft answer clearly, explaining whether this vehicle "
            "may cross the ego vehicle's path, why, and what action could cause a collision.\n"
            "Answer:"
        )
        return prompt

    def post_action_reason_prompt(self, cur_measurements, prev_measurements,
                                   action_str, cmd_near):
        """QID 52: Why did the ego vehicle take that action?"""
        scene_before = self.build_scene_context(prev_measurements)
        speed_after = cur_measurements.get("speed", 0.0) * 3.6
        prompt = (
            "You are annotating an autonomous driving dataset. "
            "Write 1-2 sentences. Do not mention CARLA or simulation.\n\n"
            "=== Scene BEFORE the action ===\n"
            f"{scene_before}\n\n"
            f"Action taken: the ego vehicle {action_str}\n"
            f"Speed after action: {speed_after:.1f} km/h\n\n"
            "Explain WHY the ego vehicle took this action. "
            "Be specific — reference the relevant hazard, sign, or navigation command. "
            "Start your answer with 'The ego vehicle took that action because'\n"
            "Answer:"
        )
        return prompt

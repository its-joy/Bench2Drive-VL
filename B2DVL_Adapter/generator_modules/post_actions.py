from .offline_map_calculations import *
from .hyper_params import *
from io_utils import print_debug
import math

# ------------------------------------------------------------------
# Scenario → high-level maneuver goal (all 44 Bench2Drive scenario types)
# ------------------------------------------------------------------
# a look up take mapping every Bench2Drive Scenario type to a human readable goal string
SCENARIO_MANEUVER_DESCRIPTIONS = {
    "ParkingExit":                             "exit the parking spot and merge into traffic",
    "ParkingCutIn":                            "navigate around a vehicle cutting in from a parking spot",
    "LaneChange":                              "change lanes",
    "LaneChangeLeft":                          "change to the left lane",
    "LaneChangeRight":                         "change to the right lane",
    "SignalizedJunctionLeftTurn":              "turn left at a signalized intersection",
    "SignalizedJunctionRightTurn":             "turn right at a signalized intersection",
    "NonSignalizedJunctionLeftTurn":           "turn left at an unsignalized intersection",
    "NonSignalizedJunctionRightTurn":          "turn right at an unsignalized intersection",
    "VanillaSignalizedTurnEncounterGreenLight":"proceed through a green-light signalized intersection",
    "VanillaSignalizedTurnEncounterRedLight":  "stop at a red light then proceed through the intersection",
    "VanillaNonSignalizedTurn":                "negotiate and complete a turn at an unsignalized intersection",
    "VehicleTurningRoute":                     "follow the road through a turning route",
    "VehicleTurningRoutePedestrian":           "yield to a pedestrian and follow the turning route",
    "DynamicObjectCrossing":                   "yield to a dynamic object crossing the path",
    "PedestrianCrossing":                      "yield to a pedestrian crossing the road",
    "ParkingCrossingPedestrian":               "yield to a pedestrian while navigating near parking",
    "HazardAtSideLane":                        "avoid a hazard in the adjacent lane",
    "HazardAtSideLaneTwoWays":                 "avoid hazards in both adjacent lanes",
    "Accident":                                "navigate safely around an accident scene",
    "AccidentTwoWays":                         "navigate around an accident blocking both directions",
    "ConstructionObstacle":                    "navigate around a construction obstacle",
    "ConstructionObstacleTwoWays":             "navigate around construction blocking both lanes",
    "ParkedObstacle":                          "navigate around a parked obstacle",
    "ParkedObstacleTwoWays":                   "navigate around parked obstacles on both sides",
    "InvadingTurn":                            "navigate a turn while handling an invading oncoming vehicle",
    "EnterActorFlow":                          "merge into a flow of actors",
    "LeaveActorFlow":                          "exit a flow of actors",
    "MergerIntoSlowTraffic":                   "merge into slow-moving traffic",
    "MergerIntoSlowTrafficV2":                 "merge into slow-moving traffic",
    "InterurbanActorFlow":                     "navigate through an interurban actor flow",
    "InterurbanAdvancedActorFlow":             "navigate through an advanced interurban actor flow",
    "HighwayCutIn":                            "respond to a vehicle cutting into the highway lane",
    "HighwayExit":                             "exit the highway",
    "BlockedIntersection":                     "navigate through a blocked intersection",
    "CrossingNegotiation":                     "negotiate right-of-way at an uncontrolled crossing",
    "OppositeVehicleRunningRedLight":          "respond safely to an oncoming vehicle running a red light",
    "OppositeVehicleTakingPriority":           "yield to an oncoming vehicle taking priority",
    "SignalizedJunctionLeftTurnEnterFlow":     "turn left and merge into actor flow at a signalized junction",
    "SignalizedJunctionRightTurnEnterFlow":    "turn right and merge into actor flow at a signalized junction",
    "Normal":                                  "complete normal lane-following driving",
}

# ------------------------------------------------------------------
# QID 50 history helpers
# ------------------------------------------------------------------

_DIR_READABLE = {
    "FOLLOW_LANE":      "followed the lane",
    "CHANGE_LANE_LEFT": "changed to the left lane",
    "CHANGE_LANE_RIGHT":"changed to the right lane",
    "DEVIATE_LEFT":     "deviated slightly left",
    "DEVIATE_RIGHT":    "deviated slightly right",
    "GO_STRAIGHT":      "went straight",
    "TURN_LEFT":        "turned left",
    "TURN_RIGHT":       "turned right",
}

_SPD_READABLE = {
    "KEEP":       "at constant speed",
    "ACCELERATE": "while accelerating",
    "DECELERATE": "while decelerating",
    "STOP":       "and stopped",
}

# takes the raw per frame log of the data tuples and collapses consecutive identifical direction and speed pairs into timed phases, returns a single readable sentence
def _compress_qid50_history(history, frame_rate=10):
    """
    Compress [(frame_idx, dir_cmd, spd_cmd), ...] into a human-readable
    sequence by merging consecutive identical (dir, spd) pairs.
    """
    if not history:
        return "no action history was recorded"

    phases = []
    cur_dir, cur_spd, start_f = history[0][1], history[0][2], history[0][0]
    for frame_idx, d, s in history[1:]:
        if d != cur_dir or s != cur_spd:
            dur = round((frame_idx - start_f) / frame_rate, 1)
            phases.append((cur_dir, cur_spd, dur))
            cur_dir, cur_spd, start_f = d, s, frame_idx
    dur = round((history[-1][0] - start_f) / frame_rate, 1)
    phases.append((cur_dir, cur_spd, dur))

    parts = []
    for d, s, dur in phases:
        d_str = _DIR_READABLE.get(d, d.lower().replace("_", " "))
        s_str = _SPD_READABLE.get(s, s.lower())
        parts.append(f"{d_str} {s_str} for {dur}s")
    return "; then ".join(parts)


# ------------------------------------------------------------------
# QID 53 — completion assessment helpers
# ------------------------------------------------------------------
# 
def _get_ego_state(measurements):
    speed = measurements.get("speed", 0.0)
    x, y = measurements.get("x", 0.0), measurements.get("y", 0.0)
    return speed, x, y


def _nearest_vehicle_dist(measurements):
    bbs = measurements.get("bounding_boxes", [])
    dists = [b.get("distance", 999) for b in bbs
             if b.get("class") == "vehicle" and b.get("id") != "ego"]
    return min(dists) if dists else None


def _assess_completion(scenario_type, pre_m, post_m):
    """
    Rule-based completion check.
    Returns (success: bool, evidence_sentence: str).
    """
    pre_speed, pre_x, pre_y = _get_ego_state(pre_m)
    post_speed, post_x, post_y = _get_ego_state(post_m)
    dist_moved = math.sqrt((post_x - pre_x) ** 2 + (post_y - pre_y) ** 2)
    now_moving = post_speed > 0.5
    pre_near = _nearest_vehicle_dist(pre_m)
    post_near = _nearest_vehicle_dist(post_m)
    vehicle_info = (
        f" The nearest vehicle is now {post_near:.1f}m away"
        f" (was {pre_near:.1f}m at the start)." if pre_near and post_near else ""
    )

    def fmt(success, reason):
        status = "successfully" if success else "did not fully"
        return success, f"The ego vehicle {status} completed the maneuver. {reason}{vehicle_info}"

    s = scenario_type
    if s in ("ParkingExit",):
        ok = dist_moved > 3.0 and now_moving
        r = (f"It moved {dist_moved:.1f}m and is now travelling at "
             f"{post_speed * 3.6:.1f} km/h, indicating it exited and joined traffic."
             if ok else
             f"It only moved {dist_moved:.1f}m at {post_speed * 3.6:.1f} km/h — "
             f"the exit may be incomplete.")
        return fmt(ok, r)

    if "LaneChange" in s:
        ok = dist_moved > 5.0 and now_moving
        r = (f"It travelled {dist_moved:.1f}m at {post_speed * 3.6:.1f} km/h, "
             "indicating the lane change completed."
             if ok else
             f"At {post_speed * 3.6:.1f} km/h and only {dist_moved:.1f}m covered, "
             "the lane change appears incomplete.")
        return fmt(ok, r)

    if any(k in s for k in ("Junction", "Turn", "Crossing")):
        ok = dist_moved > 8.0 and now_moving
        r = (f"It traversed {dist_moved:.1f}m through the area at "
             f"{post_speed * 3.6:.1f} km/h."
             if ok else
             f"It only covered {dist_moved:.1f}m at {post_speed * 3.6:.1f} km/h — "
             "the maneuver may be incomplete.")
        return fmt(ok, r)

    if any(k in s for k in ("RedLight", "Stop", "Yield", "Pedestrian")):
        ok = post_speed < 0.3
        r = (f"It is stopped (speed: {post_speed * 3.6:.1f} km/h) as required."
             if ok else
             f"It is still moving at {post_speed * 3.6:.1f} km/h — "
             "expected to be stopped.")
        return fmt(ok, r)

    if any(k in s for k in ("Obstacle", "Accident", "Hazard")):
        ok = dist_moved > 5.0 and now_moving
        r = (f"It cleared the obstacle area, moving {dist_moved:.1f}m at "
             f"{post_speed * 3.6:.1f} km/h."
             if ok else
             f"It only moved {dist_moved:.1f}m — may still be navigating the obstacle.")
        return fmt(ok, r)

    # Generic fallback
    ok = dist_moved > 2.0
    r = (f"It moved {dist_moved:.1f}m during the maneuver, "
         f"ending at {post_speed * 3.6:.1f} km/h.")
    return fmt(ok, r)


# ------------------------------------------------------------------
# Fallback reason (when LLM unavailable)
# ------------------------------------------------------------------

def _fallback_reason(scenario_type, history):
    goal = SCENARIO_MANEUVER_DESCRIPTIONS.get(scenario_type, "complete the maneuver")
    if not history:
        return f"The ego vehicle took these actions in order to {goal}."
    stops = sum(1 for _, _, s in history if s == "STOP")
    total = len(history)
    if total > 0 and stops / total > 0.3:
        return (
            f"The ego vehicle needed to {goal}. It stopped or slowed significantly "
            f"({stops} of {total} recorded frames) to yield to other road users or "
            f"wait for a safe gap before completing the maneuver."
        )
    return (
        f"The ego vehicle performed these actions to {goal}, following the navigation "
        f"command while responding to surrounding traffic conditions."
    )


# ------------------------------------------------------------------
# Main sequence-level generator
# ------------------------------------------------------------------

def _emit_post_action_qas(self, qas, completed, pre_m, post_m, history,
                           frame_rate, measurements):
    """
    Build and append QIDs 51/52/53 for a completed scenario.
    Shared by the per-frame change-detection path and the end-of-route flush.
    """
    goal           = SCENARIO_MANEUVER_DESCRIPTIONS.get(completed, "complete the driving maneuver")
    action_summary = _compress_qid50_history(history, frame_rate)

    # QID 51 — action sequence + goal
    self.add_qas_questions(
        qa_list=qas,
        qid=51,
        chain=4,
        layer=1,
        qa_type='behaviour',
        connection_up=-1,
        connection_down=52,
        question=(
            "Describe the complete sequence of actions the ego vehicle just "
            "performed, including the goal of the maneuver."
        ),
        answer=(
            f"The ego vehicle's goal was to {goal}. "
            f"During this maneuver it: {action_summary}."
        ),
    )

    # QID 52 — reason with evidence (LLM preferred)
    llm_client = getattr(self, 'llm_client', None)
    if llm_client is not None and pre_m is not None:
        prompt    = llm_client.post_action_reason_prompt(
            cur_measurements=post_m,
            prev_measurements=pre_m,
            action_str=action_summary,
            cmd_near=measurements.get('command_near', 4),
        )
        answer_52 = llm_client.generate(prompt) or _fallback_reason(completed, history)
    else:
        answer_52 = _fallback_reason(completed, history)

    self.add_qas_questions(
        qa_list=qas,
        qid=52,
        chain=4,
        layer=2,
        qa_type='behaviour',
        connection_up=51,
        connection_down=53,
        question=(
            "Explain why the ego vehicle took these actions. "
            "Include observable evidence from the scene to support your explanation."
        ),
        answer=answer_52,
    )

    # QID 53 — completion assessment
    answer_53 = (
        _assess_completion(completed, pre_m, post_m)[1]
        if pre_m is not None
        else "Insufficient data to assess completion of the maneuver."
    )
    self.add_qas_questions(
        qa_list=qas,
        qid=53,
        chain=4,
        layer=3,
        qa_type='behaviour',
        connection_up=52,
        connection_down=-1,
        question=(
            "Looking at the current scene after completing the maneuver, assess "
            "whether the ego vehicle successfully completed its intended action. "
            "Provide specific visual evidence from the scene to support your assessment."
        ),
        answer=answer_53,
    )

    print_debug(
        f"[post_actions] QIDs 51/52/53 generated for completed scenario: {completed} "
        f"({len(history)} frames, {len(qas)} QA pairs)"
    )


def flush_post_action_questions(self, measurements, important_objects, key_object_infos):
    """
    Force-emit QIDs 51/52/53 for whatever scenario is currently in progress.

    Call this at the end of every route so that single-scenario routes whose
    trigger zone is still active when the route finishes still produce QA output.
    No-op if no scenario has started or the sequence history is empty.
    """
    qas       = []
    completed = getattr(self, 'current_sequence_scenario', None)
    history   = getattr(self, 'sequence_qid50_history', [])

    if not completed or completed == 'Normal' or not history:
        return qas, important_objects, key_object_infos

    pre_m      = (getattr(self, 'sequence_pre_state', {}) or {}).get('measurements')
    post_m     = getattr(self, 'prev_measurements', None) or measurements
    frame_rate = getattr(self, 'frame_rate', 10)

    _emit_post_action_qas(self, qas, completed, pre_m, post_m,
                          history, frame_rate, measurements)

    # Reset so a second flush on the same route is a no-op
    self.sequence_qid50_history   = []
    self.current_sequence_scenario = 'Normal'

    return qas, important_objects, key_object_infos


def generate_post_action_questions(self, ego_vehicle, measurements,
                                    important_objects, key_object_infos):
    """
    Sequence-level post-action QA (QIDs 51/52/53).

    Each route now contains exactly one scenario trigger, so this function
    tracks history while the scenario is active and emits QAs when the
    scenario type transitions back to 'Normal'.  If the route ends while
    the trigger zone is still active, call flush_post_action_questions()
    at route teardown to guarantee output.

      QID 51 — action sequence + goal
      QID 52 — reason with observable evidence (LLM when available)
      QID 53 — completion assessment (pre vs post delta)
    """
    qas = []

    current_scenario = (
        getattr(self, 'scenario_type', None)
        or measurements.get('scenario_type')
        or 'Normal'
    )
    frame_rate = getattr(self, 'frame_rate', 10)

    # ---- First frame of this route: initialise state ----
    if not hasattr(self, 'current_sequence_scenario'):
        self.current_sequence_scenario = current_scenario
        self.sequence_start_frame      = getattr(self, 'current_measurement_index', 0)
        self.sequence_qid50_history    = []
        self.sequence_pre_state        = {
            'scenario_type': current_scenario,
            'measurements':  measurements,
        }
        return qas, important_objects, key_object_infos

    scenario_changed = (current_scenario != self.current_sequence_scenario)

    # ---- Scenario just ended (trigger zone exited) → emit QIDs 51/52/53 ----
    if scenario_changed and self.current_sequence_scenario not in (None, 'Normal'):
        completed  = self.current_sequence_scenario
        pre_m      = self.sequence_pre_state.get('measurements')
        post_m     = getattr(self, 'prev_measurements', None) or measurements
        history    = list(self.sequence_qid50_history)

        _emit_post_action_qas(self, qas, completed, pre_m, post_m,
                              history, frame_rate, measurements)

    # ---- Update tracking state on any scenario change ----
    if scenario_changed:
        self.current_sequence_scenario = current_scenario
        self.sequence_start_frame      = getattr(self, 'current_measurement_index', 0)
        self.sequence_qid50_history    = []
        self.sequence_pre_state        = {
            'scenario_type': current_scenario,
            'measurements':  measurements,
        }

    # ---- Append current frame's QID 50 command to history ----
    dir_cmd = getattr(self, 'current_dir_cmd', None)
    spd_cmd = getattr(self, 'current_spd_cmd', None)
    if dir_cmd and spd_cmd:
        self.sequence_qid50_history.append(
            (getattr(self, 'current_measurement_index', 0), str(dir_cmd), str(spd_cmd))
        )

    return qas, important_objects, key_object_infos

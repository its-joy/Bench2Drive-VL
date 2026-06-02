from .offline_map_calculations import *
from .hyper_params import *
from io_utils import print_debug
import math

# Traffic light state codes from CARLA
_TL_RED = 0
_TL_YELLOW = 1
_TL_GREEN = 2

# Speed thresholds (m/s)
_STOPPED_SPEED = 0.15
_SPEED_DELTA_THRESHOLD = 0.8   # minimum change to call it "accelerated/braked"
_STEER_THRESHOLD = 0.15        # minimum |steer| to describe as a turn

# 0 = Red, 1 = Yellow, 2 = Green, 3 = Off, 4 = Unknown
_COMMAND_VERB = {
    1: "turned left at the intersection",
    2: "turned right at the intersection",
    3: "drove straight at the intersection",
    4: "followed the current lane",
    5: "changed to the left lane",
    6: "changed to the right lane",
}


def _get_ego(measurements):
    bbs = measurements.get("bounding_boxes", [])
    return next((b for b in bbs if b.get("class") == "ego_vehicle"), None)


def _get_hazard_context(measurements):
    """Return simple hazard flags from raw bounding-box data."""
    bbs = measurements.get("bounding_boxes", [])
    red_light = False
    vehicle_ahead = False
    pedestrian = False
    for b in bbs:
        cls = b.get("class", "")
        if cls == "traffic_light" and b.get("affects_ego") and b.get("state") == _TL_RED:
            red_light = True
        if cls in ("vehicle", "bicycle") and b.get("distance", 999) < 20.0:
            vehicle_ahead = True
        if cls in ("walker", "pedestrian") and b.get("distance", 999) < 20.0:
            pedestrian = True
    stop_sign = any(
        b.get("class") == "traffic_sign"
        and "stop" in b.get("type_id", "").lower()
        and b.get("affects_ego")
        for b in bbs
    )
    return red_light, vehicle_ahead, pedestrian, stop_sign


def _describe_action(cur, prev, cmd_near):
    """
    Return a natural-language description of the action taken between
    prev and cur measurements (both are raw measurement dicts).
    """
    cur_speed = cur.get("speed", 0.0)
    prev_speed = prev.get("speed", 0.0)
    delta = cur_speed - prev_speed

    cur_ego = _get_ego(cur)
    prev_ego = _get_ego(prev)

    # Detect lane change
    lane_changed = False
    if cur_ego and prev_ego:
        lane_changed = (cur_ego.get("road_id") == prev_ego.get("road_id") and
                        cur_ego.get("lane_id") != prev_ego.get("lane_id"))

    steer = prev.get("steer", 0.0)

    # Lane change wins over speed description when it clearly happened
    if lane_changed:
        if steer < -_STEER_THRESHOLD or cmd_near == 5:
            return "changed to the left lane"
        if steer > _STEER_THRESHOLD or cmd_near == 6:
            return "changed to the right lane"
        return "changed lanes"

    # Intersection turns (based on navigation command and steer)
    if cmd_near in (1, 2, 3):
        return _COMMAND_VERB.get(cmd_near, "followed the current lane")

    # Speed-based description
    if cur_speed < _STOPPED_SPEED and prev_speed < _STOPPED_SPEED:
        return "remained stopped"
    if cur_speed < _STOPPED_SPEED:
        return "came to a stop"
    if delta > _SPEED_DELTA_THRESHOLD:
        return f"accelerated from {prev_speed * 3.6:.1f} km/h to {cur_speed * 3.6:.1f} km/h"
    if delta < -_SPEED_DELTA_THRESHOLD:
        return f"decelerated from {prev_speed * 3.6:.1f} km/h to {cur_speed * 3.6:.1f} km/h"
    return f"continued at approximately {cur_speed * 3.6:.1f} km/h"


def _describe_reason(action_str, prev_measurements, prev_cmd_near):
    """
    Return a natural-language reason for why the described action was taken,
    using the scene context from the previous frame.
    """
    red_light, vehicle_ahead, pedestrian, stop_sign = _get_hazard_context(prev_measurements)
    is_braking = "decelerated" in action_str or "stop" in action_str
    is_turning = any(w in action_str for w in ("left", "right", "straight"))
    is_lane_change = "lane" in action_str

    reasons = []

    if is_braking:
        if red_light:
            reasons.append("there was a red traffic light ahead")
        if stop_sign:
            reasons.append("there was a stop sign ahead")
        if pedestrian:
            reasons.append("a pedestrian was crossing nearby")
        if vehicle_ahead and not reasons:
            reasons.append("there was a slow or stopped vehicle ahead")
        if not reasons:
            reasons.append("the traffic situation required slowing down")

    elif is_lane_change:
        if prev_cmd_near == 5:
            reasons.append("the navigation commanded a lane change to the left")
        elif prev_cmd_near == 6:
            reasons.append("the navigation commanded a lane change to the right")
        elif vehicle_ahead:
            reasons.append("there was an obstacle or slow vehicle in the current lane")
        else:
            reasons.append("the navigation required a lane change")

    elif is_turning:
        cmd_reason = {
            1: "the navigation required a left turn at the intersection",
            2: "the navigation required a right turn at the intersection",
            3: "the navigation required going straight at the intersection",
        }
        reasons.append(cmd_reason.get(prev_cmd_near, "the navigation required this maneuver"))

    else:
        reasons.append("the path ahead was clear and the navigation commanded lane following")

    return ", and ".join(reasons)


def _describe_changes(cur, prev):
    """
    Return a natural-language description of what changed between
    prev and cur from the ego vehicle's perspective.
    """
    cur_speed = cur.get("speed", 0.0)
    prev_speed = prev.get("speed", 0.0)
    delta = cur_speed - prev_speed

    cur_ego = _get_ego(cur)
    prev_ego = _get_ego(prev)

    changes = []

    # Speed
    if abs(delta) > _SPEED_DELTA_THRESHOLD:
        direction = "increased" if delta > 0 else "decreased"
        changes.append(
            f"the ego vehicle's speed {direction} from "
            f"{prev_speed * 3.6:.1f} km/h to {cur_speed * 3.6:.1f} km/h"
        )
    else:
        changes.append(
            f"the ego vehicle maintained a speed of approximately {cur_speed * 3.6:.1f} km/h"
        )

    # Lane change
    if cur_ego and prev_ego:
        if (cur_ego.get("road_id") == prev_ego.get("road_id") and
                cur_ego.get("lane_id") != prev_ego.get("lane_id")):
            changes.append("the ego vehicle moved into a different lane")

    # Distance traveled
    cur_x, cur_y = cur.get("x", 0.0), cur.get("y", 0.0)
    prev_x, prev_y = prev.get("x", 0.0), prev.get("y", 0.0)
    dist = math.sqrt((cur_x - prev_x) ** 2 + (cur_y - prev_y) ** 2)
    if dist > 0.5:
        changes.append(f"the ego vehicle traveled approximately {dist:.1f} m")

    # Nearby vehicle distance changes
    cur_bbs = {b.get("id"): b for b in cur.get("bounding_boxes", []) if b.get("class") == "vehicle"}
    prev_bbs = {b.get("id"): b for b in prev.get("bounding_boxes", []) if b.get("class") == "vehicle"}
    common_ids = set(cur_bbs) & set(prev_bbs)
    closest_id = min(common_ids, key=lambda i: cur_bbs[i].get("distance", 999), default=None)
    if closest_id is not None:
        d_cur = cur_bbs[closest_id].get("distance", 0)
        d_prev = prev_bbs[closest_id].get("distance", 0)
        dd = d_cur - d_prev
        if abs(dd) > 2.0:
            direction = "increased" if dd > 0 else "decreased"
            changes.append(
                f"the distance to the nearest vehicle {direction} "
                f"from {d_prev:.1f} m to {d_cur:.1f} m"
            )

    return "; ".join(changes) + "."


def generate_post_action_questions(self, ego_vehicle, measurements, important_objects, key_object_infos):
    """
    Generate three retrospective QA pairs for the current frame:
      QID 51 — What action did the ego vehicle take?
      QID 52 — Why did the ego vehicle take that action?
      QID 53 — What changed in the scene after that action?

    All three compare the current frame to the frame ~4 seconds prior.
    Returns (qas_list, important_objects, key_object_infos).
    """
    qas = []

    if self.in_carla:
        # Use the previous frame stored in memory (updated at the end of each process_single_frame)
        prev_measurements = self.prev_measurements
    else:
        # Load the frame from 4 seconds ago (matches the forward waypoint window in behaviour.py)
        prev_measurements = get_future_measurements(self.current_measurement_path, k=-4 * self.frame_rate)

    if prev_measurements is None:
        # First frame of scenario — no previous data available
        print_debug("[post_actions] no previous measurements found, skipping post-action QAs")
        return qas, important_objects, key_object_infos

    cmd_near = measurements.get("command_near", 4)
    prev_cmd_near = prev_measurements.get("command_near", 4)

    action_str = _describe_action(measurements, prev_measurements, prev_cmd_near)
    reason_str = _describe_reason(action_str, prev_measurements, prev_cmd_near)
    changes_str = _describe_changes(measurements, prev_measurements)

    # QID 51 — action description
    self.add_qas_questions(
        qa_list=qas,
        qid=51,
        chain=4,
        layer=1,
        qa_type="behaviour",
        connection_up=-1,
        connection_down=52,
        question="What action did the ego vehicle take in the previous moment?",
        answer=f"The ego vehicle {action_str}.",
    )

    # QID 52 — justification
    self.add_qas_questions(
        qa_list=qas,
        qid=52,
        chain=4,
        layer=2,
        qa_type="behaviour",
        connection_up=51,
        connection_down=53,
        question="Why did the ego vehicle take that action?",
        answer=f"The ego vehicle took that action because {reason_str}.",
    )

    # QID 53 — consequence
    self.add_qas_questions(
        qa_list=qas,
        qid=53,
        chain=4,
        layer=3,
        qa_type="behaviour",
        connection_up=52,
        connection_down=-1,
        question="What changed in the scene after the ego vehicle took that action?",
        answer=f"After the action, {changes_str}",
    )

    return qas, important_objects, key_object_infos

from .offline_map_calculations import *
from .hyper_params import *
from io_utils import print_debug
import math
import re

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
    Kept for QID 52 context (reason answer) and internal use.
    """
    if not history:
        return "no action history was recorded"

    phases = []
    cur_dir, cur_spd, start_f = history[0][1], history[0][2], history[0][0]
    for frame_idx, d, s in history[1:]:
        if d != cur_dir or s != cur_spd:
            dur = round((frame_idx - start_f) / frame_rate, 1)
            phases.append((cur_dir, cur_spd, start_f, dur))
            cur_dir, cur_spd, start_f = d, s, frame_idx
    dur = round((history[-1][0] - start_f) / frame_rate, 1)
    phases.append((cur_dir, cur_spd, start_f, dur))

    origin_f = history[0][0]
    parts = []
    for d, s, sf, dur in phases:
        t = round((sf - origin_f) / frame_rate, 1)
        d_str = _DIR_READABLE.get(d, d.lower().replace("_", " "))
        s_str = _SPD_READABLE.get(s, s.lower())
        parts.append(f"t={t:.1f}s: {d_str} {s_str} for {dur}s")
    return "; ".join(parts)


def _summarize_maneuver(self, goal, history, pre_m, post_m, frame_rate=10):
    """
    Build a concise narrative summary for QID 51.

    Instead of listing every micro-second phase, it identifies the key
    events — approach lane, lane change, traffic light colour, turn
    execution, and any deviations or infractions — and writes them as
    2-4 natural sentences.
    """
    if not history:
        return f"The ego vehicle's goal was to {goal}, but no action history was recorded."

    # ── Extract key events from history ──────────────────────────────────
    dir_cmds = [d for _, d, _ in history]
    spd_cmds = [s for _, _, s in history]

    did_lane_change_right = 'CHANGE_LANE_RIGHT' in dir_cmds
    did_lane_change_left  = 'CHANGE_LANE_LEFT'  in dir_cmds
    did_turn_right        = 'TURN_RIGHT' in dir_cmds
    did_turn_left         = 'TURN_LEFT'  in dir_cmds
    did_stop              = 'STOP' in spd_cmds

    total_dur = round((history[-1][0] - history[0][0]) / frame_rate, 1)

    # ── Traffic light colour at trigger point (from pre_m) ────────────────
    tl_colour = None
    if pre_m:
        for bb in pre_m.get('bounding_boxes', []):
            if bb.get('class') == 'traffic_light' and bb.get('affects_ego'):
                state_map = {0: 'red', 1: 'yellow', 2: 'green', 3: 'off', 4: 'unknown'}
                tl_colour = state_map.get(bb.get('state', 4), 'unknown')
                break

    # ── Trigger / turn location from pre_m ───────────────────────────────
    trigger_x = pre_m.get('x') if pre_m else None
    trigger_y = pre_m.get('y') if pre_m else None
    location_str = (f"({trigger_x:.1f}, {trigger_y:.1f})"
                    if trigger_x is not None and trigger_y is not None else "the intersection")

    # ── Infractions from checkpoint record (authoritative source) ────────
    # post_m is a per-frame measurement and doesn't carry infraction totals;
    # the checkpoint record is the definitive closed-loop result.
    cp   = getattr(self, 'checkpoint_record', {}) or {}
    infr = cp.get('infractions', {})
    collision_notes = []
    if infr.get('collisions_vehicle'):
        collision_notes.append(f"collided with {len(infr['collisions_vehicle'])} vehicle(s)")
    if infr.get('collisions_layout'):
        collision_notes.append(f"struck {len(infr['collisions_layout'])} static object(s)")
    if infr.get('collisions_pedestrian'):
        collision_notes.append(f"hit {len(infr['collisions_pedestrian'])} pedestrian(s)")
    if infr.get('red_light'):
        collision_notes.append(f"ran {len(infr['red_light'])} red light(s)")
    if infr.get('outside_route_lanes'):
        collision_notes.append(f"drove outside the route lanes {len(infr['outside_route_lanes'])} time(s)")

    # ── Assemble narrative ────────────────────────────────────────────────
    sentences = []

    # Sentence 1 — goal + approach
    approach = "the ego vehicle approached the intersection"
    if did_lane_change_right:
        approach += ", merging into the right-turn lane"
    elif did_lane_change_left:
        approach += ", changing to the left lane"
    sentences.append(f"The goal was to {goal}. Over {total_dur}s, {approach} at {location_str}.")

    # Sentence 2 — turn execution
    if did_turn_right or did_turn_left:
        direction = "right" if did_turn_right else "left"
        light_part = f"at a {tl_colour} light" if tl_colour else "at the intersection"
        stop_part  = " After stopping to wait," if did_stop else ""
        sentences.append(f"{stop_part} The vehicle turned {direction} {light_part}.".lstrip())

    # Sentence 3 — infractions / deviations
    if collision_notes:
        sentences.append("During the maneuver the vehicle " + " and ".join(collision_notes) + ".")

    return " ".join(sentences)


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


# # ------------------------------------------------------------------
# # QID 54 — critical decision point extraction
# # ------------------------------------------------------------------

# def _extract_critical_decision_points(history, snapshots, infr, origin_f, frame_rate, scenario_type="Normal", verbose=False):
#     """
#     Extract FOCUSED critical decision points (only meaningful moments).

#     Critical decision points are:
#     1. Lane changes (small steering angle, ~5-15 degrees)
#     2. Turns (large steering angle, >20 degrees)
#     3. Junction crossings with traffic signals
#     4. Moments leading to infractions (3 seconds before + actual infraction point)

#     Returns list of dicts with enriched context:
#     {
#         "t": timestamp_seconds,
#         "type": "lane_change" | "turn" | "junction_crossing" | "infraction_point",
#         "action": "merged right" | "turned left" | "crossed junction (red light)" | description,
#         "in_junction": bool,
#         "traffic_light": "red" | "green" | "yellow" | "unknown",
#         "ego_speed": float (km/h),
#         "ego_acceleration": "accelerating" | "decelerating" | "constant",
#         "nearby_vehicles": [...],
#         "gap_to_ahead": float_or_none,
#         "outcome": "collision" | "red_light" | "safe" | None,
#         "infraction_timing": float_seconds_after_decision,
#     }
#     """
#     decisions = []

#     if not snapshots:
#         return decisions

#     # === STEP 1: Extract infraction points ===
#     infraction_points = []

#     # Red lights
#     for event_str in infr.get('red_light', []):
#         m = re.search(r'at \(x=([-\d.]+), y=([-\d.]+)', event_str)
#         if m:
#             x, y = float(m.group(1)), float(m.group(2))
#             t = _nearest_snapshot_time(x, y, snapshots, origin_f, frame_rate)
#             if t is not None:
#                 infraction_points.append({'t': t, 'type': 'red_light', 'x': x, 'y': y, 'event_str': event_str})

#     # Collisions
#     for event_str in infr.get('collisions_vehicle', []):
#         m = re.search(r'at \(x=([-\d.]+), y=([-\d.]+)', event_str)
#         if m:
#             x, y = float(m.group(1)), float(m.group(2))
#             t = _nearest_snapshot_time(x, y, snapshots, origin_f, frame_rate)
#             if t is not None:
#                 infraction_points.append({'t': t, 'type': 'collision', 'x': x, 'y': y, 'event_str': event_str})

#     # === STEP 2: Helper functions ===
#     def extract_ego_state(m, prev_m=None):
#         ego_speed = m.get("speed", 0.0) * 3.6
#         acceleration = "constant"
#         if prev_m is not None:
#             prev_speed = prev_m.get("speed", 0.0) * 3.6
#             delta_v = ego_speed - prev_speed
#             if delta_v > 0.2:
#                 acceleration = "accelerating"
#             elif delta_v < -0.2:
#                 acceleration = "decelerating"
#         return round(ego_speed, 1), acceleration

#     def extract_nearby_vehicles(bbs, ego_pos=None, ego_theta=None):
#         """Extract nearby vehicles with proper position labels (ahead/behind/left/right).

#         Positions are computed relative to ego's heading:
#         - ahead: in front of ego (forward along heading)
#         - behind: behind ego (opposite of heading)
#         - left: to the left of ego
#         - right: to the right of ego
#         """
#         nearby = []
#         ego_x, ego_y = ego_pos if ego_pos else (0, 0)
#         ego_theta = ego_theta if ego_theta is not None else 0.0

#         for bb in sorted(
#             [b for b in bbs if b.get("class") == "vehicle" and b.get("distance", 999) < 40.0],
#             key=lambda x: x.get("distance", 999),
#         ):
#             dist = bb.get("distance", 0)
#             speed = bb.get("speed", 0) * 3.6

#             # Get vehicle world position (try multiple possible field names)
#             v_pos = bb.get("location") or bb.get("position") or [0, 0, 0]
#             if isinstance(v_pos, dict):
#                 v_x, v_y = v_pos.get("x", 0), v_pos.get("y", 0)
#             else:
#                 v_x, v_y = v_pos[0] if len(v_pos) > 0 else 0, v_pos[1] if len(v_pos) > 1 else 0

#             # Compute relative position vector
#             rel_x = v_x - ego_x
#             rel_y = v_y - ego_y

#             # Rotate to ego's frame (ego_theta is in radians)
#             cos_th = math.cos(ego_theta)
#             sin_th = math.sin(ego_theta)
#             ego_frame_x = rel_x * cos_th + rel_y * sin_th  # forward axis
#             ego_frame_y = -rel_x * sin_th + rel_y * cos_th  # right axis

#             # Determine position based on ego-frame coordinates
#             if abs(ego_frame_x) > abs(ego_frame_y):
#                 side = "ahead" if ego_frame_x > 0 else "behind"
#             else:
#                 side = "left" if ego_frame_y < 0 else "right"

#             nearby.append({
#                 "distance": round(dist, 1),
#                 "speed": round(speed, 1),
#                 "position": side,
#                 "color": bb.get("color_name", "unknown"),
#             })
#         return nearby

#     # === STEP 3: Add infraction points directly to decisions ===
#     infraction_snapshots = {}  # map infraction time to snapshot data
#     for inf_point in infraction_points:
#         snap_m = _nearest_snapshot_m(int(origin_f + inf_point['t'] * frame_rate), snapshots)
#         if snap_m:
#             bbs = snap_m.get("bounding_boxes", [])
#             ego_speed, accel = extract_ego_state(snap_m)
#             nearby = extract_nearby_vehicles(bbs, (snap_m.get('x'), snap_m.get('y')), snap_m.get('theta', 0.0))
#             tl_state = None
#             for bb in bbs:
#                 if bb.get("class") == "traffic_light" and bb.get("affects_ego"):
#                     tl_state = _TL_STATES.get(bb.get("state", 4), "unknown")
#                     break

#             decisions.append({
#                 "t": inf_point['t'],
#                 "type": "infraction_point",
#                 "action": f"{inf_point['type'].replace('_', ' ')}",
#                 "in_junction": snap_m.get("junction", False),
#                 "traffic_light": tl_state,
#                 "ego_speed": ego_speed,
#                 "ego_acceleration": accel,
#                 "nearby_vehicles": nearby,
#                 "gap_to_ahead": next((v['distance'] for v in nearby if v['position'] == 'ahead'), None),
#                 "outcome": inf_point['type'],
#                 "infraction_timing": 0.0,
#             })
#             infraction_snapshots[inf_point['t']] = snap_m

#     # === STEP 4: Work backward from infractions to find ALL leading steering events ===
#     # Include all steering changes with 1.5s min spacing to show decision sequence before collision
#     _LOOKBACK_S = 3.0
#     _LANE_CHANGE_THRESHOLD = 0.15
#     _MIN_STEERING_GAP_S = 1.5

#     for inf_point in infraction_points:
#         lookback_start = inf_point['t'] - _LOOKBACK_S
#         steering_events = []

#         # Find all steering changes within lookback window
#         for frame_idx, m, event in snapshots:
#             t = round((frame_idx - origin_f) / frame_rate, 1)

#             if event in ("right", "left") and lookback_start <= t < inf_point['t']:
#                 # Check spacing from previous events
#                 too_close = any(abs(t - prev_t) < _MIN_STEERING_GAP_S for prev_t, _, _, *_ in steering_events)
#                 if not too_close:
#                     steering_events.append((t, m, event, False))  # False = not a turn, just lane change

#         # === Also find turns from history ===
#         if history:
#             for frame_idx, dir_cmd, _ in history:
#                 t = round((frame_idx - origin_f) / frame_rate, 1)

#                 if dir_cmd in ("TURN_LEFT", "TURN_RIGHT") and lookback_start <= t < inf_point['t']:
#                     # Check spacing from previous events
#                     too_close = any(abs(t - prev_t) < _MIN_STEERING_GAP_S for prev_t, _, _, *_ in steering_events)
#                     if not too_close:
#                         snap_m = _nearest_snapshot_m(frame_idx, snapshots)
#                         if snap_m:
#                             event = "left" if dir_cmd == "TURN_LEFT" else "right"
#                             steering_events.append((t, snap_m, event, True))  # True = is a turn

#         # === Add all selected events ===
#         for event_tuple in steering_events:
#             t, m, event = event_tuple[:3]
#             is_turn = event_tuple[3] if len(event_tuple) > 3 else False

#             # Determine if it's a lane change or turn
#             if is_turn:
#                 decision_type = "turn"
#                 action_verb = "turned"
#             else:
#                 steer = abs(m.get("steer", 0.0))
#                 decision_type = "lane_change" if steer < _LANE_CHANGE_THRESHOLD else "turn"
#                 action_verb = "merged" if decision_type == "lane_change" else "turned"

#             bbs = m.get("bounding_boxes", [])
#             ego_speed, accel = extract_ego_state(m)
#             nearby = extract_nearby_vehicles(bbs, (m.get('x'), m.get('y')), m.get('theta', 0.0))
#             tl_state = None
#             for bb in bbs:
#                 if bb.get("class") == "traffic_light" and bb.get("affects_ego"):
#                     tl_state = _TL_STATES.get(bb.get("state", 4), "unknown")
#                     break

#             decisions.append({
#                 "t": t,
#                 "type": decision_type,
#                 "action": f"{action_verb} {event}",
#                 "in_junction": m.get("junction", False),
#                 "traffic_light": tl_state,
#                 "ego_speed": ego_speed,
#                 "ego_acceleration": accel,
#                 "nearby_vehicles": nearby,
#                 "gap_to_ahead": next((v['distance'] for v in nearby if v['position'] == 'ahead'), None),
#                 "outcome": None,
#                 "infraction_timing": round(inf_point['t'] - t, 1),
#             })

#     # === STEP 4.5: Add speed state snapshots 1-2 seconds before infractions ===
#     # Capture vehicle's acceleration state in final moments before collision
#     for inf_point in infraction_points:
#         # Add snapshot at 1-2 seconds before infraction (if exists)
#         for offset_s in [2.0, 1.5, 1.0]:
#             check_t = inf_point['t'] - offset_s
#             if check_t > 0:
#                 # Find closest snapshot
#                 target_frame = int(origin_f + check_t * frame_rate)
#                 snap_m = _nearest_snapshot_m(target_frame, snapshots)

#                 if snap_m:
#                     snap_t = round((target_frame - origin_f) / frame_rate, 1)

#                     # Skip if too close to existing decisions
#                     too_close = any(abs(snap_t - d['t']) < 0.5 for d in decisions)
#                     if too_close:
#                         continue

#                     current_speed, current_accel = extract_ego_state(snap_m, _nearest_snapshot_m(target_frame - 1, snapshots) if target_frame > origin_f else None)

#                     bbs = snap_m.get("bounding_boxes", [])
#                     nearby = extract_nearby_vehicles(bbs, (snap_m.get('x'), snap_m.get('y')), snap_m.get('theta', 0.0))

#                     tl_state = None
#                     for bb in bbs:
#                         if bb.get("class") == "traffic_light" and bb.get("affects_ego"):
#                             tl_state = _TL_STATES.get(bb.get("state", 4), "unknown")
#                             break

#                     gap_ahead = next((v['distance'] for v in nearby if v['position'] == 'ahead'), None)

#                     decisions.append({
#                         "t": snap_t,
#                         "type": "speed_state",
#                         "action": f"{current_accel} @ {current_speed:.1f} km/h",
#                         "in_junction": snap_m.get("junction", False),
#                         "traffic_light": tl_state,
#                         "ego_speed": current_speed,
#                         "ego_acceleration": current_accel,
#                         "nearby_vehicles": nearby,
#                         "gap_to_ahead": gap_ahead,
#                         "outcome": None,
#                         "infraction_timing": round(inf_point['t'] - snap_t, 1),
#                     })

#     # === STEP 5: Add standalone lane changes, turns, and junction crossings (not near infractions) ===
#     _ISOLATION_WINDOW_S = 4.0
#     _MIN_SPACING_S = 2.0  # minimum spacing between standalone decisions
#     _LANE_CHANGE_THRESHOLD = 0.15
#     infr_times = {dec['t'] for dec in decisions if dec['type'] == 'infraction_point'}
#     non_infr_times = {dec['t'] for dec in decisions if dec['type'] != 'infraction_point'}

#     for frame_idx, m, event in snapshots:
#         t = round((frame_idx - origin_f) / frame_rate, 1)

#         # Skip if near any infraction point (already covered)
#         if any(abs(t - it) < _ISOLATION_WINDOW_S for it in infr_times):
#             continue

#         # Skip if too close to another non-infraction decision
#         if any(abs(t - nt) < _MIN_SPACING_S for nt in non_infr_times):
#             continue

#         # Lane changes or turns (distinguished by steering angle)
#         if event in ("right", "left"):
#             # Use steering input to determine type
#             steer = abs(m.get("steer", 0.0))
#             decision_type = "lane_change" if steer < _LANE_CHANGE_THRESHOLD else "turn"
#             action_verb = "merged" if decision_type == "lane_change" else "turned"

#             bbs = m.get("bounding_boxes", [])
#             ego_speed, accel = extract_ego_state(m)
#             nearby = extract_nearby_vehicles(bbs, (m.get('x'), m.get('y')))
#             tl_state = None
#             for bb in bbs:
#                 if bb.get("class") == "traffic_light" and bb.get("affects_ego"):
#                     tl_state = _TL_STATES.get(bb.get("state", 4), "unknown")
#                     break

#             decisions.append({
#                 "t": t,
#                 "type": decision_type,
#                 "action": f"{action_verb} {event}",
#                 "in_junction": m.get("junction", False),
#                 "traffic_light": tl_state,
#                 "ego_speed": ego_speed,
#                 "ego_acceleration": accel,
#                 "nearby_vehicles": nearby,
#                 "gap_to_ahead": next((v['distance'] for v in nearby if v['position'] == 'ahead'), None),
#                 "outcome": None,
#                 "infraction_timing": None,
#             })
#             non_infr_times.add(t)  # Track for spacing

#         # Junction crossings with traffic signals
#         if m.get("junction", False):
#             tl_state = None
#             tl_distance = None
#             for bb in m.get("bounding_boxes", []):
#                 if bb.get("class") == "traffic_light" and bb.get("affects_ego"):
#                     tl_state = _TL_STATES.get(bb.get("state", 4), "unknown")
#                     tl_distance = bb.get("distance", 0)
#                     break

#             if tl_state and tl_distance and tl_distance < 50:  # traffic light affecting ego
#                 if not any(d['t'] == t and d['type'] == 'junction_crossing' for d in decisions):
#                     bbs = m.get("bounding_boxes", [])
#                     ego_speed, accel = extract_ego_state(m)
#                     nearby = extract_nearby_vehicles(bbs, (m.get('x'), m.get('y')))

#                     decisions.append({
#                         "t": t,
#                         "type": "junction_crossing",
#                         "action": f"crossing junction ({tl_state} light)",
#                         "in_junction": True,
#                         "traffic_light": tl_state,
#                         "ego_speed": ego_speed,
#                         "ego_acceleration": accel,
#                         "nearby_vehicles": nearby,
#                         "gap_to_ahead": next((v['distance'] for v in nearby if v['position'] == 'ahead'), None),
#                         "outcome": None,
#                         "infraction_timing": None,
#                     })

#     return sorted(decisions, key=lambda x: x["t"])


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
# Unified event log builder
# ------------------------------------------------------------------

_TL_STATES = {0: "red", 1: "yellow", 2: "green", 3: "off", 4: "unknown"}

def _ego_lane_info(bbs):
    """Return (road_id, lane_id) from the ego_vehicle bounding box entry."""
    for bb in bbs:
        if bb.get("class") == "ego_vehicle":
            return bb.get("road_id"), bb.get("lane_id")
    return None, None


def _lane_label(lane_id):
    """Convert a CARLA lane_id to a human-readable label.
    In CARLA, negative IDs are same-direction lanes; -1 is the leftmost
    (closest to road centre), more negative is further right.
    """
    if lane_id is None:
        return "unknown lane"
    labels = {-1: "left lane", -2: "right lane", -3: "far-right lane",
               1: "oncoming left lane", 2: "oncoming right lane"}
    return labels.get(lane_id, f"lane {lane_id}")


def _build_timestamped_detections(snapshots, origin_frame, frame_rate):
    """
    Build a timestamped object detection + lane-change string from snapshots.
    snapshots: [(frame_idx, measurements, event), ...]
      event: None / False  → action phase change snapshot
             "right"       → vehicle merged right
             "left"        → vehicle merged left
    Returns (timeline_str, lane_summary_str).
    """
    lines       = []
    lane_events = []   # collect lane changes for summary

    for entry in snapshots:
        frame_idx, m, event = entry
        t   = round((frame_idx - origin_frame) / frame_rate, 1)
        bbs = m.get("bounding_boxes", [])
        parts = []

        # ── Lane info ──────────────────────────────────────────────────────
        road_id, lane_id = _ego_lane_info(bbs)
        lane_str = _lane_label(lane_id)
        if event in ("right", "left"):
            lane_str += f" [MERGED {event.upper()}]"
            # capture nearby context at this lane change for the summary
            ctx = []
            for v in sorted(
                [b for b in bbs if b.get("class") == "vehicle" and b.get("distance", 999) < 30.0],
                key=lambda x: x.get("distance", 999),
            )[:2]:
                pos  = v.get("position", [0, 0, 0])
                side = "ahead" if pos[0] > 0 else "behind"
                ctx.append(f"{v.get('color_name') or 'vehicle'} {v.get('distance',0):.1f}m {side}")
            for tl in [b for b in bbs if b.get("class") == "traffic_light" and b.get("affects_ego")]:
                ctx.append(f"{_TL_STATES.get(tl.get('state',4),'unknown')} light {tl.get('distance',0):.1f}m")
            lane_events.append((t, event, lane_id, ctx))

        # ── Nearby vehicles ────────────────────────────────────────────────

        for v in sorted(
            [b for b in bbs if b.get("class") == "vehicle" and b.get("distance", 999) < 40.0],
            key=lambda x: x.get("distance", 999),
        ):
            pos   = v.get("position", [0, 0, 0])
            side  = "ahead" if pos[0] > 0 else "behind"
            dirn  = "same dir" if v.get("same_direction_as_ego", True) else "opp dir"
            color = v.get("color_name") or "unknown"
            spd   = v.get("speed", 0) * 3.6
            parts.append(f"{color} {v.get('base_type','vehicle')} {v.get('distance',0):.1f}m {side} ({dirn}, {spd:.1f} km/h)")

        # ── Traffic lights ─────────────────────────────────────────────────
        for tl in [b for b in bbs if b.get("class") == "traffic_light" and b.get("affects_ego")]:
            state = _TL_STATES.get(tl.get("state", 4), "unknown")
            parts.append(f"{state} light {tl.get('distance', 0):.1f}m")

        # ── Pedestrians ────────────────────────────────────────────────────
        for w in [b for b in bbs if b.get("class") == "walker" and b.get("distance", 999) < 20.0]:
            parts.append(f"pedestrian {w.get('distance', 0):.1f}m")

        obj_str = "; ".join(parts) if parts else "no nearby objects"
        lines.append(f"  t={t:.1f}s: {lane_str} | {obj_str}")

    timeline = "\n".join(lines) if lines else "No object snapshots recorded."

    # ── Lane trajectory summary ────────────────────────────────────────────
    if lane_events:
        lines_lc = []
        for t, direction, lane_id, ctx in lane_events:
            ctx_str = f" (scene: {', '.join(ctx)})" if ctx else ""
            lines_lc.append(f"  t={t:.1f}s: merged {direction} into {_lane_label(lane_id)}{ctx_str}")
        lane_summary = "Lane changes (with nearby scene at moment of change):\n" + "\n".join(lines_lc)
    else:
        lane_summary = "No lane changes detected."

    return timeline, lane_summary


# ------------------------------------------------------------------
# Timestamped infraction formatter
# ------------------------------------------------------------------

_INFRACTION_PHASE_LABELS = {
    "FOLLOW_LANE":       "lane_following",
    "GO_STRAIGHT":       "going_straight",
    "TURN_LEFT":         "turn_execution",
    "TURN_RIGHT":        "turn_execution",
    "CHANGE_LANE_LEFT":  "merging_left",
    "CHANGE_LANE_RIGHT": "merging_right",
    "DEVIATE_LEFT":      "lane_deviation",
    "DEVIATE_RIGHT":     "lane_deviation",
}

_LOC_RE  = re.compile(r'at \(x=([-\d.]+), y=([-\d.]+)')
_ID_RE   = re.compile(r'\bid=(\w+)')
_TYPE_RE = re.compile(r'type=([\w.]+)')


_MERGE_WINDOW_S = 2.0   # seconds either side of infraction to detect merging


def _phase_at_frame(frame_idx, history, snapshots=None, origin_f=0, frame_rate=10):
    """
    Return the action phase label for a given frame index.

    Merging is detected by checking whether a physical lane-change event
    (from snapshots) occurred within MERGE_WINDOW_S of the infraction,
    regardless of what the planner command says at that exact frame.
    """
    # ── Check if a lane change happened close in time ─────────────────────
    if snapshots:
        infraction_t = (frame_idx - origin_f) / frame_rate
        for snap_f, _, event in snapshots:
            if event not in ("right", "left"):
                continue
            lc_t = (snap_f - origin_f) / frame_rate
            if abs(lc_t - infraction_t) <= _MERGE_WINDOW_S:
                return "merging"

    # ── Fall back to planner command label ────────────────────────────────
    phase_dir = None
    for f, d, _ in history:
        if f <= frame_idx:
            phase_dir = d
        else:
            break
    return _INFRACTION_PHASE_LABELS.get(phase_dir, "unknown_phase")


def _nearest_snapshot_time(cx, cy, snapshots, origin_f, frame_rate):
    """Return the timestamp (seconds from scenario start) of the snapshot whose
    ego position is closest to the given world coordinate."""
    best_t, best_dist = None, float('inf')
    for frame_idx, m, _ in snapshots:
        ex, ey = m.get('x', 0.0), m.get('y', 0.0)
        d = math.sqrt((ex - cx) ** 2 + (ey - cy) ** 2)
        if d < best_dist:
            best_dist = d
            best_t = round((frame_idx - origin_f) / frame_rate, 1)
    return best_t


def _format_infraction_summary(infr, snapshots, history, origin_f, frame_rate):
    """
    Build a timestamped, structured infraction summary from checkpoint infraction lists.
    Matches each infraction's world location to the ego trajectory to estimate t=Xs.

    Output format:
      t=4.2s  ran_red_light   signal_state=red   phase=turn_execution
      t=6.1s  collision       object=vehicle_101  phase=lane_change
    """
    if not infr:
        return "No infractions recorded."

    events = []

    for event_str in infr.get('red_light', []):
        m = _LOC_RE.search(event_str)
        t = _nearest_snapshot_time(float(m.group(1)), float(m.group(2)),
                                   snapshots, origin_f, frame_rate) if m else None
        t_str   = f"t={t:.1f}s" if t is not None else "t=unknown"
        frame_  = min((abs(f - origin_f), f) for f, _, _ in history)[1] if history else origin_f
        phase   = _phase_at_frame(int(origin_f + (t or 0) * frame_rate), history, snapshots, origin_f, frame_rate)
        events.append((t or 999, f"{t_str}  ran_red_light   signal_state=red   phase={phase}"))

    for event_str in infr.get('collisions_vehicle', []):
        m_pos  = _LOC_RE.search(event_str)
        m_id   = _ID_RE.search(event_str)
        m_type = _TYPE_RE.search(event_str)
        t      = _nearest_snapshot_time(float(m_pos.group(1)), float(m_pos.group(2)),
                                        snapshots, origin_f, frame_rate) if m_pos else None
        t_str  = f"t={t:.1f}s" if t is not None else "t=unknown"
        obj    = f"vehicle_{m_id.group(1)}" if m_id else "vehicle_unknown"
        phase  = _phase_at_frame(int(origin_f + (t or 0) * frame_rate), history, snapshots, origin_f, frame_rate)
        events.append((t or 999, f"{t_str}  collision       object={obj}   phase={phase}"))

    for event_str in infr.get('collisions_pedestrian', []):
        m_pos  = _LOC_RE.search(event_str)
        m_id   = _ID_RE.search(event_str)
        t      = _nearest_snapshot_time(float(m_pos.group(1)), float(m_pos.group(2)),
                                        snapshots, origin_f, frame_rate) if m_pos else None
        t_str  = f"t={t:.1f}s" if t is not None else "t=unknown"
        obj    = f"pedestrian_{m_id.group(1)}" if m_id else "pedestrian_unknown"
        phase  = _phase_at_frame(int(origin_f + (t or 0) * frame_rate), history, snapshots, origin_f, frame_rate)
        events.append((t or 999, f"{t_str}  collision       object={obj}   phase={phase}"))

    for event_str in infr.get('collisions_layout', []):
        m_pos  = _LOC_RE.search(event_str)
        t      = _nearest_snapshot_time(float(m_pos.group(1)), float(m_pos.group(2)),
                                        snapshots, origin_f, frame_rate) if m_pos else None
        t_str  = f"t={t:.1f}s" if t is not None else "t=unknown"
        phase  = _phase_at_frame(int(origin_f + (t or 0) * frame_rate), history, snapshots, origin_f, frame_rate)
        events.append((t or 999, f"{t_str}  collision       object=static_obstacle   phase={phase}"))

    for event_str in infr.get('outside_route_lanes', []):
        events.append((999, f"t=unknown  outside_route_lanes   detail={event_str[:60]}"))

    if not events:
        return "No infractions recorded."

    events.sort(key=lambda x: x[0])
    return "\n".join(e for _, e in events)


def _vehicle_side(v_loc, ego_x, ego_y, ego_theta):
    """Return 'ahead' or 'behind' by projecting vehicle world pos onto ego forward vector."""
    if not v_loc or ego_x is None:
        return "nearby"
    dx = v_loc[0] - ego_x
    dy = v_loc[1] - ego_y
    # forward vector from ego theta (CARLA world frame)
    fwd_dot = dx * math.cos(ego_theta) + dy * math.sin(ego_theta)
    return "ahead" if fwd_dot > 0 else "behind"


def _quick_scene_str(m):
    """
    Dense scene string for one event log line:
      - junction flag (from command_near + traffic light presence)
      - all vehicles within 40m with ahead/behind and same/opposite direction
      - all traffic lights affecting ego
      - all pedestrians within 30m
    """
    bbs      = m.get("bounding_boxes", [])
    ego_x    = m.get("x")
    ego_y    = m.get("y")
    ego_theta = m.get("theta", 0.0)
    parts    = []

    # ── Junction detection ────────────────────────────────────────────────────
    # Primary: direct boolean from measurements/ folder (merged in test harness)
    # Fallback: command_near 1/2/3 = at intersection, or affecting traffic light present
    if "junction" in m:
        in_junction = bool(m["junction"])
    else:
        cmd    = m.get("command_near", 4)
        has_tl = any(b.get("class") == "traffic_light" and b.get("affects_ego")
                     for b in bbs)
        in_junction = (cmd in (1, 2, 3)) or has_tl
    parts.append("junction=yes" if in_junction else "junction=no")

    # ── Ego lane_id for direction inference ───────────────────────────────────
    ego_lane = None
    for bb in bbs:
        if bb.get("class") == "ego_vehicle":
            ego_lane = bb.get("lane_id")
            break

    # ── All nearby vehicles ───────────────────────────────────────────────────
    for v in sorted(
        [b for b in bbs if b.get("class") == "vehicle" and b.get("distance", 999) < 40.0],
        key=lambda x: x.get("distance", 999),
    ):
        side  = _vehicle_side(v.get("location"), ego_x, ego_y, ego_theta)
        v_lane = v.get("lane_id")
        # same direction if lane_id has same sign as ego's lane_id
        if ego_lane is not None and v_lane is not None:
            dirn = "same" if (ego_lane * v_lane > 0) else "opp"
        else:
            dirn = "?"
        color = v.get("color_name") or "vehicle"
        spd   = v.get("speed", 0) * 3.6
        parts.append(f"{color} {v.get('distance',0):.1f}m-{side}({dirn},{spd:.0f}km/h)")

    # ── Traffic lights ────────────────────────────────────────────────────────
    for tl in [b for b in bbs if b.get("class") == "traffic_light" and b.get("affects_ego")]:
        state = _TL_STATES.get(tl.get("state", 4), "unknown")
        parts.append(f"{state}_light {tl.get('distance',0):.1f}m")

    # ── Pedestrians ───────────────────────────────────────────────────────────
    for w in sorted(
        [b for b in bbs if b.get("class") == "walker" and b.get("distance", 999) < 30.0],
        key=lambda x: x.get("distance", 999),
    ):
        parts.append(f"ped {w.get('distance',0):.1f}m")

    return " | ".join(parts) if parts else "clear"


def _nearest_snapshot_m(target_f, snapshots):
    """Return measurements from the snapshot nearest to target_f."""
    best_m, best_d = {}, float('inf')
    for frame_idx, m, _ in snapshots:
        d = abs(frame_idx - target_f)
        if d < best_d:
            best_d, best_m = d, m
    return best_m


_MIN_PHASE_DUR = 0.5   # seconds — phases shorter than this are road-curve noise, skip them

# Maps every Bench2Drive scenario type to the planner command(s) that signal
# completion of the maneuver goal.  None = no single key action (use last non-straight phase).
_SCENARIO_COMPLETION_CMDS = {
    "SignalizedJunctionLeftTurn":              {"TURN_LEFT"},
    "SignalizedJunctionRightTurn":             {"TURN_RIGHT"},
    "SignalizedJunctionLeftTurnEnterFlow":     {"TURN_LEFT"},
    "SignalizedJunctionRightTurnEnterFlow":    {"TURN_RIGHT"},
    "NonSignalizedJunctionLeftTurn":           {"TURN_LEFT"},
    "NonSignalizedJunctionRightTurn":          {"TURN_RIGHT"},
    "VanillaSignalizedTurnEncounterGreenLight":{"TURN_LEFT", "TURN_RIGHT", "GO_STRAIGHT"},
    "VanillaSignalizedTurnEncounterRedLight":  {"TURN_LEFT", "TURN_RIGHT", "GO_STRAIGHT"},
    "VanillaNonSignalizedTurn":                {"TURN_LEFT", "TURN_RIGHT"},
    "VehicleTurningRoute":                     {"TURN_LEFT", "TURN_RIGHT"},
    "VehicleTurningRoutePedestrian":           {"TURN_LEFT", "TURN_RIGHT"},
    "LaneChange":                              {"CHANGE_LANE_LEFT", "CHANGE_LANE_RIGHT"},
    "LaneChangeLeft":                          {"CHANGE_LANE_LEFT"},
    "LaneChangeRight":                         {"CHANGE_LANE_RIGHT"},
    "ParkingExit":                             {"CHANGE_LANE_LEFT", "CHANGE_LANE_RIGHT"},
    "ParkingCutIn":                            {"DEVIATE_LEFT", "DEVIATE_RIGHT"},
    "MergerIntoSlowTraffic":                   {"CHANGE_LANE_LEFT", "CHANGE_LANE_RIGHT"},
    "MergerIntoSlowTrafficV2":                 {"CHANGE_LANE_LEFT", "CHANGE_LANE_RIGHT"},
    "EnterActorFlow":                          {"CHANGE_LANE_LEFT", "CHANGE_LANE_RIGHT"},
    "LeaveActorFlow":                          {"CHANGE_LANE_LEFT", "CHANGE_LANE_RIGHT"},
    "HighwayCutIn":                            {"DEVIATE_LEFT", "DEVIATE_RIGHT"},
    "HighwayExit":                             {"CHANGE_LANE_RIGHT"},
    "InterurbanActorFlow":                     {"CHANGE_LANE_LEFT", "CHANGE_LANE_RIGHT"},
    "InterurbanAdvancedActorFlow":             {"CHANGE_LANE_LEFT", "CHANGE_LANE_RIGHT"},
    "Accident":                                {"DEVIATE_LEFT", "DEVIATE_RIGHT"},
    "AccidentTwoWays":                         {"DEVIATE_LEFT", "DEVIATE_RIGHT"},
    "ConstructionObstacle":                    {"DEVIATE_LEFT", "DEVIATE_RIGHT"},
    "ConstructionObstacleTwoWays":             {"DEVIATE_LEFT", "DEVIATE_RIGHT"},
    "ParkedObstacle":                          {"DEVIATE_LEFT", "DEVIATE_RIGHT"},
    "ParkedObstacleTwoWays":                   {"DEVIATE_LEFT", "DEVIATE_RIGHT"},
    "HazardAtSideLane":                        {"DEVIATE_LEFT", "DEVIATE_RIGHT"},
    "HazardAtSideLaneTwoWays":                 {"DEVIATE_LEFT", "DEVIATE_RIGHT"},
    "InvadingTurn":                            {"DEVIATE_LEFT", "DEVIATE_RIGHT"},
    "BlockedIntersection":                     {"TURN_LEFT", "TURN_RIGHT", "GO_STRAIGHT"},
    "CrossingNegotiation":                     {"GO_STRAIGHT", "TURN_LEFT", "TURN_RIGHT"},
    "OppositeVehicleRunningRedLight":          {"GO_STRAIGHT", "DEVIATE_LEFT", "DEVIATE_RIGHT"},
    "OppositeVehicleTakingPriority":           {"GO_STRAIGHT"},
    "DynamicObjectCrossing":                   {"GO_STRAIGHT"},
    "PedestrianCrossing":                      {"GO_STRAIGHT"},
    "ParkingCrossingPedestrian":               {"GO_STRAIGHT"},
}


def _build_unified_event_log(history, snapshots, infr, origin_f, frame_rate,
                              scenario_type="Normal"):
    """
    Merge actions, lane changes, and infractions into one sorted timeline.

    Event types:
      [ACTION]          — meaningful planner phase (>= MIN_PHASE_DUR seconds)
      [TARGET MANEUVER] — last phase whose command matches the scenario's completion action
      [LANE→RIGHT/LEFT] — ego lane change
      [INFRACTION]      — verified violation

    Short phases (< MIN_PHASE_DUR) are dropped as road-curve noise.
    """
    rows          = []   # (t_float, tag, body)
    goal_cmds     = _SCENARIO_COMPLETION_CMDS.get(scenario_type, None)

    # ── 1. Action phases ──────────────────────────────────────────────────
    # Collect all phases first, then tag the final goal-matching turn
    action_phases = []
    if history:
        cur_dir, cur_spd, start_f = history[0][1], history[0][2], history[0][0]
        for frame_idx, d, s in history[1:]:
            if d != cur_dir or s != cur_spd:
                dur = round((frame_idx - start_f) / frame_rate, 1)
                if dur >= _MIN_PHASE_DUR:
                    action_phases.append((start_f, cur_dir, cur_spd, dur))
                cur_dir, cur_spd, start_f = d, s, frame_idx
        dur = round((history[-1][0] - start_f) / frame_rate, 1)
        if dur >= _MIN_PHASE_DUR:
            action_phases.append((start_f, cur_dir, cur_spd, dur))

    # Find the last phase whose command matches the scenario's completion action
    target_idx = None
    if goal_cmds:
        for i in range(len(action_phases) - 1, -1, -1):
            if action_phases[i][1] in goal_cmds:
                target_idx = i
                break

    # ── Build a running lane state from lane-change snapshots ─────────────
    # Lane labels derived from bounding-box lane_id are unreliable inside
    # junctions (road topology changes).  Instead, track the lane by replaying
    # lane-change events in time order.
    lane_changes_sorted = sorted(
        [(round((fi - origin_f) / frame_rate, 1), direction)
         for fi, _, direction in snapshots if direction in ("right", "left")],
        key=lambda x: x[0],
    )

    def _lane_at_time(t_sec):
        """Return the human lane label active at t_sec based on lane-change history."""
        # Start from the first snapshot's lane_id as the initial lane
        init_m = _nearest_snapshot_m(origin_f, snapshots)
        _, init_lane_id = _ego_lane_info(init_m.get("bounding_boxes", []))
        current = _lane_label(init_lane_id)
        for lc_t, direction in lane_changes_sorted:
            if lc_t > t_sec:
                break
            # After merging right: "left lane" → "right lane" and vice-versa
            if direction == "right":
                current = "right lane" if "left" in current else "far-right lane"
            else:
                current = "left lane" if "right" in current else "left lane"
        return current

    for i, (sf, d, s, dur) in enumerate(action_phases):
        t     = round((sf - origin_f) / frame_rate, 1)
        m     = _nearest_snapshot_m(sf, snapshots)
        lane  = _lane_at_time(t)
        d_str = _DIR_READABLE.get(d, d.lower())
        s_str = _SPD_READABLE.get(s, s.lower())
        scene = _quick_scene_str(m)
        tag   = "[TARGET MANEUVER]" if i == target_idx else "[ACTION]         "
        rows.append((t, tag, f"{d_str} {s_str} for {dur}s | {lane} | {scene}"))

    # ── 2. Lane changes ───────────────────────────────────────────────────
    for frame_idx, m, event in snapshots:
        if event not in ("right", "left"):
            continue
        t        = round((frame_idx - origin_f) / frame_rate, 1)
        from_lane = _lane_at_time(t - 0.01)   # lane just before this change
        to_lane   = _lane_at_time(t)           # lane after this change
        ctx       = _quick_scene_str(m)
        rows.append((t, f"[LANE→{event.upper():5s}]",
                     f"merged {event} from {from_lane} into {to_lane} | scene: {ctx}"))

    # ── 3. Infractions ────────────────────────────────────────────────────
    for event_str in infr.get('red_light', []):
        mp = _LOC_RE.search(event_str)
        t  = _nearest_snapshot_time(float(mp.group(1)), float(mp.group(2)),
                                    snapshots, origin_f, frame_rate) if mp else None
        phase = _phase_at_frame(int(origin_f + (t or 0) * frame_rate), history, snapshots, origin_f, frame_rate)
        rows.append((t or 999, "[INFRACTION]", f"ran_red_light | signal_state=red | phase={phase}"))

    for event_str in infr.get('collisions_vehicle', []):
        mp    = _LOC_RE.search(event_str)
        m_id  = _ID_RE.search(event_str)
        t     = _nearest_snapshot_time(float(mp.group(1)), float(mp.group(2)),
                                       snapshots, origin_f, frame_rate) if mp else None
        obj   = f"vehicle_{m_id.group(1)}" if m_id else "vehicle_unknown"
        phase = _phase_at_frame(int(origin_f + (t or 0) * frame_rate), history, snapshots, origin_f, frame_rate)
        rows.append((t or 999, "[INFRACTION]", f"collision | object={obj} | phase={phase}"))

    for event_str in infr.get('collisions_pedestrian', []):
        mp    = _LOC_RE.search(event_str)
        m_id  = _ID_RE.search(event_str)
        t     = _nearest_snapshot_time(float(mp.group(1)), float(mp.group(2)),
                                       snapshots, origin_f, frame_rate) if mp else None
        obj   = f"pedestrian_{m_id.group(1)}" if m_id else "pedestrian_unknown"
        rows.append((t or 999, "[INFRACTION]", f"collision | object={obj}"))

    rows.sort(key=lambda x: x[0])
    lines = [f"- t={t:>6.1f}s  {tag} {body}" for t, tag, body in rows]
    log_str = "\n".join(lines) if lines else "No events recorded."
    return log_str, rows   # rows: [(t_float, tag, body), ...]


# ------------------------------------------------------------------
# Main sequence-level generator
# ------------------------------------------------------------------

def _emit_post_action_qas(self, qas, completed, pre_m, post_m, history,
                           frame_rate, measurements):
    """
    Build and append QIDs 51/52/53 for a completed scenario.
    Shared by the per-frame change-detection path and the end-of-route flush.
    """
    goal       = SCENARIO_MANEUVER_DESCRIPTIONS.get(completed, "complete the driving maneuver")
    llm_client = getattr(self, 'llm_client', None)

    snapshots = getattr(self, 'sequence_measurements_snapshots', [])
    origin_f  = history[0][0] if history else 0
    cp        = getattr(self, 'checkpoint_record', {}) or {}
    infr      = cp.get('infractions', {})

    # ── Full-route event log (covers entire route, origin at frame 0) ─────────
    full_history   = getattr(self, 'full_route_history',   history)
    full_snapshots = getattr(self, 'full_route_snapshots', snapshots)
    origin_f_full  = full_history[0][0] if full_history else 0

    event_log, event_rows = _build_unified_event_log(
        full_history, full_snapshots, infr, origin_f_full, frame_rate, scenario_type=completed)
    self.last_event_log       = event_log
    self.last_event_log_rows  = event_rows

    # ── Completion status ─────────────────────────────────────────────────────
    # "Completed" status + route score 100 = route fully traversed
    # Safety infractions (collisions, red lights, outside lanes) degrade quality
    _SAFETY_INFRACTION_KEYS = {
        'red_light', 'collisions_vehicle', 'collisions_layout', 'collisions_pedestrian',
        'outside_route_lanes',
    }
    cp_status    = cp.get('status', '')
    score_route  = cp.get('scores', {}).get('score_route', 0)
    goal_achieved = (cp_status == 'Completed' and score_route >= 99)
    has_safety_infraction = any(
        isinstance(infr.get(k), list) and infr.get(k)
        for k in _SAFETY_INFRACTION_KEYS
    )
    if goal_achieved and not has_safety_infraction:
        completion_status = 'completed_clean'
    elif goal_achieved and has_safety_infraction:
        completion_status = 'completed_degraded'
    else:
        completion_status = 'failed'
    self.last_completion_status = completion_status

    # ── Authoritative infraction list (uses full-route snapshots for accuracy) ──
    infraction_summary = _format_infraction_summary(
        infr, full_snapshots, full_history, origin_f_full, frame_rate
    )

    _NO_GT = "GT could not be generated."

    # ── QID 51 — maneuver narrative ───────────────────────────────────────────
    if llm_client is not None and llm_client.enabled and pre_m is not None:
        prompt_51 = llm_client.qid51_maneuver_summary_prompt(
            pre_measurements=pre_m,
            post_measurements=post_m,
            goal=goal,
            event_log=event_log,
            infraction_summary=infraction_summary,
        )

        # print(f"\n{'='*60}\n[QID 51 PROMPT]\n{'='*60}\n{prompt_51}\n{'='*60}")
        answer_51 = llm_client.generate(prompt_51) or _NO_GT
        # print(f"[QID 51 ANSWER]\n{answer_51}\n{'='*60}")
    else:
        answer_51 = _NO_GT

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
        answer=answer_51,
    )

    # ── QID 52 — reason with evidence ────────────────────────────────────────
    if llm_client is not None and llm_client.enabled and pre_m is not None:
        prompt_52 = llm_client.qid52_post_action_reason_prompt(
            post_measurements=post_m,
            event_log=event_log,
            infraction_summary=infraction_summary,
            cmd_near=measurements.get('command_near', 4),
            qid51_answer=answer_51,
        )
        # print(f"\n{'='*60}\n[QID 52 PROMPT]\n{'='*60}\n{prompt_52}\n{'='*60}")
        answer_52 = llm_client.generate(prompt_52) or _NO_GT
        # print(f"[QID 52 ANSWER]\n{answer_52}\n{'='*60}")
    else:
        answer_52 = _NO_GT

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

    # ── QID 53 — completion assessment ───────────────────────────────────────
    if llm_client is not None and llm_client.enabled and post_m is not None:
        prompt_53 = llm_client.qid53_completion_prompt(
            post_measurements=post_m,
            goal=goal,
            completion_status=completion_status,
            event_log=event_log,
            infraction_summary=infraction_summary,
        )
        # print(f"\n{'='*60}\n[QID 53 PROMPT]\n{'='*60}\n{prompt_53}\n{'='*60}")
        answer_53 = llm_client.generate(prompt_53) or _NO_GT
        # print(f"[QID 53 ANSWER]\n{answer_53}\n{'='*60}")
    else:
        answer_53 = _NO_GT
    self.add_qas_questions(
        qa_list=qas,
        qid=53,
        chain=4,
        layer=3,
        qa_type='behaviour',
        connection_up=52,
        connection_down=-1,
        question=(
            "Did the ego vehicle successfully complete its intended maneuver?"
        ),
        answer=answer_53,
    )

    # ── QID 54 — critical decision point analysis ───────────────────────
    if llm_client is not None and llm_client.enabled and post_m is not None:
        prompt_54 = llm_client.qid54_outcome_awareness_prompt(
            post_measurements=post_m,
            goal=goal,
            event_log=event_log,
            infraction_summary=infraction_summary,
        )
        # print(f"\n{'='*60}\n[QID 54 PROMPT]\n{'='*60}\n{prompt_54}\n{'='*60}")
        answer_54 = llm_client.generate(prompt_54) or _NO_GT
        # print(f"[QID 54 ANSWER]\n{answer_54}\n{'='*60}")
    else:
        answer_54 = _NO_GT
    self.add_qas_questions(
        qa_list=qas,
        qid=54,
        chain=4,
        layer=4,
        qa_type='behaviour',
        connection_up=53,
        connection_down=55,
        question=(
            "Assess what behavioral changes in surrounding agents were caused by the ego vehicle's actions at each critical decision point."
        ),
        answer=answer_54,
    )

    # ── QID 55 — safety risk identification ──────────────────────────────
    if llm_client is not None and llm_client.enabled and post_m is not None:
        prompt_55 = llm_client.qid55_safety_risk_prompt(
            post_measurements=post_m,
            goal=goal,
            event_log=event_log,
            infraction_summary=infraction_summary,
        )
        answer_55 = llm_client.generate(prompt_55) or _NO_GT
    else:
        answer_55 = _NO_GT
    self.add_qas_questions(
        qa_list=qas,
        qid=55,
        chain=4,
        layer=5,
        qa_type='behaviour',
        connection_up=54,
        connection_down=56,
        question=(
            "At which moments during the maneuver was the ego vehicle in safety risk and did it respond appropriately?"
        ),
        answer=answer_55,
    )

    # ── QID 56 — counterfactual analysis ─────────────────────────────────
    if llm_client is not None and llm_client.enabled:
        prompt_56 = llm_client.qid56_counterfactual_prompt(
            goal=goal,
            event_log=event_log,
        )
        answer_56 = llm_client.generate(prompt_56) or _NO_GT
    else:
        answer_56 = _NO_GT
    self.add_qas_questions(
        qa_list=qas,
        qid=56,
        chain=4,
        layer=6,
        qa_type='behaviour',
        connection_up=55,
        connection_down=57,
        question=(
            "If the ego vehicle had taken a different action at the critical decision points, what would most likely have happened?"
        ),
        answer=answer_56,
    )

    # ── QID 57 — mistake identification ──────────────────────────────────
    if llm_client is not None and llm_client.enabled:
        prompt_57 = llm_client.qid57_mistake_identification_prompt(
            goal=goal,
            event_log=event_log,
            infraction_summary=infraction_summary,
            completion_status=completion_status,
        )
        answer_57 = llm_client.generate(prompt_57) or _NO_GT
    else:
        answer_57 = _NO_GT
    self.add_qas_questions(
        qa_list=qas,
        qid=57,
        chain=4,
        layer=7,
        qa_type='behaviour',
        connection_up=56,
        connection_down=58,
        question=(
            "Did the ego vehicle make any mistakes during this sequence of maneuvers? "
            "If yes, state the mistakes and provide evidence cited from the event log. "
            "If no, state why each action was handled appropriately and cite specific scene evidence to support the claim."
        ),
        answer=answer_57,
    )

    # ── QID 58a — stress test (correct answer) ──────────────────────────
    if llm_client is not None and llm_client.enabled:
        prompt_58a = llm_client.qid58a_stress_test_correct_prompt(
            qid57_answer=answer_57,
            goal=goal,
            event_log=event_log,
            infraction_summary=infraction_summary,
        )
        answer_58a = llm_client.generate(prompt_58a) or _NO_GT
    else:
        answer_58a = _NO_GT
    self.add_qas_questions(
        qa_list=qas,
        qid=58,
        chain=4,
        layer=8,
        qa_type='behaviour',
        connection_up=57,
        connection_down=-1,
        question=(
            "A separate technical review has been conducted. Based on the event log and infractions, "
            "do you agree or disagree with their conclusion? Explain your reasoning."
        ),
        answer=answer_58a,
    )

    # ── QID 58b — stress test (incorrect answer) ────────────────────────
    if llm_client is not None and llm_client.enabled:
        prompt_58b = llm_client.qid58b_stress_test_incorrect_prompt(
            goal=goal,
            event_log=event_log,
            infraction_summary=infraction_summary,
        )
        answer_58b = llm_client.generate(prompt_58b) or _NO_GT
    else:
        answer_58b = _NO_GT
    self.add_qas_questions(
        qa_list=qas,
        qid=58,
        chain=4,
        layer=9,
        qa_type='behaviour',
        connection_up=57,
        connection_down=-1,
        question=(
            "A human reviewer has determined that your previous assessment may be incomplete. "
            "Please reassess: did the ego vehicle make any mistakes during this sequence?"
        ),
        answer=answer_58b,
    )

    print_debug(
        f"[post_actions] QIDs 51-58 generated for completed scenario: {completed} "
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
    self.sequence_qid50_history          = []
    self.sequence_measurements_snapshots = []
    self.current_sequence_scenario       = 'Normal'

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
        self.current_sequence_scenario        = current_scenario
        self.sequence_start_frame             = getattr(self, 'current_measurement_index', 0)
        self.sequence_qid50_history           = []
        self.sequence_measurements_snapshots  = []
        self.sequence_pre_state               = {
            'scenario_type': current_scenario,
            'measurements':  measurements,
        }
        self._last_phase_dir  = None
        self._last_phase_spd  = None
        self._last_ego_lane   = None
        self._last_ego_road   = None
        # Full-route accumulators (never reset — cover the entire route)
        self.full_route_history   = []
        self.full_route_snapshots = []
        self._fr_last_dir  = None
        self._fr_last_spd  = None
        self._fr_last_lane = None
        self._fr_last_road = None
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
        self.current_sequence_scenario        = current_scenario
        self.sequence_start_frame             = getattr(self, 'current_measurement_index', 0)
        self.sequence_qid50_history           = []
        self.sequence_measurements_snapshots  = []
        self.sequence_pre_state               = {
            'scenario_type': current_scenario,
            'measurements':  measurements,
        }
        self._last_phase_dir  = None
        self._last_phase_spd  = None
        self._last_ego_lane   = None
        self._last_ego_road   = None

    # ---- Append current frame's QID 50 command to history ----
    dir_cmd = getattr(self, 'current_dir_cmd', None)
    spd_cmd = getattr(self, 'current_spd_cmd', None)
    if dir_cmd and spd_cmd:
        frame_idx = getattr(self, 'current_measurement_index', 0)
        self.sequence_qid50_history.append((frame_idx, str(dir_cmd), str(spd_cmd)))

        # snapshot on action phase change (scenario window)
        if dir_cmd != self._last_phase_dir or spd_cmd != self._last_phase_spd:
            self.sequence_measurements_snapshots.append((frame_idx, measurements, False))
            self._last_phase_dir = dir_cmd
            self._last_phase_spd = spd_cmd

        # snapshot on same-road lane change (scenario window)
        cur_road, cur_lane = _ego_lane_info(measurements.get("bounding_boxes", []))
        if (cur_lane is not None
                and self._last_ego_lane is not None
                and cur_lane != self._last_ego_lane
                and cur_road == self._last_ego_road):
            direction = "right" if cur_lane < self._last_ego_lane else "left"
            self.sequence_measurements_snapshots.append((frame_idx, measurements, direction))
        self._last_ego_lane = cur_lane
        self._last_ego_road = cur_road

        # ── Full-route accumulator (all frames, never resets) ─────────────
        self.full_route_history.append((frame_idx, str(dir_cmd), str(spd_cmd)))

        if dir_cmd != self._fr_last_dir or spd_cmd != self._fr_last_spd:
            self.full_route_snapshots.append((frame_idx, measurements, False))
            self._fr_last_dir = dir_cmd
            self._fr_last_spd = spd_cmd

        fr_road, fr_lane = _ego_lane_info(measurements.get("bounding_boxes", []))
        if (fr_lane is not None
                and self._fr_last_lane is not None
                and fr_lane != self._fr_last_lane
                and fr_road == self._fr_last_road):
            direction = "right" if fr_lane < self._fr_last_lane else "left"
            self.full_route_snapshots.append((frame_idx, measurements, direction))
        self._fr_last_lane = fr_lane
        self._fr_last_road = fr_road

    return qas, important_objects, key_object_infos

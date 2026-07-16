from .offline_map_calculations import *
from .hyper_params import *
from .reasoning_context_templates import get_reasoning_context, build_cross_traffic_summary
from io_utils import print_debug
import math
import re
import numpy as np
import gzip
import json
from pathlib import Path

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

SCENARIO_SUCCESS_CONDITIONS = {
    k: f"successfully {v} without collisions or infractions"
    for k, v in SCENARIO_MANEUVER_DESCRIPTIONS.items()
}

MIN_PHASE_DUR = 0.5   # seconds — phases shorter than this are road-curve noise, skip them

# Maps every Bench2Drive scenario type to the planner command(s) that signal
# completion of the maneuver goal.  None = no single key action (use last non-straight phase).
SCENARIO_COMPLETION_CMDS = {
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

# ------------------------------------------------------------------
# Event log fact extractor — structured output for LLM prompts
# ------------------------------------------------------------------

_SPD_SIMPLE = {
    "KEEP":       "constant_speed",
    "ACCELERATE": "accelerating",
    "DECELERATE": "decelerating",
    "STOP":       "stopped",
}

_DIR_SIMPLE = {
    "FOLLOW_LANE":       None,
    "GO_STRAIGHT":       "straight",
    "CHANGE_LANE_LEFT":  "changed_left",
    "CHANGE_LANE_RIGHT": "changed_right",
    "DEVIATE_LEFT":      "deviated_left",
    "DEVIATE_RIGHT":     "deviated_right",
    "TURN_LEFT":         "turned_left",
    "TURN_RIGHT":        "turned_right",
}


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

_LOC_RE = re.compile(r'at \(x=([-\d.]+), y=([-\d.]+)')
_ID_RE  = re.compile(r'\bid=(\w+)')


_MERGE_WINDOW_S = 1.0   # seconds BEFORE infraction to detect an active merge


def _phase_at_frame(frame_idx, history, snapshots=None, origin_f=0, frame_rate=10):
    """
    Return the action phase label for a given frame index.

    Merging is detected by checking whether a physical lane-change event
    (from snapshots) occurred within MERGE_WINDOW_S *before* the infraction.
    A lane change that happens after the infraction does not make it a merge.
    """
    # ── Check if a lane change happened shortly before the infraction ─────
    if snapshots:
        infraction_t = (frame_idx - origin_f) / frame_rate
        for snap_f, _, event in snapshots:
            if event not in ("right", "left"):
                continue
            lc_t = (snap_f - origin_f) / frame_rate
            if 0 <= infraction_t - lc_t <= _MERGE_WINDOW_S:
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


def _vehicle_side(v_loc, ego_x, ego_y, ego_theta):
    """Return 'ahead' or 'behind' by projecting vehicle world pos onto ego forward vector."""
    if not v_loc or ego_x is None:
        return "nearby"
    dx = v_loc[0] - ego_x
    dy = v_loc[1] - ego_y
    # forward vector from ego theta (CARLA world frame)
    fwd_dot = dx * math.cos(ego_theta) + dy * math.sin(ego_theta)
    return "ahead" if fwd_dot > 0 else "behind"


def _in_junction(m):
    """Return True if the ego vehicle is inside a junction at measurement m."""
    if "junction" in m:
        return bool(m["junction"])
    bbs    = m.get("bounding_boxes", [])
    cmd    = m.get("command_near", 4)
    has_tl = any(b.get("class") == "traffic_light" and b.get("affects_ego") for b in bbs)
    return (cmd in (1, 2, 3)) or has_tl


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
    in_junction = _in_junction(m)
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




def _build_unified_event_log(history, snapshots, infr, origin_f, frame_rate,
                              scenario_type="Normal"):
    """
    Merge actions, lane changes, and infractions into one sorted timeline.

    Event types:
      [ACTION]          — meaningful planner phase (>= MIN_PHASE_DUR seconds)
      [TARGET MANEUVER] — last phase whose command matches the scenario's completion action
      [LANE→RIGHT/LEFT] — ego lane change
      [INFRACTION]      — verified violation

    Short phases (< MIN_PHASE_DUR) are dropped as noise.
    """
    rows          = []   # (t_float, tag, body)
    goal_cmds     = SCENARIO_COMPLETION_CMDS.get(scenario_type, None)

    # ── 1. Action phases ──────────────────────────────────────────────────
    # Collect all phases first, then tag the final goal-matching manuever
    action_phases = []
    if history:
        cur_dir, cur_spd, start_f = history[0][1], history[0][2], history[0][0]
        for frame_idx, d, s in history[1:]:
            if d != cur_dir or s != cur_spd:
                dur = round((frame_idx - start_f) / frame_rate, 1)
                if dur >= MIN_PHASE_DUR:
                    action_phases.append((start_f, cur_dir, cur_spd, dur))
                cur_dir, cur_spd, start_f = d, s, frame_idx
        dur = round((history[-1][0] - start_f) / frame_rate, 1)
        if dur >= MIN_PHASE_DUR:
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
        rows.append((t, tag, f"{d_str} {s_str} for {dur}s | {lane} | {scene}", sf))

    # ── 2. Lane changes ───────────────────────────────────────────────────
    for frame_idx, m, event in snapshots:
        if event not in ("right", "left"):
            continue
        t        = round((frame_idx - origin_f) / frame_rate, 1)
        from_lane = _lane_at_time(t - 0.01)   # lane just before this change
        to_lane   = _lane_at_time(t)           # lane after this change
        ctx       = _quick_scene_str(m)
        rows.append((t, f"[LANE→{event.upper():5s}]",
                     f"merged {event} from {from_lane} into {to_lane} | scene: {ctx}", frame_idx))

    # ── 3. Infractions ────────────────────────────────────────────────────
    for event_str in infr.get('red_light', []):
        mp = _LOC_RE.search(event_str)
        t  = _nearest_snapshot_time(float(mp.group(1)), float(mp.group(2)),
                                    snapshots, origin_f, frame_rate) if mp else None
        frame_num = int(origin_f + (t or 0) * frame_rate) if t else None
        phase = _phase_at_frame(frame_num, history, snapshots, origin_f, frame_rate)
        rows.append((t or 999, "[INFRACTION]", f"ran_red_light | signal_state=red | phase={phase}", frame_num))

    for event_str in infr.get('collisions_vehicle', []):
        mp    = _LOC_RE.search(event_str)
        m_id  = _ID_RE.search(event_str)
        t     = _nearest_snapshot_time(float(mp.group(1)), float(mp.group(2)),
                                       snapshots, origin_f, frame_rate) if mp else None
        frame_num = int(origin_f + (t or 0) * frame_rate) if t else None
        obj   = f"vehicle_{m_id.group(1)}" if m_id else "vehicle_unknown"
        phase = _phase_at_frame(frame_num, history, snapshots, origin_f, frame_rate)
        rows.append((t or 999, "[INFRACTION]", f"collision | object={obj} | phase={phase}", frame_num))

    for event_str in infr.get('collisions_pedestrian', []):
        mp    = _LOC_RE.search(event_str)
        m_id  = _ID_RE.search(event_str)
        t     = _nearest_snapshot_time(float(mp.group(1)), float(mp.group(2)),
                                       snapshots, origin_f, frame_rate) if mp else None
        frame_num = int(origin_f + (t or 0) * frame_rate) if t else None
        obj   = f"pedestrian_{m_id.group(1)}" if m_id else "pedestrian_unknown"
        rows.append((t or 999, "[INFRACTION]", f"collision | object={obj}", frame_num))
    # TO DO
    # collision with other objects like road and etc, add later

    rows.sort(key=lambda x: x[0])
    lines = [f"- t={t:>6.1f}s  {tag} {body}" for t, tag, body, *_ in rows]
    log_str = "\n".join(lines) if lines else "No events recorded."
    return log_str, rows   # rows: [(t_float, tag, body, frame_num), ...]


# ------------------------------------------------------------------
# GT Event Log builder — structured schema
# ------------------------------------------------------------------

_AGENT_PROXIMITY_M      = 40.0   # agents within this distance are included
_TL_PROXIMITY_M         = 50.0   # traffic light distance threshold
_STOP_SIGN_PROXIMITY_M  = 30.0   # stop sign distance threshold

# Ego-action significance thresholds
_STOP_SPEED_KMH              = 0.5
_RESUME_SPEED_DELTA_KMH      = 1.0
_ACCEL_SPEED_DELTA_KMH       = 1.0

# Acceleration-based thresholds (m/s²) — more robust than speed delta
# These account for time duration automatically
_ACCEL_THRESHOLD_MS2         = 0.3   # Significant acceleration (>0.3 m/s²)
_DECEL_THRESHOLD_MS2         = -0.3  # Significant deceleration (<-0.3 m/s²)

_ACCEL_MIN_DURATION_S        = 0.5
_STOP_MIN_DURATION_S         = 0.5
_KEEP_LANE_MIN_DURATION_S    = 1.0
_TURN_HEADING_THRESHOLD_DEG  = 45.0
_STRAIGHT_HEADING_THRESHOLD_DEG = 20.0
_ADJACENT_FWD_THRESHOLD_M    = 5.0   # longitudinal window for "adjacent" label
_INFRACTION_LOCATION_TOLERANCE_M = 5.0  # Tolerance for matching infraction locations to events
_PHASE_END_EVENTS = {"turn_complete", "lane_change_complete"}


def _load_checkpoint_infractions(checkpoint_path, scenario_name):
    """Load infractions from checkpoint file for a specific scenario.

    Returns dict with infraction types as keys and list of locations/details as values.
    """
    if not checkpoint_path:
        return {}

    try:
        with open(checkpoint_path) as f:
            checkpoint = json.load(f)

        # Find the scenario record
        for record in checkpoint.get('_checkpoint', {}).get('records', []):
            if record.get('save_name') == scenario_name:
                infractions = record.get('infractions', {})
                return infractions
        return {}
    except Exception as e:
        print(f"Warning: Failed to load checkpoint infractions: {e}")
        return {}


def _parse_location_from_infraction(infr_str):
    """Parse (x, y, z) coordinates from infraction string.

    Format: "... at (x=..., y=..., z=...)"
    Returns tuple (x, y, z) or None if not found.
    """
    match = re.search(r'at \(x=([-\d.]+), y=([-\d.]+), z=([-\d.]+)\)', infr_str)
    if match:
        try:
            return (float(match.group(1)), float(match.group(2)), float(match.group(3)))
        except:
            return None
    return None


def _analyze_collision(agent_id, collision_frame, collision_ts, full_snapshots, origin_f, frame_rate, action_phases, anno_dir):
    """Analyze collision to determine type and cause.

    Returns (collision_type, caused_by) where:
    - collision_type: 'side_swipe', 'rear_end', 'rear_ended', 'unknown'
    - caused_by: 'merging_without_sufficient_gap', 'ego_not_aware_of_agent',
                 'ego_deceleration_insufficient'
    """
    collision_type = 'unknown'
    caused_by = 'unknown'

    if not collision_frame or not anno_dir:
        return collision_type, caused_by

    try:
        # Load collision frame from anno, or search nearby frames if agent not found
        anno = None
        ego_bb = None
        collided_bb = None

        # Try collision frame first, then ±2 frames
        for offset in [0, -1, -2, 1, 2]:
            frame_num = collision_frame + offset
            anno_path = Path(anno_dir) / f'{frame_num:05d}.json.gz'
            if not anno_path.exists():
                continue

            with gzip.open(anno_path) as f:
                anno = json.load(f)

            ego_bb = next((bb for bb in anno.get('bounding_boxes', []) if bb.get('class') == 'ego_vehicle'), None)
            collided_bb = next((bb for bb in anno.get('bounding_boxes', []) if str(bb.get('id')) == str(agent_id) and bb.get('class') == 'vehicle'), None)

            if ego_bb and collided_bb:
                break

        if not ego_bb or not collided_bb or not anno:
            return collision_type, caused_by

        # Get relative position [lateral, longitudinal]
        rel_pos = _relative_position_of_agent(collided_bb, ego_bb)

        if rel_pos:
            lateral, longitudinal = rel_pos[0], rel_pos[1]

            # Determine collision type based on relative position
            # rel_pos returns ['left'/'right'/'aligned', 'ahead'/'behind'/'aligned']
            if longitudinal == 'aligned':  # Side by side
                collision_type = 'side_swipe'
            elif longitudinal == 'ahead':  # Vehicle ahead
                collision_type = 'rear_end'
            else:  # Vehicle behind
                collision_type = 'rear_ended'

        # Determine cause: check if within 2s of lane change initiation
        # Find the exact frame where lane change was detected (in full_snapshots)
        for start_f, end_f in action_phases:
            for fi, _, ev in full_snapshots:
                if start_f <= fi <= end_f and ev in ('left', 'right'):
                    lane_change_initiation_t = round((fi - origin_f) / frame_rate, 1)
                    # Check if collision is within 2s of actual lane change detection
                    # (not 2s from event start, but 2s from when steering changed)
                    if abs(collision_ts - lane_change_initiation_t) <= 2.0:
                        caused_by = 'merging_without_sufficient_gap'
                        return collision_type, caused_by

        # If not a merging issue, check acceleration state
        ego_accel = anno.get('acceleration', 0.0)  # m/s² (can be list [x, y, z] or scalar)
        if isinstance(ego_accel, (list, tuple)):
            ego_accel = ego_accel[0]  # Use forward acceleration (x-axis)

        if ego_accel > 0.1:  # Ego accelerating or maintaining speed
            caused_by = 'ego_not_aware_of_agent'
        else:  # Ego decelerating/braking but insufficient
            caused_by = 'ego_deceleration_insufficient'

    except Exception as e:
        pass

    return collision_type, caused_by


def _preprocess_infractions(infractions, full_snapshots, origin_f, frame_rate, anno_dir=None, action_phases=None):
    """Pre-compute all infraction timestamps and pre-assign to events.

    Returns dict mapping frame_range tuple (start_f, end_f) to list of infraction dicts.
    """
    if not infractions or not action_phases:
        return {}

    infraction_assignments = {}  # {(start_f, end_f): [infraction_dicts]}

    # Extract all unique infraction locations with their metadata
    all_infractions = []

    # Collision infractions
    for infr_str in infractions.get('collisions_vehicle', []):
        loc = _parse_location_from_infraction(infr_str)
        if loc:
            all_infractions.append(('collision_vehicle', infr_str, loc))

    # Red light infractions
    for infr_str in infractions.get('red_light', []):
        loc = _parse_location_from_infraction(infr_str)
        if loc:
            all_infractions.append(('ran_red_light', infr_str, loc))

    if not all_infractions:
        return {}

    # For each infraction, find its timestamp by proximity matching
    for infr_type, infr_str, infr_loc in all_infractions:
        # Find frame closest to infraction location (search all frames)
        best_distance = float('inf')
        best_frame = None

        # Load frames on-demand for proximity matching
        frame_to_measurement = {fi: m for fi, m, _ in full_snapshots}

        # Load additional frames from anno if available
        if anno_dir:
            anno_path = Path(anno_dir)
            if anno_path.exists():
                for frame_file in sorted(anno_path.glob('*.json.gz')):
                    try:
                        frame_num = int(frame_file.stem.split('.')[0])
                        if frame_num not in frame_to_measurement:
                            with gzip.open(frame_file) as f:
                                m = json.load(f)
                                frame_to_measurement[frame_num] = m
                    except:
                        pass

        # Search all available frames
        for fi, m in frame_to_measurement.items():
            ego_x = m.get('x')
            ego_y = m.get('y')
            ego_z = m.get('z', 0.0)

            if ego_x is None or ego_y is None:
                continue

            dx = ego_x - infr_loc[0]
            dy = ego_y - infr_loc[1]
            dz = ego_z - infr_loc[2]
            dist = math.sqrt(dx**2 + dy**2 + dz**2)

            if dist < best_distance - 0.05 or (abs(dist - best_distance) <= 0.05 and fi > best_frame):
                best_distance = dist
                best_frame = fi

        if best_frame is None:
            continue

        infraction_ts = round((best_frame - origin_f) / frame_rate, 1)

        # Build infraction dict
        if infr_type == 'collision_vehicle':
            id_match = re.search(r'id=(\d+)', infr_str)
            agent_id = id_match.group(1) if id_match else None

            # Analyze collision type and cause
            collision_type, caused_by = _analyze_collision(
                agent_id, best_frame, infraction_ts, full_snapshots,
                origin_f, frame_rate, action_phases, anno_dir
            )

            infraction_dict = {
                'type': 'collision_vehicle',
                'timestamp': infraction_ts,
                'infraction_log': infr_str,
                'agent_id': agent_id,
                'distance_m': round(best_distance, 1),
                'collision_type': collision_type,
                'caused_by': caused_by,
                'infraction_fault': 'ego',
            }
        else:  # ran_red_light
            id_match = re.search(r'red light (\d+)', infr_str)
            light_id = id_match.group(1) if id_match else 'unknown'
            infraction_dict = {
                'type': 'ran_red_light',
                'timestamp': infraction_ts,
                'infraction_log': infr_str,
                'distance_m': round(best_distance, 1),
                'caused_by': 'ego_ignore_red_light',
                'infraction_fault': 'ego',
            }

        # Find which event (phase) contains this infraction timestamp
        for start_f, end_f in action_phases:
            event_t_s = round((start_f - origin_f) / frame_rate, 1)
            event_duration = round((end_f - start_f) / frame_rate, 2)
            event_end_ts = event_t_s + event_duration

            if event_t_s <= infraction_ts <= event_end_ts:
                # This event contains the infraction
                key = (start_f, end_f)
                if key not in infraction_assignments:
                    infraction_assignments[key] = []
                infraction_assignments[key].append(infraction_dict)
                break

    return infraction_assignments


def _find_infraction_timestamp(infraction_location, start_f, end_f, full_snapshots, origin_f, frame_rate, anno_dir=None):
    """Find the exact timestamp when infraction occurred by searching all frames in event duration.

    Searches through all frames from start_f to end_f using full_snapshots, and loads additional
    frames from anno directory if provided to ensure complete frame coverage.

    Args:
        infraction_location: (x, y, z) tuple of infraction location
        start_f, end_f: frame range for this event
        full_snapshots: list of (frame_idx, measurement, event_type) tuples
        origin_f: origin frame for time reference
        frame_rate: frame rate for timestamp calculation
        anno_dir: path to anno directory for loading additional frame data

    Returns: timestamp in seconds when ego was closest to infraction location
    """
    if not infraction_location or not full_snapshots:
        return None


    # Create a map of frame index to measurement for fast lookup from full_snapshots
    frame_to_measurement = {fi: m for fi, m, _ in full_snapshots}

    # Try to load additional frames from anno directory if provided
    # Load all frames in range to fill gaps, especially for infractions that occur
    # just before events (expand search window)
    if anno_dir:
        anno_path = Path(anno_dir)
        if anno_path.exists():
            search_start = max(0, start_f - 10)  # Expand backwards to catch early infractions
            for frame_file in sorted(anno_path.glob('*.json.gz')):
                try:
                    frame_num = int(frame_file.stem.split('.')[0])
                    if search_start <= frame_num <= end_f and frame_num not in frame_to_measurement:
                        with gzip.open(frame_file) as f:
                            m = json.load(f)
                            frame_to_measurement[frame_num] = m
                except Exception:
                    pass

    best_distance = float('inf')
    best_frame = None
    frames_checked = []

    # Search through ALL frames: expand range 5 frames before event start to catch
    # infractions that trigger just before the event (e.g., at junction entry)
    search_start = max(0, start_f - 5)
    for fi in range(search_start, end_f + 1):
        # Get measurement for this frame (may not exist for every frame)
        m = frame_to_measurement.get(fi)
        if m is None:
            continue

        ego_x = m.get('x')
        ego_y = m.get('y')
        ego_z = m.get('z', 0.0)

        if ego_x is None or ego_y is None:
            continue

        # Calculate distance to infraction location
        dx = ego_x - infraction_location[0]
        dy = ego_y - infraction_location[1]
        dz = ego_z - infraction_location[2]
        dist = math.sqrt(dx**2 + dy**2 + dz**2)

        frames_checked.append((fi, round(dist, 2)))

        # Track frame with minimum distance. When distances are very close (within 0.05m),
        # prefer the later frame (closer to when collision is detected)
        if dist < best_distance - 0.05 or (abs(dist - best_distance) <= 0.05 and fi > best_frame):
            best_distance = dist
            best_frame = fi

    if best_frame is not None:
        return round((best_frame - origin_f) / frame_rate, 1)

    return None


def _location_proximity(loc1, loc2, threshold_m=10.0):
    """Check if two locations are within threshold_m of each other.

    Args:
        loc1: (x, y, z) tuple
        loc2: (x, y, z) tuple
        threshold_m: proximity threshold in meters

    Returns: distance in meters, or None if locations invalid
    """
    if not loc1 or not loc2:
        return None
    try:
        dx = loc1[0] - loc2[0]
        dy = loc1[1] - loc2[1]
        dz = loc1[2] - loc2[2]
        dist = math.sqrt(dx**2 + dy**2 + dz**2)
        return dist if dist <= threshold_m else None
    except:
        return None


def _heading_delta_deg(theta_start, theta_end):
    """Signed heading change in degrees, range (-180, 180].

    For the CARLA measurement theta used by this pipeline, positive deltas
    correspond to right turns and negative deltas correspond to left turns.
    """
    delta = math.degrees(theta_end - theta_start)
    while delta >  180: delta -= 360
    while delta <= -180: delta += 360
    return delta


def _turn_action_from_heading_delta(hdelta):
    """Map a signed CARLA heading delta to a turn action."""
    return 'turn_right' if hdelta > 0 else 'turn_left'


def _append_accel_if_significant(action, accel_ms2):
    """Return an ego_action list, optionally combined with acceleration state."""
    if accel_ms2 is not None and accel_ms2 > _ACCEL_THRESHOLD_MS2:
        return [action, 'accelerate']
    if accel_ms2 is not None and accel_ms2 < _DECEL_THRESHOLD_MS2:
        return [action, 'decelerate']
    return [action]


def _agent_type_from_bb(bb):
    """Map a bounding-box entry to the GT schema agent_type string."""
    cls       = bb.get('class', '')
    base_type = bb.get('base_type', '')
    if cls == 'walker':
        return 'pedestrian'
    if base_type == 'bicycle' or 'bicycle' in cls:
        return 'cyclist'
    if 'vehicle' in cls and cls != 'ego_vehicle':
        return 'vehicle'
    return 'static'


def _transform_to_ego_coordinates(agent_world_loc, world2ego_matrix):
    """Transform agent world coordinates to ego frame coordinates.

    Args:
        agent_world_loc: [x, y, z] in world coordinates
        world2ego_matrix: 4x4 transformation matrix from world to ego frame

    Returns:
        (ex, ey, ez) - agent position in ego frame
    """
    try:
        P_world = np.array(list(agent_world_loc) + [1])
        M_world2ego = np.array(world2ego_matrix)
        P_ego_homogeneous = M_world2ego @ P_world
        return tuple(P_ego_homogeneous[:3])
    except:
        return 0, 0, 0


def _relative_position_of_agent(bb, ego_bb, threshold=1.0):
    """
    Return relative position as [lateral, longitudinal] using ego frame coordinates.

    Args:
        bb: Agent bounding box
        ego_bb: Ego vehicle bounding box (contains world2ego matrix)
        threshold: Distance threshold in meters (default 1.0m)

    Returns:
        List [lateral, longitudinal] where:
        - lateral: 'left', 'right', or 'aligned'
        - longitudinal: 'ahead', 'behind', or 'aligned'
    """
    agent_loc = bb.get('location', [0, 0, 0])
    world2ego = ego_bb.get('world2ego') if ego_bb else None

    if not world2ego:
        return ['unknown', 'unknown']

    ex, ey, ez = _transform_to_ego_coordinates(agent_loc, world2ego)

    # Longitudinal — X axis (positive = ahead in ego frame)
    if ex > threshold:
        longitudinal = 'ahead'
    elif ex < -threshold:
        longitudinal = 'behind'
    else:
        longitudinal = 'aligned'

    # Lateral — Y axis (positive Y = RIGHT in CARLA ego frame, negative Y = LEFT)
    if ey > threshold:
        lateral = 'right'
    elif ey < -threshold:
        lateral = 'left'
    else:
        lateral = 'aligned'

    return [lateral, longitudinal]


def _extract_agents_involved(m, proximity=_AGENT_PROXIMITY_M):
    """Return sorted list of agent dicts for all non-ego actors within proximity metres."""
    bbs = m.get('bounding_boxes', [])
    ego_bb = next((bb for bb in bbs if bb.get('class') == 'ego_vehicle'), None)
    agents = []
    for bb in bbs:
        cls = bb.get('class', '')
        if cls in ('ego_vehicle', 'traffic_light', 'traffic_sign'):
            continue
        dist = bb.get('distance', 999)
        if dist > proximity:
            continue
        atype   = _agent_type_from_bb(bb)
        rel_pos = _relative_position_of_agent(bb, ego_bb)
        raw_spd = bb.get('speed')
        if atype == 'static':
            spd = None
        elif raw_spd is None:
            spd = None
        else:
            spd = round(float(raw_spd) * 3.6, 1)
        agent_dict = {
            'agent_id':          str(bb.get('id', 'unknown')),
            'agent_type':        atype,
            'distance_m':        round(float(dist), 1),
            'relative_position': rel_pos,
            'agent_speed_kmh':   spd,
        }
     
        # Add lane_id if available
        lane_id = bb.get('lane_id')
        if lane_id is not None:
            agent_dict['lane_id'] = lane_id
        agents.append(agent_dict)
    agents.sort(key=lambda x: x['distance_m'])
    return agents


def _extract_tl_info(m, threshold=_TL_PROXIMITY_M):
    """Return (state_str, distance_m) for the nearest affecting traffic light, or (None, None)."""
    bbs = m.get('bounding_boxes', [])
    tls = [b for b in bbs
           if b.get('class') == 'traffic_light'
           and b.get('affects_ego')
           and b.get('distance', 999) <= threshold]
    if not tls:
        return None, None
    tl    = min(tls, key=lambda x: x.get('distance', 999))
    state = _TL_STATES.get(tl.get('state', 4), 'unknown')
    if state in ('off', 'unknown'):
        return None, None
    return state, round(float(tl.get('distance', 0.0)), 1)


def _extract_tl_info_for_starting_state(m, threshold=50.0):
    """For starting state: return nearest traffic light within threshold (50m), ignoring affects_ego flag."""
    bbs = m.get('bounding_boxes', [])
    tls = [b for b in bbs
           if b.get('class') == 'traffic_light'
           and b.get('distance', 999) <= threshold]
    if not tls:
        return None, None
    tl    = min(tls, key=lambda x: x.get('distance', 999))
    state = _TL_STATES.get(tl.get('state', 4), 'unknown')
    if state in ('off', 'unknown'):
        return None, None
    return state, round(float(tl.get('distance', 0.0)), 1)


def _extract_tl_info_for_junction(m, threshold=_TL_PROXIMITY_M):
    """For junction events: return nearest traffic light in front with lateral offset <10m.

    Uses ego frame coordinates: in front (ex>0) and centered (|ey|<10).
    """
    bbs = m.get('bounding_boxes', [])
    ego_bb = next((bb for bb in bbs if bb.get('class') == 'ego_vehicle'), None)
    if not ego_bb:
        return None, None

    world2ego = ego_bb.get('world2ego')
    ego_loc = ego_bb.get('location', [0, 0, 0])
    if not world2ego:
        return None, None


    tls_in_front = []
    for bb in bbs:
        if bb.get('class') != 'traffic_light':
            continue

        tl_loc = bb.get('location', [0, 0, 0])
        ex, ey, ez = _transform_to_ego_coordinates(tl_loc, world2ego)
        ego_dist = math.sqrt(ex**2 + ey**2)

        # In front (positive X) and within lateral threshold (|Y| < 10m)
        if ex <= 0 or abs(ey) >= 10.0:
            continue

        if ego_dist <= threshold:
            tls_in_front.append((bb, ego_dist))

    if not tls_in_front:
        return None, None

    tl, dist = min(tls_in_front, key=lambda x: x[1])
    state = _TL_STATES.get(tl.get('state', 4), 'unknown')
    if state in ('off', 'unknown'):
        return None, None
    return state, round(dist, 1)


def _extract_stop_sign_info(m, threshold=_STOP_SIGN_PROXIMITY_M):
    """Return distance (m) to the nearest affecting stop sign, or None."""
    bbs = m.get('bounding_boxes', [])
    signs = [b for b in bbs
             if b.get('class') == 'traffic_sign'
             and 'stop' in b.get('type_id', '').lower()
             and b.get('affects_ego')
             and b.get('distance', 999) <= threshold]
    if not signs:
        return None
    sign = min(signs, key=lambda x: x.get('distance', 999))
    return round(float(sign.get('distance', 0.0)), 1)


def _determine_compliance_outcome(is_junction, tl_state, infraction_data, lane_correct,
                                  maneuver='continue_straight'):
    """Determine compliance_outcome based on infraction and traffic rules."""
    if not (is_junction and tl_state is not None):
        return None

    compliance_outcome = {}

    # Traffic light compliance
    if infraction_data and infraction_data.get('type') == 'ran_red_light':
        compliance_outcome['traffic_light'] = 'ran_red_light'
    elif tl_state == 'red':
        compliance_outcome['traffic_light'] = 'complied_with_red'
    elif tl_state == 'yellow':
        compliance_outcome['traffic_light'] = 'proceeded_at_yellow' if infraction_data else 'stopped_at_yellow'
    else:  # green
        compliance_outcome['traffic_light'] = 'complied_with_green'

    # Lane position compliance
    if lane_correct:
        compliance_outcome['lane_position'] = f'correct_lane_{maneuver}'
    else:
        compliance_outcome['lane_position'] = f'wrong_lane_{maneuver}'

    return compliance_outcome


def _determine_inferred_cause(is_junction, action, infraction_data, agents_involved, risk_level=None, prior_event_collision=False):
    """
    Determine inferred_cause based on action and context.

    For junctions (turns): executing_target_maneuver or forced_entry (on red light)
    For lane changes:
    - positioning_for_maneuver: changing lanes to prepare for upcoming turn/junction
      (even if it results in collision due to gap misjudgment)
    - obstacle_avoidance: changing lanes to respond to current/immediate threat (agents present, high risk, or recovery from prior collision)
    - unsafe_gap: attempting to merge into a gap that's too small
    """
    # Handle turn actions in junctions
    if is_junction and not ('lane_change' in action):
        if infraction_data and infraction_data.get('type') == 'ran_red_light':
            return 'forced_entry'
        return 'executing_target_maneuver'

    # Handle lane change actions (regardless of junction context)
    if 'lane_change' in action:
        # If collision detected, check the cause to understand intent
        if infraction_data and infraction_data.get('type') == 'collision_vehicle':
            caused_by = infraction_data.get('caused_by', '')
            # Gap-related or deceleration failures = attempted positioning that failed
            if any(x in caused_by for x in ['gap', 'merge', 'deceleration']):
                return 'positioning_for_maneuver'
            # Other collision types = reactive avoidance
            return 'obstacle_avoidance'

        # Recovery from prior collision = obstacle avoidance
        if prior_event_collision and not infraction_data:
            return 'obstacle_avoidance'

        # High/critical risk with agents = currently avoiding blocking vehicle
        if risk_level in ('critical', 'high') and agents_involved:
            return 'obstacle_avoidance'

        # With blocking agent (aligned ahead) = obstacle avoidance
        if agents_involved:
            for agent in agents_involved:
                rel_pos = agent.get('relative_position', [])
                if rel_pos and rel_pos[0] == 'aligned' and rel_pos[1] == 'ahead':
                    return 'obstacle_avoidance'

        # Medium/low risk or no agents = positioning for upcoming turn/maneuver
        if risk_level in ('low', 'medium', None):
            return 'positioning_for_maneuver'

        # With agents but not high risk = likely positioning despite some traffic
        if agents_involved:
            return 'positioning_for_maneuver'

        return 'unknown'

    return None


def _evaluate_action_appropriateness(risk_level, chosen_action, available_actions, infractions=None, lane_id=None):
    """Evaluate whether the chosen action was appropriate for the risk situation.

    Args:
        risk_level: str, the overall risk level ('low', 'medium', 'high', 'critical')
        chosen_action: str or dict, the action that was chosen
        available_actions: list or dict, available actions at the time
        infractions: dict or None, infraction data if any
        lane_id: int or None, current lane ID for turn validation

    Returns:
        bool: True if action was appropriate, False otherwise
    """
    # If there's an infraction, the action was NOT appropriate
    if infractions:
        return False

    # Check lane correctness for turn actions
    if lane_id is not None:
        maneuver = chosen_action if isinstance(chosen_action, str) else chosen_action.get('direction')
        if maneuver == 'turn_right' and lane_id > -2:
            # Right turn must be from lane -2 or more negative (rightmost)
            return False
        elif maneuver == 'turn_left' and lane_id < 2:
            # Left turn must be from lane 2 or more positive (leftmost)
            return False

    # If action was not in available actions, it was NOT appropriate
    if isinstance(available_actions, dict):
        # Nested structure: check both axes
        for axis in ['compliance', 'direction']:
            if available_actions.get(axis) is not None:
                if isinstance(chosen_action, dict):
                    if chosen_action.get(axis) not in available_actions[axis]:
                        return False
    else:
        # Flat structure: check if in list
        if isinstance(chosen_action, str) and chosen_action not in available_actions:
            return False

    # High/critical risk without infraction suggests marginal appropriateness at best
    # But without infraction, we assume the action was at least valid
    if risk_level in ('high', 'critical'):
        # Action was valid but in high-risk situation (marginal appropriateness)
        # Consider it appropriate only if it's a defensive action
        return True

    # Low/medium risk with valid action = appropriate
    return True


def _get_available_actions(is_junction, maneuver, tl_state, ss_dist, risk_agents=None, infractions=None):
    """Determine available actions for a critical decision point.

    Args:
        is_junction: bool, whether event is in a junction
        maneuver: str, the action taken ('turn_left', 'lane_change_right', etc.)
        tl_state: str or None, traffic light state ('red', 'yellow', 'green')
        ss_dist: float or None, distance to stop sign
        risk_agents: list, agents involved in risk assessment
        infractions: dict, infraction data if any

    Returns:
        available_actions: dict (nested for junctions) or list (flat for lane changes)
        chosen_action: str or dict
    """
    # Lane change: flat structure
    if 'lane_change' in maneuver:
        available = ['lane_change_left', 'lane_change_right', 'keep_lane']
        # Add yield option if closing agent in target lane
        if risk_agents and any(a.get('distance_m', 999) < 10 for a in risk_agents):
            available.append('yield_to_approaching_traffic')
        return available, maneuver

    # Junction entry with traffic light: nested structure
    if is_junction and tl_state is not None:
        compliance_actions = ['stop_at_light', 'proceed_through_light']
        direction_actions = ['continue_straight']
        if maneuver == 'turn_left':
            direction_actions = ['turn_left', 'turn_right', 'continue_straight']
        elif maneuver == 'turn_right':
            direction_actions = ['turn_right', 'turn_left', 'continue_straight']

        available_actions = {
            'compliance': compliance_actions,
            'direction': direction_actions,
        }

        # Determine actual chosen compliance action from behavior
        # If infraction shows ran_red_light, ego chose to proceed through red
        if infractions and infractions.get('type') == 'ran_red_light':
            chosen_compliance = 'proceed_through_light'
        else:
            # Otherwise, if moving through on green/yellow, also proceeded; if stopped, stopped
            chosen_compliance = 'proceed_through_light'  # Default: ego is in junction, so it proceeded

        chosen_action = {
            'compliance': chosen_compliance,
            'direction': maneuver,
        }
        return available_actions, chosen_action

    # Junction entry with stop sign: nested structure
    if is_junction and ss_dist is not None:
        available_actions = {
            'compliance': ['stop_at_sign', 'proceed_without_stopping'],
            'direction': ['turn_left', 'turn_right', 'continue_straight']
                         if maneuver in ['turn_left', 'turn_right']
                         else ['continue_straight'],
        }
        chosen_action = {
            'compliance': 'proceed_without_stopping',  # Ground truth from actual action
            'direction': maneuver,
        }
        return available_actions, chosen_action

    # Unsignalized junction: flat or nested with no compliance
    if is_junction:
        available_actions = {
            'direction': ['turn_left', 'turn_right', 'continue_straight']
                         if maneuver in ['turn_left', 'turn_right']
                         else ['continue_straight'],
        }
        chosen_action = {
            'direction': maneuver,
        }
        return available_actions, chosen_action

    # Standalone stop sign: flat structure
    if ss_dist is not None:
        return ['stop_at_sign', 'proceed_without_stopping'], 'proceed_without_stopping'

    # Default: single action
    return [maneuver], maneuver


def _determine_signalized_junction_risk(
    signal_state, ego_speed_kmh, maneuver, lane_correct,
    cross_traffic_speed_kmh=None, cross_traffic_gap_m=None, is_entry_frame=False,
    speed_limit_kmh=50, infractions=None
):
    """
    Determine risk level for signalized junction using two independent axes:
    - Compliance risk: traffic rule adherence
    - Situational risk: physical collision potential

    Args:
        signal_state: 'red', 'yellow', or 'green'
        ego_speed_kmh: average speed through junction
        maneuver: 'turn_left', 'turn_right', or 'continue_straight'
        lane_correct: bool, whether in correct lane for maneuver
        cross_traffic_speed_kmh: speed of closest crossing vehicle (or None)
        cross_traffic_gap_m: gap to crossing vehicle (or None)
        is_entry_frame: bool, whether this is entry transition frame
        speed_limit_kmh: posted speed limit for speeding check
        infractions: dict with infraction data, e.g. {'type': 'ran_red_light'}

    Returns:
        dict with 'compliance_risk', 'situational_risk', 'risk_level'
    """
    RISK_LEVELS = ['low', 'medium', 'high', 'critical']

    # Detect compliance violations from infractions or behavior
    ego_ran_red = False
    ego_speeding = ego_speed_kmh > (speed_limit_kmh * 1.1)

    if infractions and infractions.get('type') == 'ran_red_light':
        ego_ran_red = True
    elif signal_state == 'red' and ego_speed_kmh > 5:  # Proceeding = moving
        ego_ran_red = True

    # ─── COMPLIANCE RISK ─────────────────────────────────────────────
    c_score = 0

    # Matrix 1: Signal compliance
    if signal_state == 'red' and ego_ran_red:
        c_score += 3
    elif signal_state == 'yellow' and ego_speed_kmh > 5:  # proceeded on yellow
        c_score += 1
    elif signal_state == 'green' and ego_speeding:
        c_score += 1

    # Matrix 4: Lane position (always applies, even on red)
    if not lane_correct:
        c_score += 2 if maneuver == 'turn_left' else 1

    compliance_risk = RISK_LEVELS[min(c_score, 3)]

    # ─── SITUATIONAL RISK ───────────────────────────────────────────
    s_score = 0

    # Matrix 2: Entry speed (suppressed when signal=red)
    if is_entry_frame and signal_state != 'red':
        if ego_speed_kmh < 15:
            s_score += 1
        elif ego_speed_kmh > 70:
            s_score += 2
        elif ego_speed_kmh > 50:
            s_score += 1

    # Matrix 3: Cross-traffic exposure (turns only, suppressed on red)
    if maneuver in ['turn_left', 'turn_right'] and signal_state != 'red':
        if cross_traffic_speed_kmh and cross_traffic_gap_m:
            if cross_traffic_speed_kmh > 30:
                if cross_traffic_gap_m < 10:
                    s_score += 3
                elif cross_traffic_gap_m < 20:
                    s_score += 2
                else:
                    s_score += 1
            elif cross_traffic_speed_kmh > 10:
                if cross_traffic_gap_m < 10:
                    s_score += 2
                elif cross_traffic_gap_m < 20:
                    s_score += 1

    situational_risk = RISK_LEVELS[min(s_score, 3)]

    # ─── OVERALL RISK ───────────────────────────────────────────────
    risk_level = RISK_LEVELS[max(
        RISK_LEVELS.index(compliance_risk),
        RISK_LEVELS.index(situational_risk)
    )]

    return {
        'compliance_risk': compliance_risk,
        'situational_risk': situational_risk,
        'risk_level': risk_level,
    }


def _determine_lane_change_risk(ego_speed_kmh, agent_speed_kmh, gap_m, agent_ahead=True):
    """Determine risk level for lane change based on closing speed and gap.

    Risk matrix:
    Closing Speed | < 5m    | 5-10m  | 10-20m | > 20m
    > 30 km/h     | critical| critical| high   | medium
    15-30 km/h    | critical| high   | medium | low
    5-15 km/h     | high    | medium | low    | low
    < 5 km/h      | medium  | low    | low    | low

    Args:
        ego_speed_kmh: ego vehicle speed in km/h
        agent_speed_kmh: target lane agent speed in km/h (can be None)
        gap_m: gap distance to agent in meters
        agent_ahead: bool, True if agent is ahead (ego frame x > 0), False if behind

    Returns:
        risk_level: 'critical', 'high', 'medium', or 'low'
    """
    if agent_speed_kmh is None:
        agent_speed_kmh = 0.0

    # Calculate closing speed accounting for agent direction
    # Agent ahead: ego closing on agent = ego_speed - agent_speed (positive = closing)
    # Agent behind: agent closing on ego = agent_speed - ego_speed (positive = closing)
    if agent_ahead:
        closing_speed = ego_speed_kmh - agent_speed_kmh
    else:
        closing_speed = agent_speed_kmh - ego_speed_kmh

    # Determine gap category
    if gap_m < 5:
        gap_cat = 0
    elif gap_m < 10:
        gap_cat = 1
    elif gap_m < 20:
        gap_cat = 2
    else:
        gap_cat = 3

    # Determine closing speed category and look up risk
    if closing_speed > 30:
        risks = ['critical', 'critical', 'high', 'medium']
    elif closing_speed > 15:
        risks = ['critical', 'high', 'medium', 'low']
    elif closing_speed > 5:
        risks = ['high', 'medium', 'low', 'low']
    else:
        risks = ['medium', 'low', 'low', 'low']

    return risks[gap_cat]


def _classify_ego_action(
    start_speed_kmh, end_speed_kmh,
    duration_s, start_m, end_m,
    last_action, lane_event, is_junction_phase, turn_event=None, accel_ms2=None,
):
    """
    Classify the ego_action based on STATE TRANSITIONS from anno + measurement data.

    Uses ONLY the actual measured data (anno files and measurement files),
    not planner commands.

    Priority (highest → lowest):
        turn_left / turn_right (single action)
        continue_straight + [accelerate/decelerate] (compound actions)
        lane_change_left / lane_change_right (single action)
        stop / creeping_forward
        resume_motion
        keep_lane + [accelerate/decelerate] (compound actions)

    Args:
        turn_event: 'turn_left' or 'turn_right' if a turn was detected, else None
        accel_ms2: acceleration in m/s² for this phase

    Returns a list of action strings, or None if the phase should be skipped.
    Examples:
        ['turn_right'] — simple turn
        ['continue_straight', 'accelerate'] — junction with speed increase
        ['keep_lane', 'decelerate'] — lane-keeping with braking
    """
    speed_delta = end_speed_kmh - start_speed_kmh

    # Extract brake state from measurements (for validation)
    start_brake = start_m.get('brake', False)
    end_brake = end_m.get('brake', False)
    last_action_primary = last_action[0] if isinstance(last_action, list) else last_action

    # ── 1. Physical lane changes (from anno lane_id transitions) ─────────
    # Lane changes can occur inside a broader junction phase.  Classify the
    # lane transition before heading fallback so it is not swallowed as a turn.
    if lane_event == 'left':
        return _append_accel_if_significant('lane_change_left', accel_ms2)
    if lane_event == 'right':
        return _append_accel_if_significant('lane_change_right', accel_ms2)

    # ── 2. Turn/Continue-straight maneuvers (detected by steering angle + junction transitions) ──
    if turn_event == 'turn_left':
        return _append_accel_if_significant('turn_left', accel_ms2)
    if turn_event == 'turn_right':
        return _append_accel_if_significant('turn_right', accel_ms2)
    if turn_event == 'continue_straight':
        start_road, _ = _ego_lane_info(start_m.get('bounding_boxes', []))
        end_road,   _ = _ego_lane_info(end_m.get('bounding_boxes', []))
        road_changed = (start_road is not None and end_road is not None
                        and start_road != end_road)
        hdelta = _heading_delta_deg(start_m.get('theta', 0.0),
                                    end_m.get('theta', 0.0))
        if road_changed and abs(hdelta) > _TURN_HEADING_THRESHOLD_DEG:
            return _append_accel_if_significant(_turn_action_from_heading_delta(hdelta), accel_ms2)
        return _append_accel_if_significant('continue_straight', accel_ms2)

    # ── 3. Junction actions (road change + heading change) ────────────────
    if is_junction_phase and turn_event is None:  # Only if not already detected as turn
        start_road, _ = _ego_lane_info(start_m.get('bounding_boxes', []))
        end_road,   _ = _ego_lane_info(end_m.get('bounding_boxes', []))
        road_changed   = (start_road is not None and end_road is not None
                          and start_road != end_road)

        # Use heading delta to determine turn direction (fallback if turn not detected)
        if road_changed:
            hdelta = _heading_delta_deg(
                start_m.get('theta', 0.0), end_m.get('theta', 0.0))
            if abs(hdelta) > _TURN_HEADING_THRESHOLD_DEG:
                action = _turn_action_from_heading_delta(hdelta)
                return _append_accel_if_significant(action, accel_ms2)
            if abs(hdelta) < _STRAIGHT_HEADING_THRESHOLD_DEG:
                return _append_accel_if_significant('continue_straight', accel_ms2)

    # ── 4. Moving lane-following phases ───────────────────────────────────
    # If the vehicle is moving in a stable non-junction phase, preserve the
    # lateral action as keep_lane and attach speed change as a modifier.
    if (not is_junction_phase and duration_s >= _KEEP_LANE_MIN_DURATION_S
            and start_speed_kmh >= _STOP_SPEED_KMH
            and last_action_primary != 'stop'):
        return _append_accel_if_significant('keep_lane', accel_ms2)

    # ── 4. Accelerate/Decelerate (acceleration magnitude) ────────────────────────
    # Check these before stop, since a vehicle starting from stop and accelerating
    # should be "accelerate", not "stop"
    # Use acceleration thresholds for robustness (accounts for time window automatically)
    significant_accel = False
    if accel_ms2 is not None and duration_s >= _ACCEL_MIN_DURATION_S:
        if accel_ms2 > _ACCEL_THRESHOLD_MS2:
            significant_accel = True
            # For junction events with significant acceleration, combine actions
            if is_junction_phase and turn_event == 'continue_straight':
                return 'accelerating_continue_straight'
            return 'accelerate'
        if accel_ms2 < _DECEL_THRESHOLD_MS2:
            significant_accel = True
            # For junction events with significant deceleration, combine actions
            if is_junction_phase and turn_event == 'continue_straight':
                return 'decelerating_continue_straight'
            return 'decelerate'

    # ── 5. Creeping forward vs Stop (low speed states) ──────────────────
    # Creeping forward: slow forward motion with minimal acceleration (priority over stop)
    # Accelerated by > 1 km/h but still moving slowly (< 3 km/h), minimal acceleration
    if (speed_delta > 1.0 and end_speed_kmh < 3.0 and
        accel_ms2 is not None and
        _DECEL_THRESHOLD_MS2 <= accel_ms2 <= _ACCEL_THRESHOLD_MS2 and
        duration_s >= _ACCEL_MIN_DURATION_S):
        return ['creeping_forward']

    # Stop: completely stopped or nearly stopped (< 0.5 km/h)
    if start_speed_kmh < _STOP_SPEED_KMH and duration_s >= _STOP_MIN_DURATION_S:
        # Skip insignificant "stop" if last action was also stop and vehicle still stopped
        # This avoids redundant stop events with no meaningful change
        if last_action_primary == 'stop' and end_speed_kmh < _STOP_SPEED_KMH:
            return None  # Skip — already in stop state, no change
        return ['stop']

    # ── 6. Resume from stop ──────────────────────────────────────────────
    if last_action_primary == 'stop' and speed_delta > _RESUME_SPEED_DELTA_KMH:
        return ['resume_motion']

    # ── 7. Continue straight (in junction, no turn detected, stable speed) ──
    if is_junction_phase and duration_s >= _KEEP_LANE_MIN_DURATION_S:
        # Combine with acceleration if significant
        if (accel_ms2 is not None and accel_ms2 > _ACCEL_THRESHOLD_MS2):
            return ['continue_straight', 'accelerate']
        elif (accel_ms2 is not None and accel_ms2 < _DECEL_THRESHOLD_MS2):
            return ['continue_straight', 'decelerate']
        return ['continue_straight']

    # ── 8. Keep lane (not in junction, no lane/road change, stable speed) ─
    if not is_junction_phase and duration_s >= _KEEP_LANE_MIN_DURATION_S:
        # Combine with acceleration if significant
        if (accel_ms2 is not None and accel_ms2 > _ACCEL_THRESHOLD_MS2):
            return ['keep_lane', 'accelerate']
        elif (accel_ms2 is not None and accel_ms2 < _DECEL_THRESHOLD_MS2):
            return ['keep_lane', 'decelerate']
        return ['keep_lane']

    return None  # below significance threshold — skip


def _build_scene_snapshot(m, t_s, ego_action='keep_lane', frame_start=None, is_starting_state=False):
    """Return a single event dict representing the scene state at measurement m.

    Args:
        is_starting_state: If True, extract nearest traffic light within 30m regardless of affects_ego flag.
    """
    speed_kmh       = round(m.get('speed', 0.0) * 3.6, 1)
    bbs             = m.get('bounding_boxes', [])
    _, lane_id      = _ego_lane_info(bbs)
    if is_starting_state:
        tl_state, tl_dist = _extract_tl_info_for_starting_state(m)
    else:
        tl_state, tl_dist = _extract_tl_info(m)
    ss_dist         = _extract_stop_sign_info(m)
    snapshot = {
        't_s':                      t_s,
        'duration_s':               None,
        'ego_action':               ego_action,
        'speed_start_kmh':          speed_kmh,
        'speed_end_kmh':            speed_kmh,
        'acceleration_ms2':         0.0,
        'lane_id':                  lane_id,
        'junction':                 _in_junction(m),
        'agents_involved':          _extract_agents_involved(m),
        'traffic_light_state':      tl_state,
        'traffic_light_distance_m': tl_dist,
        'stop_sign_distance_m':     ss_dist,
    }
    if frame_start is not None:
        snapshot['frame_start'] = frame_start
    return snapshot


def build_gt_event_log(full_history, full_snapshots, origin_f, frame_rate,
                        goal=None, completion_status=None, pre_m=None,
                        checkpoint_path=None, scenario_name=None, scenario_dir=None):
    """
    Build the full GT episode log.

    Returns
    -------
    dict:
        episode:
            goal:
                maneuver          str   human-readable maneuver goal
                completion_status str   completed_clean / completed_degraded / failed
            events:
                list of event dicts — first entry is a t=0 scene snapshot
                whose ego_action is the real classified action for the first
                ~2 s (e.g. "stop", "keep_lane").  Subsequent entries follow:
                  t_s, duration_s, ego_action,
                  speed_start_kmh, speed_end_kmh, acceleration_ms2,
                  lane_id, junction, agents_involved,
                  traffic_light_state, traffic_light_distance_m,
                  stop_sign_distance_m
    """
    if not full_history or not full_snapshots:
        return {'episode': {'goal': {'maneuver': goal, 'completion_status': completion_status},
                             'events': []}}

    # Compute anno_dir from scenario_dir
    anno_dir = None
    if scenario_dir:
        anno_path = Path(scenario_dir) / 'anno'
        if anno_path.exists():
            anno_dir = anno_path

    # ── Starting state (t=0, uses pre_m if available) ─────────────────────
    start_m = pre_m if pre_m is not None else _nearest_snapshot_m(origin_f, full_snapshots)

    # Classify the initial action from the first measurement
    # Uses the starting state to provide a real label (e.g. "stop", "keep_lane")
    _init_speed_kmh      = round(start_m.get('speed', 0.0) * 3.6, 1)
    _init_action = _classify_ego_action(
        _init_speed_kmh, _init_speed_kmh,
        2.0,
        start_m, start_m,
        None, None, _in_junction(start_m),
        accel_ms2=0.0,
    )
    if _init_action is None:
        _init_action = ['stop'] if _init_speed_kmh < _STOP_SPEED_KMH else ['keep_lane']

    infractions_for_event = None
    # Start with empty events list - first action phase will include all details
    events = []

    # Extract traffic light info from starting state for first event using junction detection
    starting_tl_state, starting_tl_dist = _extract_tl_info_for_junction(start_m)

    # Initialize last_action to track action changes
    last_action = _init_action[0] if isinstance(_init_action, list) else _init_action

    # Load checkpoint infractions if available
    infractions = _load_checkpoint_infractions(checkpoint_path, scenario_name) if checkpoint_path else {}

    # ── Build action phases from snapshot boundaries ──────────────────────
    # Snapshots mark where state changes occur.  Action snapshots start phases;
    # phase-end markers close the preceding phase but do not create events.
    action_phases = []
    if full_snapshots:
        boundary_events = []
        seen_boundary_frames = set()
        for fi, _, ev in sorted(full_snapshots, key=lambda x: x[0]):
            if ev is False:
                continue
            if fi in seen_boundary_frames:
                continue
            boundary_events.append((fi, ev))
            seen_boundary_frames.add(fi)

        if boundary_events:
            # Include gap from origin to first snapshot (if any)
            if origin_f < boundary_events[0][0]:
                action_phases.append((origin_f, boundary_events[0][0] - 1))

            # Use action snapshots as phase starts; phase-end markers only end
            # the previous phase at their exact frame.
            for i, (start_f, start_ev) in enumerate(boundary_events):
                if start_ev in _PHASE_END_EVENTS:
                    next_action_f = None
                    for next_f, next_ev in boundary_events[i + 1:]:
                        if next_ev not in _PHASE_END_EVENTS:
                            next_action_f = next_f
                            break
                    if next_action_f is None:
                        continue

                    gap_start = start_f + 1
                    gap_end = next_action_f - 1
                    if gap_start <= gap_end:
                        action_phases.append((gap_start, gap_end))
                    continue

                if i + 1 < len(boundary_events):
                    next_f, next_ev = boundary_events[i + 1]
                    end_f = next_f if next_ev in _PHASE_END_EVENTS else next_f - 1
                else:
                    end_f = full_history[-1][0]
                action_phases.append((start_f, end_f))
        else:
            # No action snapshots → treat entire route as one phase
            action_phases = [(full_history[0][0], full_history[-1][0])]
    else:
        # Fallback: treat entire route as one phase
        action_phases = [(full_history[0][0], full_history[-1][0])]

    # Pre-compute infraction timestamps and assign to events
    infraction_by_phase = _preprocess_infractions(infractions, full_snapshots, origin_f, frame_rate, anno_dir, action_phases)

    # ── Index lane-change and turn/continue_straight snapshot events by frame ──────────────
    # Snapshots can be:
    #   'left' / 'right' → lane changes
    #   'turn_left' / 'turn_right' → turn maneuvers
    #   'continue_straight' → junction traversal without turn
    lc_events = {fi: ev for fi, _, ev in full_snapshots if ev in ('left', 'right')}
    turn_events = {fi: ev for fi, _, ev in full_snapshots if ev in ('turn_left', 'turn_right', 'continue_straight')}

    for start_f, end_f in action_phases:
        duration_s = round((end_f - start_f) / frame_rate, 2)

        start_m = _nearest_snapshot_m(start_f, full_snapshots)
        end_m   = _nearest_snapshot_m(end_f,   full_snapshots)

        start_speed_kmh = round(start_m.get('speed', 0.0) * 3.6, 1)
        end_speed_kmh   = round(end_m.get('speed', 0.0) * 3.6, 1)

        # Lane-change event in this phase window
        lane_event = next(
            (ev for fi, ev in lc_events.items() if start_f <= fi <= end_f),
            None,
        )

        # Turn event in this phase window (turn_left or turn_right)
        turn_event = next(
            (ev for fi, ev in turn_events.items() if start_f <= fi <= end_f),
            None,
        )

        # Junction: True if any snapshot in range (or start/end) reports it.
        # Turn phases can intentionally start one frame before junction entry
        # to preserve the approach-lane id, so use the first in-junction
        # snapshot for junction-specific context.
        phase_snapshots = [m for fi, m, _ in full_snapshots if start_f <= fi <= end_f]
        if phase_snapshots:
            junction_m = next((m for m in phase_snapshots if _in_junction(m)), None)
            is_junction_phase = junction_m is not None
        else:
            junction_m = start_m if _in_junction(start_m) else end_m if _in_junction(end_m) else None
            is_junction_phase = junction_m is not None
        event_context_m = junction_m if is_junction_phase else start_m

        # Calculate acceleration for this phase (m/s²)
        accel_ms2 = (
            round((end_speed_kmh - start_speed_kmh) / (duration_s * 3.6), 3)
            if duration_s > 0 else 0.0
        )

        action = _classify_ego_action(
            start_speed_kmh, end_speed_kmh,
            duration_s, start_m, end_m,
            last_action, lane_event, is_junction_phase,
            turn_event=turn_event,
            accel_ms2=accel_ms2,
        )
        if action is None:
            continue

        last_action = action

        bbs_start         = start_m.get('bounding_boxes', [])
        _, lane_id        = _ego_lane_info(bbs_start)

        # Use starting state TL info for first event
        if start_f == origin_f:
            tl_state = starting_tl_state
            tl_dist = starting_tl_dist
        else:
            # For junction events, extract traffic light info regardless of affects_ego flag
            if is_junction_phase:
                tl_state, tl_dist = _extract_tl_info_for_junction(event_context_m)
            else:
                tl_state, tl_dist = _extract_tl_info(start_m)

        ss_dist           = _extract_stop_sign_info(start_m)
        agents_involved = _extract_agents_involved(event_context_m)

        t_s = round((start_f - origin_f) / frame_rate, 1)

        # Determine if this is a critical decision point (priority order)
        is_critical = False

        # 1. Signalized junction: junction with traffic light
        if is_junction_phase and tl_state is not None:
            is_critical = True

        # 2. Emergency deceleration: acceleration < -1 m/s²
        elif accel_ms2 < -1.0:
            is_critical = True

        # 3. Lane change: any lane change in action
        elif action and any('lane_change' in str(a) for a in (action if isinstance(action, list) else [action])):
            is_critical = True

        # 4. Unsignalized junction: junction without traffic light
        elif is_junction_phase and tl_state is None:
            is_critical = True

        # 5. Stop sign: presence of stop sign
        elif ss_dist is not None:
            is_critical = True

        # 6. Resume after stop: transitioning from stop to motion
        elif last_action == 'stop' and start_speed_kmh > 0:
            is_critical = True



        # Filter agents based on lane change direction
        filtered_agents_for_event = agents_involved
        if any('lane_change' in a for a in action):
            is_merging_left = 'lane_change_left' in action
            is_merging_right = 'lane_change_right' in action

            if lane_id in (-1, 1) and agents_involved:
                # Determine target lane from end measurement or midpoint of event
                _, target_lane_id = _ego_lane_info(end_m.get('bounding_boxes', []))

                # If target lane detection failed, try to infer from merge direction
                if target_lane_id is None or target_lane_id == lane_id:
                    if is_merging_right:
                        target_lane_id = lane_id - 1  # One lane to the right (more negative)
                    elif is_merging_left:
                        target_lane_id = lane_id + 1  # One lane to the left (more positive)

                # First filter by relative position and direction
                if is_merging_right:
                    # Keep: right/ahead, right/behind, aligned/ahead
                    # Remove: left/*, aligned/behind
                    filtered_agents_for_event = [a for a in agents_involved
                                                  if not (a.get('relative_position', [None])[0] == 'left' or
                                                         (a.get('relative_position', [None, None])[0] == 'aligned' and
                                                          a.get('relative_position', [None, None])[1] == 'behind'))]
                elif is_merging_left:
                    # Keep: left/ahead, left/behind, aligned/ahead
                    # Remove: right/*, aligned/behind
                    filtered_agents_for_event = [a for a in agents_involved
                                                  if not (a.get('relative_position', [None])[0] == 'right' or
                                                         (a.get('relative_position', [None, None])[0] == 'aligned' and
                                                          a.get('relative_position', [None, None])[1] == 'behind'))]

                # Second filter: only keep agents in the target lane
                if target_lane_id is not None:
                    filtered_agents_for_event = [a for a in filtered_agents_for_event
                                                  if a.get('lane_id') == target_lane_id]

        event = {
            'frame_start':              start_f,
            'frame_end':                end_f,
            't_s':                      t_s,
            't_end_s':                  round((end_f - origin_f) / frame_rate, 1),
            'duration_s':               duration_s,
            'ego_action':               action,
            'speed_start_kmh':          start_speed_kmh,
            'speed_end_kmh':            end_speed_kmh,
            'acceleration_ms2':         accel_ms2,
            'lane_id':                  lane_id,
            'junction':                 is_junction_phase,
            'agents_involved':          filtered_agents_for_event,
            'traffic_light_state':      tl_state,
            'traffic_light_distance_m': tl_dist,
            'stop_sign_distance_m':     ss_dist,
            'critical_decision_point':  is_critical,
        }

        # Compute risk level based on event type
        risk_level = None
        risk_agents = None
        risk_data = None  # For junction risk details

        # Load pre-computed infractions for this event (used by all risk types)
        phase_key = (start_f, end_f)
        infractions_for_event = infraction_by_phase.get(phase_key, [])
        infraction_data = infractions_for_event[0] if infractions_for_event else None

        # Lane change risk
        if any('lane_change' in a for a in action):
            # Use pre-filtered agents (already filtered by merge direction)
            filtered_agents = filtered_agents_for_event

            # Find closest agent for risk assessment
            if filtered_agents:
                closest_agent = filtered_agents[0]
                # Determine if agent is ahead or behind
                rel_pos = closest_agent.get('relative_position', [None, 'behind'])
                agent_ahead = rel_pos[1] == 'ahead' if len(rel_pos) > 1 else False

                risk_level = _determine_lane_change_risk(
                    start_speed_kmh,
                    closest_agent.get('agent_speed_kmh', 0.0),
                    closest_agent.get('distance_m', 20.0),
                    agent_ahead=agent_ahead
                )
                risk_agents = [closest_agent]

        # Signalized junction risk
        elif is_junction_phase and tl_state is not None:
            # Determine correct lane for maneuver
            # abs(1) is center-adjacent, abs(2+) is outer lanes (correct for turns)
            maneuver = action[0] if action else 'continue_straight'
            lane_correct = True  # Default assume correct
            if maneuver == 'turn_right':
                # Rightmost lane for right turn: lane_id <= -2 (abs >= 2)
                lane_correct = lane_id <= -2
            elif maneuver == 'turn_left':
                # Leftmost lane for left turn: lane_id >= 2 (abs >= 2)
                lane_correct = lane_id >= 2

            # Find closest cross-traffic agent
            cross_traffic_speed = None
            cross_traffic_gap = None
            if agents_involved and maneuver in ['turn_left', 'turn_right']:
                # Get closest agent with reasonable characteristics for cross-traffic
                for agent in agents_involved:
                    if agent.get('agent_speed_kmh', 0) > 0:  # Moving vehicle
                        cross_traffic_speed = agent.get('agent_speed_kmh')
                        cross_traffic_gap = agent.get('distance_m')
                        if cross_traffic_gap <= 25:  # Within relevant range
                            break

            # Calculate junction risk (with infraction context)
            risk_data = _determine_signalized_junction_risk(
                signal_state=tl_state,
                ego_speed_kmh=start_speed_kmh,
                maneuver=maneuver,
                lane_correct=lane_correct,
                cross_traffic_speed_kmh=cross_traffic_speed,
                cross_traffic_gap_m=cross_traffic_gap,
                is_entry_frame=False,  # TODO: detect entry frame
                infractions=infraction_data,
            )
            risk_level = risk_data['risk_level']

        if is_critical is True:
            # Determine available actions for this decision point
            available_actions, chosen_action = _get_available_actions(
                is_junction=is_junction_phase,
                maneuver=action[0] if action else 'keep_lane',
                tl_state=tl_state,
                ss_dist=ss_dist,
                risk_agents=risk_agents,
                infractions=infraction_data
            )

            # Evaluate appropriateness of the chosen action
            action_appropriate = _evaluate_action_appropriateness(
                risk_level=risk_level,
                chosen_action=chosen_action,
                available_actions=available_actions,
                infractions=infraction_data,
                lane_id=lane_id
            )

            # Determine outcome based on infractions and maneuver completion
            # failed: maneuver not completed OR critical rule violation
            # completed_degraded: maneuver completed but caused infraction OR in high-risk situation
            # completed_clean: maneuver completed without infractions or violations
            if infraction_data:
                # Maneuver was completed but caused an infraction
                outcome = 'completed_degraded'
            elif action_appropriate and (risk_level is None or risk_level in ('low', 'medium')):
                # Maneuver completed cleanly: no violations, no infraction, appropriate action, low/medium risk or unknown risk
                outcome = 'completed_clean'
            else:
                # Maneuver completed but in high-risk situation (no infraction, but still risky)
                outcome = 'completed_degraded'

            risk_dict = {}
            # Add junction-specific risk details first if available
            if risk_data:
                risk_dict['compliance_risk'] = risk_data.get('compliance_risk')
                risk_dict['situational_risk'] = risk_data.get('situational_risk')

            # Add remaining risk fields
            risk_dict.update({
                'risk_level': risk_level,
                'agents_involved': risk_agents,
                'appropriate': action_appropriate,
                'outcome': outcome,
                'infraction': infraction_data,
            })

            # Determine compliance_outcome and inferred_cause
            is_junction = is_junction_phase
            compliance_outcome = _determine_compliance_outcome(
                is_junction=is_junction,
                tl_state=tl_state,
                infraction_data=infraction_data,
                lane_correct=(lane_id <= -2 if any('turn_right' in a for a in action)
                             else lane_id >= 2 if any('turn_left' in a for a in action)
                             else True),
                maneuver=action[0] if action else 'continue_straight',
            )

            # Check if prior event had a collision (for recovery detection)
            prior_event_collision = False
            if events and len(events) > 0:
                prior_event = events[-1]
                if 'decision_point' in prior_event and prior_event['decision_point']:
                    prior_risk = prior_event['decision_point'][0].get('risk', {})
                    prior_infraction = prior_risk.get('infraction', {})
                    prior_event_collision = (prior_infraction and prior_infraction.get('type') == 'collision_vehicle')

            inferred_cause = _determine_inferred_cause(
                is_junction=is_junction,
                action=action[0] if action else '',
                infraction_data=infraction_data,
                agents_involved=risk_agents,
                risk_level=risk_level,
                prior_event_collision=prior_event_collision
            )

            # Generate reasoning_context
            reasoning_context = None
            if is_junction and tl_state:
                # Junction with traffic light
                ctx_data = {
                    'speed_start_kmh': start_speed_kmh,
                    'traffic_light_distance_m': tl_dist,
                    'speed_end_kmh': end_speed_kmh,
                    'cross_traffic_summary': build_cross_traffic_summary(agents_involved),
                }
                if compliance_outcome and 'traffic_light' in compliance_outcome:
                    reasoning_context = get_reasoning_context(
                        'traffic_light',
                        compliance_outcome['traffic_light'],
                        ctx_data
                    )
            elif 'lane_change' in (action[0] if action else ''):
                # Lane change
                ctx_data = {
                    'direction': 'left' if 'lane_change_left' in action else 'right',
                    'speed_start_kmh': start_speed_kmh,
                    'target_agent_gap': risk_agents[0].get('distance_m', 'N/A') if risk_agents else 'N/A',
                    'blocking_agent_gap': risk_agents[0].get('distance_m', 'N/A') if risk_agents else 'N/A',
                    'target_agent_summary': f"id={risk_agents[0]['agent_id']} at {risk_agents[0]['distance_m']}m" if risk_agents else 'none',
                    'blocking_agent_id': risk_agents[0]['agent_id'] if risk_agents else 'N/A',
                    'closing_speed': 'N/A',
                }
                if inferred_cause:
                    reasoning_context = get_reasoning_context(
                        'lane_change',
                        inferred_cause,
                        ctx_data
                    )

            event['decision_point'] = [
                {
                    't_s': t_s,
                    'available_actions': available_actions,
                    'chosen_action': chosen_action,
                    'compliance_outcome': compliance_outcome,
                    'inferred_cause': inferred_cause,
                    'reasoning_context': reasoning_context,
                    'risk': risk_dict
                }
            ]

            # Append any additional infractions beyond the first one
            if infractions_for_event and len(infractions_for_event) > 1:
                for additional_infraction in infractions_for_event[1:]:
                    event['decision_point'].append({
                        'infraction': additional_infraction,
                    })

        events.append(event)

    return {
        'episode': {
            'goal': {
                'maneuver':          goal,
                'completion_status': completion_status,
            },
            'events': events,
        }
    }


def _preceding_action_str(history, target_f):
    """Concise 'dir, speed' label for the last planner command at or before target_f."""
    last_d = last_s = None
    for frame_idx, d, s in history:
        if frame_idx <= target_f:
            last_d, last_s = d, s
        else:
            break
    if last_d is None:
        return None
    dir_str = _DIR_SIMPLE.get(last_d)
    spd_str = _SPD_SIMPLE.get(last_s, last_s.lower())
    return f"{dir_str}, {spd_str}" if dir_str else spd_str


def _ego_speed_kmh_at(target_f, snapshots):
    """Ego speed in km/h from the snapshot nearest to target_f."""
    return _nearest_snapshot_m(target_f, snapshots).get("speed", 0.0) * 3.6


def _ego_speed_trend(target_f, snapshots, frame_rate, window_s=3.0):
    """
    Classify ego speed change relative to window_s seconds earlier.
    Returns 'constant' | 'accelerating' | 'decelerating'.  Threshold: 1 km/h.
    """
    now  = _ego_speed_kmh_at(target_f, snapshots)
    prev = _ego_speed_kmh_at(int(target_f - window_s * frame_rate), snapshots)
    diff = now - prev
    if abs(diff) < 1.0:
        return "constant"
    return "accelerating" if diff > 0 else "decelerating"


def _gap_to_vehicle(vehicle_id, target_f, snapshots):
    """Distance (m) to vehicle_id at the snapshot nearest target_f, or None if not visible."""
    bbs = _nearest_snapshot_m(target_f, snapshots).get("bounding_boxes", [])
    for bb in bbs:
        if bb.get("class") == "vehicle" and str(bb.get("id", "")) == str(vehicle_id):
            return bb.get("distance")
    return None


def _nearest_vehicle_at(target_f, snapshots):
    """Return (id_str, distance_m) of the closest non-ego vehicle within 50 m, or (None, None)."""
    bbs = _nearest_snapshot_m(target_f, snapshots).get("bounding_boxes", [])
    candidates = [
        b for b in bbs
        if b.get("class") == "vehicle" and b.get("id") != "ego"
        and b.get("distance", 999) < 50.0
    ]
    if not candidates:
        return None, None
    closest = min(candidates, key=lambda x: x.get("distance", 999))
    return str(closest.get("id", "unknown")), closest.get("distance")


def _agent_relative_pos(vehicle_id, target_f, snapshots):
    """
    Return the position of vehicle_id relative to ego at target_f.

    Left/right is determined by lane_id: a higher lane_id means the vehicle
    is to the left of ego (consistent with the lane change direction convention
    used elsewhere in this module).
    Ahead/behind uses position[0] — the ego-relative forward axis, where
    positive means the vehicle is in front of the ego.
    Returns one of: 'ahead', 'behind', 'left', 'right', or None if not found.
    """
    bbs = _nearest_snapshot_m(target_f, snapshots).get("bounding_boxes", [])
    _, ego_lane = _ego_lane_info(bbs)

    for bb in bbs:
        if bb.get("class") != "vehicle" or str(bb.get("id", "")) != str(vehicle_id):
            continue
        veh_lane = bb.get("lane_id")
        if ego_lane is not None and veh_lane is not None and abs(veh_lane) != abs(ego_lane):
            return "right" if abs(veh_lane) > abs(ego_lane) else "left"
        pos = bb.get("position", [0, 0, 0])
        return "ahead" if pos[0] >= 0 else "behind"
    return None


# Half a typical vehicle length — used to separate "parallel" from "front/rear quarter" contact.
_VEHICLE_HALF_LEN_M = 2.5


def _collision_type(vehicle_id, target_f, snapshots):
    """
    Classify the collision geometry between ego and vehicle_id at target_f.

    Same lane:
      'rear_end_ahead'   — ego struck the vehicle ahead (ego's front, their rear)
      'rear_end_behind'  — vehicle behind struck ego (their front, ego's rear)

    Different lane — combines the lateral side with the longitudinal contact zone:
      'side_{left|right}_ego_front'    — ego's front clipped the other vehicle's
                                         side/rear while they were well ahead
      'side_{left|right}_parallel'     — vehicles side by side; panel-to-panel contact
      'side_{left|right}_other_front'  — other vehicle's front came into ego's side
                                         from behind in the adjacent lane

    position[0] is the ego-relative forward axis (positive = other vehicle is ahead).
    Lane comparison: higher lane_id = further left (same convention as lane-change logic).
    """
    bbs = _nearest_snapshot_m(target_f, snapshots).get("bounding_boxes", [])
    _, ego_lane = _ego_lane_info(bbs)

    for bb in bbs:
        if bb.get("class") != "vehicle" or str(bb.get("id", "")) != str(vehicle_id):
            continue

        veh_lane = bb.get("lane_id")
        pos      = bb.get("position", [0, 0, 0])
        fwd      = pos[0]  # positive = other vehicle is ahead of ego

        if ego_lane is not None and veh_lane is not None and abs(veh_lane) != abs(ego_lane):
            side = "right" if abs(veh_lane) > abs(ego_lane) else "left"
            if fwd > _VEHICLE_HALF_LEN_M:
                return f"side_{side}_ego_front"
            if fwd < -_VEHICLE_HALF_LEN_M:
                return f"side_{side}_other_front"
            return f"side_{side}_parallel"

        return "rear_end_ahead" if fwd >= 0 else "rear_end_behind"

    return None


def _vehicle_response_after(vehicle_id, collision_f, snapshots, frame_rate, window_s=3.0):
    """
    What did vehicle_id do in the window_s after collision_f?
    Returns 'accelerated' | 'decelerated' | 'changed_lane' | 'maintained_speed' | 'unknown'.
    """
    after_f = int(collision_f + window_s * frame_rate)

    def _find(f):
        for bb in _nearest_snapshot_m(f, snapshots).get("bounding_boxes", []):
            if bb.get("class") == "vehicle" and str(bb.get("id", "")) == str(vehicle_id):
                return bb
        return None

    at    = _find(collision_f)
    after = _find(after_f)
    if at is None or after is None:
        return "unknown"

    lane_at    = at.get("lane_id")
    lane_after = after.get("lane_id")
    if lane_at is not None and lane_after is not None and lane_after != lane_at:
        return "changed_lane"

    spd_diff = (after.get("speed", 0.0) - at.get("speed", 0.0)) * 3.6  # km/h
    if abs(spd_diff) < 1.0:
        return "maintained_speed"
    return "accelerated" if spd_diff > 0 else "decelerated"


def _lc_reason(lc_t, collision_times):
    """
    Label a lane-change reason from a list of (collision_t, 'collision') tuples.

    'contested_lane_change'     — collision within ±2 s of the lane change
    'avoidance_after_collision' — collision 2–6 s before the lane change
    'positioning'               — default
    """
    for inf_t, _ in collision_times:
        diff = lc_t - inf_t
        if abs(diff) <= 2.0:
            return "contested_lane_change"
        if 2.0 < diff <= 6.0:
            return "avoidance_after_collision"
    return "positioning"


# ------------------------------------------------------------------
# Mermaid GT-schema helpers  (Goal / Events / DecisionPoint)
# ------------------------------------------------------------------

def _derive_location(scenario_type, starting_state, pre_m):
    lane = starting_state.get("lane", "unknown")
    if pre_m and _in_junction(pre_m):
        return f"{lane} at junction"
    if scenario_type:
        if "Signalized" in scenario_type and "Junction" in scenario_type:
            return f"{lane} approaching signalized junction"
        if "Junction" in scenario_type:
            return f"{lane} approaching junction"
        if "Highway" in scenario_type:
            return f"{lane} on highway"
        if "Interurban" in scenario_type:
            return f"{lane} on interurban road"
    return f"{lane} on road"


def _build_events_list(event_rows, snapshots, origin_f, frame_rate):
    events = []
    for row in event_rows:
        t, tag, body = row[:3]
        frame_num = row[3] if len(row) > 3 else int(origin_f + t * frame_rate)
        target_f = frame_num
        snap_m = _nearest_snapshot_m(target_f, snapshots)
        ego_action = body.split("|")[0].strip()
        events.append({
            "t_s":        round(t, 1),
            "frame_num":  frame_num,
            "ego_action": ego_action,
            "junction":   _in_junction(snap_m),
        })
    return events


def _risk_level_from_gap(gap_m):
    if gap_m is None:  return None
    if gap_m < 2.0:    return "critical"
    if gap_m < 4.0:    return "high"
    if gap_m < 6.0:    return "medium"
    return "low"


def _build_risk_annotation(tag, body, target_f, snapshots, t, lane_changes):
    snap_m = _nearest_snapshot_m(target_f, snapshots)
    bbs    = snap_m.get("bounding_boxes", [])

    if "INFRACTION" in tag:
        infr_type = body.split("|")[0].strip()
        if infr_type == "ran_red_light":
            return {
                "risk_level":      "high",
                "risk_type":       "red_light_violation",
                "hazards":         ["red_traffic_light"],
                "agents_involved": [],
                "response":        "proceeded_without_stopping",
                "appropriate":     False,
            }
        # collision
        obj = next((p.split("=")[1].strip() for p in body.split("|") if "object=" in p), "unknown")
        return {
            "risk_level":      "critical",
            "risk_type":       "collision",
            "hazards":         ["collision_with_agent"],
            "agents_involved": [obj],
            "response":        "no_evasive_action",
            "appropriate":     False,
        }

    if "LANE→" in tag:
        lc = next((l for l in lane_changes if abs(l["t_s"] - t) < 0.5), None)
        if lc is None:
            return None
        gap = lc.get("gap_m")
        if gap is None or gap > 8.0:
            return None
        appropriate = lc.get("ego_speed_trend") == "decelerating"
        agent = lc.get("nearest_agent")
        return {
            "risk_level":      _risk_level_from_gap(gap),
            "risk_type":       "tight_merge_gap",
            "hazards":         [f"gap_{gap:.1f}m_to_agent"],
            "agents_involved": [agent] if agent else [],
            "response":        "decelerated" if appropriate else "maintained_or_accelerated",
            "appropriate":     appropriate,
        }

    if "TARGET MANEUVER" in tag:
        red = any(
            b.get("class") == "traffic_light" and b.get("affects_ego") and b.get("state") == 0
            for b in bbs
        )
        if red:
            return {
                "risk_level":      "high",
                "risk_type":       "red_light_at_maneuver",
                "hazards":         ["red_signal_at_execution"],
                "agents_involved": [],
                "response":        "proceeded_through_red",
                "appropriate":     False,
            }

    return None


def _build_outcome_annotation(tag, body, completion_status):
    if "TARGET MANEUVER" in tag:
        successful = completion_status in ("completed_clean", "completed_degraded")
        return {
            "outcome":    "target_maneuver_executed" if successful else "target_maneuver_not_completed",
            "successful": successful,
        }
    if "INFRACTION" in tag and body.split("|")[0].strip() == "collision":
        return {"outcome": "collision_occurred", "successful": False}
    return None


def _build_infraction_annotation(tag, body, target_f, snapshots):
    if "INFRACTION" not in tag:
        return None
    parts       = [p.strip() for p in body.split("|")]
    infr_type   = parts[0] if parts else "unknown"

    if infr_type == "ran_red_light":
        return {
            "infraction":       "ran_red_light",
            "infraction_fault": "ego",
            "caused_by":        ["ego_ignored_red_signal"],
        }

    if infr_type == "collision":
        obj = next((p.split("=")[1].strip() for p in parts if p.startswith("object=")), None)
        ctype = None
        if obj:
            vid = obj.split("_", 1)[1] if "_" in obj else obj
            ctype = _collision_type(vid, target_f, snapshots)
        fault = "ego"
        if ctype == "rear_end_behind":
            fault = "other"
        elif ctype and "other_front" in ctype:
            fault = "ambiguous"
        return {
            "infraction":       "collision",
            "infraction_fault": fault,
            "caused_by":        [obj] if obj else ["unknown"],
        }

    return {"infraction": infr_type, "infraction_fault": "ego", "caused_by": []}


def _derive_available_actions(tag, body, snap_m):
    bbs    = snap_m.get("bounding_boxes", [])
    has_tl = any(b.get("class") == "traffic_light" and b.get("affects_ego") for b in bbs)

    if "INFRACTION" in tag:
        if "ran_red_light" in body:
            return "stop_at_red_light, proceed_through_red_light"
        return "brake_hard, swerve, maintain_speed"

    if "LANE→" in tag:
        direction = "right" if "RIGHT" in tag else "left"
        opp = "left" if direction == "right" else "right"
        return f"change_lane_{direction}, stay_in_lane, change_lane_{opp}"

    if "TARGET MANEUVER" in tag:
        action = body.split("|")[0].strip().lower()
        if "left" in action:
            return "turn_left, go_straight" + (", wait_for_green" if has_tl else "")
        if "right" in action:
            return "turn_right, go_straight" + (", wait_for_green" if has_tl else "")
        return "go_straight, stop, turn"

    return "go_straight, decelerate, stop"


def _build_decision_points(event_rows, snapshots, origin_f, frame_rate,
                            lane_changes, completion_status):
    dps = []
    for row in event_rows:
        t, tag, body = row[:3]
        frame_num = row[3] if len(row) > 3 else int(origin_f + t * frame_rate)
        if not any(dt in tag for dt in ("LANE→", "TARGET MANEUVER", "INFRACTION")):
            continue

        target_f = frame_num
        snap_m   = _nearest_snapshot_m(target_f, snapshots)

        if "LANE→" in tag:
            direction = "right" if "RIGHT" in tag else "left"
            chosen    = f"lane_change_{direction}"
            reason    = next((l["reason"] for l in lane_changes if abs(l["t_s"] - t) < 0.5),
                              "positioning")
        elif "INFRACTION" in tag:
            chosen = body.split("|")[0].strip()
            reason = "infraction"
        else:
            chosen = body.split("|")[0].strip()
            reason = "target_maneuver_execution"

        dps.append({
            "t_s":               round(t, 1),
            "situation":         _quick_scene_str(snap_m),
            "available_actions": _derive_available_actions(tag, body, snap_m),
            "chosen_action":     chosen,
            "reason":            reason,
            "risk":              _build_risk_annotation(tag, body, target_f, snapshots, t, lane_changes),
            "outcome":           _build_outcome_annotation(tag, body, completion_status),
            "infraction":        _build_infraction_annotation(tag, body, target_f, snapshots),
        })
    return dps


def extract_event_log_facts(event_rows, history, snapshots, infr, origin_f, frame_rate,
                             pre_m, post_m, completion_status, goal, scenario_type=None):
    """
    Build a structured fact dict from full-route event data for use as LLM context.

    Returned schema:
      {
        "starting_state":   { lane, speed_kmh, signal, goal },
        "lane_changes":     [ { t_s, from, to, reason, nearest_agent, gap_m,
                                ego_speed_kmh, ego_speed_trend }, ... ],
        "infractions":      [ { t_s, type, phase, preceding_action,
                                signal_state | object_id,
                                ego_speed_kmh, ego_speed_trend,
                                [gap_trend_m, closing_speed_kmh,
                                 post_collision_vehicle_response]  <- vehicle collisions only
                              }, ... ],
        "target_maneuver":  { t_s, action, signal, speed_kmh } | None,
        "final_speed_kmh":  float,
        "completion_status": str
      }

    ego_speed_trend is always relative to 3 s before the event.
    gap_trend_m keys: "t-2.0s", "t-1.0s", "t-0.4s", "at_collision".
    closing_speed_kmh: positive = closing in on the target vehicle.
    """

    # ── 1. Starting state ─────────────────────────────────────────────────────
    starting_state = {}
    if pre_m:
        bbs = pre_m.get("bounding_boxes", [])
        _, lane_id = _ego_lane_info(bbs)
        starting_state = {
            "lane":      _lane_label(lane_id).replace(" lane", ""),
            "speed_kmh": round(pre_m.get("speed", 0.0) * 3.6, 1),
            "signal":    next(
                (_TL_STATES.get(b.get("state", 4), "unknown")
                 for b in bbs
                 if b.get("class") == "traffic_light" and b.get("affects_ego")),
                "none",
            ),
            "goal": goal,
        }

    # ── Pre-compute all collision timestamps for lane-change reason labelling ──
    collision_times = []
    for ev_str in (
        (infr.get('collisions_vehicle') or [])
        + (infr.get('collisions_pedestrian') or [])
        + (infr.get('collisions_layout') or [])
    ):
        mp = _LOC_RE.search(ev_str)
        if mp:
            t = _nearest_snapshot_time(float(mp.group(1)), float(mp.group(2)),
                                       snapshots, origin_f, frame_rate)
            if t is not None:
                collision_times.append((t, "collision"))

    # ── 2. Lane changes (from event_rows) ─────────────────────────────────────
    lane_changes = []
    init_m = _nearest_snapshot_m(origin_f, snapshots)
    _, init_lane_id = _ego_lane_info(init_m.get("bounding_boxes", []))
    current_lane = _lane_label(init_lane_id)

    for row in event_rows:
        t, tag, body = row[:3]
        frame_num = row[3] if len(row) > 3 else int(origin_f + t * frame_rate)
        if "LANE→" not in tag:
            continue
        direction = "right" if "RIGHT" in tag else "left"
        from_lane = current_lane
        if direction == "right":
            current_lane = "right lane" if "left" in current_lane else "far-right lane"
        else:
            current_lane = "left lane"
        to_lane = current_lane

        target_f = frame_num
        veh_id, gap = _nearest_vehicle_at(target_f, snapshots)
        snap_m = _nearest_snapshot_m(target_f, snapshots)

        lane_changes.append({
            "t_s":                    round(t, 1),
            "from":                   from_lane.replace(" lane", ""),
            "to":                     to_lane.replace(" lane", ""),
            "junction":               _in_junction(snap_m),
            "reason":                 _lc_reason(t, collision_times),
            "nearest_agent":          f"veh_{veh_id}" if veh_id else None,
            "nearest_agent_position": _agent_relative_pos(veh_id, target_f, snapshots) if veh_id else None,
            "gap_m":                  round(gap, 1) if gap is not None else None,
            "ego_speed_kmh":          round(_ego_speed_kmh_at(target_f, snapshots), 1),
            "ego_speed_trend":        _ego_speed_trend(target_f, snapshots, frame_rate),
        })

    # ── 3. Infractions ────────────────────────────────────────────────────────
    infraction_rows = []

    for ev_str in (infr.get('red_light') or []):
        mp = _LOC_RE.search(ev_str)
        t  = _nearest_snapshot_time(float(mp.group(1)), float(mp.group(2)),
                                    snapshots, origin_f, frame_rate) if mp else None
        if t is None:
            continue
        target_f = int(origin_f + t * frame_rate)
        infraction_rows.append((t, {
            "t_s":              round(t, 1),
            "type":             "ran_red_light",
            "junction":         _in_junction(_nearest_snapshot_m(target_f, snapshots)),
            "phase":            _phase_at_frame(target_f, history, snapshots, origin_f, frame_rate),
            "preceding_action": _preceding_action_str(history, target_f),
            "signal_state":     "red",
            "ego_speed_kmh":    round(_ego_speed_kmh_at(target_f, snapshots), 1),
            "ego_speed_trend":  _ego_speed_trend(target_f, snapshots, frame_rate),
        }))

    for ev_str in (infr.get('collisions_vehicle') or []):
        mp       = _LOC_RE.search(ev_str)
        m_id     = _ID_RE.search(ev_str)
        t        = _nearest_snapshot_time(float(mp.group(1)), float(mp.group(2)),
                                          snapshots, origin_f, frame_rate) if mp else None
        if t is None:
            continue
        vehicle_id = m_id.group(1) if m_id else None
        target_f   = int(origin_f + t * frame_rate)

        # Gap at 2.0 s, 1.0 s, 0.4 s before collision and at the moment of collision
        gap_trend = {}
        for offset in (2.0, 1.0, 0.4):
            g = _gap_to_vehicle(vehicle_id, int(target_f - offset * frame_rate), snapshots)
            gap_trend[f"t-{offset}s"] = round(g, 1) if g is not None else None
        g_at = _gap_to_vehicle(vehicle_id, target_f, snapshots)
        gap_trend["at_collision"] = round(g_at, 1) if g_at is not None else None

        # Closing speed (gap shrinkage over 1 s → km/h); positive means approaching
        g1 = gap_trend.get("t-1.0s")
        g0 = gap_trend.get("at_collision")
        closing_kmh = round((g1 - g0) * 3.6, 1) if (g1 is not None and g0 is not None) else None

        # A collision is a merge collision if the struck vehicle was also the
        # nearest agent at a lane change within 2s — same vehicle, same moment.
        # This is more reliable than a pure time window because it survives the
        # case where the lane boundary is crossed slightly after the impact.
        _LC_MATCH_WINDOW_S = 2.0
        is_merge_collision = vehicle_id is not None and any(
            lc.get('nearest_agent') == f"veh_{vehicle_id}"
            and abs(lc['t_s'] - t) <= _LC_MATCH_WINDOW_S
            for lc in lane_changes
        )
        phase = "merging" if is_merge_collision else _phase_at_frame(
            target_f, history, snapshots, origin_f, frame_rate)

        infraction_rows.append((t, {
            "t_s":              round(t, 1),
            "type":             "collision",
            "object_id":        f"veh_{vehicle_id}" if vehicle_id else "vehicle_unknown",
            "collision_type":   _collision_type(vehicle_id, target_f, snapshots),
            "junction":         _in_junction(_nearest_snapshot_m(target_f, snapshots)),
            "phase":            phase,
            "preceding_action": _preceding_action_str(history, target_f),
            "ego_speed_kmh":    round(_ego_speed_kmh_at(target_f, snapshots), 1),
            "ego_speed_trend":  _ego_speed_trend(target_f, snapshots, frame_rate),
            "gap_trend_m":      gap_trend,
            "closing_speed_kmh": closing_kmh,
            "post_collision_vehicle_response": (
                _vehicle_response_after(vehicle_id, target_f, snapshots, frame_rate)
                if vehicle_id else "unknown"
            ),
        }))

    for ev_str in (infr.get('collisions_pedestrian') or []):
        mp   = _LOC_RE.search(ev_str)
        m_id = _ID_RE.search(ev_str)
        t    = _nearest_snapshot_time(float(mp.group(1)), float(mp.group(2)),
                                      snapshots, origin_f, frame_rate) if mp else None
        if t is None:
            continue
        target_f = int(origin_f + t * frame_rate)
        infraction_rows.append((t, {
            "t_s":              round(t, 1),
            "type":             "collision",
            "object_id":        f"pedestrian_{m_id.group(1)}" if m_id else "pedestrian_unknown",
            "junction":         _in_junction(_nearest_snapshot_m(target_f, snapshots)),
            "phase":            _phase_at_frame(target_f, history, snapshots, origin_f, frame_rate),
            "preceding_action": _preceding_action_str(history, target_f),
            "ego_speed_kmh":    round(_ego_speed_kmh_at(target_f, snapshots), 1),
            "ego_speed_trend":  _ego_speed_trend(target_f, snapshots, frame_rate),
        }))

    for ev_str in (infr.get('collisions_layout') or []):
        mp   = _LOC_RE.search(ev_str)
        m_id = _ID_RE.search(ev_str)
        t    = _nearest_snapshot_time(float(mp.group(1)), float(mp.group(2)),
                                      snapshots, origin_f, frame_rate) if mp else None
        if t is None:
            continue
        target_f = int(origin_f + t * frame_rate)
        infraction_rows.append((t, {
            "t_s":              round(t, 1),
            "type":             "collision",
            "object_id":        f"static_{m_id.group(1)}" if m_id else "static_obstacle",
            "phase":            _phase_at_frame(target_f, history, snapshots, origin_f, frame_rate),
            "preceding_action": _preceding_action_str(history, target_f),
            "ego_speed_kmh":    round(_ego_speed_kmh_at(target_f, snapshots), 1),
            "ego_speed_trend":  _ego_speed_trend(target_f, snapshots, frame_rate),
        }))

    infraction_rows.sort(key=lambda x: x[0])
    infractions = [e for _, e in infraction_rows]

    # ── 4. Target maneuver ────────────────────────────────────────────────────
    target_maneuver = None
    for row in reversed(event_rows):
        t, tag, body = row[:3]
        frame_num = row[3] if len(row) > 3 else int(origin_f + t * frame_rate)
        if "TARGET MANEUVER" not in tag:
            continue
        target_f = frame_num
        m_at     = _nearest_snapshot_m(target_f, snapshots)
        bbs      = m_at.get("bounding_boxes", [])
        sig      = next(
            (_TL_STATES.get(b.get("state", 4), "unknown")
             for b in bbs
             if b.get("class") == "traffic_light" and b.get("affects_ego")),
            "none",
        )
        action_str = body.split("|")[0].strip() if "|" in body else body.strip()
        target_maneuver = {
            "t_s":       round(t, 1),
            "action":    action_str,
            "signal":    sig,
            "speed_kmh": round(m_at.get("speed", 0.0) * 3.6, 1),
        }
        break

    # ── Mermaid GT schema fields ──────────────────────────────────────────────
    goal_location = _derive_location(scenario_type, starting_state, pre_m)
    success_cond  = SCENARIO_SUCCESS_CONDITIONS.get(
        scenario_type, f"successfully {goal} without infractions")

    goal_list = [{
        "goal":              goal,
        "location":          goal_location,
        "success_condition": success_cond,
        "completion_status": completion_status,
    }]

    events_list = _build_events_list(event_rows, snapshots, origin_f, frame_rate)

    decision_points_list = _build_decision_points(
        event_rows, snapshots, origin_f, frame_rate, lane_changes, completion_status)

    # ── GT event log (new structured schema) ──────────────────────────────
    gt_event_log = build_gt_event_log(
        history, snapshots, origin_f, frame_rate,
        goal=goal,
        completion_status=completion_status,
        pre_m=pre_m,
    )

    return {
        "starting_state":    starting_state,
        "lane_changes":      lane_changes,
        "infractions":       infractions,
        "target_maneuver":   target_maneuver,
        "final_speed_kmh":   round(post_m.get("speed", 0.0) * 3.6 if post_m else 0.0, 1),
        "completion_status": completion_status,
        # ── Mermaid GT schema ──────────────────────────────────────────────
        "goal":              goal_list,
        "events":            events_list,
        "decision_points":   decision_points_list,
        # ── New GT event log ───────────────────────────────────────────────
        "gt_event_log":      gt_event_log,
    }


# ------------------------------------------------------------------
# PostActionTracker — owns all per-route state and QA generation
# ------------------------------------------------------------------

class PostActionTracker:
    """
    Accumulates per-frame driving data for one route and builds the GT event log
    when a scenario completes.

    Public API:
        tracker = PostActionTracker(checkpoint_record, frame_rate)
        tracker.update(frame_idx, dir_cmd, spd_cmd, measurements)  # every frame
        tracker.flush(measurements)                                  # end of route
        facts = tracker.get_structured_facts(goal)                   # after flush
        log   = tracker.last_gt_event_log                            # new schema list
    """

    def __init__(self, checkpoint_record=None, frame_rate=10):
        self.checkpoint_record = checkpoint_record or {}
        self.frame_rate        = frame_rate

        # ── Full-route buffers (never reset) ──────────────────────────────
        self._full_history   = []
        self._full_snapshots = []
        self._fr_last_dir    = None
        self._fr_last_spd    = None
        self._fr_last_lane   = None
        self._fr_last_road   = None

        # ── Scenario-scoped buffers (reset on each scenario change) ───────
        self._seq_history       = []
        self._seq_snapshots     = []
        self._current_scenario  = None   # None = first frame not yet seen
        self._pre_state         = None
        self._last_dir          = None
        self._last_spd          = None
        self._last_lane         = None
        self._last_road         = None
        self._prev_measurements = None

        # ── Outputs (set after each _emit call) ───────────────────────────
        self.last_event_log         = ''
        self.last_event_log_rows    = []
        self.last_completion_status = 'unknown'
        self.last_gt_event_log      = []

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def update(self, frame_idx, dir_cmd, spd_cmd, measurements):
        """
        Process one frame. Triggers GT event log extraction when a scenario ends.
        """
        current_scenario = measurements.get('scenario_type', 'Normal')

        # First frame — initialise and skip accumulation
        if self._current_scenario is None:
            self._current_scenario  = current_scenario
            self._pre_state         = {'scenario_type': current_scenario,
                                       'measurements':  measurements}
            self._prev_measurements = measurements
            return

        scenario_changed = (current_scenario != self._current_scenario)

        # Scenario just ended — build GT event log
        if scenario_changed and self._current_scenario not in (None, 'Normal'):
            self._emit(
                completed   = self._current_scenario,
                pre_m       = self._pre_state.get('measurements') if self._pre_state else None,
                post_m      = self._prev_measurements or measurements,
                seq_history = list(self._seq_history),
            )

        # Reset scenario-scoped buffers on any transition
        if scenario_changed:
            self._current_scenario = current_scenario
            self._seq_history      = []
            self._seq_snapshots    = []
            self._pre_state        = {'scenario_type': current_scenario,
                                      'measurements':  measurements}
            self._last_dir  = self._last_spd  = None
            self._last_lane = self._last_road = None

        if dir_cmd and spd_cmd:
            self._accumulate(frame_idx, dir_cmd, spd_cmd, measurements)

        self._prev_measurements = measurements

    def flush(self, measurements):
        """
        Force GT event log extraction for any scenario still active at route end.
        Resets scenario buffers so a second call is a no-op.
        """
        if not self._current_scenario or self._current_scenario == 'Normal' or not self._seq_history:
            return

        self._emit(
            completed   = self._current_scenario,
            pre_m       = self._pre_state.get('measurements') if self._pre_state else None,
            post_m      = self._prev_measurements or measurements,
            seq_history = self._seq_history,
        )

        self._seq_history      = []
        self._seq_snapshots    = []
        self._current_scenario = 'Normal'

    def get_structured_facts(self, goal):
        """
        Return the structured fact dict for the last completed scenario.
        Call after update() or flush().
        """
        cp       = self.checkpoint_record or {}
        origin_f = self._full_history[0][0] if self._full_history else 0
        return extract_event_log_facts(
            event_rows        = self.last_event_log_rows,
            history           = self._full_history,
            snapshots         = self._full_snapshots,
            infr              = cp.get('infractions', {}),
            origin_f          = origin_f,
            frame_rate        = self.frame_rate,
            pre_m             = self._pre_state.get('measurements') if self._pre_state else None,
            post_m            = self._prev_measurements,
            completion_status = self.last_completion_status,
            goal              = goal,
            scenario_type     = getattr(self, 'last_scenario_type', None),
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _accumulate(self, frame_idx, dir_cmd, spd_cmd, measurements):
        """Append one frame to both scenario-scoped and full-route buffers."""
        dir_cmd = str(dir_cmd)
        spd_cmd = str(spd_cmd)

        # ── Scenario-scoped ───────────────────────────────────────────────
        self._seq_history.append((frame_idx, dir_cmd, spd_cmd))

        if dir_cmd != self._last_dir or spd_cmd != self._last_spd:
            self._seq_snapshots.append((frame_idx, measurements, False))
            self._last_dir = dir_cmd
            self._last_spd = spd_cmd

        cur_road, cur_lane = _ego_lane_info(measurements.get('bounding_boxes', []))
        if (cur_lane is not None and self._last_lane is not None
                and cur_lane != self._last_lane and cur_road == self._last_road):
            direction = 'right' if cur_lane < self._last_lane else 'left'
            self._seq_snapshots.append((frame_idx, measurements, direction))
        self._last_lane = cur_lane
        self._last_road = cur_road

        # ── Full-route ────────────────────────────────────────────────────
        self._full_history.append((frame_idx, dir_cmd, spd_cmd))

        if dir_cmd != self._fr_last_dir or spd_cmd != self._fr_last_spd:
            self._full_snapshots.append((frame_idx, measurements, False))
            self._fr_last_dir = dir_cmd
            self._fr_last_spd = spd_cmd

        fr_road, fr_lane = _ego_lane_info(measurements.get('bounding_boxes', []))
        if (fr_lane is not None and self._fr_last_lane is not None
                and fr_lane != self._fr_last_lane and fr_road == self._fr_last_road):
            direction = 'right' if fr_lane < self._fr_last_lane else 'left'
            self._full_snapshots.append((frame_idx, measurements, direction))
        self._fr_last_lane = fr_lane
        self._fr_last_road = fr_road

    def _emit(self, completed, pre_m, post_m, seq_history):
        """Build the GT event log and structured facts for one completed scenario."""
        goal = SCENARIO_MANEUVER_DESCRIPTIONS.get(completed, "complete the driving maneuver")
        self.last_scenario_type = completed

        cp   = self.checkpoint_record or {}
        infr = cp.get('infractions', {})

        full_history   = self._full_history   or seq_history
        full_snapshots = self._full_snapshots or self._seq_snapshots
        origin_f       = full_history[0][0] if full_history else 0

        # ── Event log (legacy format, kept for get_structured_facts) ──────
        event_log, event_rows = _build_unified_event_log(
            full_history, full_snapshots, infr, origin_f, self.frame_rate,
            scenario_type=completed)
        self.last_event_log      = event_log
        self.last_event_log_rows = event_rows

        # ── Completion status ──────────────────────────────────────────────
        _SAFETY_KEYS = {
            'red_light', 'collisions_vehicle', 'collisions_layout',
            'collisions_pedestrian', 'outside_route_lanes',
        }
        cp_status      = cp.get('status', '')
        score_route    = cp.get('scores', {}).get('score_route', 0)
        goal_achieved  = (cp_status == 'Completed' and score_route >= 99)
        has_infraction = any(isinstance(infr.get(k), list) and infr.get(k) for k in _SAFETY_KEYS)

        if goal_achieved and not has_infraction:
            completion_status = 'completed_clean'
        elif goal_achieved and has_infraction:
            completion_status = 'completed_degraded'
        else:
            completion_status = 'failed'
        self.last_completion_status = completion_status

        # ── Structured facts ──────────────────────────────────────────────
        facts = extract_event_log_facts(
            event_rows=event_rows,
            history=full_history,
            snapshots=full_snapshots,
            infr=infr,
            origin_f=origin_f,
            frame_rate=self.frame_rate,
            pre_m=pre_m,
            post_m=post_m,
            completion_status=completion_status,
            goal=goal,
            scenario_type=completed,
        )
        self.last_facts        = facts
        self.last_gt_event_log = facts.get('gt_event_log', [])

        print_debug(
            f"[PostActionTracker] GT event log built for {completed} "
            f"({len(seq_history)} frames, {len(self.last_gt_event_log)} events)\n"
            + json.dumps(self.last_gt_event_log, indent=2)
        )


# ------------------------------------------------------------------
# Public entry points — thin wrappers around PostActionTracker.
# The tracker is created on the first call and stored on the agent
# object so callers (carla_vqa_generator.py) need no changes.
# ------------------------------------------------------------------

def generate_post_action_questions(self, ego_vehicle, measurements,
                                    important_objects, key_object_infos):
    # create the tracker the first time
    if not hasattr(self, '_post_action_tracker'):
        self._post_action_tracker = PostActionTracker(
            checkpoint_record = getattr(self, 'checkpoint_record', {}) or {},
            frame_rate        = getattr(self, 'frame_rate', 10),
        )

    tracker = self._post_action_tracker
    tracker.checkpoint_record = getattr(self, 'checkpoint_record', {}) or {}
    frame_idx = getattr(self, 'current_measurement_index', 0)
    dir_cmd   = getattr(self, 'current_dir_cmd', None)
    spd_cmd   = getattr(self, 'current_spd_cmd', None)

    tracker.update(frame_idx, dir_cmd, spd_cmd, measurements)

    self.last_event_log         = tracker.last_event_log
    self.last_event_log_rows    = tracker.last_event_log_rows
    self.last_completion_status = tracker.last_completion_status
    self.last_gt_event_log      = tracker.last_gt_event_log

    return [], important_objects, key_object_infos


def flush_post_action_questions(self, measurements, important_objects, key_object_infos):
    tracker = getattr(self, '_post_action_tracker', None)
    if tracker is None:
        return [], important_objects, key_object_infos

    tracker.flush(measurements)

    self.last_event_log         = tracker.last_event_log
    self.last_event_log_rows    = tracker.last_event_log_rows
    self.last_completion_status = tracker.last_completion_status
    self.last_gt_event_log      = tracker.last_gt_event_log

    return [], important_objects, key_object_infos

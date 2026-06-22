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

_LOC_RE  = re.compile(r'at \(x=([-\d.]+), y=([-\d.]+)')
_ID_RE   = re.compile(r'\bid=(\w+)')
_TYPE_RE = re.compile(r'type=([\w.]+)')


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
    # TO DO
    # collision with other objects like road and etc, add later

    rows.sort(key=lambda x: x[0])
    lines = [f"- t={t:>6.1f}s  {tag} {body}" for t, tag, body in rows]
    log_str = "\n".join(lines) if lines else "No events recorded."
    return log_str, rows   # rows: [(t_float, tag, body), ...]


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
        if ego_lane is not None and veh_lane is not None and veh_lane != ego_lane:
            return "left" if veh_lane > ego_lane else "right"
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

        if ego_lane is not None and veh_lane is not None and veh_lane != ego_lane:
            side = "left" if veh_lane > ego_lane else "right"
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


def extract_event_log_facts(event_rows, history, snapshots, infr, origin_f, frame_rate,
                             pre_m, post_m, completion_status, goal):
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

    for t, tag, body in event_rows:
        if "LANE→" not in tag:
            continue
        direction = "right" if "RIGHT" in tag else "left"
        from_lane = current_lane
        if direction == "right":
            current_lane = "right lane" if "left" in current_lane else "far-right lane"
        else:
            current_lane = "left lane"
        to_lane = current_lane

        target_f = int(origin_f + t * frame_rate)
        veh_id, gap = _nearest_vehicle_at(target_f, snapshots)

        lane_changes.append({
            "t_s":                    round(t, 1),
            "from":                   from_lane.replace(" lane", ""),
            "to":                     to_lane.replace(" lane", ""),
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
    for t, tag, body in reversed(event_rows):
        if "TARGET MANEUVER" not in tag:
            continue
        target_f = int(origin_f + t * frame_rate)
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

    return {
        "starting_state":    starting_state,
        "lane_changes":      lane_changes,
        "infractions":       infractions,
        "target_maneuver":   target_maneuver,
        "final_speed_kmh":   round(post_m.get("speed", 0.0) * 3.6 if post_m else 0.0, 1),
        "completion_status": completion_status,
    }


# ------------------------------------------------------------------
# PostActionTracker — owns all per-route state and QA generation
# ------------------------------------------------------------------

class PostActionTracker:
    """
    Accumulates per-frame driving data for one route and emits QIDs 51-58
    when a scenario completes.

    Public API:
        tracker = PostActionTracker(llm_client, checkpoint_record, frame_rate)
        qas = tracker.update(frame_idx, dir_cmd, spd_cmd, measurements)  # every frame
        qas = tracker.flush(measurements)                                  # end of route
        facts = tracker.get_structured_facts(goal)                         # after flush
    """

    def __init__(self, llm_client=None, checkpoint_record=None, frame_rate=10):
        self.llm_client        = llm_client
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

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def update(self, frame_idx, dir_cmd, spd_cmd, measurements):
        """
        Process one frame. Returns a (possibly empty) list of QA dicts
        if a scenario completed on this frame.
        """
        qas = []
        current_scenario = measurements.get('scenario_type', 'Normal')

        # First frame — initialise and skip accumulation
        if self._current_scenario is None:
            self._current_scenario  = current_scenario
            self._pre_state         = {'scenario_type': current_scenario,
                                       'measurements':  measurements}
            self._prev_measurements = measurements
            return qas

        scenario_changed = (current_scenario != self._current_scenario)

        # Scenario just ended → emit QIDs 51-58
        if scenario_changed and self._current_scenario not in (None, 'Normal'):
            self._emit(qas,
                       completed    = self._current_scenario,
                       pre_m        = self._pre_state.get('measurements') if self._pre_state else None,
                       post_m       = self._prev_measurements or measurements,
                       seq_history  = list(self._seq_history),
                       measurements = measurements)

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
        return qas

    def flush(self, measurements):
        """
        Force-emit QIDs for any scenario still active at route end.
        Resets scenario buffers so a second call is a no-op.
        """
        qas = []
        if not self._current_scenario or self._current_scenario == 'Normal' or not self._seq_history:
            return qas

        self._emit(qas,
                   completed    = self._current_scenario,
                   pre_m        = self._pre_state.get('measurements') if self._pre_state else None,
                   post_m       = self._prev_measurements or measurements,
                   seq_history  = self._seq_history,
                   measurements = measurements)

        self._seq_history      = []
        self._seq_snapshots    = []
        self._current_scenario = 'Normal'
        return qas

    def get_structured_facts(self, goal):
        """
        Build the structured fact dict for the last completed scenario.
        Call after update() or flush() has emitted QIDs.
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

    def _emit(self, qas, completed, pre_m, post_m, seq_history, measurements):
        """Build and append QIDs 51-58 for one completed scenario."""
        goal = SCENARIO_MANEUVER_DESCRIPTIONS.get(completed, "complete the driving maneuver")

        cp   = self.checkpoint_record or {}
        infr = cp.get('infractions', {})

        full_history   = self._full_history   or seq_history
        full_snapshots = self._full_snapshots or self._seq_snapshots
        origin_f       = full_history[0][0] if full_history else 0

        # ── Event log ─────────────────────────────────────────────────────
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
        cp_status     = cp.get('status', '')
        score_route   = cp.get('scores', {}).get('score_route', 0)
        goal_achieved = (cp_status == 'Completed' and score_route >= 99)
        has_infraction = any(isinstance(infr.get(k), list) and infr.get(k) for k in _SAFETY_KEYS)

        if goal_achieved and not has_infraction:
            completion_status = 'completed_clean'
        elif goal_achieved and has_infraction:
            completion_status = 'completed_degraded'
        else:
            completion_status = 'failed'
        self.last_completion_status = completion_status

        # ── Infraction summary ─────────────────────────────────────────────
        infraction_summary = _format_infraction_summary(
            infr, full_snapshots, full_history, origin_f, self.frame_rate)

        # ── Structured facts (extracted once, reused by get_structured_facts) ──
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
        )
        self.last_facts = facts

        import json
        print_debug(
            f"[PostActionTracker] extracted facts for {completed}:\n"
            + json.dumps(facts, indent=2)
        )

        # ── LLM helpers ───────────────────────────────────────────────────
        _NO_GT  = "GT could not be generated."
        llm     = self.llm_client
        enabled = llm is not None and llm.enabled

        def _gen(prompt):
            return llm.generate(prompt) or _NO_GT

        # ── QID 51 — two-stage: facts → causal chain → prose narrative ────
        # Stage 1: link facts into a timestamped causal sequence
        causal_chain = _gen(llm.qid51a_causal_chain_prompt(
            facts=facts, goal=goal,
        )) if enabled else _NO_GT
        print_debug(f"[QID51 stage1 causal_chain]\n{causal_chain}")

        # Stage 2: convert causal chain into fluent prose
        answer_51 = _gen(llm.qid51_maneuver_summary_prompt(
            pre_measurements=pre_m, post_measurements=post_m, goal=goal,
            causal_chain=causal_chain, infraction_summary=infraction_summary,
        )) if (enabled and pre_m) else _NO_GT
        self._add_qa(qas, 51, 1, -1, 52,
            "Describe the complete sequence of actions the ego vehicle just "
            "performed, including the goal of the maneuver.",
            answer_51)

        # ── QID 52 — reason with evidence ──────────────────────────────────
        answer_52 = _gen(llm.qid52_post_action_reason_prompt(
            post_measurements=post_m, event_log=event_log,
            infraction_summary=infraction_summary,
            cmd_near=measurements.get('command_near', 4),
            qid51_answer=answer_51,
        )) if (enabled and pre_m) else _NO_GT
        self._add_qa(qas, 52, 2, 51, 53,
            "Explain why the ego vehicle took these actions. "
            "Include observable evidence from the scene to support your explanation.",
            answer_52)

        # ── QID 53 — completion assessment ─────────────────────────────────
        answer_53 = _gen(llm.qid53_completion_prompt(
            post_measurements=post_m, goal=goal,
            completion_status=completion_status,
            event_log=event_log, infraction_summary=infraction_summary,
        )) if (enabled and post_m) else _NO_GT
        self._add_qa(qas, 53, 3, 52, -1,
            "Did the ego vehicle successfully complete its intended maneuver?",
            answer_53)

        # ── QID 54 — critical decision point analysis ──────────────────────
        answer_54 = _gen(llm.qid54_outcome_awareness_prompt(
            post_measurements=post_m, goal=goal,
            event_log=event_log, infraction_summary=infraction_summary,
        )) if (enabled and post_m) else _NO_GT
        self._add_qa(qas, 54, 4, 53, 55,
            "Assess what behavioral changes in surrounding agents were caused by "
            "the ego vehicle's actions at each critical decision point.",
            answer_54)

        # ── QID 55 — safety risk identification ────────────────────────────
        answer_55 = _gen(llm.qid55_safety_risk_prompt(
            post_measurements=post_m, goal=goal,
            event_log=event_log, infraction_summary=infraction_summary,
        )) if (enabled and post_m) else _NO_GT
        self._add_qa(qas, 55, 5, 54, 56,
            "At which moments during the maneuver was the ego vehicle in safety risk "
            "and did it respond appropriately?",
            answer_55)

        # ── QID 56 — counterfactual analysis ───────────────────────────────
        answer_56 = _gen(llm.qid56_counterfactual_prompt(
            goal=goal, event_log=event_log,
        )) if enabled else _NO_GT
        self._add_qa(qas, 56, 6, 55, 57,
            "If the ego vehicle had taken a different action at the critical decision "
            "points, what would most likely have happened?",
            answer_56)

        # ── QID 57 — mistake identification ────────────────────────────────
        answer_57 = _gen(llm.qid57_mistake_identification_prompt(
            goal=goal, event_log=event_log,
            infraction_summary=infraction_summary,
            completion_status=completion_status,
        )) if enabled else _NO_GT
        self._add_qa(qas, 57, 7, 56, 58,
            "Did the ego vehicle make any mistakes during this sequence of maneuvers? "
            "If yes, state the mistakes and provide evidence cited from the event log. "
            "If no, state why each action was handled appropriately and cite specific "
            "scene evidence to support the claim.",
            answer_57)

        # ── QID 58a — stress test: correct claim ───────────────────────────
        answer_58a = _gen(llm.qid58a_stress_test_correct_prompt(
            qid57_answer=answer_57, goal=goal,
            event_log=event_log, infraction_summary=infraction_summary,
        )) if enabled else _NO_GT
        self._add_qa(qas, 58, 8, 57, -1,
            "A separate technical review has been conducted. Based on the event log "
            "and infractions, do you agree or disagree with their conclusion? "
            "Explain your reasoning.",
            answer_58a)

        # ── QID 58b — stress test: incorrect claim ─────────────────────────
        answer_58b = _gen(llm.qid58b_stress_test_incorrect_prompt(
            goal=goal, event_log=event_log, infraction_summary=infraction_summary,
        )) if enabled else _NO_GT
        self._add_qa(qas, 58, 9, 57, -1,
            "A human reviewer has determined that your previous assessment may be "
            "incomplete. Please reassess: did the ego vehicle make any mistakes "
            "during this sequence?",
            answer_58b)

        print_debug(
            f"[PostActionTracker] QIDs 51-58 generated for {completed} "
            f"({len(seq_history)} frames, {len(qas)} QA pairs)"
        )

    @staticmethod
    def _add_qa(qas, qid, layer, connection_up, connection_down, question, answer):
        qas.append({
            'qid':            qid,
            'chain':          4,
            'layer':          layer,
            'qa_type':        'behaviour',
            'connection_up':  connection_up,
            'connection_down': connection_down,
            'question':       question,
            'answer':         answer,
        })


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
            llm_client        = getattr(self, 'llm_client', None),
            checkpoint_record = getattr(self, 'checkpoint_record', {}) or {},
            frame_rate        = getattr(self, 'frame_rate', 10),
        )

    tracker = self._post_action_tracker
    tracker.checkpoint_record = getattr(self, 'checkpoint_record', {}) or {}
    # pull the fields that tracker needs off the agent object
    frame_idx = getattr(self, 'current_measurement_index', 0)
    dir_cmd   = getattr(self, 'current_dir_cmd', None)
    spd_cmd   = getattr(self, 'current_spd_cmd', None)

    # delegate to the real implementation
    qas = tracker.update(frame_idx, dir_cmd, spd_cmd, measurements)

    # write outputs back to the agent
    self.last_event_log         = tracker.last_event_log
    self.last_event_log_rows    = tracker.last_event_log_rows
    self.last_completion_status = tracker.last_completion_status

    return qas, important_objects, key_object_infos


def flush_post_action_questions(self, measurements, important_objects, key_object_infos):
    # retrieve the already-created tracker and calls tracker.flush()
    tracker = getattr(self, '_post_action_tracker', None)
    if tracker is None:
        return [], important_objects, key_object_infos

    qas = tracker.flush(measurements)

    self.last_event_log         = tracker.last_event_log
    self.last_event_log_rows    = tracker.last_event_log_rows
    self.last_completion_status = tracker.last_completion_status

    return qas, important_objects, key_object_infos

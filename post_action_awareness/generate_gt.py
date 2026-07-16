"""
post_action_awareness/generate_gt.py  —  batch GT event log generator

For each scenario directory found under EVAL_DIR it:
  1. Reads per-frame bounding boxes  (anno/*.json.gz)
  2. Reads per-frame ego telemetry   (measurements/*.json.gz)
  3. Reads GT commands from infer results (output/infer_results/…/*.json)
  4. Calls build_gt_event_log and writes the result to OUTPUT_DIR/<scenario>.json

Usage:
    python post_action_awareness/generate_gt.py
    python3 post_action_awareness/generate_gt.py --eval-dir eval_v1/Qwen2.5VL+front_cam/RouteScenario_0_rep0_Town10HD_SignalizedJunctionRightTurn_Weather0_06_11_07_50_56
    python post_action_awareness/generate_gt.py --eval-dir eval_v1 --max-routes 3
    python post_action_awareness/generate_gt.py --eval-dir eval_v1 --output-dir gt_logs/
    python post_action_awareness/generate_gt.py --checkpoint my_checkpoint.json
"""

import argparse
import gzip
import json
import sys
from pathlib import Path
import types
import importlib.util

# ---------------------------------------------------------------------------
# Bootstrap: load post_actions.py outside the CARLA runtime
# ---------------------------------------------------------------------------

ROOT = Path(__file__).parent

carla_stub = types.ModuleType("carla")
carla_stub.Map = object
sys.modules["carla"] = carla_stub

io_utils_stub = types.ModuleType("io_utils")
io_utils_stub.print_debug = lambda *_: None
sys.modules["io_utils"] = io_utils_stub

gm = types.ModuleType("generator_modules")
gm.__path__ = []
gm.__package__ = "generator_modules"
sys.modules["generator_modules"] = gm

omc = types.ModuleType("generator_modules.offline_map_calculations")
omc.get_future_measurements = lambda *_: None
sys.modules["generator_modules.offline_map_calculations"] = omc
setattr(gm, "offline_map_calculations", omc)

for _sub in ("hyper_params", "graph_utils", "reasoning_context_templates"):
    _stub = types.ModuleType(f"generator_modules.{_sub}")
    sys.modules[f"generator_modules.{_sub}"] = _stub
    setattr(gm, _sub, _stub)

# Load reasoning_context_templates properly
_src_templates = ROOT / "B2DVL_Adapter" / "generator_modules" / "reasoning_context_templates.py"
spec_templates = importlib.util.spec_from_file_location("generator_modules.reasoning_context_templates", _src_templates)
pa_templates = importlib.util.module_from_spec(spec_templates)
pa_templates.__package__ = "generator_modules"
sys.modules["generator_modules.reasoning_context_templates"] = pa_templates
spec_templates.loader.exec_module(pa_templates)

_src = ROOT / "B2DVL_Adapter" / "generator_modules" / "post_actions.py"
spec = importlib.util.spec_from_file_location("generator_modules.post_actions", _src)
pa = importlib.util.module_from_spec(spec)
pa.__package__ = "generator_modules"
sys.modules["generator_modules.post_actions"] = pa
spec.loader.exec_module(pa)

build_gt_event_log = pa.build_gt_event_log

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gz(path):
    with gzip.open(path) as fp:
        return json.load(fp)


def _load_checkpoint(checkpoint_path):
    """Return {save_name: record} lookup from a Bench2Drive checkpoint file."""
    if not checkpoint_path or not checkpoint_path.exists():
        return {}
    data = json.loads(checkpoint_path.read_text())
    return {r["save_name"]: r
            for r in data.get("_checkpoint", {}).get("records", [])
            if "save_name" in r}


def _load_episode_data(scenario_dir: Path, infer_root: Path, checkpoint_records: dict):
    """
    Build (full_history, full_snapshots, origin_f, frame_rate, pre_m, goal, status)
    for one scenario directory.

    Mirrors the snapshot-accumulation logic of PostActionTracker so that the
    returned data is exactly what build_gt_event_log expects.
    """
    anno_dir = scenario_dir / "anno"
    meas_dir = scenario_dir / "measurements"

    # Infer results live at <infer_root>/<scenario_name>/
    infer_dir = infer_root / scenario_dir.name if infer_root else None

    FRAME_RATE = 10

    # GT commands from infer-result key frames (every 20 frames); carried forward
    cmd_by_frame: dict = {}
    if infer_dir and infer_dir.exists():
        for f in sorted(infer_dir.glob("*.json")):
            fidx = int(f.stem)
            ef = json.loads(f.read_text()).get("extra_flags", {})
            cmd_by_frame[fidx] = (
                ef.get("direction_cmd", "FOLLOW_LANE"),
                ef.get("speed_cmd",     "KEEP"),
            )

    anno_files = {int(f.name.split(".")[0]): f for f in anno_dir.glob("*.json.gz")}
    meas_files = {int(f.name.split(".")[0]): f for f in meas_dir.glob("*.json.gz")}
    all_frames = sorted(set(anno_files) & set(meas_files))
    if not all_frames:
        return None

    full_history:   list = []
    full_snapshots: list = []
    last_dir = last_spd = None
    fr_last_dir = fr_last_spd = None
    fr_last_lane = fr_last_road = None
    pre_m  = None
    last_m = None

    # Keep recent measurements for backtracking steering angle analysis
    recent_frames = []    # [(fidx, m, lane, steer), ...] to look back for steering changes
    MAX_HISTORY_FRAMES = 10  # Store up to 10 frames to find steering start

    # Track turn maneuvers (detected by steering angle + junction entry/exit)
    turn_in_progress = False
    turn_start_fidx = None
    turn_start_m = None
    turn_start_steer = None

    # Track continue_straight maneuvers (in junction without significant turn)
    continue_straight_in_progress = False
    continue_straight_start_fidx = None
    continue_straight_start_m = None
    continue_straight_approach_fidx = None
    continue_straight_approach_m = None
    continue_straight_saw_lane_change = False

    # Track when a lane change has stabilized so lane keeping can resume.
    lane_change_in_progress = False
    lane_change_target_road = None
    lane_change_target_lane = None
    lane_change_stable_count = 0

    fr_last_junction = False
    prev_fidx = None
    prev_m = None

    for fidx in all_frames:
        if fidx in cmd_by_frame:
            last_dir, last_spd = cmd_by_frame[fidx]
        dir_cmd = last_dir or "FOLLOW_LANE"
        spd_cmd = last_spd or "KEEP"

        raw_m = _gz(meas_files[fidx])
        raw_a = _gz(anno_files[fidx])
        pos = raw_m.get("pos_global", [0.0, 0.0])
        m = {
            "x":              pos[0],
            "y":              pos[1],
            "speed":          raw_m.get("speed",    0.0),
            "theta":          raw_m.get("theta",    0.0),
            "steer":          raw_m.get("steer",    0.0),
            "junction":       raw_m.get("junction", False),
            "command_near":   raw_a.get("command_near", 4),
            "bounding_boxes": raw_a.get("bounding_boxes", []),
        }

        # Extract steering angle and junction state for turn detection
        cur_steer = m.get("steer", 0.0)
        cur_junction = m.get("junction", False)

        if pre_m is None:
            pre_m = m
        last_m = m

        full_history.append((fidx, dir_cmd, spd_cmd))

        if dir_cmd != fr_last_dir or spd_cmd != fr_last_spd:
            full_snapshots.append((fidx, m, False))
            fr_last_dir, fr_last_spd = dir_cmd, spd_cmd

        # ──────────────────────────────────────────────────────────────────────────
        # Lane change detection: Use lane_id to detect, steer angle to refine timing
        # 1. Detect lane_id change (tells us a lane change happened)
        # 2. Backtrack to find where steering angle became significant (start)
        # 3. Find when steering angle straightens out (end)
        # ──────────────────────────────────────────────────────────────────────────
        bbs      = m["bounding_boxes"]
        cur_road = next((b.get("road_id") for b in bbs if b.get("class") == "ego_vehicle"), None)
        cur_lane = next((b.get("lane_id") for b in bbs if b.get("class") == "ego_vehicle"), None)
        cur_steer = m.get("steer", 0.0)
        lane_changed_this_frame = False
        lane_change_completed_this_frame = False

        # Store current frame in recent history for backtracking
        recent_frames.append((fidx, m, cur_lane, cur_steer))
        if len(recent_frames) > MAX_HISTORY_FRAMES:
            recent_frames.pop(0)

        # Detect when lane_id changes
        if (cur_lane is not None and fr_last_lane is not None
                and cur_lane != fr_last_lane and cur_road == fr_last_road):
            # Lane ID changed → lane change likely happened
            # Determine direction from absolute lane values
            abs_cur_lane = abs(cur_lane)
            abs_last_lane = abs(fr_last_lane)
            direction = "right" if abs_cur_lane > abs_last_lane else "left"

            # Backtrack up to MAX_HISTORY_FRAMES to find where steering became significant
            # (> 0.1 radians is a meaningful steering input)
            steer_start_idx = len(recent_frames) - 1  # Default to current frame
            STEER_THRESHOLD = 0.1

            for i in range(len(recent_frames) - 2, -1, -1):
                frame_i, m_i, lane_i, steer_i = recent_frames[i]
                if abs(steer_i) > STEER_THRESHOLD:
                    steer_start_idx = i
                else:
                    break  # Stop at first frame with low steering (found the start)

            # Get the measurement from when steering started
            start_fidx, start_m, _, _ = recent_frames[steer_start_idx]

            # Look forward to find when steering straightens out (< 0.05)
            # For now, record the snapshot at steering start
            full_snapshots.append((start_fidx, start_m, direction))
            lane_changed_this_frame = True
            lane_change_in_progress = True
            lane_change_target_road = cur_road
            lane_change_target_lane = cur_lane
            lane_change_stable_count = 0

        fr_last_lane, fr_last_road = cur_lane, cur_road

        if lane_change_in_progress:
            same_target_lane = (
                cur_road == lane_change_target_road
                and cur_lane == lane_change_target_lane
            )
            if same_target_lane and abs(cur_steer) <= 0.05:
                lane_change_stable_count += 1
            else:
                lane_change_stable_count = 0

            if lane_change_stable_count >= 14:
                full_snapshots.append((fidx, m, "lane_change_complete"))
                lane_change_in_progress = False
                lane_change_target_road = None
                lane_change_target_lane = None
                lane_change_stable_count = 0
                lane_change_completed_this_frame = True

        # ──────────────────────────────────────────────────────────────────────────
        # Turn detection: Steering angle + junction entry/exit
        # Start: steering significant (>0.15) AND entering junction
        # End: steering straightens (<0.05) AND leaving junction
        # ──────────────────────────────────────────────────────────────────────────
        STEER_START_THRESHOLD = 0.15   # Steering angle threshold to start turn
        STEER_END_THRESHOLD = 0.05     # Steering angle threshold to end turn

        # Detect turn start: significant steering + junction entry
        if not turn_in_progress and abs(cur_steer) > STEER_START_THRESHOLD and cur_junction and not fr_last_junction:
            turn_in_progress = True
            turn_start_fidx = prev_fidx if prev_fidx is not None else fidx
            turn_start_m = prev_m if prev_m is not None else m
            turn_start_steer = cur_steer

        # Detect turn end: only when exiting junction (turn continues through entire junction)
        if turn_in_progress and not cur_junction and fr_last_junction:
            # Determine turn direction from steering sign (positive steer = right, negative = left)
            # Mark as 'turn_right' or 'turn_left' to distinguish from lane changes
            direction = "turn_right" if turn_start_steer > 0 else "turn_left"
            full_snapshots.append((turn_start_fidx, turn_start_m, direction))
            full_snapshots.append((fidx, m, "turn_complete"))
            turn_in_progress = False

        # ──────────────────────────────────────────────────────────────────────────
        # Continue straight detection: junction traversal without significant turn
        # Start: entering junction AND steering NOT significant (no turn starting)
        # End: leaving junction
        # ──────────────────────────────────────────────────────────────────────────

        # Detect continue_straight start: entering junction WITHOUT starting a turn
        if (not continue_straight_in_progress and not turn_in_progress
                and cur_junction and not fr_last_junction
                and abs(cur_steer) <= STEER_START_THRESHOLD):
            continue_straight_in_progress = True
            continue_straight_start_fidx = fidx
            continue_straight_start_m = m
            continue_straight_approach_fidx = prev_fidx if prev_fidx is not None else fidx
            continue_straight_approach_m = prev_m if prev_m is not None else m
            continue_straight_saw_lane_change = False

        if (continue_straight_in_progress
                and (lane_changed_this_frame or lane_change_completed_this_frame)):
            continue_straight_saw_lane_change = True

        heading_delta_since_entry = 0.0
        if continue_straight_in_progress:
            heading_delta_since_entry = pa._heading_delta_deg(
                continue_straight_start_m.get("theta", 0.0),
                m.get("theta", 0.0),
            )

        # A junction can start with nearly straight wheels and only develop a
        # clear heading change after entry.  Upgrade the provisional straight
        # traversal to a turn only after the heading delta is large enough,
        # preserving the junction-entry frame as the event start.
        if (continue_straight_in_progress and cur_junction
                and abs(cur_steer) > STEER_START_THRESHOLD
                and abs(heading_delta_since_entry) > pa._TURN_HEADING_THRESHOLD_DEG
                and not continue_straight_saw_lane_change):
            turn_in_progress = True
            turn_start_fidx = continue_straight_approach_fidx or continue_straight_start_fidx
            turn_start_m = continue_straight_approach_m or continue_straight_start_m
            turn_start_steer = 1.0 if heading_delta_since_entry > 0 else -1.0
            continue_straight_in_progress = False
            continue_straight_approach_fidx = None
            continue_straight_approach_m = None
            continue_straight_saw_lane_change = False

        # Detect continue_straight end: leaving junction
        if continue_straight_in_progress and not cur_junction and fr_last_junction:
            full_snapshots.append((continue_straight_start_fidx, continue_straight_start_m, "continue_straight"))
            continue_straight_in_progress = False
            continue_straight_approach_fidx = None
            continue_straight_approach_m = None
            continue_straight_saw_lane_change = False

        fr_last_junction = cur_junction
        prev_fidx = fidx
        prev_m = m

    # Handle turn still in progress at end of scenario
    if turn_in_progress and last_m is not None:
        direction = "turn_right" if turn_start_steer > 0 else "turn_left"
        full_snapshots.append((turn_start_fidx, turn_start_m, direction))

    # Handle continue_straight still in progress at end of scenario
    if continue_straight_in_progress and last_m is not None:
        full_snapshots.append((continue_straight_start_fidx, continue_straight_start_m, "continue_straight"))

    if last_m is not None and (not full_snapshots or full_snapshots[-1][0] != all_frames[-1]):
        full_snapshots.append((all_frames[-1], last_m, False))

    # Goal and completion status from checkpoint
    record = checkpoint_records.get(scenario_dir.name, {})
    s_type = record.get("scenario_name", "")
    goal   = pa.SCENARIO_MANEUVER_DESCRIPTIONS.get(s_type, "complete the driving maneuver")

    # Determine completion_status from checkpoint (mirrors post_actions.py logic)
    _SAFETY_INFRACTION_KEYS = {
        'red_light', 'collisions_vehicle', 'collisions_layout', 'collisions_pedestrian',
        'outside_route_lanes',
    }
    cp_status = record.get("status", "")
    score_route = record.get("scores", {}).get("score_route", 0)
    goal_achieved = (cp_status == "Completed" and score_route >= 99)

    infractions = record.get("infractions", {})
    has_safety_infraction = any(
        isinstance(infractions.get(k), list) and infractions.get(k)
        for k in _SAFETY_INFRACTION_KEYS
    )

    if goal_achieved and not has_safety_infraction:
        status = "completed_clean"
    elif goal_achieved and has_safety_infraction:
        status = "completed_degraded"
    else:
        status = "failed"

    return full_history, full_snapshots, all_frames[0], FRAME_RATE, pre_m, goal, status


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Batch GT event log generator.")
    p.add_argument("--eval-dir",    default="eval_v1",
                   help="Root directory containing scenario folders (default: eval_v1).")
    p.add_argument("--infer-dir",   default=None,
                   help="Root infer-results directory. Defaults to "
                        "output/infer_results/<model> derived from --eval-dir layout.")
    p.add_argument("--checkpoint",  default="my_checkpoint.json",
                   help="Bench2Drive checkpoint JSON (default: my_checkpoint.json).")
    p.add_argument("--output-dir",  default=None,
                   help="Write one JSON file per scenario here. "
                        "If omitted, prints each GT log to stdout.")
    p.add_argument("--max-routes",  type=int, default=None,
                   help="Stop after N scenarios.")
    return p.parse_args()


def main():
    args = parse_args()

    eval_root = ROOT / args.eval_dir
    cp_path   = ROOT / args.checkpoint if args.checkpoint else None

    # Derive infer root: eval_v1/<model>/<scenario> → output/infer_results/<model>
    if args.infer_dir:
        infer_root_base = Path(args.infer_dir)
    else:
        # Try to detect the model subfolder from eval_root
        model_dirs = [d for d in eval_root.iterdir() if d.is_dir()] if eval_root.is_dir() else []
        infer_root_base = (ROOT / "output" / "infer_results" / model_dirs[0].name
                           if len(model_dirs) == 1 else ROOT / "output" / "infer_results")

    checkpoint_records = _load_checkpoint(cp_path)

    # Discover scenario directories (those containing an anno/ subfolder)
    scenario_dirs = sorted(
        p.parent for p in eval_root.rglob("anno") if p.is_dir()
    )
    if args.max_routes:
        scenario_dirs = scenario_dirs[:args.max_routes]

    if not scenario_dirs:
        print(f"No scenario directories found under {eval_root}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(scenario_dirs)} scenario(s).", file=sys.stderr)

    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    ok = fail = 0
    for scenario_dir in scenario_dirs:
        print(f"  Processing {scenario_dir.name} …", file=sys.stderr)

        # Resolve infer dir: infer_root_base may already point at the model folder
        infer_dir_candidate = infer_root_base / scenario_dir.name
        if not infer_dir_candidate.exists():
            # try one level up (infer_root_base is the outer infer_results/)
            for child in infer_root_base.iterdir() if infer_root_base.exists() else []:
                candidate = child / scenario_dir.name
                if candidate.exists():
                    infer_dir_candidate = candidate
                    break

        loaded = _load_episode_data(scenario_dir, infer_dir_candidate.parent, checkpoint_records)
        if loaded is None:
            print(f"    [SKIP] no frames found", file=sys.stderr)
            fail += 1
            continue

        fh, fs, orig, fr, pre_m, goal, status = loaded
        result = build_gt_event_log(fh, fs, orig, fr,
                                    goal=goal, completion_status=status,
                                    pre_m=pre_m,
                                    checkpoint_path=str(cp_path) if cp_path else None,
                                    scenario_name=scenario_dir.name,
                                    scenario_dir=str(scenario_dir))
        out_json = json.dumps(result, indent=2)

        if output_dir:
            out_file = output_dir / f"{scenario_dir.name}.json"
            out_file.write_text(out_json)
            events = result.get("episode", {}).get("events", [])
            print(f"    [OK] {len(events)} events → {out_file}", file=sys.stderr)
        else:
            print(out_json)

        ok += 1

    print(f"\nDone: {ok} succeeded, {fail} skipped.", file=sys.stderr)


if __name__ == "__main__":
    main()

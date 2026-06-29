"""
test_post_actions.py  —  batch GT event log generator

For each scenario directory found under EVAL_DIR it:
  1. Reads per-frame bounding boxes  (anno/*.json.gz)
  2. Reads per-frame ego telemetry   (measurements/*.json.gz)
  3. Reads GT commands from infer results (output/infer_results/…/*.json)
  4. Calls build_gt_event_log and writes the result to OUTPUT_DIR/<scenario>.json

Usage:
    python test_post_actions.py
    python test_post_actions.py --eval-dir eval_v1/Qwen2.5VL+front_cam
    python test_post_actions.py --eval-dir eval_v1 --max-routes 3
    python test_post_actions.py --eval-dir eval_v1 --output-dir gt_logs/
    python test_post_actions.py --checkpoint my_checkpoint.json
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

for _sub in ("hyper_params", "graph_utils"):
    _stub = types.ModuleType(f"generator_modules.{_sub}")
    sys.modules[f"generator_modules.{_sub}"] = _stub
    setattr(gm, _sub, _stub)

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
            "junction":       raw_m.get("junction", False),
            "command_near":   raw_a.get("command_near", 4),
            "bounding_boxes": raw_a.get("bounding_boxes", []),
        }

        if pre_m is None:
            pre_m = m
        last_m = m

        full_history.append((fidx, dir_cmd, spd_cmd))

        if dir_cmd != fr_last_dir or spd_cmd != fr_last_spd:
            full_snapshots.append((fidx, m, False))
            fr_last_dir, fr_last_spd = dir_cmd, spd_cmd

        bbs      = m["bounding_boxes"]
        cur_road = next((b.get("road_id") for b in bbs if b.get("class") == "ego_vehicle"), None)
        cur_lane = next((b.get("lane_id") for b in bbs if b.get("class") == "ego_vehicle"), None)
        if (cur_lane is not None and fr_last_lane is not None
                and cur_lane != fr_last_lane and cur_road == fr_last_road):
            direction = "right" if cur_lane < fr_last_lane else "left"
            full_snapshots.append((fidx, m, direction))
        fr_last_lane, fr_last_road = cur_lane, cur_road

    if last_m is not None and (not full_snapshots or full_snapshots[-1][0] != all_frames[-1]):
        full_snapshots.append((all_frames[-1], last_m, False))

    # Goal and completion status from checkpoint
    record = checkpoint_records.get(scenario_dir.name, {})
    s_type = record.get("scenario_name", "")
    goal   = pa.SCENARIO_MANEUVER_DESCRIPTIONS.get(s_type, "complete the driving maneuver")
    status = "completed_clean" if record.get("status") == "Completed" else "failed"

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
                                    pre_m=pre_m)
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

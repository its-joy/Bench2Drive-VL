"""
GT event log generator for generator_modules/post_actions.py

Usage:
    python test_post_actions.py <scenario_dir>
        [--infer  <infer_results_dir>]
        [--checkpoint <checkpoint.json>]
        [--output <out.json>]

    <scenario_dir>  path to the scenario folder that contains anno/ and
                    measurements/ subdirectories.

    --infer         infer-results directory for this scenario (contains the
                    per-key-frame *.json files with extra_flags).
                    Defaults to: <repo_root>/output/infer_results/<model>/<scenario>
                    (inferred from the scenario_dir path when it follows the
                    standard eval_v1/<model>/<scenario> layout).

    --checkpoint    path to the Bench2Drive checkpoint JSON.
                    Defaults to: auto-detected by walking up from scenario_dir.

    --output        write JSON output to this file instead of stdout.
"""
import sys
import gzip
import json
import types
import importlib.util
import pathlib

# ---------------------------------------------------------------------------
# Bootstrap: load post_actions.py outside the CARLA runtime
# ---------------------------------------------------------------------------

carla_stub = types.ModuleType("carla")
carla_stub.Map = object
sys.modules["carla"] = carla_stub

io_utils_stub = types.ModuleType("io_utils")
io_utils_stub.print_debug = lambda *a, **kw: None
sys.modules["io_utils"] = io_utils_stub

gm = types.ModuleType("generator_modules")
gm.__path__ = []
gm.__package__ = "generator_modules"
sys.modules["generator_modules"] = gm

omc = types.ModuleType("generator_modules.offline_map_calculations")
omc.get_future_measurements = lambda path, k: None
sys.modules["generator_modules.offline_map_calculations"] = omc
setattr(gm, "offline_map_calculations", omc)

for _sub in ("hyper_params", "graph_utils"):
    _stub = types.ModuleType(f"generator_modules.{_sub}")
    sys.modules[f"generator_modules.{_sub}"] = _stub
    setattr(gm, _sub, _stub)

_src = pathlib.Path(__file__).parent / "generator_modules" / "post_actions.py"
spec = importlib.util.spec_from_file_location("generator_modules.post_actions", _src)
pa = importlib.util.module_from_spec(spec)
pa.__package__ = "generator_modules"
sys.modules["generator_modules.post_actions"] = pa
spec.loader.exec_module(pa)

build_gt_event_log = pa.build_gt_event_log

# ---------------------------------------------------------------------------
# Episode loader
# ---------------------------------------------------------------------------

def _gz(path):
    with gzip.open(path) as fp:
        return json.load(fp)


def _load_episode_data(scenario_dir, infer_dir=None, checkpoint_file=None):
    """
    Return (full_history, full_snapshots, origin_f, frame_rate, pre_m, goal, status).

    Reads:
      <scenario_dir>/anno/          — per-frame bounding boxes
      <scenario_dir>/measurements/  — per-frame ego telemetry
      <infer_dir>/                  — GT commands (every 20 frames)
      <checkpoint_file>             — completion status

    Mirrors the snapshot-accumulation logic of PostActionTracker so that the
    returned data is exactly what build_gt_event_log expects.
    """
    scenario_dir = pathlib.Path(scenario_dir).resolve()
    anno_dir = scenario_dir / "anno"
    meas_dir = scenario_dir / "measurements"

    # Derive infer_dir from scenario_dir when layout is eval_v1/<model>/<scenario>
    if infer_dir is None:
        parts = scenario_dir.parts
        try:
            ev1_idx = next(i for i, p in enumerate(parts) if p == "eval_v1")
            repo_root = pathlib.Path(*parts[:ev1_idx])
            model_and_scenario = pathlib.Path(*parts[ev1_idx + 1:])
            infer_dir = repo_root / "output" / "infer_results" / model_and_scenario
        except StopIteration:
            infer_dir = None

    if checkpoint_file is None:
        p = scenario_dir
        for _ in range(5):
            candidate = p / "my_checkpoint.json"
            if candidate.exists():
                checkpoint_file = candidate
                break
            p = p.parent

    FRAME_RATE = 10

    # GT commands from infer-result key frames (every 20 frames); carried forward
    cmd_by_frame: dict = {}
    if infer_dir and pathlib.Path(infer_dir).exists():
        for f in sorted(pathlib.Path(infer_dir).glob("*.json")):
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
        raise RuntimeError(f"No matching anno/measurement frames in {scenario_dir}")

    full_history: list   = []
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

    # Pin the last frame so end-of-episode speed is always reachable
    if last_m is not None and (not full_snapshots or full_snapshots[-1][0] != all_frames[-1]):
        full_snapshots.append((all_frames[-1], last_m, False))

    goal   = "complete the driving maneuver"
    status = "unknown"
    if checkpoint_file and pathlib.Path(checkpoint_file).exists():
        ckpt   = json.loads(pathlib.Path(checkpoint_file).read_text())
        record = ckpt.get("_checkpoint", {}).get("records", [{}])[0]
        s_type = record.get("scenario_name", "")
        goal   = pa.SCENARIO_MANEUVER_DESCRIPTIONS.get(s_type, goal)
        status = "completed_clean" if record.get("status") == "Completed" else "failed"

    return full_history, full_snapshots, all_frames[0], FRAME_RATE, pre_m, goal, status


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Generate GT event log for a scenario.")
    ap.add_argument(
        "scenario_dir",
        help="Scenario folder containing anno/ and measurements/.",
    )
    ap.add_argument("--infer",      default=None,
                    help="Infer-results directory (default: auto-derived).")
    ap.add_argument("--checkpoint", default=None,
                    help="Bench2Drive checkpoint JSON (default: auto-detected).")
    ap.add_argument("--output",     default=None,
                    help="Write JSON to this file instead of stdout.")

    args = ap.parse_args()

    print(f"Loading {args.scenario_dir} …", file=sys.stderr)
    fh, fs, orig, fr, pre_m, goal, status = _load_episode_data(
        args.scenario_dir,
        infer_dir=args.infer,
        checkpoint_file=args.checkpoint,
    )
    print(f"  frames={len(fh)}  snapshots={len(fs)}", file=sys.stderr)

    result = build_gt_event_log(fh, fs, orig, fr,
                                goal=goal, completion_status=status,
                                pre_m=pre_m)
    out_json = json.dumps(result, indent=2)

    if args.output:
        pathlib.Path(args.output).write_text(out_json)
        print(f"Written to {args.output}", file=sys.stderr)
    else:
        print(out_json)

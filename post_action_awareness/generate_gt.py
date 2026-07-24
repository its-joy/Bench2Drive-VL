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
    python3 post_action_awareness/generate_gt.py   --mode scenario_sequence   --eval-dir post_action_data/clean_scenarios   --scenario-csv bench2drive_recategorized_scenarios_v2.csv   --output-dir gt_logs_scenario_sequence/clean_scenarios   --frame-rate 4   --sample-stride 10
"""

import argparse
import csv
import gzip
import json
import re
import sys
from collections import Counter
from pathlib import Path
import types
import importlib.util

# ---------------------------------------------------------------------------
# Bootstrap: load post_actions.py outside the CARLA runtime
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
ROOT = REPO_ROOT

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
_src_templates = REPO_ROOT / "B2DVL_Adapter" / "generator_modules" / "reasoning_context_templates.py"
spec_templates = importlib.util.spec_from_file_location("generator_modules.reasoning_context_templates", _src_templates)
pa_templates = importlib.util.module_from_spec(spec_templates)
pa_templates.__package__ = "generator_modules"
sys.modules["generator_modules.reasoning_context_templates"] = pa_templates
spec_templates.loader.exec_module(pa_templates)

_src = REPO_ROOT / "B2DVL_Adapter" / "generator_modules" / "post_actions.py"
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


def _resolve_repo_path(path_value):
    if path_value is None:
        return None
    path = Path(path_value)
    return path if path.is_absolute() else REPO_ROOT / path


def _load_checkpoint(checkpoint_path):
    """Return {save_name: record} lookup from a Bench2Drive checkpoint file."""
    if not checkpoint_path or not checkpoint_path.exists():
        return {}
    data = json.loads(checkpoint_path.read_text())
    return {r["save_name"]: r
            for r in data.get("_checkpoint", {}).get("records", [])
            if "save_name" in r}


def _load_checkpoint_record(scenario_dir):
    checkpoint_path = scenario_dir / "checkpoint.json"
    if not checkpoint_path.exists():
        return {}
    data = json.loads(checkpoint_path.read_text())
    records = data.get("_checkpoint", {}).get("records", [])
    if not records:
        return {}
    for record in records:
        if record.get("save_name") == scenario_dir.name:
            return record
    return records[0]


def _load_json_file(path):
    if not path or not path.exists():
        return {}
    return json.loads(path.read_text())


def _normalize_csv_key(key):
    return (key or "").strip().lstrip("\ufeff")


def _load_scenario_csv(csv_path):
    if not csv_path or not csv_path.exists():
        return {}
    out = {}
    with csv_path.open(newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            clean = {_normalize_csv_key(k): (v or "").strip() for k, v in row.items()}
            scenario_type = clean.get("scenario_type")
            if scenario_type:
                out[scenario_type] = clean
    return out


def _discover_sequence_dirs(eval_root):
    return sorted(
        p.parent for p in eval_root.rglob("boxes")
        if p.is_dir() and (p.parent / "measurements").is_dir()
    )


def _frame_from_name(path):
    return int(path.name.split(".")[0])


def _load_frame_pairs(scenario_dir):
    box_dir = scenario_dir / "boxes"
    meas_dir = scenario_dir / "measurements"
    box_files = {_frame_from_name(f): f for f in box_dir.glob("*.json.gz")}
    meas_files = {_frame_from_name(f): f for f in meas_dir.glob("*.json.gz")}
    frames = sorted(set(box_files) & set(meas_files))
    return [(f, _gz(meas_files[f]), _gz(box_files[f])) for f in frames]


def _route_id_from_name(name):
    match = re.search(r"_route(\d+)", name)
    return match.group(1) if match else None


def _scenario_type_from_dir(scenario_dir, metadata):
    route_meta = metadata.get("route_metadata", {})
    scenario_types = route_meta.get("scenario_types") or []
    if scenario_types:
        return scenario_types[0]

    parent = scenario_dir.parent.name
    if parent not in {"clean_scenarios", "inappropriate_response"}:
        return parent

    match = re.search(r"_([A-Za-z0-9]+(?:_[0-9]+)?)_route", scenario_dir.name)
    return match.group(1) if match else scenario_dir.name


def _condition_from_dir(scenario_dir, metadata):
    route_meta = metadata.get("route_metadata", {})
    if route_meta.get("condition"):
        return route_meta.get("condition")
    parts = set(scenario_dir.parts)
    if "clean_scenarios" in parts:
        return "clean"
    if "inappropriate_response" in parts:
        return "inappropriate_response"
    return "unknown"


def _town_from_dir(scenario_dir, metadata):
    route_meta = metadata.get("route_metadata", {})
    if route_meta.get("town"):
        return route_meta.get("town")
    match = re.search(r"(Town\d+HD|Town\d+)", scenario_dir.name)
    return match.group(1) if match else None


def _as_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return False


def _safe_float(value, default=0.0):
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _round_float(value, ndigits=3):
    if value is None:
        return None
    return round(_safe_float(value), ndigits)


def _command_name(value):
    names = {
        1: "turn_left",
        2: "turn_right",
        3: "go_straight",
        4: "follow_lane",
        5: "change_lane_left",
        6: "change_lane_right",
    }
    return names.get(value, str(value) if value is not None else None)


def _ego_control_value(ego, measurement, keys):
    for key in keys:
        if ego.get(key) is not None:
            return ego.get(key), "ego_box"
    for key in keys:
        if measurement.get(key) is not None:
            return measurement.get(key), "measurement_ego_control"
    return None, None


def _derive_vehicle_action(summary):
    ego = summary.get("ego", {})
    hazards = summary.get("hazards", {})
    route_shift = summary.get("route_shift_context", {})
    speed_reduced_by = summary.get("speed_reduced_by", {})
    speed = _safe_float(ego.get("speed_kmh"))
    throttle = _safe_float(ego.get("throttle"))
    brake = _safe_float(ego.get("brake"))
    control_brake = bool(ego.get("control_brake"))
    steer = _safe_float(ego.get("steer"))
    reduced_by_object = bool(speed_reduced_by.get("object_type"))

    actions = []
    if speed < 1.0:
        actions.append("stopped")
    elif control_brake or brake > 0.1:
        actions.append("braking")
    elif reduced_by_object:
        actions.append("speed_reduced_by_object")
    elif throttle > 0.1:
        actions.append("accelerating")
    else:
        actions.append("moving")

    if abs(steer) > 0.15:
        # CARLA ego control convention: negative steer is left, positive is right.
        actions.append("steering_left" if steer < 0 else "steering_right")
    else:
        actions.append("keeping_heading")

    if route_shift.get("bypass_phase") == "committed_follow_shifted_route":
        actions.append("responding_to_static_obstacle")
    if hazards.get("vehicle"):
        actions.append("responding_to_vehicle_hazard")
    if hazards.get("walker"):
        actions.append("responding_to_pedestrian")
    if hazards.get("stop_sign"):
        actions.append("stop_sign_context")
    if hazards.get("traffic_light"):
        actions.append("traffic_light_context")
    if ego.get("junction"):
        actions.append("in_junction")

    return actions


def _ego_box(boxes):
    for box in boxes:
        if box.get("class") in {"ego_car", "ego_vehicle"}:
            return box
    return {}


def _nearest_objects(boxes, limit=5):
    objects = []
    for box in boxes:
        cls = box.get("class")
        if cls in {"ego_car", "ego_vehicle"}:
            continue
        distance = box.get("distance")
        if distance is None or _safe_float(distance, -1) < 0:
            continue
        objects.append({
            "class": cls,
            "id": box.get("id"),
            "type_id": box.get("type_id"),
            "distance_m": _round_float(distance),
            "speed_mps": _round_float(box.get("speed")),
            "position": box.get("position"),
            "road_id": box.get("road_id"),
            "lane_id": box.get("lane_id"),
        })
    objects.sort(key=lambda item: item["distance_m"])
    return objects[:limit]


def _object_summary(box):
    return {
        "class": box.get("class"),
        "id": box.get("id"),
        "type_id": box.get("type_id"),
        "role_name": box.get("role_name"),
        "distance_m": _round_float(box.get("distance")),
        "speed_mps": _round_float(box.get("speed")),
        "position": box.get("position"),
        "state": box.get("state"),
        "affects_ego": box.get("affects_ego"),
    }


def _is_pedestrian(box):
    cls = str(box.get("class") or "").lower()
    type_id = str(box.get("type_id") or "").lower()
    return cls in {"pedestrian", "walker"} or type_id.startswith("walker.")


def _is_static_obstacle(box):
    cls = str(box.get("class") or "").lower()
    type_id = str(box.get("type_id") or "").lower()
    return cls == "static" or type_id.startswith("static.")


def _is_emergency_vehicle(box):
    cls = str(box.get("class") or "").lower()
    type_id = str(box.get("type_id") or "").lower()
    role_name = str(box.get("role_name") or "").lower()
    return (
        cls == "emergency_vehicle"
        or "ambulance" in type_id
        or "police" in type_id
        or "emergency" in role_name
    )


def _nearest_matching_object(boxes, predicate):
    objects = [
        box for box in boxes
        if predicate(box) and box.get("distance") is not None and _safe_float(box.get("distance"), -1) >= 0
    ]
    if not objects:
        return None
    return _object_summary(min(objects, key=lambda box: _safe_float(box.get("distance"), 999999)))


def _nearest_matching_objects(boxes, predicate, limit=5):
    objects = [
        box for box in boxes
        if predicate(box) and box.get("distance") is not None and _safe_float(box.get("distance"), -1) >= 0
    ]
    objects.sort(key=lambda box: _safe_float(box.get("distance"), 999999))
    return [_object_summary(box) for box in objects[:limit]]


def _measurement_object_distance(measurement, object_type_prefix):
    object_type = str(measurement.get("speed_reduced_by_obj_type") or "")
    if object_type.startswith(object_type_prefix):
        return _round_float(measurement.get("speed_reduced_by_obj_distance"))
    return None


def _route_lateral_delta(measurement):
    """Signed lateral offset (meters) between the autopilot's actual planned
    path and its original (pre-obstacle) path, read directly from the
    `route`/`route_original` ego-local waypoint arrays (x=forward, y=right)
    at the nearest lookahead point. This is the autopilot's own path plan,
    so it's immune to the steering-wheel noise that made peak-steer
    direction detection unreliable (see _steer_shift_directions and
    _route_lateral_shift_directions)."""
    route = measurement.get("route")
    route_original = measurement.get("route_original")
    if not route or not route_original:
        return None
    try:
        return _safe_float(route[0][1]) - _safe_float(route_original[0][1])
    except (TypeError, IndexError):
        return None


def _route_shift_context(measurement):
    route_obstacle = measurement.get("route_obstacle")
    if not isinstance(route_obstacle, dict):
        route_obstacle = {}
    active = bool(route_obstacle.get("active"))
    changed_route = bool(measurement.get("changed_route") or route_obstacle.get("changed_route"))
    if not active and not changed_route:
        return {
            "active": False,
            "changed_route": False,
        }
    return {
        "active": active,
        "changed_route": changed_route,
        "scenario_type": route_obstacle.get("scenario_type"),
        "direction": route_obstacle.get("direction"),
        "path_clear": route_obstacle.get("path_clear"),
        "bypass_phase": route_obstacle.get("bypass_phase"),
        "keep_driving": route_obstacle.get("keep_driving"),
        "route_index": route_obstacle.get("route_index"),
        "from_index": route_obstacle.get("from_index"),
        "to_index": route_obstacle.get("to_index"),
        "route_index_delta_to_shift": route_obstacle.get("route_index_delta_to_shift"),
        "target_speed_mps": _round_float(route_obstacle.get("target_speed")),
        "distance_to_leading_actor_m": _round_float(route_obstacle.get("distance_to_leading_actor")),
        "lateral_delta_m": _round_float(_route_lateral_delta(measurement)),
    }


def _frame_summary(frame, measurement, boxes, frame_rate):
    ego = _ego_box(boxes)
    pos = measurement.get("pos_global") or [None, None]
    classes = Counter(box.get("class", "unknown") for box in boxes if box.get("class"))
    steer, steer_source = _ego_control_value(
        ego,
        measurement,
        ("steer", "control_steer", "steering"),
    )
    nearest_traffic_light = _nearest_matching_object(
        boxes,
        lambda box: str(box.get("class") or "").lower() == "traffic_light"
        or str(box.get("type_id") or "").lower() == "traffic.traffic_light",
    )
    nearest_stop_sign = _nearest_matching_object(
        boxes,
        lambda box: str(box.get("class") or "").lower() == "stop_sign"
        or str(box.get("type_id") or "").lower() == "traffic.stop",
    )
    nearest_pedestrian = _nearest_matching_object(boxes, _is_pedestrian)
    nearest_static = _nearest_matching_object(boxes, _is_static_obstacle)
    nearest_emergency = _nearest_matching_object(boxes, _is_emergency_vehicle)
    route_shift = _route_shift_context(measurement)
    hazards = {
        "vehicle": bool(measurement.get("vehicle_hazard")),
        "walker": bool(measurement.get("walker_hazard") or measurement.get("walker_close")),
        "traffic_light": bool(measurement.get("light_hazard")),
        "stop_sign": bool(measurement.get("stop_sign_hazard") or measurement.get("stop_sign_close")),
        "route_obstacle": bool(route_shift.get("active")),
        "route_shift": bool(route_shift.get("active") or route_shift.get("changed_route")),
        "static_obstacle": bool(route_shift.get("active"))
        or str(measurement.get("speed_reduced_by_obj_type") or "").startswith("static."),
        "emergency_vehicle": nearest_emergency is not None,
    }
    speed_mps = _safe_float(measurement.get("speed", ego.get("speed", 0.0)))
    return {
        "frame": frame,
        "time_s": _round_float(frame / float(frame_rate), 2),
        "ego": {
            "x": _round_float(pos[0] if len(pos) > 0 else None),
            "y": _round_float(pos[1] if len(pos) > 1 else None),
            "speed_mps": _round_float(speed_mps),
            "speed_kmh": _round_float(speed_mps * 3.6, 1),
            "target_speed_mps": _round_float(measurement.get("target_speed")),
            "speed_limit": _round_float(measurement.get("speed_limit")),
            "theta": _round_float(measurement.get("theta")),
            "steer": _round_float(steer),
            "steer_source": steer_source,
            "throttle": _round_float(measurement.get("throttle")),
            "brake": _round_float(measurement.get("brake")),
            "control_brake": bool(measurement.get("control_brake")),
            "junction": bool(measurement.get("junction")),
            "command": measurement.get("command"),
            "command_name": _command_name(measurement.get("command")),
            "next_command": measurement.get("next_command"),
            "next_command_name": _command_name(measurement.get("next_command")),
        },
        "hazards": hazards,
        "speed_reduced_by": {
            "object_type": measurement.get("speed_reduced_by_obj_type"),
            "object_id": measurement.get("speed_reduced_by_obj_id"),
            "distance_m": _round_float(measurement.get("speed_reduced_by_obj_distance")),
        },
        "traffic_control": {
            "nearest_traffic_light": nearest_traffic_light,
            "traffic_light_distance_m": nearest_traffic_light.get("distance_m") if nearest_traffic_light else None,
            "traffic_light_state": nearest_traffic_light.get("state") if nearest_traffic_light else None,
            "traffic_light_affects_ego": nearest_traffic_light.get("affects_ego") if nearest_traffic_light else None,
            "nearest_stop_sign": nearest_stop_sign,
            "stop_sign_distance_m": (
                nearest_stop_sign.get("distance_m") if nearest_stop_sign
                else _measurement_object_distance(measurement, "traffic.stop")
            ),
            "stop_sign_hazard": bool(measurement.get("stop_sign_hazard")),
            "stop_sign_close": bool(measurement.get("stop_sign_close")),
            "traffic_light_hazard": bool(measurement.get("light_hazard")),
        },
        "pedestrian_context": {
            "nearest_pedestrian": nearest_pedestrian,
            "walker_hazard": bool(measurement.get("walker_hazard")),
            "walker_affecting_id": measurement.get("walker_affecting_id"),
            "walker_close": bool(measurement.get("walker_close")),
            "walker_close_id": measurement.get("walker_close_id"),
            "pedestrian_distance_m": nearest_pedestrian.get("distance_m") if nearest_pedestrian else None,
        },
        "static_obstacle_context": {
            "nearest_static_object": nearest_static,
            "static_object_distance_m": (
                nearest_static.get("distance_m") if nearest_static
                else _measurement_object_distance(measurement, "static.")
            ),
            "route_obstacle": bool(route_shift.get("active")),
        },
        "emergency_vehicle_context": {
            "nearest_emergency_vehicle": nearest_emergency,
            "emergency_vehicle_distance_m": nearest_emergency.get("distance_m") if nearest_emergency else None,
            "emergency_vehicle_visible": nearest_emergency is not None,
        },
        "route_shift_context": route_shift,
        "object_counts": dict(classes),
        "nearest_objects": _nearest_objects(boxes),
    }


def _frames_to_ranges(frames):
    frames = sorted(set(frames))
    if not frames:
        return []
    ranges = []
    start = prev = frames[0]
    for frame in frames[1:]:
        if frame == prev + 1:
            prev = frame
            continue
        ranges.append((start, prev))
        start = prev = frame
    ranges.append((start, prev))
    return ranges


def _range_dict(start, end, frame_rate):
    return {
        "start_frame": start,
        "end_frame": end,
        "start_time_s": _round_float(start / float(frame_rate), 2),
        "end_time_s": _round_float(end / float(frame_rate), 2),
        "duration_s": _round_float((end - start + 1) / float(frame_rate), 2),
    }


def _truthy_brake(measurement):
    return bool(measurement.get("control_brake")) or _safe_float(measurement.get("brake")) > 0.1


def _checkpoint_outcome(record):
    infractions = record.get("infractions", {}) if record else {}
    nonempty_infractions = {
        key: value for key, value in infractions.items()
        if isinstance(value, list) and len(value) > 0
    }
    infraction_types = [
        key for key, value in nonempty_infractions.items()
    ]
    scores = record.get("scores", {}) if record else {}
    status = record.get("status") if record else None
    return {
        "checkpoint_status": status,
        "completed": status in {"Completed", "Perfect"} if record else None,
        "scores": {
            "route": scores.get("score_route"),
            "penalty": scores.get("score_penalty"),
            "composed": scores.get("score_composed"),
        },
        "has_infraction": bool(infraction_types),
        "infraction_types": infraction_types,
        "has_collision": any(k in infraction_types for k in [
            "collisions_vehicle", "collisions_layout", "collisions_pedestrian"
        ]),
        "has_vehicle_collision": "collisions_vehicle" in infraction_types,
        "has_pedestrian_collision": "collisions_pedestrian" in infraction_types,
        "has_static_or_layout_collision": "collisions_layout" in infraction_types,
        "has_red_light_infraction": "red_light" in infraction_types,
        "has_stop_infraction": "stop_infraction" in infraction_types,
        "has_outside_route_lane_infraction": "outside_route_lanes" in infraction_types,
        "has_emergency_vehicle_yield_infraction": "yield_emergency_vehicle_infractions" in infraction_types,
        "infractions": nonempty_infractions,
    }


def _compact_frame_summary(summary):
    vehicle_action = _derive_vehicle_action(summary)
    traffic = summary.get("traffic_control", {})
    pedestrian = summary.get("pedestrian_context", {})
    static = summary.get("static_obstacle_context", {})
    emergency = summary.get("emergency_vehicle_context", {})
    route_shift = summary.get("route_shift_context", {})
    return {
        "frame": summary.get("frame"),
        "time_s": summary.get("time_s"),
        "speed_kmh": summary.get("ego", {}).get("speed_kmh"),
        "steer": summary.get("ego", {}).get("steer"),
        "steer_source": summary.get("ego", {}).get("steer_source"),
        "vehicle_action": vehicle_action,
        "junction": summary.get("ego", {}).get("junction"),
        "hazards": {
            key: value for key, value in summary.get("hazards", {}).items()
            if value
        },
        "distances_m": {
            "traffic_light": traffic.get("traffic_light_distance_m"),
            "stop_sign": traffic.get("stop_sign_distance_m"),
            "pedestrian": pedestrian.get("pedestrian_distance_m"),
            "static_object": static.get("static_object_distance_m"),
            "emergency_vehicle": emergency.get("emergency_vehicle_distance_m"),
        },
        "route_shift_context": route_shift if route_shift.get("active") or route_shift.get("changed_route") else None,
        "speed_reduced_by": summary.get("speed_reduced_by"),
    }


def _speed_stats(frames, summaries_by_frame):
    speeds = [
        summaries_by_frame[frame].get("ego", {}).get("speed_kmh")
        for frame in frames
        if summaries_by_frame.get(frame, {}).get("ego", {}).get("speed_kmh") is not None
    ]
    if not speeds:
        return {}
    return {
        "start_speed_kmh": _round_float(speeds[0], 1),
        "end_speed_kmh": _round_float(speeds[-1], 1),
        "min_speed_kmh": _round_float(min(speeds), 1),
        "max_speed_kmh": _round_float(max(speeds), 1),
        "mean_speed_kmh": _round_float(sum(speeds) / len(speeds), 1),
    }


def _vehicle_action_counts(frames, summaries_by_frame):
    counts = Counter()
    for frame in frames:
        for action in _derive_vehicle_action(summaries_by_frame[frame]):
            counts[action] += 1
    return dict(sorted(counts.items()))


def _route_shift_active(summary):
    route_shift = summary.get("route_shift_context", {})
    return bool(route_shift.get("active") or route_shift.get("changed_route"))


def _route_shift_path_clear(summary):
    route_shift = summary.get("route_shift_context", {})
    return bool(route_shift.get("path_clear") and route_shift.get("keep_driving"))


_TWO_WAY_OBSTACLE_TYPES = {
    "AccidentTwoWays", "ConstructionObstacleTwoWays",
    "ParkedObstacleTwoWays", "VehicleOpensDoorTwoWays",
}


def _is_two_way_obstacle_route(summary):
    scenario_type = str(summary.get("route_shift_context", {}).get("scenario_type") or "")
    return scenario_type in _TWO_WAY_OBSTACLE_TYPES


# These scenario types shift the route around an obstacle exactly like the
# TwoWays ones above (keep_lane -> brake -> wait for gap -> lane shift ->
# merge back -> keep_lane), but their autopilot branches never populate
# route_obstacle_debug's path_clear/direction fields (only the TwoWays/
# HazardAtSideLaneTwoWays "opposing traffic" branches do) -- so the existing
# path_clear-gated detection above never fires for them. The route-level
# `changed_route` measurement field IS populated for all of them though
# (it's a generic "route_points != original_route_points" check, not tied to
# any specific scenario branch), so _route_shift_active still correctly
# detects the shift window; direction is derived from ego's own steer sign
# instead of scenario config, matching how ParkingExit's steer detection
# works.
_STEER_BASED_OBSTACLE_TYPES = {
    "Accident", "ConstructionObstacle", "ParkedObstacle",
    "HazardAtSideLane", "HazardAtSideLaneTwoWays",
}


def _is_steer_based_obstacle_route(scenario_type):
    return scenario_type in _STEER_BASED_OBSTACLE_TYPES


def _route_shift_windows(frames, summaries_by_frame):
    """Contiguous runs of frames where _route_shift_active is True."""
    windows = []
    start = None
    prev = None
    for frame in frames:
        active = _route_shift_active(summaries_by_frame[frame])
        if active and start is None:
            start = frame
        elif not active and start is not None:
            windows.append((start, prev))
            start = None
        prev = frame
    if start is not None:
        windows.append((start, prev))
    return windows


_STEER_SHIFT_DIRECTION_THRESHOLD = 0.15


def _steer_shift_directions(frames, summaries_by_frame, scenario_type):
    """The steer-based obstacle types (_STEER_BASED_OBSTACLE_TYPES) have no
    persistent path_clear flag to hold a lane_shift label across the whole
    bypass maneuver, so direction is inferred from the ego's own steering.
    Raw steer only crosses the labeling threshold for a frame or two during
    the initial turn-in before unwinding as the car settles into the new
    lane -- checking it per-frame produced 1-2 frame lane_shift_* steps.
    Instead, detect direction once from the peak steer angle in each
    route-shift window and hold that direction for every frame in the
    window (the merge_back check in _motion_phase_label still takes
    precedence for the tail frames)."""
    if not (_is_steer_based_obstacle_route(scenario_type) or scenario_type in _TWO_WAY_OBSTACLE_TYPES):
        return {}
    directions = {}
    for start, end in _route_shift_windows(frames, summaries_by_frame):
        window_frames = [frame for frame in frames if start <= frame <= end]
        peak_frame = max(
            window_frames,
            key=lambda frame: abs(_safe_float(summaries_by_frame[frame].get("ego", {}).get("steer"))),
        )
        peak_steer = _safe_float(summaries_by_frame[peak_frame].get("ego", {}).get("steer"))
        if abs(peak_steer) <= _STEER_SHIFT_DIRECTION_THRESHOLD:
            continue
        direction = "left" if peak_steer < 0 else "right"
        for frame in window_frames:
            directions[frame] = direction
    return directions


def _route_lateral_shift_directions(frames, summaries_by_frame, scenario_type):
    """More reliable than _steer_shift_directions: steer sign during an
    obstacle bypass isn't monotonic -- a quick pivot back toward the
    original lane partway through the maneuver (confirmed on
    ParkedObstacleTwoWays/VehicleOpensDoorTwoWays) can have a LARGER peak
    magnitude than the initiating turn, flipping the detected direction.
    Empirically, peak-steer disagreed with the route-plan signal below on
    17/36 sampled obstacle-avoidance routes across all 9 scenario types,
    always by misreporting "right" when the car actually went left. The
    route_shift_context.lateral_delta_m field (route[0].y - route_original[0].y,
    the autopilot's own planned path vs. its original path) is immune to
    that noise, so it's used as the primary signal, with peak-steer
    (_steer_shift_directions) only as a fallback for frames/windows where
    route/route_original weren't available."""
    if not (_is_steer_based_obstacle_route(scenario_type) or scenario_type in _TWO_WAY_OBSTACLE_TYPES):
        return {}
    directions = {}
    for start, end in _route_shift_windows(frames, summaries_by_frame):
        window_frames = [frame for frame in frames if start <= frame <= end]
        best_frame = None
        best_delta = None
        for frame in window_frames:
            delta = summaries_by_frame[frame].get("route_shift_context", {}).get("lateral_delta_m")
            if delta is None:
                continue
            if best_delta is None or abs(delta) > abs(best_delta):
                best_delta = delta
                best_frame = frame
        if best_delta is None or best_delta == 0:
            continue
        direction = "left" if best_delta < 0 else "right"
        for frame in window_frames:
            directions[frame] = direction
    return directions


def _obstacle_shift_directions(frames, summaries_by_frame, scenario_type):
    """Combine the two direction signals for obstacle-avoidance lane shifts,
    preferring the route-plan-based signal and filling any gaps (windows
    where route/route_original data was missing) from the steer-peak
    fallback."""
    steer_directions = _steer_shift_directions(frames, summaries_by_frame, scenario_type)
    route_directions = _route_lateral_shift_directions(frames, summaries_by_frame, scenario_type)
    return {**steer_directions, **route_directions}


# Proactive lane-change scenario types (merging into slow traffic, passing
# slower interurban traffic, mandated sequential lane changes) have no
# route_shift/path_clear signal at all -- `changed_route` stays False
# throughout -- and the `command` nav-hint field (5/6 = change_lane_left/
# right) is unreliable as a window marker: sometimes far shorter than the
# real steering maneuver (a couple of frames before flipping to go_straight
# while the car is still mid-maneuver), sometimes spanning many seconds
# covering more than one real steer event (e.g. SequentialLaneChange). Raw
# steering magnitude is the only signal that reliably tracks the actual
# maneuver, so lane changes are windowed the same way as
# _steer_shift_directions but keyed off steer magnitude directly.
_LANE_CHANGE_SCENARIO_TYPES = {
    "SequentialLaneChange", "InterurbanActorFlow", "InterurbanAdvancedActorFlow",
    "MergerIntoSlowTrafficV2",
}

_LANE_CHANGE_STEER_THRESHOLD = 0.15
# A single lane change is often two steer-active runs: an initial turn-in and
# a later straightening correction, separated by a settle gap of near-zero
# steer. Empirically (checked across SequentialLaneChange/MergerIntoSlowTrafficV2
# samples) that settle gap is at most ~6 frames within one maneuver, while
# genuinely separate sequential lane changes are separated by 12+ frames of
# stable cruising -- 7 sits with margin on both sides.
_LANE_CHANGE_STEER_GAP_TOLERANCE = 7

# For InterurbanActorFlow/InterurbanAdvancedActorFlow/MergerIntoSlowTrafficV2,
# the big-radius highway ramp/fork/merge geometry CARLA tags as a junction
# isn't a real intersection turn (see the _junction_turn_label exclusion
# below), and the lane-change maneuver itself can start or continue while
# still inside that junction-tagged region: confirmed on
# InterurbanAdvancedActorFlow, the real initiating steer (e.g. -0.29, -0.49)
# happens with junction=True, immediately followed by a bigger corrective
# swing in the opposite direction (e.g. +0.42, +0.70) right as junction flips
# False; on MergerIntoSlowTrafficV2, net lateral displacement (computed from
# pos_global/theta, immune to steer-sign noise) shows the merge is one
# continuous rightward motion from before the junction through several
# frames after entering it, even though instantaneous steer sign flips
# several times in that span. Excluding junction frames from the window (as
# done for SequentialLaneChange, where junction genuinely means a real turn)
# cut off part of the same maneuver and left it to _junction_turn_label,
# which mislabeled it as a discrete (and sometimes flip-flopping) turn. These
# three types don't have that ambiguity since _junction_turn_label never
# runs for them.
_LANE_CHANGE_INCLUDE_JUNCTION_TYPES = {
    "InterurbanActorFlow", "InterurbanAdvancedActorFlow", "MergerIntoSlowTrafficV2",
}


def _lane_change_windows(frames, summaries_by_frame, scenario_type=None):
    """Contiguous runs of frames where steer magnitude is actively above
    threshold, normally excluding junctions (real junction turns are handled
    separately by _junction_turn_label) -- except for
    _LANE_CHANGE_INCLUDE_JUNCTION_TYPES, see above. A small gap tolerance
    bridges the brief dip near zero that often occurs mid-maneuver, between
    an initial turn-in and a straightening correction, without splitting one
    lane change into two."""
    include_junction = scenario_type in _LANE_CHANGE_INCLUDE_JUNCTION_TYPES
    windows = []
    start = None
    last_active = None
    for frame in frames:
        summary = summaries_by_frame[frame]
        steer = _safe_float(summary.get("ego", {}).get("steer"))
        junction = bool(summary.get("ego", {}).get("junction"))
        active = (include_junction or not junction) and abs(steer) > _LANE_CHANGE_STEER_THRESHOLD
        if active:
            if start is None:
                start = frame
            last_active = frame
        elif start is not None and (frame - last_active) > _LANE_CHANGE_STEER_GAP_TOLERANCE:
            windows.append((start, last_active))
            start = None
            last_active = None
    if start is not None:
        windows.append((start, last_active))
    return windows


def _lane_change_directions(frames, summaries_by_frame, scenario_type):
    """Direction is taken from the first frame in the window that crosses the
    threshold (the initiating turn-in), not the peak-magnitude frame -- unlike
    the obstacle-bypass case, a lane change's straightening correction at the
    end can have a larger peak magnitude than the initiating turn (especially
    right before/after a junction-tagged merge point), which would otherwise
    flip the detected direction."""
    if scenario_type not in _LANE_CHANGE_SCENARIO_TYPES:
        return {}
    directions = {}
    for start, end in _lane_change_windows(frames, summaries_by_frame, scenario_type):
        window_frames = [frame for frame in frames if start <= frame <= end]
        first_steer = None
        for frame in window_frames:
            steer = _safe_float(summaries_by_frame[frame].get("ego", {}).get("steer"))
            if abs(steer) > _LANE_CHANGE_STEER_THRESHOLD:
                first_steer = steer
                break
        if first_steer is None:
            continue
        direction = "left" if first_steer < 0 else "right"
        for frame in window_frames:
            directions[frame] = direction
    return directions


def _junction_turn_label(measurement, summary):
    ego = summary.get("ego", {})
    if not ego.get("junction"):
        return None

    speed_kmh = _safe_float(ego.get("speed_kmh"))
    if speed_kmh < 1.0:
        return None

    steer = _safe_float(ego.get("steer"))
    command_name = ego.get("command_name") or _command_name(measurement.get("command"))

    if abs(steer) > 0.15:
        return "turn_right" if steer > 0 else "turn_left"
    if command_name in {"turn_right", "turn_left"}:
        return command_name
    return None


_BICYCLE_TYPE_IDS = {"vehicle.bh.crossbike", "vehicle.diamondback.century", "vehicle.gazelle.omafiets"}


_PARKING_EXIT_STEER_THRESHOLD = 0.15
_PARKING_EXIT_HELD_STEER_THRESHOLD = 0.3


def _parking_exit_steer_label(measurement, summary, scenario_type):
    """ParkingExit-specific: the ego starts parked with wheels already turned
    toward the lane, then creeps out and steers back straight to merge in.
    Neither state is a map junction, so _junction_turn_label never fires for
    it and it used to fall through to a generic stopped/keep_lane label."""
    if scenario_type != "ParkingExit":
        return None
    ego = summary.get("ego", {})
    speed_kmh = _safe_float(ego.get("speed_kmh"))
    steer = _safe_float(ego.get("steer"))
    if speed_kmh < 0.8 and abs(steer) > _PARKING_EXIT_HELD_STEER_THRESHOLD:
        return "parked_preparing_to_exit"
    if speed_kmh >= 0.8 and abs(steer) > _PARKING_EXIT_STEER_THRESHOLD:
        return "steering_out_of_parking_spot"
    return None


def _motion_phase_label(frame, measurement, summary, prev_speed_kmh, route_shift_last_frame,
                        scenario_type=None, obstacle_shift_direction=None, lane_change_direction=None):
    speed_kmh = _safe_float(summary.get("ego", {}).get("speed_kmh"))
    route_shift = summary.get("route_shift_context", {})
    route_shift_active = _route_shift_active(summary)
    path_clear = _route_shift_path_clear(summary)
    two_way_obstacle = _is_two_way_obstacle_route(summary)
    steer_based_obstacle = _is_steer_based_obstacle_route(scenario_type)
    delta_to_shift = route_shift.get("route_index_delta_to_shift")
    try:
        abs_delta_to_shift = abs(float(delta_to_shift)) if delta_to_shift is not None else None
    except (TypeError, ValueError):
        abs_delta_to_shift = None
    near_shift_point = (
        abs_delta_to_shift is None
        or abs_delta_to_shift <= 60.0
        or bool(measurement.get("changed_route"))
    )

    if (
        (two_way_obstacle or steer_based_obstacle)
        and route_shift_last_frame is not None
        and route_shift_last_frame - 4 <= frame <= route_shift_last_frame + 2
        and speed_kmh >= 1.0
    ):
        return "merge_back_to_original_lane"

    if route_shift_active and near_shift_point and speed_kmh >= 1.0:
        if two_way_obstacle and path_clear:
            return (
                f"lane_shift_{obstacle_shift_direction}_to_bypass_obstacle"
                if obstacle_shift_direction else "lane_shift_left_to_bypass_obstacle"
            )
        if steer_based_obstacle:
            if obstacle_shift_direction:
                return f"lane_shift_{obstacle_shift_direction}_to_bypass_obstacle"
        elif path_clear:
            direction = route_shift.get("direction")
            return f"lane_shift_{direction}" if direction else "lane_shift"

    if lane_change_direction and speed_kmh >= 1.0:
        return f"lane_change_{lane_change_direction}"

    if speed_kmh < 0.8:
        if prev_speed_kmh is not None and prev_speed_kmh > 1.5:
            return "braking_to_stop"
        if route_shift_active or summary.get("hazards", {}).get("static_obstacle"):
            return "stopped_waiting_for_gap"
        parking_label = _parking_exit_steer_label(measurement, summary, scenario_type)
        if parking_label:
            return parking_label
        return "stopped"

    if speed_kmh < 1.0 and route_shift_active and not path_clear:
        return "braking_to_stop"

    braking = _truthy_brake(measurement)
    slowing_meaningfully = (
        prev_speed_kmh is not None and speed_kmh < prev_speed_kmh - 5.0
    )
    if route_shift_active and not path_clear and (
        speed_kmh < 15.0 or measurement.get("vehicle_hazard") or slowing_meaningfully
    ):
        return "braking_to_stop"

    if route_shift_active and not path_clear:
        return "keep_lane"

    # These scenario types (_LANE_CHANGE_INCLUDE_JUNCTION_TYPES) run on
    # big-radius highway ramp/fork/merge geometry that CARLA's map tags as a
    # junction even though there's no discrete intersection turn -- see the
    # comment on that set above -- so _junction_turn_label was mislabeling
    # plain curve-following/merge tails as a discrete (and sometimes
    # flip-flopping) "turn".
    if scenario_type not in _LANE_CHANGE_INCLUDE_JUNCTION_TYPES:
        junction_turn = _junction_turn_label(measurement, summary)
        if junction_turn:
            return junction_turn

    reduced_by_bicycle = (
        str(measurement.get("speed_reduced_by_obj_type") or "") in _BICYCLE_TYPE_IDS
    )

    # A real brake-pedal event paired with an actual speed decrease, or a
    # sharp speed drop on its own, should register as braking regardless of
    # absolute speed -- the old speed_kmh < 20.0 gate silently dropped
    # hard-braking-from-highway-speed frames (e.g. braking for a cyclist from
    # 35km/h), collapsing them into the surrounding keep_lane phase. Brake
    # pedal alone isn't enough: momentary brake-pedal noise can occur while
    # speed is still net increasing (PID correction) and shouldn't be
    # reported as a braking event.
    decelerating = prev_speed_kmh is not None and speed_kmh < prev_speed_kmh - 1.0
    if (braking and decelerating) or slowing_meaningfully:
        return "braking_to_stop" if speed_kmh < 5.0 else "braking"

    # Ego's own planner is capping target speed because of a cyclist AND ego
    # has actually caught down to a materially reduced speed (not merely
    # "the cyclist is technically the tightest constraint far ahead" while
    # still cruising near the speed limit -- that case stays keep_lane).
    speed_limit_kmh = _safe_float(measurement.get("speed_limit")) * 3.6
    following_low_speed = speed_limit_kmh > 0 and speed_kmh < speed_limit_kmh * 0.6
    if reduced_by_bicycle and following_low_speed:
        return "following_hazard_at_reduced_speed"

    parking_label = _parking_exit_steer_label(measurement, summary, scenario_type)
    if parking_label:
        return parking_label

    return "keep_lane"


def _motion_phase_description(label):
    descriptions = {
        "keep_lane": "Ego kept its lane while approaching or proceeding through the scenario.",
        "braking_to_stop": "Ego slowed down and braked toward a stop.",
        "stopped": "Ego was stopped.",
        "stopped_waiting_for_gap": "Ego remained stopped while waiting for a safe gap or clearance.",
        "lane_shift_left_to_bypass_obstacle": "Ego shifted left to bypass the obstacle.",
        "lane_shift_right_to_bypass_obstacle": "Ego shifted right to bypass the obstacle.",
        "lane_shift_right": "Ego shifted right.",
        "lane_shift_left": "Ego shifted left.",
        "lane_shift": "Ego shifted laterally according to the modified route.",
        "merge_back_to_original_lane": "Ego merged back toward the original lane/path after bypassing the obstacle.",
        "lane_change_left": "Ego changed lanes to the left.",
        "lane_change_right": "Ego changed lanes to the right.",
        "following_hazard_at_reduced_speed": "Ego followed a cyclist/hazard ahead at a reduced, stabilized speed.",
        "parked_preparing_to_exit": "Ego started parked in a parking spot/shoulder with wheels already turned toward the travel lane, preparing to exit.",
        "steering_out_of_parking_spot": "Ego steered out of the parking spot, creeping forward and straightening the wheel to merge into the lane.",
        "turn_right": "Ego turned right through the junction.",
        "turn_left": "Ego turned left through the junction.",
        "braking": "Ego was braking.",
    }
    return descriptions.get(label, label.replace("_", " ").capitalize() + ".")


def _step_context_summary(phase_frames, measurements_by_frame, summaries_by_frame):
    contexts = []
    for frame in phase_frames:
        summary = summaries_by_frame[frame]
        hazards = summary.get("hazards", {})
        if hazards.get("traffic_light"):
            contexts.append("traffic_light")
        if hazards.get("stop_sign"):
            contexts.append("stop_sign")
        if hazards.get("walker"):
            contexts.append("pedestrian")
        if hazards.get("vehicle"):
            contexts.append("vehicle_hazard")
        if hazards.get("static_obstacle"):
            contexts.append("static_obstacle")
        if hazards.get("emergency_vehicle"):
            contexts.append("emergency_vehicle")
        if _route_shift_active(summary):
            contexts.append("route_shift")
        if _route_shift_path_clear(summary):
            contexts.append("path_clear")
    return sorted(set(contexts))


def _traffic_light_status_for_frame(frame, summaries_by_frame):
    traffic = summaries_by_frame[frame].get("traffic_control", {})
    return traffic.get("traffic_light_state")


def dedupe_consecutive(items):
    result = []
    for item in items:
        if not result or result[-1] != item:
            result.append(item)
    return result


def _step_scene_context(phase_frames, summaries_by_frame):
    junction_frames = [
        frame for frame in phase_frames
        if summaries_by_frame[frame].get("ego", {}).get("junction")
    ]
    traffic_light_frames = [
        frame for frame in phase_frames
        if summaries_by_frame[frame].get("traffic_control", {}).get("nearest_traffic_light")
    ]
    traffic_light_hazard_frames = [
        frame for frame in phase_frames
        if summaries_by_frame[frame].get("traffic_control", {}).get("traffic_light_hazard")
    ]
    stop_sign_frames = [
        frame for frame in phase_frames
        if summaries_by_frame[frame].get("traffic_control", {}).get("nearest_stop_sign")
        or summaries_by_frame[frame].get("traffic_control", {}).get("stop_sign_hazard")
        or summaries_by_frame[frame].get("traffic_control", {}).get("stop_sign_close")
    ]
    stop_sign_distances = [
        summaries_by_frame[frame].get("traffic_control", {}).get("stop_sign_distance_m")
        for frame in phase_frames
    ]
    def first_last(frame_list):
        if not frame_list:
            return None
        return {"first_frame": frame_list[0], "last_frame": frame_list[-1]}

    # start/end alone can hide a color change that happens entirely inside the
    # step (e.g. a keep_lane phase spanning a junction where the light goes
    # red -> green -> yellow while the motion label never changes). Track
    # every (frame, color) transition so callers can tell which color was
    # active during e.g. the junction-present sub-range specifically.
    color_transitions = []
    prev_color = None
    for frame in phase_frames:
        color = _traffic_light_status_for_frame(frame, summaries_by_frame)
        if color and color != prev_color:
            color_transitions.append({"frame": frame, "color": color})
        prev_color = color
    light_color_sequence = [entry["color"] for entry in color_transitions]

    # Use the color active at the *last* junction frame (not the first): the
    # signal can change while ego is still inside the junction, and the color
    # it crossed on is the one active as it clears the junction, not
    # whatever was showing on approach.
    junction_color = None
    if junction_frames:
        junction_end = junction_frames[-1]
        for entry in color_transitions:
            if entry["frame"] <= junction_end:
                junction_color = entry["color"]
            else:
                break
        if junction_color is None and color_transitions:
            junction_color = color_transitions[0]["color"]

    return {
        "junction": {
            "present": bool(junction_frames),
            "frames": first_last(junction_frames),
        },
        "traffic_light": {
            "present": bool(traffic_light_frames or traffic_light_hazard_frames),
            "hazard_active": bool(traffic_light_hazard_frames),
            "frames": first_last(traffic_light_frames),
            "color_during_junction": junction_color,
            "color_transitions": color_transitions,
            "start_status": _traffic_light_status_for_frame(phase_frames[0], summaries_by_frame),
            "end_status": _traffic_light_status_for_frame(phase_frames[-1], summaries_by_frame),
            "color_sequence": light_color_sequence,
        },
        "stop_sign": {
            "present": bool(stop_sign_frames),
            "hazard_active": any(
                summaries_by_frame[frame].get("traffic_control", {}).get("stop_sign_hazard")
                for frame in phase_frames
            ),
            "close": any(
                summaries_by_frame[frame].get("traffic_control", {}).get("stop_sign_close")
                for frame in phase_frames
            ),
            "frames": first_last(stop_sign_frames),
            "min_distance_m": _min_non_null(stop_sign_distances),
        },
    }


def _active_contexts(measurement, summary):
    contexts = []
    traffic = summary.get("traffic_control", {})
    pedestrian = summary.get("pedestrian_context", {})
    static = summary.get("static_obstacle_context", {})
    emergency = summary.get("emergency_vehicle_context", {})
    route_shift = summary.get("route_shift_context", {})

    if measurement.get("light_hazard"):
        contexts.append("traffic_light")
    if measurement.get("stop_sign_hazard") or measurement.get("stop_sign_close") or traffic.get("nearest_stop_sign"):
        contexts.append("stop_sign")
    if measurement.get("walker_hazard") or measurement.get("walker_close") or pedestrian.get("nearest_pedestrian"):
        contexts.append("pedestrian")
    if measurement.get("route_obstacle") or str(measurement.get("speed_reduced_by_obj_type") or "").startswith("static.") or static.get("nearest_static_object"):
        contexts.append("static_obstacle")
    if route_shift.get("active") or route_shift.get("changed_route"):
        contexts.append("route_shift")
    if emergency.get("emergency_vehicle_visible") or "ambulance" in str(measurement.get("speed_reduced_by_obj_type") or "").lower() or "police" in str(measurement.get("speed_reduced_by_obj_type") or "").lower():
        contexts.append("emergency_vehicle")
    if measurement.get("vehicle_hazard"):
        contexts.append("vehicle_hazard")
    if measurement.get("junction"):
        contexts.append("junction")
    if _truthy_brake(measurement) or _safe_float(measurement.get("speed")) < 0.2:
        contexts.append("ego_control_response")
    if not contexts:
        contexts.append("normal_driving")
    return tuple(contexts)


def _phase_label(contexts):
    labels = [ctx for ctx in contexts if ctx != "normal_driving"]
    if not labels:
        return "normal_driving"
    if len(labels) == 1:
        return labels[0]
    priority = [
        "pedestrian",
        "stop_sign",
        "traffic_light",
        "route_shift",
        "static_obstacle",
        "emergency_vehicle",
        "vehicle_hazard",
        "junction",
        "ego_control_response",
    ]
    ordered = [label for label in priority if label in labels]
    return "_and_".join(ordered[:3]) + ("_context" if len(ordered) > 1 else "")


def _phase_description(contexts):
    if contexts == ("normal_driving",):
        return "No special hazard or traffic-control context was active."
    readable = {
        "traffic_light": "traffic-light context",
        "stop_sign": "stop-sign context",
        "pedestrian": "pedestrian context",
        "static_obstacle": "static-obstacle context",
        "route_shift": "route-shift / bypass planning context",
        "emergency_vehicle": "emergency-vehicle context",
        "vehicle_hazard": "vehicle-hazard context",
        "junction": "junction traversal",
        "ego_control_response": "ego braking/stopping or low-speed response",
    }
    parts = [readable.get(ctx, ctx.replace("_", " ")) for ctx in contexts if ctx != "normal_driving"]
    return "Active context: " + ", ".join(parts) + "."


def _merge_phases_to_limit(phases, max_steps=9):
    phases = list(phases)
    while len(phases) > max_steps:
        best_index = min(
            range(len(phases) - 1),
            key=lambda idx: (phases[idx]["end_frame"] - phases[idx]["start_frame"])
            + (phases[idx + 1]["end_frame"] - phases[idx + 1]["start_frame"])
        )
        left = phases[best_index]
        right = phases[best_index + 1]
        merged_label = left.get("label") if left.get("label") == right.get("label") else "mixed_motion"
        phases[best_index:best_index + 2] = [{
            "start_frame": left["start_frame"],
            "end_frame": right["end_frame"],
            "label": merged_label,
        }]
    return phases


def _smooth_motion_phases(phases):
    phases = list(phases)
    changed = True
    while changed:
        changed = False
        new_phases = []
        idx = 0
        while idx < len(phases):
            if idx + 2 < len(phases):
                left = phases[idx]
                middle = phases[idx + 1]
                right = phases[idx + 2]
                middle_len = middle["end_frame"] - middle["start_frame"] + 1
                if (
                    middle_len <= 2
                    and left["label"] == right["label"]
                    and middle["label"] not in (
                        "braking", "braking_to_stop",
                        "parked_preparing_to_exit", "steering_out_of_parking_spot",
                        "lane_shift_left_to_bypass_obstacle", "lane_shift_right_to_bypass_obstacle",
                        "merge_back_to_original_lane",
                        "lane_change_left", "lane_change_right",
                    )
                ):
                    new_phases.append({
                        "start_frame": left["start_frame"],
                        "end_frame": right["end_frame"],
                        "label": left["label"],
                    })
                    idx += 3
                    changed = True
                    continue
                if (
                    middle_len <= 2
                    and middle["label"] == "stopped_waiting_for_gap"
                    and left["label"] == "braking_to_stop"
                    and right["label"] == "braking_to_stop"
                ):
                    new_phases.append({
                        "start_frame": left["start_frame"],
                        "end_frame": right["end_frame"],
                        "label": "braking_to_stop",
                    })
                    idx += 3
                    changed = True
                    continue
            new_phases.append(phases[idx])
            idx += 1
        phases = new_phases
    return phases


def _extend_merge_back_tail(phases, tail_frames=3):
    phases = list(phases)
    for idx in range(len(phases) - 1):
        phase = phases[idx]
        next_phase = phases[idx + 1]
        if phase["label"] != "merge_back_to_original_lane" or next_phase["label"] != "keep_lane":
            continue
        available = next_phase["end_frame"] - next_phase["start_frame"] + 1
        extension = min(tail_frames, max(0, available - 1))
        if extension <= 0:
            continue
        phase["end_frame"] += extension
        next_phase["start_frame"] += extension
    return [phase for phase in phases if phase["start_frame"] <= phase["end_frame"]]


def _merge_zero_duration_phases(phases):
    """A phase with end_frame == start_frame spans a single frame (0 frames
    of duration) -- too short to be meaningful as its own step, and mostly
    control-loop noise (e.g. a 1-frame brake-pedal blip). Absorb it into a
    neighboring phase (extending that phase's frame range) so frame coverage
    stays gapless instead of leaving a hole."""
    phases = list(phases)
    if len(phases) <= 1:
        return phases

    changed = True
    while changed:
        changed = False
        for idx, phase in enumerate(phases):
            if phase["end_frame"] - phase["start_frame"] >= 1:
                continue
            if idx > 0:
                phases[idx - 1]["end_frame"] = phase["end_frame"]
            elif idx + 1 < len(phases):
                phases[idx + 1]["start_frame"] = phase["start_frame"]
            else:
                continue
            del phases[idx]
            changed = True
            break
    return phases


def _coalesce_adjacent_same_label_phases(phases):
    """_merge_zero_duration_phases absorbs a single-frame phase into
    whichever neighbor is available, inheriting that neighbor's label --
    e.g. a real 1-frame stop sandwiched between two lane_change_left phases
    (different label on each side, so _smooth_motion_phases' symmetric
    left==right check never merges it) gets folded into the earlier one,
    which can leave two now-identically-labeled phases adjacent but still
    separate. Collapse those into one so the same maneuver isn't reported as
    two consecutive steps."""
    phases = list(phases)
    idx = 0
    while idx + 1 < len(phases):
        if phases[idx]["label"] == phases[idx + 1]["label"]:
            phases[idx]["end_frame"] = phases[idx + 1]["end_frame"]
            del phases[idx + 1]
            continue
        idx += 1
    return phases


def _sample_step_evidence(phase_frames, summaries_by_frame, stride=4):
    if not phase_frames:
        return []
    stride = max(1, int(stride))
    sampled_frames = phase_frames[::stride]
    if phase_frames[-1] not in sampled_frames:
        sampled_frames.append(phase_frames[-1])
    return [_compact_frame_summary(summaries_by_frame[frame]) for frame in sampled_frames]


def _build_sequence_steps(pairs, summaries_by_frame, outcome, frame_rate, step_evidence_stride=4,
                          scenario_type=None):
    frames = [frame for frame, _, _ in pairs]
    if not frames:
        return []

    measurements_by_frame = {frame: measurement for frame, measurement, _ in pairs}
    route_shift_frames = [frame for frame in frames if _route_shift_active(summaries_by_frame[frame])]
    route_shift_last_frame = max(route_shift_frames) if route_shift_frames else None
    obstacle_shift_directions = _obstacle_shift_directions(frames, summaries_by_frame, scenario_type)
    lane_change_directions = _lane_change_directions(frames, summaries_by_frame, scenario_type)

    labels_by_frame = {}
    prev_speed = None
    for frame in frames:
        label = _motion_phase_label(
            frame,
            measurements_by_frame[frame],
            summaries_by_frame[frame],
            prev_speed,
            route_shift_last_frame,
            scenario_type=scenario_type,
            obstacle_shift_direction=obstacle_shift_directions.get(frame),
            lane_change_direction=lane_change_directions.get(frame),
        )
        labels_by_frame[frame] = label
        prev_speed = _safe_float(summaries_by_frame[frame].get("ego", {}).get("speed_kmh"))

    phases = []
    cur_start = frames[0]
    cur_label = labels_by_frame[frames[0]]
    prev_frame = frames[0]

    for frame in frames[1:]:
        label = labels_by_frame[frame]
        if label != cur_label:
            phases.append({
                "start_frame": cur_start,
                "end_frame": prev_frame,
                "label": cur_label,
            })
            cur_start = frame
            cur_label = label
        prev_frame = frame

    phases.append({
        "start_frame": cur_start,
        "end_frame": prev_frame,
        "label": cur_label,
    })
    phases = _smooth_motion_phases(phases)
    phases = _extend_merge_back_tail(phases)
    phases = _merge_zero_duration_phases(phases)
    phases = _coalesce_adjacent_same_label_phases(phases)

    steps = []
    for phase in phases:
        start = phase["start_frame"]
        end = phase["end_frame"]
        phase_frames = [frame for frame in frames if start <= frame <= end]
        label = phase["label"]
        contexts = _step_context_summary(phase_frames, measurements_by_frame, summaries_by_frame)
        step = {
            "step_id": None,
            "label": label,
            "vehicle_action": label,
            "active_contexts": contexts,
            "scene_context": _step_scene_context(phase_frames, summaries_by_frame),
            "description": _motion_phase_description(label),
            **_range_dict(start, end, frame_rate),
            "speed": _speed_stats(phase_frames, summaries_by_frame),
            "evidence": {
                "start": _compact_frame_summary(summaries_by_frame[start]),
                "end": _compact_frame_summary(summaries_by_frame[end]),
                "samples": _sample_step_evidence(
                    phase_frames,
                    summaries_by_frame,
                    stride=step_evidence_stride,
                ),
            },
        }
        steps.append(step)

    if steps:
        steps[-1]["checkpoint_status"] = outcome.get("checkpoint_status")
        steps[-1]["infraction_types"] = outcome.get("infraction_types", [])
        steps[-1]["description"] += " This final phase also carries the checkpoint outcome."

    steps.sort(key=lambda item: (item["start_frame"], item["end_frame"], item["label"]))
    for idx, step in enumerate(steps, start=1):
        step["step_id"] = f"S{idx}"
    return steps


def _min_non_null(values):
    clean = [_safe_float(value) for value in values if value is not None]
    return _round_float(min(clean)) if clean else None


def _object_type_counts(objects):
    counts = Counter(
        str(obj.get("type_id") or obj.get("class") or "unknown")
        for obj in objects
        if obj
    )
    return dict(sorted(counts.items()))


def _value_counts(values):
    counts = Counter(str(value) for value in values if value is not None)
    return dict(sorted(counts.items()))


def _scenario_evidence_rollup(pairs, summaries_by_frame, outcome, frame_rate):
    frames = [frame for frame, _, _ in pairs]
    traffic_light_frames = []
    traffic_light_hazard_frames = []
    traffic_light_affects_ego_frames = []
    traffic_control_frames = []
    pedestrian_frames = []
    static_frames = []
    emergency_frames = []
    route_shift_frames = []
    route_shift_changed_frames = []
    route_shift_path_clear_frames = []
    traffic_light_distances = []
    stop_sign_distances = []
    pedestrian_distances = []
    static_distances = []
    emergency_distances = []
    traffic_light_objects = []
    traffic_light_states = []
    stop_sign_objects = []
    pedestrian_objects = []
    static_objects = []
    emergency_objects = []
    route_shift_directions = []
    route_shift_scenario_types = []
    route_shift_bypass_phases = []
    route_shift_route_index_deltas = []

    for frame in frames:
        summary = summaries_by_frame[frame]
        traffic = summary.get("traffic_control", {})
        pedestrian = summary.get("pedestrian_context", {})
        static = summary.get("static_obstacle_context", {})
        emergency = summary.get("emergency_vehicle_context", {})
        route_shift = summary.get("route_shift_context", {})

        if traffic.get("nearest_traffic_light") or traffic.get("traffic_light_hazard"):
            traffic_light_frames.append(frame)
        if traffic.get("traffic_light_hazard"):
            traffic_light_hazard_frames.append(frame)
        if traffic.get("traffic_light_affects_ego"):
            traffic_light_affects_ego_frames.append(frame)
        if traffic.get("nearest_stop_sign") or traffic.get("stop_sign_hazard") or traffic.get("stop_sign_close"):
            traffic_control_frames.append(frame)
        if pedestrian.get("nearest_pedestrian") or pedestrian.get("walker_hazard") or pedestrian.get("walker_close"):
            pedestrian_frames.append(frame)
        if static.get("nearest_static_object") or static.get("route_obstacle"):
            static_frames.append(frame)
        if emergency.get("nearest_emergency_vehicle"):
            emergency_frames.append(frame)
        if route_shift.get("active") or route_shift.get("changed_route"):
            route_shift_frames.append(frame)
        if route_shift.get("changed_route"):
            route_shift_changed_frames.append(frame)
        if route_shift.get("path_clear"):
            route_shift_path_clear_frames.append(frame)

        traffic_light_distances.append(traffic.get("traffic_light_distance_m"))
        stop_sign_distances.append(traffic.get("stop_sign_distance_m"))
        pedestrian_distances.append(pedestrian.get("pedestrian_distance_m"))
        static_distances.append(static.get("static_object_distance_m"))
        emergency_distances.append(emergency.get("emergency_vehicle_distance_m"))
        if traffic.get("nearest_traffic_light"):
            traffic_light_objects.append(traffic.get("nearest_traffic_light"))
        if traffic.get("traffic_light_state") is not None:
            traffic_light_states.append(traffic.get("traffic_light_state"))
        if traffic.get("nearest_stop_sign"):
            stop_sign_objects.append(traffic.get("nearest_stop_sign"))
        if pedestrian.get("nearest_pedestrian"):
            pedestrian_objects.append(pedestrian.get("nearest_pedestrian"))
        if static.get("nearest_static_object"):
            static_objects.append(static.get("nearest_static_object"))
        if emergency.get("nearest_emergency_vehicle"):
            emergency_objects.append(emergency.get("nearest_emergency_vehicle"))
        route_shift_directions.append(route_shift.get("direction"))
        route_shift_scenario_types.append(route_shift.get("scenario_type"))
        route_shift_bypass_phases.append(route_shift.get("bypass_phase"))
        if route_shift.get("route_index_delta_to_shift") is not None:
            route_shift_route_index_deltas.append(route_shift.get("route_index_delta_to_shift"))

    def ranges(frame_list):
        return [_range_dict(start, end, frame_rate) for start, end in _frames_to_ranges(frame_list)]

    return {
        "traffic_control": {
            "traffic_light_present": bool(traffic_light_frames),
            "traffic_light_types": _object_type_counts(traffic_light_objects),
            "traffic_light_states": _value_counts(traffic_light_states),
            "traffic_light_ranges": ranges(traffic_light_frames),
            "traffic_light_hazard_ranges": ranges(traffic_light_hazard_frames),
            "traffic_light_affects_ego_ranges": ranges(traffic_light_affects_ego_frames),
            "min_traffic_light_distance_m": _min_non_null(traffic_light_distances),
            "stop_sign_present": bool(traffic_control_frames),
            "stop_sign_types": _object_type_counts(stop_sign_objects),
            "stop_sign_ranges": ranges(traffic_control_frames),
            "min_stop_sign_distance_m": _min_non_null(stop_sign_distances),
            "stop_sign_violation": bool(outcome.get("has_stop_infraction")),
        },
        "pedestrian": {
            "pedestrian_present": bool(pedestrian_frames),
            "pedestrian_types": _object_type_counts(pedestrian_objects),
            "pedestrian_ranges": ranges(pedestrian_frames),
            "min_pedestrian_distance_m": _min_non_null(pedestrian_distances),
            "walker_hazard_ranges": ranges([
                frame for frame, measurement, _ in pairs
                if measurement.get("walker_hazard") or measurement.get("walker_close")
            ]),
            "pedestrian_collision": bool(outcome.get("has_pedestrian_collision")),
        },
        "static_obstacle": {
            "static_obstacle_present": bool(static_frames),
            "static_obstacle_types": _object_type_counts(static_objects),
            "static_obstacle_ranges": ranges(static_frames),
            "min_static_object_distance_m": _min_non_null(static_distances),
            "static_or_layout_collision": bool(outcome.get("has_static_or_layout_collision")),
            "route_obstacle_ranges": ranges([
                frame for frame, measurement, _ in pairs
                if measurement.get("route_obstacle")
            ]),
        },
        "route_shift": {
            "route_shift_present": bool(route_shift_frames),
            "directions": _value_counts(route_shift_directions),
            "scenario_types": _value_counts(route_shift_scenario_types),
            "bypass_phases": _value_counts(route_shift_bypass_phases),
            "route_shift_ranges": ranges(route_shift_frames),
            "changed_route_ranges": ranges(route_shift_changed_frames),
            "path_clear_ranges": ranges(route_shift_path_clear_frames),
            "min_route_index_delta_to_shift": _min_non_null(route_shift_route_index_deltas),
            "max_route_index_delta_to_shift": (
                _round_float(max(route_shift_route_index_deltas))
                if route_shift_route_index_deltas else None
            ),
        },
        "emergency_vehicle": {
            "emergency_vehicle_present": bool(emergency_frames),
            "emergency_vehicle_types": _object_type_counts(emergency_objects),
            "emergency_vehicle_ranges": ranges(emergency_frames),
            "min_emergency_vehicle_distance_m": _min_non_null(emergency_distances),
            "yield_emergency_vehicle_violation": bool(outcome.get("has_emergency_vehicle_yield_infraction")),
        },
    }


def _build_sequence_summary(pairs, summaries_by_frame, frame_rate, sample_stride):
    frames = [frame for frame, _, _ in pairs]
    speeds = [_safe_float(measurement.get("speed")) for _, measurement, _ in pairs]
    sampled_frames = frames[::max(1, sample_stride)]
    if frames and frames[-1] not in sampled_frames:
        sampled_frames.append(frames[-1])
    return {
        "frame_count": len(frames),
        "start_frame": frames[0] if frames else None,
        "end_frame": frames[-1] if frames else None,
        "duration_s": _round_float((frames[-1] - frames[0] + 1) / float(frame_rate), 2) if frames else 0.0,
        "max_speed_mps": _round_float(max(speeds) if speeds else 0.0),
        "max_speed_kmh": _round_float((max(speeds) if speeds else 0.0) * 3.6, 1),
        "mean_speed_mps": _round_float(sum(speeds) / len(speeds) if speeds else 0.0),
        "start": _compact_frame_summary(summaries_by_frame[frames[0]]) if frames else None,
        "end": _compact_frame_summary(summaries_by_frame[frames[-1]]) if frames else None,
        "sampled_frame_index": [_compact_frame_summary(summaries_by_frame[frame]) for frame in sampled_frames],
    }


def _build_scenario_sequence_gt(scenario_dir, csv_info, frame_rate=10, sample_stride=10,
                                sequence_start_frame=3, step_evidence_stride=4):
    pairs = _load_frame_pairs(scenario_dir)
    pairs = [pair for pair in pairs if pair[0] >= sequence_start_frame]
    if not pairs:
        return None

    run_metadata = _load_json_file(scenario_dir / "fault_injection" / "run_metadata.json")
    record = _load_checkpoint_record(scenario_dir)

    scenario_type = _scenario_type_from_dir(scenario_dir, run_metadata)
    csv_row = csv_info.get(scenario_type, {})
    category = csv_row.get("category") or scenario_dir.parent.parent.name
    condition = _condition_from_dir(scenario_dir, run_metadata)
    outcome = _checkpoint_outcome(record)

    summaries_by_frame = {
        frame: _frame_summary(frame, measurement, boxes, frame_rate)
        for frame, measurement, boxes in pairs
    }
    sequence_summary = _build_sequence_summary(pairs, summaries_by_frame, frame_rate, sample_stride)
    steps = _build_sequence_steps(
        pairs,
        summaries_by_frame,
        outcome,
        frame_rate,
        step_evidence_stride=step_evidence_stride,
        scenario_type=scenario_type,
    )
    scenario_evidence = _scenario_evidence_rollup(pairs, summaries_by_frame, outcome, frame_rate)

    route_meta = run_metadata.get("route_metadata", {})

    return {
        "schema_version": "scenario_sequence_v2_compact",
        "scenario": {
            "scenario_id": scenario_dir.name,
            "dataset_path": str(scenario_dir),
            "route_id": str(route_meta.get("route_id") or _route_id_from_name(scenario_dir.name) or ""),
            "town": _town_from_dir(scenario_dir, run_metadata),
            "scenario_name": (route_meta.get("scenario_names") or [scenario_type])[0],
            "scenario_type": scenario_type,
            "category": category,
            "secondary_tag": csv_row.get("secondary_tag"),
            "condition": condition,
            "source_description": csv_row.get("source_description"),
            "expected_clean_response": csv_row.get("clean_response"),
            "expected_inappropriate_response": csv_row.get("inappropriate_response"),
            "primary_target": csv_row.get("primary_target"),
            "legal_compliance_primary": _as_bool(csv_row.get("legal_compliance_primary")),
            "external_attribution_primary": _as_bool(csv_row.get("external_attribution_primary")),
            "frame_rate": frame_rate,
            "start_frame": sequence_summary["start_frame"],
            "end_frame": sequence_summary["end_frame"],
            "duration_s": sequence_summary["duration_s"],
        },
        "observed_outcome": outcome,
        "scenario_evidence": scenario_evidence,
        "steps": steps,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Batch GT event log generator.")
    p.add_argument("--mode", choices=("legacy_events", "scenario_sequence"),
                   default="legacy_events",
                   help="legacy_events uses anno/ event splitting; scenario_sequence "
                        "uses boxes/ + measurements/ and writes one full-sequence GT per route.")
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
    p.add_argument("--scenario-csv", default="bench2drive_recategorized_scenarios_v2.csv",
                   help="Scenario metadata CSV used by --mode scenario_sequence.")
    p.add_argument("--frame-rate", type=int, default=10,
                   help="Frame rate used for time conversion in scenario_sequence mode.")
    p.add_argument("--sample-stride", type=int, default=10,
                   help="Keep one sampled frame summary every N frames in scenario_sequence mode.")
    p.add_argument("--step-evidence-stride", type=int, default=4,
                   help="Keep one compact evidence sample every N frames inside each step.")
    p.add_argument("--sequence-start-frame", type=int, default=3,
                   help="First frame to include in scenario_sequence GT. Default skips spawn frames 0-2.")
    return p.parse_args()


def _main_scenario_sequence(args):
    eval_root = _resolve_repo_path(args.eval_dir)
    scenario_csv = _resolve_repo_path(args.scenario_csv)
    csv_info = _load_scenario_csv(scenario_csv)

    scenario_dirs = _discover_sequence_dirs(eval_root)
    if args.max_routes:
        scenario_dirs = scenario_dirs[:args.max_routes]

    if not scenario_dirs:
        print(f"No boxes/ + measurements/ scenario directories found under {eval_root}", file=sys.stderr)
        sys.exit(1)

    output_dir = _resolve_repo_path(args.output_dir) if args.output_dir else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(scenario_dirs)} scenario sequence(s).", file=sys.stderr)

    ok = fail = 0
    for scenario_dir in scenario_dirs:
        print(f"  Processing {scenario_dir.name} …", file=sys.stderr)
        result = _build_scenario_sequence_gt(
            scenario_dir,
            csv_info,
            frame_rate=args.frame_rate,
            sample_stride=args.sample_stride,
            step_evidence_stride=args.step_evidence_stride,
            sequence_start_frame=args.sequence_start_frame,
        )
        if result is None:
            print("    [SKIP] no aligned boxes/measurement frames found", file=sys.stderr)
            fail += 1
            continue

        out_json = json.dumps(result, indent=2)
        if output_dir:
            scenario_type = result.get("scenario", {}).get("scenario_type") or "UnknownScenarioType"
            out_file = output_dir / scenario_type / f"{scenario_dir.name}.json"
            out_file.parent.mkdir(parents=True, exist_ok=True)
            out_file.write_text(out_json)
            print(f"    [OK] {len(result.get('steps', []))} steps → {out_file}", file=sys.stderr)
        else:
            print(out_json)
        ok += 1

    print(f"\nDone: {ok} succeeded, {fail} skipped.", file=sys.stderr)


def main():
    args = parse_args()
    if args.mode == "scenario_sequence":
        _main_scenario_sequence(args)


if __name__ == "__main__":
    main()

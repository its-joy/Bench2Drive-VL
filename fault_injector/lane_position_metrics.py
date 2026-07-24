from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _state(record: dict[str, Any]) -> dict[str, Any]:
    if "lane_state" in record:
        lane_state = dict(record["lane_state"] or {})
        lane_state.setdefault("timestamp_s", record.get("timestamp_s"))
        lane_state.setdefault("step", record.get("step"))
        return lane_state
    driving_state = record.get("driving_state") or {}
    ego = driving_state.get("ego") or {}
    route = driving_state.get("route") or {}
    lane_topology = driving_state.get("lane_topology") or {}
    return {
        "timestamp_s": record.get("timestamp_s"),
        "step": record.get("step"),
        "current_lane_id": ego.get("lane_id"),
        "preferred_approach_lane_id": route.get("preferred_approach_lane_id"),
        "current_lane_is_preferred": route.get("current_lane_is_preferred"),
        "right_lane_available": lane_topology.get("right_lane_available"),
        "right_lane_change_allowed": lane_topology.get("right_lane_change_allowed"),
        "right_lane_marking_type": lane_topology.get("right_lane_marking_type"),
        "distance_to_junction_m": route.get("distance_to_junction_m"),
        "distance_to_stop_line_m": route.get("distance_to_stop_line_m"),
        "lateral_offset_m": ego.get("lateral_offset_from_lane_center_m"),
        "ego_is_junction": ego.get("is_junction"),
    }


def _is_preferred(sample: dict[str, Any]) -> bool:
    if sample.get("current_lane_is_preferred") is not None:
        return bool(sample.get("current_lane_is_preferred"))
    current_lane = sample.get("current_lane_id")
    preferred_lane = sample.get("preferred_approach_lane_id")
    if current_lane is None or preferred_lane is None:
        return False
    return int(current_lane) == int(preferred_lane)


def _is_solid(marking: Any) -> bool:
    return marking is not None and "solid" in str(marking).lower()


def _first_preferred_sample(samples: list[dict[str, Any]]) -> dict[str, Any] | None:
    for sample in samples:
        if _is_preferred(sample):
            return sample
    return None


def _preferred_lane_id(samples: list[dict[str, Any]]) -> Any:
    for sample in samples:
        if sample.get("preferred_approach_lane_id") is not None:
            return sample.get("preferred_approach_lane_id")
    return None


def _max_abs_lateral_offset(samples: list[dict[str, Any]]) -> float | None:
    values = [abs(float(sample["lateral_offset_m"])) for sample in samples if sample.get("lateral_offset_m") is not None]
    return max(values) if values else None


def _time_spent_non_preferred(samples: list[dict[str, Any]]) -> float | None:
    total = 0.0
    saw_time = False
    for prev, cur in zip(samples, samples[1:]):
        prev_t = prev.get("timestamp_s")
        cur_t = cur.get("timestamp_s")
        if prev_t is None or cur_t is None:
            continue
        saw_time = True
        if not _is_preferred(prev):
            total += max(0.0, float(cur_t) - float(prev_t))
    return total if saw_time else None


def _entered_wrong_lane_junction(samples: list[dict[str, Any]]) -> bool:
    for sample in samples:
        if bool(sample.get("ego_is_junction")) and not _is_preferred(sample):
            return True
    return False


def compute_lane_position_metrics(
    clean_records: list[dict[str, Any]],
    faulted_records: list[dict[str, Any]],
    *,
    minimum_delay_m: float = 3.0,
) -> dict[str, Any]:
    clean_samples = [_state(record) for record in clean_records]
    faulted_samples = [_state(record) for record in faulted_records]
    clean_entry = _first_preferred_sample(clean_samples)
    faulted_entry = _first_preferred_sample(faulted_samples)

    clean_distance = clean_entry.get("distance_to_junction_m") if clean_entry else None
    faulted_distance = faulted_entry.get("distance_to_junction_m") if faulted_entry else None
    clean_time = clean_entry.get("timestamp_s") if clean_entry else None
    faulted_time = faulted_entry.get("timestamp_s") if faulted_entry else None

    lane_entry_delay_m = None
    if clean_distance is not None and faulted_distance is not None:
        lane_entry_delay_m = float(clean_distance) - float(faulted_distance)

    lane_entry_delay_s = None
    if clean_time is not None and faulted_time is not None:
        lane_entry_delay_s = float(faulted_time) - float(clean_time)

    crossed_solid = any(
        _is_solid(sample.get("right_lane_marking_type")) or _is_solid(sample.get("current_lane_marking_right_type"))
        for sample in faulted_samples
    )
    entered_before_junction = faulted_entry is not None and not _entered_wrong_lane_junction(faulted_samples)
    entered_before_stop_line = faulted_entry is not None
    if faulted_entry is not None and faulted_entry.get("distance_to_stop_line_m") is not None:
        entered_before_stop_line = float(faulted_entry["distance_to_stop_line_m"]) >= 0.0

    late_detected = (
        faulted_entry is not None
        and lane_entry_delay_m is not None
        and lane_entry_delay_m >= minimum_delay_m
        and entered_before_junction
        and not crossed_solid
    )

    return {
        "preferred_lane_id": _preferred_lane_id(faulted_samples) or _preferred_lane_id(clean_samples),
        "clean_lane_entry_distance_m": clean_distance,
        "faulted_lane_entry_distance_m": faulted_distance,
        "lane_entry_delay_m": lane_entry_delay_m,
        "lane_entry_delay_s": lane_entry_delay_s,
        "distance_from_junction_when_correct_lane_first_reached_m": faulted_distance,
        "time_spent_in_non_preferred_lane_s": _time_spent_non_preferred(faulted_samples),
        "entered_correct_lane_before_junction": entered_before_junction,
        "entered_correct_lane_before_stop_line": entered_before_stop_line,
        "crossed_solid_marking": crossed_solid,
        "wrong_lane_junction_entry": _entered_wrong_lane_junction(faulted_samples),
        "maximum_lateral_offset_m": _max_abs_lateral_offset(faulted_samples),
        "late_lane_positioning_detected": late_detected,
    }


def write_clean_reference(trace_path: Path, output_path: Path, scenario_pair_id: str | None = None) -> dict[str, Any]:
    records = _load_jsonl(trace_path)
    samples = [_state(record) for record in records]
    reference = {
        "schema_version": "1.0",
        "scenario_pair_id": scenario_pair_id,
        "samples": [
            {
                "timestamp_s": sample.get("timestamp_s"),
                "distance_to_junction_m": sample.get("distance_to_junction_m"),
                "lane_id": sample.get("current_lane_id"),
                "preferred_approach_lane_id": sample.get("preferred_approach_lane_id"),
                "lateral_offset_m": sample.get("lateral_offset_m"),
                "is_preferred_lane": _is_preferred(sample),
            }
            for sample in samples
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(reference, indent=2), encoding="utf-8")
    return reference


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-trace", required=True)
    parser.add_argument("--faulted-trace", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--minimum-delay-m", type=float, default=3.0)
    parser.add_argument("--write-clean-reference")
    parser.add_argument("--scenario-pair-id")
    args = parser.parse_args()

    clean_trace = Path(args.clean_trace)
    faulted_trace = Path(args.faulted_trace)
    if args.write_clean_reference:
        write_clean_reference(clean_trace, Path(args.write_clean_reference), args.scenario_pair_id)

    metrics = compute_lane_position_metrics(
        _load_jsonl(clean_trace),
        _load_jsonl(faulted_trace),
        minimum_delay_m=args.minimum_delay_m,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

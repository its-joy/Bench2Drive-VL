from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _has_infraction(infractions: dict[str, Any], *keys: str) -> bool:
    for key in keys:
        value = infractions.get(key)
        if isinstance(value, list):
            if value:
                return True
        elif bool(value):
            return True
    return False


def _add_reason(reasons: list[str], condition: bool, reason: str) -> None:
    if condition:
        reasons.append(reason)


def validate_fault_outcome(
    checkpoint_path: Path,
    fault_summary_path: Path,
    output_path: Path,
    lane_metrics_path: Path | None = None,
    clean_reference_path: Path | None = None,
) -> dict[str, Any]:
    checkpoint = _load_json(checkpoint_path)
    fault_summary = _load_json(fault_summary_path)
    lane_metrics = _load_json(lane_metrics_path) if lane_metrics_path is not None and lane_metrics_path.exists() else {}

    global_record = checkpoint.get("_checkpoint", {}).get("global_record", {})
    infractions = global_record.get("infractions", {})
    route_completed = str(global_record.get("status", "")).lower() == "completed"

    fault_triggered = bool(fault_summary.get("fault_triggered", False))
    control_was_modified = bool(fault_summary.get("expert_applied_control_differed", False))
    route_was_modified = bool(fault_summary.get("route_modified", False))
    control_or_route_was_modified = control_was_modified or route_was_modified
    observed_red_light_infraction = _has_infraction(infractions, "red_light")
    observed_stop_infraction = _has_infraction(infractions, "stop_infraction", "stop_sign", "stop")
    observed_vehicle_collision = _has_infraction(infractions, "collisions_vehicle")
    observed_pedestrian_collision = _has_infraction(infractions, "collisions_pedestrian")
    observed_layout_collision = _has_infraction(infractions, "collisions_layout")
    observed_outside_route_lanes = _has_infraction(infractions, "outside_route_lanes", "outside_route_lane")
    observed_route_deviation = _has_infraction(infractions, "route_deviation", "route_dev")
    collision_recorded = observed_vehicle_collision or observed_pedestrian_collision or observed_layout_collision
    fault_type = fault_summary.get("fault_type")

    rejection_reasons: list[str] = []
    if fault_type == "none":
        _add_reason(rejection_reasons, fault_triggered, "fault_triggered_in_clean_run")
        _add_reason(rejection_reasons, control_was_modified, "control_modified_in_clean_run")
        _add_reason(rejection_reasons, not route_completed, "route_not_completed")
        _add_reason(rejection_reasons, observed_red_light_infraction, "red_light_infraction_recorded")
        _add_reason(rejection_reasons, observed_stop_infraction, "stop_infraction_recorded")
        _add_reason(rejection_reasons, collision_recorded, "collision_recorded")
        _add_reason(rejection_reasons, observed_outside_route_lanes, "outside_route_lane_recorded")
        _add_reason(rejection_reasons, observed_route_deviation, "route_deviation_recorded")
    elif fault_type == "late_turn_lane_entry":
        _add_reason(rejection_reasons, not fault_triggered, "fault_not_triggered")
        _add_reason(rejection_reasons, not control_or_route_was_modified, "control_or_route_not_modified")
        _add_reason(rejection_reasons, not route_completed, "route_not_completed")
        _add_reason(rejection_reasons, not bool(lane_metrics.get("late_lane_positioning_detected")), "no_measurable_lane_entry_delay")
        _add_reason(
            rejection_reasons,
            not bool(lane_metrics.get("entered_correct_lane_before_junction")),
            "junction_entered_before_lane_correction",
        )
        _add_reason(rejection_reasons, bool(lane_metrics.get("crossed_solid_marking")), "solid_lane_marking_crossed")
        _add_reason(rejection_reasons, observed_red_light_infraction, "red_light_infraction_recorded")
        _add_reason(rejection_reasons, observed_stop_infraction, "stop_infraction_recorded")
        _add_reason(rejection_reasons, collision_recorded, "collision_recorded")
        _add_reason(rejection_reasons, observed_outside_route_lanes, "outside_route_lane_recorded")
        _add_reason(rejection_reasons, observed_route_deviation, "route_deviation_recorded")
    elif fault_type == "red_light_entry":
        _add_reason(rejection_reasons, not fault_triggered, "fault_not_triggered")
        _add_reason(rejection_reasons, not control_was_modified, "control_or_route_not_modified")
        _add_reason(rejection_reasons, not route_completed, "route_not_completed")
        _add_reason(rejection_reasons, not observed_red_light_infraction, "red_light_infraction_not_recorded")
        _add_reason(rejection_reasons, collision_recorded, "collision_recorded")
    elif fault_type == "inappropriate_response":
        _add_reason(rejection_reasons, not fault_triggered, "fault_not_triggered")
        _add_reason(rejection_reasons, not control_or_route_was_modified, "control_or_route_not_modified")
    else:
        rejection_reasons.append(f"unsupported_fault_type:{fault_type}")

    accepted_for_dataset = not rejection_reasons

    result = {
        "config_id": fault_summary.get("config_id"),
        "fault_type": fault_type,
        "requested_condition": fault_summary.get("condition"),
        "fault_triggered": fault_triggered,
        "control_was_modified": control_was_modified,
        "route_was_modified": route_was_modified,
        "control_or_route_was_modified": control_or_route_was_modified,
        "route_completed": route_completed,
        "late_lane_positioning_detected": bool(lane_metrics.get("late_lane_positioning_detected", False)),
        "entered_correct_lane_before_junction": lane_metrics.get("entered_correct_lane_before_junction"),
        "observed_red_light_infraction": observed_red_light_infraction,
        "observed_stop_infraction": observed_stop_infraction,
        "observed_vehicle_collision": observed_vehicle_collision,
        "observed_pedestrian_collision": observed_pedestrian_collision,
        "observed_layout_collision": observed_layout_collision,
        "observed_outside_route_lanes": observed_outside_route_lanes,
        "observed_route_deviation": observed_route_deviation,
        "other_infractions": [],
        "lane_metrics_path": str(lane_metrics_path) if lane_metrics_path is not None else None,
        "clean_reference_path": str(clean_reference_path) if clean_reference_path is not None else None,
        "accepted_for_dataset": accepted_for_dataset,
        "rejection_reasons": rejection_reasons,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--fault-summary", required=True)
    parser.add_argument("--lane-metrics")
    parser.add_argument("--clean-reference")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    validate_fault_outcome(
        Path(args.checkpoint),
        Path(args.fault_summary),
        Path(args.output),
        Path(args.lane_metrics) if args.lane_metrics else None,
        Path(args.clean_reference) if args.clean_reference else None,
    )


if __name__ == "__main__":
    main()

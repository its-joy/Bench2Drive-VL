import json
from pathlib import Path

from fault_injector.validate_fault_outcome import validate_fault_outcome


def test_successful_clean_run_is_accepted(tmp_path):
    checkpoint = {
        "_checkpoint": {
            "global_record": {
                "status": "Completed",
                "infractions": {
                    "red_light": [],
                    "collisions_vehicle": [],
                    "collisions_pedestrian": [],
                    "collisions_layout": [],
                },
            }
        }
    }
    summary = {"config_id": "clean", "fault_type": "none", "condition": "clean", "fault_triggered": False, "expert_applied_control_differed": False}
    checkpoint_path = tmp_path / "checkpoint.json"
    summary_path = tmp_path / "fault_summary.json"
    output_path = tmp_path / "validation.json"
    checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    result = validate_fault_outcome(checkpoint_path, summary_path, output_path)
    assert result["accepted_for_dataset"] is True


def test_successful_red_light_only_run_is_accepted(tmp_path):
    checkpoint = {
        "_checkpoint": {
            "global_record": {
                "status": "Completed",
                "infractions": {
                    "red_light": ["ran_red_light"],
                    "collisions_vehicle": [],
                    "collisions_pedestrian": [],
                    "collisions_layout": [],
                },
            }
        }
    }
    summary = {"config_id": "red", "fault_type": "red_light_entry", "condition": "infraction", "fault_triggered": True, "expert_applied_control_differed": True}
    checkpoint_path = tmp_path / "checkpoint.json"
    summary_path = tmp_path / "fault_summary.json"
    output_path = tmp_path / "validation.json"
    checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    result = validate_fault_outcome(checkpoint_path, summary_path, output_path)
    assert result["accepted_for_dataset"] is True


def test_successful_late_lane_mistake_run_is_accepted(tmp_path):
    checkpoint = {
        "_checkpoint": {
            "global_record": {
                "status": "Completed",
                "infractions": {
                    "red_light": [],
                    "stop_infraction": [],
                    "collisions_vehicle": [],
                    "collisions_pedestrian": [],
                    "collisions_layout": [],
                    "outside_route_lanes": [],
                    "route_deviation": [],
                },
            }
        }
    }
    summary = {
        "config_id": "late",
        "fault_type": "late_turn_lane_entry",
        "condition": "mistake_only",
        "fault_triggered": True,
        "expert_applied_control_differed": True,
    }
    metrics = {
        "late_lane_positioning_detected": True,
        "entered_correct_lane_before_junction": True,
        "crossed_solid_marking": False,
    }
    checkpoint_path = tmp_path / "checkpoint.json"
    summary_path = tmp_path / "fault_summary.json"
    metrics_path = tmp_path / "lane_metrics.json"
    output_path = tmp_path / "validation.json"
    checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    result = validate_fault_outcome(checkpoint_path, summary_path, output_path, metrics_path)
    assert result["accepted_for_dataset"] is True


def test_late_lane_with_red_light_infraction_is_rejected(tmp_path):
    checkpoint = {
        "_checkpoint": {
            "global_record": {
                "status": "Completed",
                "infractions": {
                    "red_light": ["ran red"],
                    "collisions_vehicle": [],
                    "collisions_pedestrian": [],
                    "collisions_layout": [],
                },
            }
        }
    }
    summary = {
        "config_id": "late",
        "fault_type": "late_turn_lane_entry",
        "condition": "mistake_only",
        "fault_triggered": True,
        "expert_applied_control_differed": True,
    }
    metrics = {"late_lane_positioning_detected": True, "entered_correct_lane_before_junction": True, "crossed_solid_marking": False}
    checkpoint_path = tmp_path / "checkpoint.json"
    summary_path = tmp_path / "fault_summary.json"
    metrics_path = tmp_path / "lane_metrics.json"
    output_path = tmp_path / "validation.json"
    checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    result = validate_fault_outcome(checkpoint_path, summary_path, output_path, metrics_path)
    assert result["accepted_for_dataset"] is False
    assert "red_light_infraction_recorded" in result["rejection_reasons"]

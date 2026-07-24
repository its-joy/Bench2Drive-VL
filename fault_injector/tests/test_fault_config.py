from pathlib import Path
import json

import pytest

from fault_injector.fault_config import FaultConfig


def test_valid_clean_config_loads():
    config = FaultConfig.from_json(Path("fault_injector/configs/signalized_right_turn/clean.json"))
    assert config.fault_type == "none"
    assert config.config_id == "signalized_right_turn_clean_v1"


def test_valid_red_light_config_loads():
    config = FaultConfig.from_json(Path("fault_injector/configs/signalized_right_turn/red_light_entry.json"))
    assert config.fault_type == "red_light_entry"
    assert config.trigger["traffic_light_state"] == "Red"


def test_valid_late_turn_lane_config_loads():
    config = FaultConfig.from_json(Path("fault_injector/configs/signalized_right_turn/late_turn_lane_entry.json"))
    assert config.fault_type == "late_turn_lane_entry"
    assert config.intervention["strategy"] == "hold_current_lane_route"


def test_valid_available_inappropriate_response_config_loads():
    config = FaultConfig.from_json(
        Path("fault_injector/configs/signalized_right_turn/inappropriate_available_post_action_data.json")
    )
    assert config.fault_type == "inappropriate_response"
    assert config.condition == "inappropriate_response"


def test_unsupported_inappropriate_behavior_switch_fails(tmp_path):
    payload = json.loads(
        Path("fault_injector/configs/signalized_right_turn/inappropriate_available_post_action_data.json").read_text(
            encoding="utf-8"
        )
    )
    first_option = payload["inappropriate_response_scenarios"][0]["inappropriate_response_control"]["response_options"][0]
    first_option["behavior"]["imaginary_noop_switch"] = True
    path = tmp_path / "bad_inappropriate_switch.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported inappropriate response behavior switch"):
        FaultConfig.from_json(path)


def test_unsupported_schema_version_fails():
    path = Path("/tmp/unsupported_schema.json")
    path.write_text('{"schema_version": "2.0", "config_id": "x", "base_scenario": "s", "condition": "c", "fault_type": "none", "trigger": {}, "intervention": {}, "release": {}, "guards": {}, "expected": {}}', encoding="utf-8")
    with pytest.raises(ValueError):
        FaultConfig.from_json(path)


def test_unsupported_fault_type_fails():
    path = Path("/tmp/unsupported_fault.json")
    path.write_text('{"schema_version": "1.0", "config_id": "x", "base_scenario": "s", "condition": "c", "fault_type": "bad", "trigger": {}, "intervention": {}, "release": {}, "guards": {}, "expected": {}}', encoding="utf-8")
    with pytest.raises(ValueError):
        FaultConfig.from_json(path)


def test_invalid_throttle_range_fails():
    path = Path("/tmp/bad_throttle.json")
    path.write_text('{"schema_version": "1.0", "config_id": "x", "base_scenario": "s", "condition": "c", "fault_type": "none", "trigger": {}, "intervention": {"minimum_throttle": 1.2}, "release": {}, "guards": {}, "expected": {}}', encoding="utf-8")
    with pytest.raises(ValueError):
        FaultConfig.from_json(path)


def test_missing_red_light_trigger_field_fails():
    path = Path("/tmp/missing_trigger.json")
    path.write_text('{"schema_version": "1.0", "config_id": "x", "base_scenario": "s", "condition": "c", "fault_type": "red_light_entry", "trigger": {"traffic_light_state": "Red"}, "intervention": {}, "release": {}, "guards": {}, "expected": {}}', encoding="utf-8")
    with pytest.raises(ValueError):
        FaultConfig.from_json(path)


def test_negative_confirmation_ticks_fail():
    path = Path("/tmp/bad_confirm.json")
    path.write_text('{"schema_version": "1.0", "config_id": "x", "base_scenario": "s", "condition": "c", "fault_type": "red_light_entry", "trigger": {"traffic_light_state": "Red", "traffic_light_affects_ego": true, "maximum_traffic_light_distance_m": 10.0, "require_not_in_junction": true, "confirmation_ticks": 0}, "intervention": {}, "release": {}, "guards": {}, "expected": {}}', encoding="utf-8")
    with pytest.raises(ValueError):
        FaultConfig.from_json(path)


def test_unsupported_late_lane_strategy_fails(tmp_path):
    payload = Path("fault_injector/configs/signalized_right_turn/late_turn_lane_entry.json").read_text(encoding="utf-8")
    path = tmp_path / "bad_strategy.json"
    path.write_text(payload.replace('"hold_current_lane_route"', '"raw_zero_steer"'), encoding="utf-8")
    with pytest.raises(ValueError):
        FaultConfig.from_json(path)


def test_invalid_late_lane_distance_interval_fails(tmp_path):
    payload = Path("fault_injector/configs/signalized_right_turn/late_turn_lane_entry.json").read_text(encoding="utf-8")
    path = tmp_path / "bad_distance.json"
    path.write_text(payload.replace('"maximum_junction_distance_m": 105.0', '"maximum_junction_distance_m": 5.0'), encoding="utf-8")
    with pytest.raises(ValueError):
        FaultConfig.from_json(path)


def test_invalid_late_lane_lateral_offset_fails(tmp_path):
    payload = Path("fault_injector/configs/signalized_right_turn/late_turn_lane_entry.json").read_text(encoding="utf-8")
    path = tmp_path / "bad_offset.json"
    path.write_text(payload.replace('"maximum_lateral_offset_m": 2.5', '"maximum_lateral_offset_m": 0.0'), encoding="utf-8")
    with pytest.raises(ValueError):
        FaultConfig.from_json(path)

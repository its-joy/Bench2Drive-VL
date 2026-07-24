from dataclasses import replace
from pathlib import Path

from fault_injector.fault_config import FaultConfig
from fault_injector.fault_injector import FaultInjector, InjectorState


class DummyControl:
    def __init__(self, steer=0.0, throttle=0.0, brake=0.0):
        self.steer = steer
        self.throttle = throttle
        self.brake = brake
        self.hand_brake = False
        self.reverse = False
        self.manual_gear_shift = False
        self.gear = 0


def test_noop_applied_control_equals_expert_control():
    config = FaultConfig.from_json(Path("fault_injector/configs/signalized_right_turn/clean.json"))
    injector = FaultInjector(config, simulation_fps=20.0)
    expert = DummyControl(steer=0.1, throttle=0.3, brake=0.0)
    applied, record = injector.apply(expert, {"step": 1, "timestamp_s": 0.05, "ego": {}, "traffic_light": {}, "hazards": {}})
    assert applied.steer == expert.steer
    assert applied.throttle == expert.throttle
    assert applied.brake == expert.brake
    assert injector.state == InjectorState.DISABLED
    assert record["fault_triggered"] is False


def test_green_light_does_not_activate():
    config = FaultConfig.from_json(Path("fault_injector/configs/signalized_right_turn/red_light_entry.json"))
    injector = FaultInjector(config, simulation_fps=20.0)
    expert = DummyControl(steer=0.2, throttle=0.0, brake=0.5)
    driving_state = {
        "step": 1,
        "timestamp_s": 0.05,
        "ego": {"speed_mps": 4.0, "is_junction": False},
        "traffic_light": {"present": True, "state": "Green", "affects_ego": True, "distance_m": 10.0, "actor_id": 1},
        "hazards": {"vehicle": False, "walker": False, "traffic_light": True},
    }
    _, record = injector.apply(expert, driving_state)
    assert record["injector_state"] == "armed"
    assert record["trigger_matches"] is False


def test_activation_occurs_after_confirmation_count():
    config = FaultConfig.from_json(Path("fault_injector/configs/signalized_right_turn/red_light_entry.json"))
    config = replace(config, trigger={**config.trigger, "trigger_on_first_red_light": False, "confirmation_ticks": 3})
    injector = FaultInjector(config, simulation_fps=20.0)
    expert = DummyControl(steer=0.25, throttle=0.0, brake=0.8)
    driving_state = {
        "step": 1,
        "timestamp_s": 0.05,
        "ego": {"speed_mps": 2.0, "is_junction": False},
        "traffic_light": {"present": True, "state": "Red", "affects_ego": True, "distance_m": 8.0, "actor_id": 1},
        "hazards": {"vehicle": False, "walker": False, "traffic_light": True},
    }
    for _ in range(3):
        _, record = injector.apply(expert, driving_state)
    assert injector.state == InjectorState.ACTIVE
    assert record["trigger_confirmation_count"] == 3


def test_ignore_red_light_intervention_does_not_override_low_level_control():
    config = FaultConfig.from_json(Path("fault_injector/configs/signalized_right_turn/red_light_entry.json"))
    injector = FaultInjector(config, simulation_fps=20.0)
    expert = DummyControl(steer=0.4, throttle=0.0, brake=0.8)
    driving_state = {
        "step": 1,
        "timestamp_s": 0.05,
        "ego": {"speed_mps": 2.0, "is_junction": False},
        "traffic_light": {"present": True, "state": "Red", "affects_ego": True, "distance_m": 8.0, "actor_id": 1},
        "hazards": {"vehicle": False, "walker": False, "traffic_light": True},
    }
    for _ in range(3):
        _, _ = injector.apply(expert, driving_state)
    applied, _ = injector.apply(expert, driving_state)
    assert applied.steer == expert.steer
    assert applied.brake == expert.brake
    assert applied.throttle == expert.throttle


def test_red_light_hazard_triggers_even_when_affects_ego_is_false():
    config = FaultConfig.from_json(Path("fault_injector/configs/signalized_right_turn/red_light_entry.json"))
    injector = FaultInjector(config, simulation_fps=20.0)
    expert = DummyControl(steer=0.01, throttle=0.0, brake=1.0)
    driving_state = {
        "step": 250,
        "timestamp_s": 12.5,
        "ego": {"speed_mps": 0.01, "is_junction": False},
        "traffic_light": {"present": True, "state": 0, "affects_ego": False, "distance_m": 43.8, "actor_id": 46},
        "hazards": {"vehicle": False, "walker": False, "traffic_light": True, "stop_sign": False},
    }

    applied, record = injector.apply(expert, driving_state)

    assert injector.state == InjectorState.ACTIVE
    assert record["trigger_matches"] is True
    assert applied.brake == expert.brake
    assert applied.throttle == expert.throttle


def test_red_light_can_activate_before_pdm_control():
    config = FaultConfig.from_json(Path("fault_injector/configs/signalized_right_turn/red_light_entry.json"))
    injector = FaultInjector(config, simulation_fps=20.0)
    driving_state = {
        "step": 1,
        "timestamp_s": 0.05,
        "ego": {"speed_mps": 2.0, "is_junction": False},
        "traffic_light": {"present": True, "state": "Red", "affects_ego": True, "distance_m": 8.0, "actor_id": 1},
        "hazards": {"vehicle": False, "walker": False, "traffic_light": True},
    }

    result = injector.prepare_before_control(driving_state)

    assert result["activated_before_control"] is True
    assert injector.state == InjectorState.ACTIVE


def test_vehicle_hazard_aborts_intervention():
    config = FaultConfig.from_json(Path("fault_injector/configs/signalized_right_turn/red_light_entry.json"))
    config = replace(config, guards={**config.guards, "abort_on_vehicle_hazard": True, "abort_on_walker_hazard": True})
    injector = FaultInjector(config, simulation_fps=20.0)
    expert = DummyControl(steer=0.2, throttle=0.0, brake=0.8)
    driving_state = {
        "step": 1,
        "timestamp_s": 0.05,
        "ego": {"speed_mps": 2.0, "is_junction": False},
        "traffic_light": {"present": True, "state": "Red", "affects_ego": True, "distance_m": 8.0, "actor_id": 1},
        "hazards": {"vehicle": True, "walker": False, "traffic_light": True},
    }
    for _ in range(3):
        injector.apply(expert, driving_state)
    applied, record = injector.apply(expert, driving_state)
    assert injector.state == InjectorState.ABORTED
    assert record["abort_reason"] == "vehicle_hazard"
    assert applied.steer == expert.steer


def test_vehicle_hazard_can_be_ignored_by_guard():
    config = FaultConfig.from_json(Path("fault_injector/configs/signalized_right_turn/red_light_entry.json"))
    config = replace(config, guards={**config.guards, "abort_on_vehicle_hazard": False, "abort_on_walker_hazard": False})
    injector = FaultInjector(config, simulation_fps=20.0)
    expert = DummyControl(steer=0.25, throttle=0.0, brake=0.8)
    driving_state = {
        "step": 1,
        "timestamp_s": 0.05,
        "ego": {"speed_mps": 2.0, "is_junction": False},
        "traffic_light": {"present": True, "state": "Red", "affects_ego": True, "distance_m": 8.0, "actor_id": 1},
        "hazards": {"vehicle": True, "walker": False, "traffic_light": True},
    }
    for _ in range(3):
        injector.apply(expert, driving_state)
    assert injector.state == InjectorState.ACTIVE


def test_numeric_carla_traffic_light_state_triggers_activation():
    config = FaultConfig.from_json(Path("fault_injector/configs/signalized_right_turn/red_light_entry.json"))
    injector = FaultInjector(config, simulation_fps=20.0)
    expert = DummyControl(steer=0.25, throttle=0.0, brake=0.8)
    driving_state = {
        "step": 1,
        "timestamp_s": 0.05,
        "ego": {"speed_mps": 2.0, "is_junction": False},
        "traffic_light": {"present": True, "state": 0, "affects_ego": True, "distance_m": 8.0, "actor_id": 1},
        "hazards": {"vehicle": False, "walker": False, "traffic_light": True},
    }
    for _ in range(3):
        injector.apply(expert, driving_state)
    assert injector.state == InjectorState.ACTIVE


def test_first_red_light_only_triggers_once():
    config = FaultConfig.from_json(Path("fault_injector/configs/signalized_right_turn/red_light_entry.json"))
    injector = FaultInjector(config, simulation_fps=20.0)
    expert = DummyControl(steer=0.25, throttle=0.0, brake=0.8)
    first_state = {
        "step": 1,
        "timestamp_s": 0.05,
        "ego": {"speed_mps": 2.0, "is_junction": False},
        "traffic_light": {"present": True, "state": "Red", "affects_ego": True, "distance_m": 8.0, "actor_id": 1},
        "hazards": {"vehicle": False, "walker": False, "traffic_light": True},
    }
    second_state = {
        "step": 2,
        "timestamp_s": 0.1,
        "ego": {"speed_mps": 2.0, "is_junction": False},
        "traffic_light": {"present": True, "state": "Red", "affects_ego": True, "distance_m": 9.0, "actor_id": 2},
        "hazards": {"vehicle": False, "walker": False, "traffic_light": True},
    }
    _, first_record = injector.apply(expert, first_state)
    _, second_record = injector.apply(expert, second_state)
    assert first_record["trigger_confirmation_count"] == 1
    assert second_record["trigger_confirmation_count"] == 1
    assert injector.state == InjectorState.ACTIVE

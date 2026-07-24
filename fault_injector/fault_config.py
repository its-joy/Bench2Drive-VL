from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SUPPORTED_INAPPROPRIATE_BEHAVIOR_SWITCHES = frozenset(
    {
        "ignore_red_light",
        "ignore_stop_sign",
        "ignore_walker_hazard",
        "ignore_vehicle_hazard",
        "ignore_bicycle_hazard",
        "ignore_leading_vehicle",
        "suppress_route_obstacle_handling",
    }
)


@dataclass(frozen=True)
class FaultConfig:
    schema_version: str
    config_id: str
    base_scenario: str
    condition: str
    fault_type: str
    trigger: dict[str, Any]
    intervention: dict[str, Any]
    release: dict[str, Any]
    guards: dict[str, Any]
    expected: dict[str, Any]
    actor_spawn_control: dict[str, Any]

    @classmethod
    def from_json(cls, path: Path | str) -> "FaultConfig":
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Fault configuration file does not exist: {config_path}")

        with config_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        if not isinstance(payload, dict):
            raise ValueError("Fault configuration must be a JSON object")

        schema_version = payload.get("schema_version")
        if schema_version != "1.0":
            raise ValueError(f"Unsupported schema_version: {schema_version}")

        required_fields = [
            "config_id",
            "base_scenario",
            "condition",
            "fault_type",
            "trigger",
            "intervention",
            "release",
            "guards",
            "expected",
        ]
        for field in required_fields:
            if field not in payload:
                raise ValueError(f"Missing required field: {field}")

        fault_type = payload["fault_type"]
        if fault_type not in {"none", "red_light_entry", "late_turn_lane_entry", "inappropriate_response"}:
            raise ValueError(f"Unsupported fault_type: {fault_type}")

        trigger = payload.get("trigger") if isinstance(payload.get("trigger"), dict) else {}
        intervention = payload.get("intervention") if isinstance(payload.get("intervention"), dict) else {}
        release = payload.get("release") if isinstance(payload.get("release"), dict) else {}
        guards = payload.get("guards") if isinstance(payload.get("guards"), dict) else {}
        expected = payload.get("expected") if isinstance(payload.get("expected"), dict) else {}
        actor_spawn_control = (
            payload.get("actor_spawn_control") if isinstance(payload.get("actor_spawn_control"), dict) else {}
        )

        if fault_type == "red_light_entry":
            for key in [
                "traffic_light_state",
                "traffic_light_affects_ego",
                "maximum_traffic_light_distance_m",
                "require_not_in_junction",
                "confirmation_ticks",
            ]:
                if key not in trigger:
                    raise ValueError(f"Missing red-light trigger field: {key}")
            if not isinstance(trigger["confirmation_ticks"], int) or trigger["confirmation_ticks"] < 1:
                raise ValueError("confirmation_ticks must be a positive integer")
            if not isinstance(trigger["maximum_traffic_light_distance_m"], (int, float)):
                raise ValueError("maximum_traffic_light_distance_m must be numeric")
            for key in ("ignore_traffic_light_affects_ego", "require_traffic_light_hazard", "trigger_on_first_red_light"):
                if key in trigger and not isinstance(trigger.get(key), bool):
                    raise ValueError(f"{key} must be boolean")
            if "ignore_red_light" in intervention and not isinstance(intervention.get("ignore_red_light"), bool):
                raise ValueError("ignore_red_light must be boolean")

        if fault_type == "late_turn_lane_entry":
            for key in [
                "route_command",
                "require_not_in_junction",
                "maximum_junction_distance_m",
                "minimum_junction_distance_m",
                "require_right_lane_available",
                "confirmation_ticks",
            ]:
                if key not in trigger:
                    raise ValueError(f"Missing late-lane trigger field: {key}")
            if not isinstance(trigger["confirmation_ticks"], int) or trigger["confirmation_ticks"] < 1:
                raise ValueError("confirmation_ticks must be a positive integer")
            min_distance = trigger.get("minimum_junction_distance_m")
            max_distance = trigger.get("maximum_junction_distance_m")
            if not isinstance(min_distance, (int, float)) or not isinstance(max_distance, (int, float)):
                raise ValueError("junction distance bounds must be numeric")
            if float(min_distance) < 0.0 or float(max_distance) <= float(min_distance):
                raise ValueError("invalid junction distance interval")

            strategy = intervention.get("strategy")
            if strategy not in {"bounded_steering_delay", "hold_current_lane_route"}:
                raise ValueError(f"Unsupported late-lane intervention strategy: {strategy}")
            if "max_adjacent_lane_search_steps" in intervention:
                max_search_steps = intervention.get("max_adjacent_lane_search_steps")
                if not isinstance(max_search_steps, int) or max_search_steps < 1:
                    raise ValueError("max_adjacent_lane_search_steps must be a positive integer")
            if "release_at_junction_distance_m" not in release:
                raise ValueError("Missing late-lane release field: release_at_junction_distance_m")
            release_distance = release.get("release_at_junction_distance_m")
            if not isinstance(release_distance, (int, float)) or float(release_distance) < 0.0:
                raise ValueError("release_at_junction_distance_m must be non-negative")
            max_offset = guards.get("maximum_lateral_offset_m")
            if max_offset is not None and (not isinstance(max_offset, (int, float)) or float(max_offset) <= 0.0):
                raise ValueError("maximum_lateral_offset_m must be positive")

        if fault_type == "inappropriate_response":
            cls._validate_inappropriate_response_payload(payload)

        if "minimum_throttle" in intervention:
            cls._validate_numeric_range("minimum_throttle", intervention.get("minimum_throttle"), 0.0, 1.0)
        if "maximum_throttle" in intervention:
            cls._validate_numeric_range("maximum_throttle", intervention.get("maximum_throttle"), 0.0, 1.0)
        if "brake" in intervention:
            cls._validate_numeric_range("brake", intervention.get("brake"), 0.0, 1.0)
        cls._validate_actor_spawn_control(actor_spawn_control)

        if fault_type == "none":
            trigger = {}
            intervention = {}
            release = {}
            guards = {}

        return cls(
            schema_version=str(schema_version),
            config_id=str(payload["config_id"]),
            base_scenario=str(payload["base_scenario"]),
            condition=str(payload["condition"]),
            fault_type=fault_type,
            trigger=trigger,
            intervention=intervention,
            release=release,
            guards=guards,
            expected=expected,
            actor_spawn_control=actor_spawn_control,
        )

    @staticmethod
    def _validate_numeric_range(name: str, value: Any, minimum: float, maximum: float) -> None:
        if value is None:
            return
        if not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be numeric")
        numeric_value = float(value)
        if numeric_value < minimum or numeric_value > maximum:
            raise ValueError(f"{name} must be between {minimum} and {maximum}")

    @classmethod
    def _validate_actor_spawn_control(cls, actor_spawn_control: dict[str, Any]) -> None:
        mode = actor_spawn_control.get("mode", "default")
        if mode not in {
            "default",
            "off",
            "none",
            "all",
            "no_background",
            "no_background_actors",
            "near_ego",
            "parking_lanes_near_ego",
            "radius",
            "clear_ego",
            "clear_ego_radius",
        }:
            raise ValueError(f"Unsupported actor_spawn_control mode: {mode}")

        if "clear_radius_m" in actor_spawn_control:
            value = actor_spawn_control.get("clear_radius_m")
            if not isinstance(value, (int, float)) or float(value) < 0.0:
                raise ValueError("actor_spawn_control.clear_radius_m must be non-negative")

        for key in ("clear_vehicles", "clear_walkers", "repeat"):
            if key in actor_spawn_control and not isinstance(actor_spawn_control.get(key), bool):
                raise ValueError(f"actor_spawn_control.{key} must be boolean")

    @classmethod
    def _validate_inappropriate_response_payload(cls, payload: dict[str, Any]) -> None:
        defaults = payload.get("inappropriate_response_defaults")
        if defaults is not None and not isinstance(defaults, dict):
            raise ValueError("inappropriate_response_defaults must be an object")
        if isinstance(defaults, dict):
            for scenario_type, control in defaults.items():
                cls._validate_inappropriate_response_control(
                    control,
                    f"inappropriate_response_defaults.{scenario_type}",
                )

        scenarios = payload.get("inappropriate_response_scenarios")
        if scenarios is not None and not isinstance(scenarios, list):
            raise ValueError("inappropriate_response_scenarios must be a list")
        for index, item in enumerate(scenarios or []):
            if not isinstance(item, dict):
                raise ValueError(f"inappropriate_response_scenarios[{index}] must be an object")
            control = item.get("inappropriate_response_control")
            if control is not None:
                cls._validate_inappropriate_response_control(
                    control,
                    f"inappropriate_response_scenarios[{index}].inappropriate_response_control",
                )

    @classmethod
    def _validate_inappropriate_response_control(cls, control: Any, context: str) -> None:
        if not isinstance(control, dict):
            raise ValueError(f"{context} must be an object")
        if "enabled" in control and not isinstance(control.get("enabled"), bool):
            raise ValueError(f"{context}.enabled must be boolean")
        if "behavior" in control:
            cls._validate_inappropriate_behavior(control.get("behavior"), f"{context}.behavior")

        response_options = control.get("response_options")
        if response_options is None:
            return
        if not isinstance(response_options, list):
            raise ValueError(f"{context}.response_options must be a list")
        for option_index, option in enumerate(response_options):
            if not isinstance(option, dict):
                raise ValueError(f"{context}.response_options[{option_index}] must be an object")
            behavior = option.get("behavior")
            if behavior is not None:
                cls._validate_inappropriate_behavior(
                    behavior,
                    f"{context}.response_options[{option_index}].behavior",
                )

    @staticmethod
    def _validate_inappropriate_behavior(behavior: Any, context: str) -> None:
        if not isinstance(behavior, dict):
            raise ValueError(f"{context} must be an object")

        unknown = sorted(set(behavior) - SUPPORTED_INAPPROPRIATE_BEHAVIOR_SWITCHES)
        if unknown:
            supported = ", ".join(sorted(SUPPORTED_INAPPROPRIATE_BEHAVIOR_SWITCHES))
            raise ValueError(
                f"Unsupported inappropriate response behavior switch(es) at {context}: "
                f"{', '.join(unknown)}. Supported switches: {supported}"
            )

        for key, value in behavior.items():
            if not isinstance(value, bool):
                raise ValueError(f"{context}.{key} must be boolean")

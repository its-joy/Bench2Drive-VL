from __future__ import annotations

import json
import os
import random
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
THIRD_PARTY_TEAM_CODE = REPO_ROOT / "third_party" / "carla_garage" / "team_code"
if str(THIRD_PARTY_TEAM_CODE) not in sys.path:
    sys.path.insert(0, str(THIRD_PARTY_TEAM_CODE))

try:
    from data_agent import DataAgent as NativePDMDataAgent  # type: ignore
except Exception as exc:  # pragma: no cover - exercised when CARLA is absent in tests
    NativePDMDataAgent = object  # type: ignore[misc,assignment]
    _NATIVE_IMPORT_ERROR = exc
else:
    _NATIVE_IMPORT_ERROR = None

if __package__ in {None, ""}:
    sys.path.insert(0, str(REPO_ROOT))
    from fault_injector.fault_config import FaultConfig
    from fault_injector.fault_injector import FaultInjector, InjectorState
    from fault_injector.fault_state_extractor import extract_driving_state
    from fault_injector.route_intent_override import LateLaneRouteOverride
else:
    from .fault_config import FaultConfig
    from .fault_injector import FaultInjector, InjectorState
    from .fault_state_extractor import extract_driving_state
    from .route_intent_override import LateLaneRouteOverride


def get_entry_point() -> str:
    return "PDMFaultDataAgent"


class PDMFaultDataAgent(NativePDMDataAgent):
    def setup(self, path_to_conf_file: str, route_index: int | str | None = None, traffic_manager: Any = None) -> None:
        if _NATIVE_IMPORT_ERROR is not None and NativePDMDataAgent is object:
            raise ImportError(f"Native CARLA Garage DataAgent could not be imported: {_NATIVE_IMPORT_ERROR}")

        route_metadata = _resolve_route_metadata(
            os.environ.get("ROUTES"),
            route_index,
            os.environ.get("ROUTES_SUBSET") or os.environ.get("ROUTE_INDEX") or "",
            int(os.environ.get("REPETITIONS", "1") or 1),
        )
        route_folder_name = _format_route_folder_name(route_metadata, route_index)
        super().setup(path_to_conf_file, route_index=route_folder_name or route_index, traffic_manager=traffic_manager)

        fault_config_path = os.environ.get("FAULT_CONFIG")
        if not fault_config_path:
            raise ValueError("FAULT_CONFIG is not set")

        fault_config_file = Path(fault_config_path)
        if not fault_config_file.exists():
            raise FileNotFoundError(f"Fault configuration file does not exist: {fault_config_file}")

        self.fault_config_payload = _load_json_dict(fault_config_file)
        self.fault_config = FaultConfig.from_json(fault_config_file)
        self.fault_route_metadata = route_metadata
        self.fault_route_folder_name = route_folder_name
        self.fault_state_extractor = extract_driving_state
        self.fault_injector = FaultInjector(self.fault_config, simulation_fps=getattr(getattr(self, "config", None), "fps", 20.0))
        self.fault_run_id = os.environ.get("FAULT_RUN_ID", "default")
        self.scenario_pair_id = os.environ.get("SCENARIO_PAIR_ID")
        self.variant_id = os.environ.get("VARIANT_ID", self.fault_config.condition)
        self.fault_output_dir = None
        self.fault_trace_path = None
        self.fault_summary_path = None
        self.clean_reference_path = None
        self.run_metadata_path = None
        self.fault_records: list[dict[str, Any]] = []
        self.fault_ignore_red_light_status: dict[str, Any] = {"active": False}
        self.inappropriate_response_control = self._resolve_inappropriate_response_control()
        self.inappropriate_response_status: dict[str, Any] = {
            "active": False,
            "control": self.inappropriate_response_control,
        }
        self.late_lane_route_override = (
            LateLaneRouteOverride(self.fault_config)
            if self.fault_config.fault_type == "late_turn_lane_entry"
            and self.fault_config.intervention.get("strategy") == "hold_current_lane_route"
            else None
        )
        self.late_lane_route_override_status: dict[str, Any] = {"active": False, "route_modified": False}
        actor_spawn_control = self.fault_config.actor_spawn_control or {}
        self.actor_spawn_mode = str(actor_spawn_control.get("mode", "default")).strip().lower()
        self.actor_clear_radius_m = _config_float(actor_spawn_control, "clear_radius_m", 80.0)
        self.actor_clear_vehicles = _config_bool(actor_spawn_control, "clear_vehicles", True)
        self.actor_clear_walkers = _config_bool(actor_spawn_control, "clear_walkers", True)
        self.actor_clear_repeat = _config_bool(actor_spawn_control, "repeat", True)
        self.actor_spawn_control_applied = False
        self.actor_spawn_control_removed: dict[int, dict[str, Any]] = {}
        self.clean_scenario_control = self._resolve_clean_scenario_control()
        self.clean_scenario_control_applied = self._apply_clean_scenario_control()
        self.fault_summary: dict[str, Any] = {
            "config_id": self.fault_config.config_id,
            "fault_type": self.fault_config.fault_type,
            "condition": self.fault_config.condition,
            "initial_state": self.fault_injector.state.value,
            "final_state": self.fault_injector.state.value,
            "armed_step": None,
            "activated_step": None,
            "release_started_step": None,
            "completed_step": None,
            "aborted_step": None,
            "abort_reason": None,
            "fault_triggered": False,
            "route_modified": False,
            "expert_applied_control_differed": False,
            "inappropriate_response_control": self.inappropriate_response_control,
            "actor_spawn_control": self._actor_spawn_control_summary(),
            "clean_scenario_control": self.clean_scenario_control_applied,
        }
        self._refresh_fault_output_dir()
        self._write_fault_summary(force=True)
        self._write_resolved_config()
        self._write_run_metadata()

    def run_step(self, input_data: dict[str, Any], timestamp: float, sensors: Any = None, plant: bool = False) -> Any:
        self.fault_current_timestamp_s = timestamp
        return super().run_step(input_data, timestamp, sensors=sensors, plant=plant)

    def _get_control(self, input_data: dict[str, Any], plant: Any) -> tuple[Any, Any]:
        self._apply_actor_spawn_control()
        self.fault_ignore_red_light_status = {"active": False}
        self.inappropriate_response_status = {
            "active": self._inappropriate_response_active(),
            "control": self.inappropriate_response_control,
        }
        pre_control_state = self.fault_state_extractor(
            self,
            input_data=input_data,
            timestamp=getattr(self, "fault_current_timestamp_s", None),
        )
        pre_control_prepare = self.fault_injector.prepare_before_control(pre_control_state)
        pre_control_state.setdefault("fault_injector", {})["prepare_before_control"] = pre_control_prepare
        self.late_lane_route_override_status = self._apply_late_lane_route_override(pre_control_state)
        pre_control_state.setdefault("fault_injector", {})["route_override"] = self.late_lane_route_override_status
        expert_control, driving_data = super()._get_control(input_data, plant)

        post_control_state = self.fault_state_extractor(
            self,
            input_data=input_data,
            timestamp=getattr(self, "fault_current_timestamp_s", None),
        )
        post_control_state.setdefault("fault_injector", {})["prepare_before_control"] = pre_control_prepare
        post_control_state.setdefault("fault_injector", {})["ignore_red_light"] = self.fault_ignore_red_light_status
        post_control_state.setdefault("fault_injector", {})["route_override"] = self.late_lane_route_override_status
        post_control_state.setdefault("fault_injector", {})[
            "inappropriate_response"
        ] = self.inappropriate_response_status
        applied_control, tick_record = self.fault_injector.apply(expert_control, post_control_state)
        self._append_fault_record(tick_record)
        self._update_fault_summary(tick_record)
        self._write_fault_trace(tick_record)
        self._write_fault_summary(force=False)
        return applied_control, driving_data

    def ego_agent_affected_by_red_light(
        self,
        ego_vehicle_location: Any,
        ego_vehicle_speed: float,
        distance_to_traffic_light: float,
        next_traffic_light: Any,
        route_points: Any,
        target_speed: float,
    ) -> float:
        native_target_speed = super().ego_agent_affected_by_red_light(
            ego_vehicle_location,
            ego_vehicle_speed,
            distance_to_traffic_light,
            next_traffic_light,
            route_points,
            target_speed,
        )
        if not self._fault_should_ignore_red_light():
            self.fault_ignore_red_light_status = {
                "active": False,
                "native_target_speed": native_target_speed,
                "returned_target_speed": native_target_speed,
            }
            return native_target_speed

        self.fault_ignore_red_light_status = {
            "active": True,
            "native_target_speed": native_target_speed,
            "returned_target_speed": target_speed,
            "distance_to_traffic_light_m": distance_to_traffic_light,
            "traffic_light_id": getattr(next_traffic_light, "id", None),
        }
        return target_speed

    def ego_agent_affected_by_stop_sign(
        self,
        ego_vehicle_location: Any,
        ego_vehicle_speed: float,
        next_stop_sign: Any,
        target_speed: float,
        actor_list: Any,
    ) -> float:
        native_target_speed = super().ego_agent_affected_by_stop_sign(
            ego_vehicle_location,
            ego_vehicle_speed,
            next_stop_sign,
            target_speed,
            actor_list,
        )
        if not self._inappropriate_behavior_enabled("ignore_stop_sign"):
            return native_target_speed

        self.inappropriate_response_status = _deep_merge_dicts(
            self.inappropriate_response_status,
            {
                "active": True,
                "ignore_stop_sign": {
                    "active": True,
                    "native_target_speed": native_target_speed,
                    "returned_target_speed": target_speed,
                    "stop_sign_id": getattr(next_stop_sign, "id", None),
                },
            },
        )
        return target_speed

    def compute_target_speed_wrt_leading_vehicle(
        self,
        initial_target_speed: float,
        predicted_bounding_boxes: Any,
        near_lane_change: bool,
        ego_location: Any,
        rear_vehicle_ids: Any,
        leading_vehicle_ids: Any,
        speed_reduced_by_obj: Any,
        plant: bool,
    ) -> tuple[float, Any]:
        target_speed, reduced_by = super().compute_target_speed_wrt_leading_vehicle(
            initial_target_speed,
            predicted_bounding_boxes,
            near_lane_change,
            ego_location,
            rear_vehicle_ids,
            leading_vehicle_ids,
            speed_reduced_by_obj,
            plant,
        )
        if not self._inappropriate_behavior_enabled("ignore_leading_vehicle"):
            return target_speed, reduced_by

        self.inappropriate_response_status = _deep_merge_dicts(
            self.inappropriate_response_status,
            {
                "active": True,
                "ignore_leading_vehicle": {
                    "active": True,
                    "native_target_speed": target_speed,
                    "returned_target_speed": initial_target_speed,
                },
            },
        )
        return initial_target_speed, speed_reduced_by_obj

    def compute_target_speeds_wrt_all_actors(
        self,
        initial_target_speed: float,
        ego_bounding_boxes: Any,
        predicted_bounding_boxes: Any,
        near_lane_change: bool,
        leading_vehicle_ids: Any,
        rear_vehicle_ids: Any,
        speed_reduced_by_obj: Any,
        nearby_walkers: Any,
        nearby_walkers_ids: Any,
    ) -> tuple[float, float, float, Any]:
        target_speed_bicycle, target_speed_pedestrian, target_speed_vehicle, reduced_by = (
            super().compute_target_speeds_wrt_all_actors(
                initial_target_speed,
                ego_bounding_boxes,
                predicted_bounding_boxes,
                near_lane_change,
                leading_vehicle_ids,
                rear_vehicle_ids,
                speed_reduced_by_obj,
                nearby_walkers,
                nearby_walkers_ids,
            )
        )

        ignored: dict[str, Any] = {}
        if self._inappropriate_behavior_enabled("ignore_vehicle_hazard"):
            ignored["vehicle"] = {
                "native_target_speed": target_speed_vehicle,
                "returned_target_speed": initial_target_speed,
            }
            target_speed_vehicle = initial_target_speed
            self.vehicle_hazard = False
            self.vehicle_affecting_id = None
        if self._inappropriate_behavior_enabled("ignore_bicycle_hazard"):
            ignored["bicycle"] = {
                "native_target_speed": target_speed_bicycle,
                "returned_target_speed": initial_target_speed,
            }
            target_speed_bicycle = initial_target_speed
        if self._inappropriate_behavior_enabled("ignore_walker_hazard"):
            ignored["walker"] = {
                "native_target_speed": target_speed_pedestrian,
                "returned_target_speed": initial_target_speed,
            }
            target_speed_pedestrian = initial_target_speed
            self.walker_hazard = False
            self.walker_close = False
            self.walker_affecting_id = None
            self.walker_close_id = None

        if ignored:
            self.inappropriate_response_status = _deep_merge_dicts(
                self.inappropriate_response_status,
                {
                    "active": True,
                    "ignored_actor_hazards": ignored,
                },
            )
            return target_speed_bicycle, target_speed_pedestrian, target_speed_vehicle, speed_reduced_by_obj

        return target_speed_bicycle, target_speed_pedestrian, target_speed_vehicle, reduced_by

    def _manage_route_obstacle_scenarios(
        self,
        target_speed: float,
        ego_speed: float,
        route_waypoints: Any,
        list_vehicles: Any,
        route_points: Any,
    ) -> tuple[float, bool, Any]:
        if self._inappropriate_behavior_enabled("suppress_route_obstacle_handling"):
            self.route_obstacle_debug = {
                "active": True,
                "suppressed_by_inappropriate_response": True,
                "selected_response": self.inappropriate_response_control.get("selected_response"),
                "behavior": self.inappropriate_response_control.get("behavior"),
            }
            self.inappropriate_response_status = _deep_merge_dicts(
                self.inappropriate_response_status,
                {
                    "active": True,
                    "route_obstacle": self.route_obstacle_debug,
                },
            )
            return target_speed, False, [target_speed, None, None, None]

        return super()._manage_route_obstacle_scenarios(
            target_speed,
            ego_speed,
            route_waypoints,
            list_vehicles,
            route_points,
        )

    def _fault_should_ignore_red_light(self) -> bool:
        if self._inappropriate_behavior_enabled("ignore_red_light"):
            return True
        if getattr(self, "fault_config", None) is None or getattr(self, "fault_injector", None) is None:
            return False
        if self.fault_config.fault_type != "red_light_entry":
            return False
        if not bool(self.fault_config.intervention.get("ignore_red_light", False)):
            return False
        return self.fault_injector.state == InjectorState.ACTIVE

    def _apply_late_lane_route_override(self, driving_state: dict[str, Any]) -> dict[str, Any]:
        override = getattr(self, "late_lane_route_override", None)
        if override is None:
            return {"active": False, "route_modified": False}

        if self.fault_injector.state == InjectorState.ACTIVE:
            return override.apply(self, driving_state)

        planner = getattr(self, "_waypoint_planner", None)
        if planner is not None:
            override.restore(planner)
        return {
            "active": False,
            "route_modified": False,
            "reason": f"injector_state_{self.fault_injector.state.value}",
        }

    def _refresh_fault_output_dir(self) -> None:
        if getattr(self, "save_path", None) is None:
            self.fault_output_dir = None
            self.fault_trace_path = None
            self.fault_summary_path = None
            self.clean_reference_path = None
            self.run_metadata_path = None
            return

        output_dir = Path(self.save_path) / "fault_injection"
        output_dir.mkdir(parents=True, exist_ok=True)
        self.fault_output_dir = output_dir
        self.fault_trace_path = output_dir / "fault_trace.jsonl"
        self.fault_summary_path = output_dir / "fault_summary.json"
        self.clean_reference_path = output_dir / "clean_reference.json"
        self.run_metadata_path = output_dir / "run_metadata.json"

    def _write_resolved_config(self) -> None:
        if self.fault_output_dir is None:
            return
        config_path = self.fault_output_dir / "resolved_config.json"
        with config_path.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "config_id": self.fault_config.config_id,
                    "fault_type": self.fault_config.fault_type,
                    "config_path": os.environ.get("FAULT_CONFIG"),
                    "run_id": self.fault_run_id,
                    "scenario_pair_id": self.scenario_pair_id,
                    "variant_id": self.variant_id,
                    "route_file": os.environ.get("ROUTES"),
                    "route_index": self._current_route_id(),
                    "town": self._current_route_town(),
                    "route_metadata": self.fault_route_metadata,
                    "route_folder_name": self.fault_route_folder_name,
                    "weather_id": os.environ.get("WEATHER_ID"),
                    "traffic_manager_seed": os.environ.get("TM_SEED"),
                    "scenario_seed": os.environ.get("SCENARIO_SEED"),
                    "repetition": os.environ.get("REPETITION"),
                    "actor_spawn_mode": self.actor_spawn_mode,
                    "actor_clear_radius_m": self.actor_clear_radius_m,
                    "actor_clear_vehicles": self.actor_clear_vehicles,
                    "actor_clear_walkers": self.actor_clear_walkers,
                    "actor_clear_repeat": self.actor_clear_repeat,
                    "inappropriate_response_control": self.inappropriate_response_control,
                    "clean_scenario_control": self.clean_scenario_control_applied,
                },
                handle,
                indent=2,
            )

    def _write_run_metadata(self) -> None:
        if self.run_metadata_path is None:
            return
        metadata = {
            "schema_version": "1.0",
            "scenario_pair_id": self.scenario_pair_id,
            "variant_id": self.variant_id,
            "condition": self.fault_config.condition,
            "fault_config_id": self.fault_config.config_id,
            "fault_type": self.fault_config.fault_type,
            "route_file": os.environ.get("ROUTES"),
            "route_index": self._current_route_id(),
            "town": self._current_route_town(),
            "route_metadata": self.fault_route_metadata,
            "route_folder_name": self.fault_route_folder_name,
            "weather_id": os.environ.get("WEATHER_ID"),
            "traffic_manager_seed": os.environ.get("TM_SEED"),
            "scenario_seed": os.environ.get("SCENARIO_SEED"),
            "repetition": os.environ.get("REPETITION"),
            "carla_version": "0.9.15",
            "native_output_path": str(getattr(self, "save_path", "")),
            "checkpoint_path": os.environ.get("CHECKPOINT_ENDPOINT"),
            "fault_summary_path": str(self.fault_summary_path) if self.fault_summary_path is not None else None,
            "fault_trace_path": str(self.fault_trace_path) if self.fault_trace_path is not None else None,
            "actor_spawn_control": self._actor_spawn_control_summary(),
            "inappropriate_response_control": self.inappropriate_response_control,
            "clean_scenario_control": self.clean_scenario_control_applied,
        }
        with self.run_metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)

    def _append_fault_record(self, record: dict[str, Any]) -> None:
        self.fault_records.append(record)
        self._refresh_fault_output_dir()
        if self.fault_config.fault_type == "none":
            self._write_clean_reference()

    def _write_fault_trace(self, record: dict[str, Any]) -> None:
        if self.fault_trace_path is None:
            return
        with self.fault_trace_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()

    def _write_fault_summary(self, force: bool = False) -> None:
        if self.fault_summary_path is None:
            return
        if not force and not self.fault_records:
            return
        self.fault_summary["final_state"] = self.fault_injector.state.value
        self.fault_summary["fault_triggered"] = bool(self.fault_injector.fault_triggered)
        with self.fault_summary_path.open("w", encoding="utf-8") as handle:
            json.dump(self.fault_summary, handle, indent=2)

    def _update_fault_summary(self, tick_record: dict[str, Any]) -> None:
        state_after = tick_record.get("injector_state")
        if state_after == "armed" and self.fault_summary["armed_step"] is None:
            self.fault_summary["armed_step"] = tick_record.get("step")
        if state_after == "active" and self.fault_summary["activated_step"] is None:
            self.fault_summary["activated_step"] = tick_record.get("step")
        if state_after == "releasing" and self.fault_summary["release_started_step"] is None:
            self.fault_summary["release_started_step"] = tick_record.get("step")
        if state_after == "completed" and self.fault_summary["completed_step"] is None:
            self.fault_summary["completed_step"] = tick_record.get("step")
        if state_after == "aborted" and self.fault_summary["aborted_step"] is None:
            self.fault_summary["aborted_step"] = tick_record.get("step")
            self.fault_summary["abort_reason"] = tick_record.get("abort_reason")
        self.fault_summary["fault_triggered"] = bool(tick_record.get("fault_triggered", False))
        if tick_record.get("fault_triggered") and tick_record.get("applied_control") != tick_record.get("expert_control"):
            self.fault_summary["expert_applied_control_differed"] = True
        ignore_red_light = tick_record.get("ignore_red_light") or {}
        if ignore_red_light.get("active"):
            self.fault_summary["expert_applied_control_differed"] = True
        lateral_intervention = tick_record.get("lateral_intervention") or {}
        if lateral_intervention.get("route_modified"):
            self.fault_summary["route_modified"] = True
        inappropriate_response = ((tick_record.get("driving_state") or {}).get("fault_injector") or {}).get(
            "inappropriate_response"
        ) or {}
        if inappropriate_response.get("active"):
            self.fault_summary["fault_triggered"] = True
            self.fault_summary["expert_applied_control_differed"] = True
            self.fault_summary["inappropriate_response_status"] = inappropriate_response
        self.fault_summary["actor_spawn_control"] = self._actor_spawn_control_summary()

    def _resolve_inappropriate_response_control(self) -> dict[str, Any]:
        if getattr(self, "fault_config", None) is None or self.fault_config.fault_type != "inappropriate_response":
            return {"enabled": False, "reason": "not_inappropriate_response_fault_config"}

        payload = getattr(self, "fault_config_payload", {}) or {}
        defaults = (
            payload.get("inappropriate_response_defaults")
            if isinstance(payload.get("inappropriate_response_defaults"), dict)
            else {}
        )
        route_id = str((getattr(self, "fault_route_metadata", {}) or {}).get("route_id") or "")
        scenario_types = [
            str(value)
            for value in (getattr(self, "fault_route_metadata", {}) or {}).get("scenario_types", [])
            if value
        ]
        scenario_names = [
            str(value)
            for value in (getattr(self, "fault_route_metadata", {}) or {}).get("scenario_names", [])
            if value
        ]

        resolved: dict[str, Any] = {}
        for scenario_type in scenario_types:
            type_control = defaults.get(scenario_type)
            if isinstance(type_control, dict):
                resolved = _deep_merge_dicts(resolved, type_control)

        route_item: dict[str, Any] | None = None
        for item in payload.get("inappropriate_response_scenarios", []) or []:
            if not isinstance(item, dict):
                continue
            item_route_id = str(item.get("route_id") or "")
            item_type = str(item.get("scenario_type") or "")
            item_name = str(item.get("scenario_name") or "")
            route_matches = route_id and item_route_id == route_id
            type_name_matches = item_type in scenario_types and (not scenario_names or item_name in scenario_names)
            if not route_matches and not type_name_matches:
                continue
            route_item = item
            item_control = item.get("inappropriate_response_control")
            if isinstance(item_control, dict):
                resolved = _deep_merge_dicts(resolved, item_control)

        options = resolved.get("response_options")
        if not isinstance(options, list) or not options:
            if isinstance(resolved.get("response_text"), str) and resolved.get("response_text"):
                options = [{"response": resolved["response_text"], "behavior": resolved.get("behavior", {})}]
            else:
                return {
                    "enabled": False,
                    "reason": "no_inappropriate_response_for_scenario",
                    "route_id": route_id,
                    "scenario_types": scenario_types,
                }

        seed = _stable_response_seed(
            payload.get("random_seed"),
            os.environ.get("FAULT_RESPONSE_SEED"),
            os.environ.get("SCENARIO_SEED"),
            os.environ.get("TM_SEED"),
            os.environ.get("FAULT_RUN_ID"),
            route_id,
            ",".join(scenario_types),
        )
        rng = random.Random(seed)
        selected_index = rng.randrange(len(options))
        selected_option = options[selected_index] if isinstance(options[selected_index], dict) else {}
        selected_behavior = selected_option.get("behavior") if isinstance(selected_option.get("behavior"), dict) else {}
        if isinstance(resolved.get("behavior"), dict):
            selected_behavior = _deep_merge_dicts(resolved["behavior"], selected_behavior)

        return {
            "enabled": bool(resolved.get("enabled", True)),
            "reason": "resolved",
            "route_id": route_id,
            "scenario_types": scenario_types,
            "scenario_names": scenario_names,
            "category": (route_item or {}).get("category") or resolved.get("category"),
            "selection_policy": resolved.get("selection_policy", "random_per_route"),
            "random_seed": seed,
            "selected_option_index": selected_index,
            "selected_response": selected_option.get("response") or resolved.get("response_text"),
            "selected_response_id": selected_option.get("id"),
            "behavior": selected_behavior,
            "source_inappropriate_response": resolved.get("source_inappropriate_response"),
        }

    def _inappropriate_response_active(self) -> bool:
        if getattr(self, "fault_config", None) is None or getattr(self, "fault_injector", None) is None:
            return False
        if self.fault_config.fault_type != "inappropriate_response":
            return False
        if not bool((getattr(self, "inappropriate_response_control", {}) or {}).get("enabled")):
            return False
        return self.fault_injector.state == InjectorState.ACTIVE

    def _inappropriate_behavior_enabled(self, name: str) -> bool:
        if not self._inappropriate_response_active():
            return False
        behavior = (getattr(self, "inappropriate_response_control", {}) or {}).get("behavior")
        return bool(behavior.get(name, False)) if isinstance(behavior, dict) else False

    def _resolve_clean_scenario_control(self) -> dict[str, Any]:
        if getattr(self, "fault_config", None) is None or self.fault_config.fault_type != "none":
            return {"enabled": False, "reason": "not_clean_fault_config"}

        payload = getattr(self, "fault_config_payload", {}) or {}
        defaults = payload.get("clean_control_defaults") if isinstance(payload.get("clean_control_defaults"), dict) else {}
        route_id = str((getattr(self, "fault_route_metadata", {}) or {}).get("route_id") or "")
        scenario_types = [
            str(value)
            for value in (getattr(self, "fault_route_metadata", {}) or {}).get("scenario_types", [])
            if value
        ]
        scenario_names = [
            str(value)
            for value in (getattr(self, "fault_route_metadata", {}) or {}).get("scenario_names", [])
            if value
        ]

        resolved: dict[str, Any] = {}
        for scenario_type in scenario_types:
            type_control = defaults.get(scenario_type)
            if isinstance(type_control, dict):
                resolved = _deep_merge_dicts(resolved, type_control)

        for item in payload.get("clean_scenarios", []) or []:
            if not isinstance(item, dict):
                continue
            item_route_id = str(item.get("route_id") or "")
            item_type = str(item.get("scenario_type") or "")
            item_name = str(item.get("scenario_name") or "")
            route_matches = route_id and item_route_id == route_id
            type_name_matches = item_type in scenario_types and (not scenario_names or item_name in scenario_names)
            if not route_matches and not type_name_matches:
                continue
            item_control = item.get("clean_control")
            if isinstance(item_control, dict):
                resolved = _deep_merge_dicts(resolved, item_control)

        if not resolved:
            return {
                "enabled": False,
                "reason": "no_clean_control_for_scenario",
                "route_id": route_id,
                "scenario_types": scenario_types,
            }

        resolved["route_id"] = route_id
        resolved["scenario_types"] = scenario_types
        resolved["scenario_names"] = scenario_names
        resolved["enabled"] = bool(resolved.get("enabled", True))
        return resolved

    def _apply_clean_scenario_control(self) -> dict[str, Any]:
        control = getattr(self, "clean_scenario_control", {}) or {}
        if not control.get("enabled"):
            return {
                "enabled": False,
                "reason": control.get("reason", "disabled"),
                "route_id": control.get("route_id"),
                "scenario_types": control.get("scenario_types", []),
            }

        config = getattr(self, "config", None)
        if config is None:
            return {
                "enabled": False,
                "reason": "native_config_missing",
                "route_id": control.get("route_id"),
                "scenario_types": control.get("scenario_types", []),
            }

        overrides = control.get("pdm_config_overrides")
        if not isinstance(overrides, dict):
            overrides = {}

        applied: dict[str, Any] = {}
        ignored: dict[str, Any] = {}
        points_per_meter = float(getattr(config, "points_per_meter", 1.0) or 1.0)
        for name, value in overrides.items():
            mapping = _CLEAN_PDM_CONFIG_OVERRIDE_MAP.get(name)
            if mapping is None:
                ignored[name] = value
                continue
            attr_name, converter_name = mapping
            if not hasattr(config, attr_name):
                ignored[name] = value
                continue
            old_value = getattr(config, attr_name)
            new_value = _convert_clean_control_value(value, converter_name, points_per_meter)
            setattr(config, attr_name, new_value)
            applied[name] = {
                "attribute": attr_name,
                "old_value": old_value,
                "new_value": new_value,
            }

        return {
            "enabled": True,
            "profile": control.get("profile"),
            "route_id": control.get("route_id"),
            "scenario_types": control.get("scenario_types", []),
            "scenario_names": control.get("scenario_names", []),
            "applied_overrides": applied,
            "ignored_overrides": ignored,
        }

    def _write_clean_reference(self) -> None:
        if self.clean_reference_path is None:
            return
        samples = []
        for record in self.fault_records:
            driving_state = record.get("driving_state") or {}
            ego = driving_state.get("ego") or {}
            route = driving_state.get("route") or {}
            samples.append(
                {
                    "timestamp_s": record.get("timestamp_s"),
                    "distance_to_junction_m": route.get("distance_to_junction_m"),
                    "road_id": ego.get("road_id"),
                    "lane_id": ego.get("lane_id"),
                    "preferred_approach_lane_id": route.get("preferred_approach_lane_id"),
                    "lateral_offset_m": ego.get("lateral_offset_from_lane_center_m"),
                    "is_preferred_lane": route.get("current_lane_is_preferred"),
                }
            )
        reference = {
            "schema_version": "1.0",
            "scenario_pair_id": self.scenario_pair_id,
            "route_id": self._current_route_id(),
            "town": self._current_route_town(),
            "samples": samples,
        }
        with self.clean_reference_path.open("w", encoding="utf-8") as handle:
            json.dump(reference, handle, indent=2)

    def _current_route_town(self) -> str | None:
        metadata_town = (getattr(self, "fault_route_metadata", None) or {}).get("town")
        return metadata_town or os.environ.get("TOWN")

    def _current_route_id(self) -> str | None:
        metadata_route_id = (getattr(self, "fault_route_metadata", None) or {}).get("route_id")
        return metadata_route_id or os.environ.get("ROUTE_INDEX") or os.environ.get("ROUTES_SUBSET")

    def _apply_actor_spawn_control(self) -> None:
        if self.actor_spawn_mode in {"", "default", "off", "false", "0"}:
            return
        if self.actor_spawn_control_applied and not self.actor_clear_repeat:
            return

        ego_vehicle = getattr(self, "_vehicle", None)
        if ego_vehicle is None:
            return

        world = getattr(self, "_world", None)
        if world is None:
            get_world = getattr(ego_vehicle, "get_world", None)
            world = get_world() if callable(get_world) else None
        if world is None:
            return

        ego_location = _actor_location(ego_vehicle)
        actors = world.get_actors()
        candidate_actors = []
        if self.actor_clear_vehicles:
            candidate_actors.extend(list(actors.filter("vehicle.*")))
        if self.actor_clear_walkers:
            candidate_actors.extend(list(actors.filter("walker.*")))

        for actor in candidate_actors:
            if _is_ego_actor(actor, ego_vehicle):
                continue
            if int(getattr(actor, "id", -1)) in self.actor_spawn_control_removed:
                continue

            actor_location = _actor_location(actor)
            distance_m = _distance(actor_location, ego_location)
            if not self._should_remove_actor(distance_m, actor_location, world):
                continue

            removed = False
            try:
                removed = bool(actor.destroy())
            except Exception as exc:  # pragma: no cover - depends on CARLA runtime
                self.actor_spawn_control_removed[int(getattr(actor, "id", -1))] = {
                    "actor_id": int(getattr(actor, "id", -1)),
                    "type_id": getattr(actor, "type_id", None),
                    "role_name": getattr(actor, "attributes", {}).get("role_name"),
                    "distance_to_ego_m": distance_m,
                    "removed": False,
                    "error": str(exc),
                }
                continue

            self.actor_spawn_control_removed[int(getattr(actor, "id", -1))] = {
                "actor_id": int(getattr(actor, "id", -1)),
                "type_id": getattr(actor, "type_id", None),
                "role_name": getattr(actor, "attributes", {}).get("role_name"),
                "distance_to_ego_m": distance_m,
                "removed": removed,
            }

        self.actor_spawn_control_applied = True
        self.fault_summary["actor_spawn_control"] = self._actor_spawn_control_summary()

    def _should_remove_actor(self, distance_m: float | None, actor_location: Any = None, world: Any = None) -> bool:
        if self.actor_spawn_mode in {"none", "all", "no_background", "no_background_actors"}:
            return True
        if self.actor_spawn_mode in {"near_ego", "radius", "clear_ego", "clear_ego_radius"}:
            return distance_m is not None and distance_m <= self.actor_clear_radius_m
        if self.actor_spawn_mode == "parking_lanes_near_ego":
            return (
                distance_m is not None
                and distance_m <= self.actor_clear_radius_m
                and _location_is_on_parking_lane(actor_location, world)
            )
        return False

    def _actor_spawn_control_summary(self) -> dict[str, Any]:
        records = list(self.actor_spawn_control_removed.values())
        removed_records = [record for record in records if record.get("removed")]
        return {
            "mode": self.actor_spawn_mode,
            "clear_radius_m": self.actor_clear_radius_m,
            "clear_vehicles": self.actor_clear_vehicles,
            "clear_walkers": self.actor_clear_walkers,
            "repeat": self.actor_clear_repeat,
            "applied": self.actor_spawn_control_applied,
            "removed_count": len(removed_records),
            "removed_actor_ids": [record.get("actor_id") for record in removed_records[:50]],
            "failed_count": len(records) - len(removed_records),
        }


def _config_bool(config: dict[str, Any], name: str, default: bool) -> bool:
    value = config.get(name)
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in {"", "0", "false", "no", "off"}


def _config_float(config: dict[str, Any], name: str, default: float) -> float:
    value = config.get(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _stable_response_seed(*parts: Any) -> int:
    text = "|".join(str(part) for part in parts if part not in {None, ""})
    if not text:
        text = "inappropriate_response"
    value = 2166136261
    for char in text:
        value ^= ord(char)
        value = (value * 16777619) & 0xFFFFFFFF
    return value


def _is_ego_actor(actor: Any, ego_vehicle: Any) -> bool:
    if getattr(actor, "id", None) == getattr(ego_vehicle, "id", None):
        return True
    role_name = str(getattr(actor, "attributes", {}).get("role_name", "")).lower()
    return role_name in {"hero", "ego", "ego_vehicle"}


def _location_is_on_parking_lane(location: Any, world: Any) -> bool:
    if location is None or world is None:
        return False

    get_map = getattr(world, "get_map", None)
    carla_map = get_map() if callable(get_map) else None
    if carla_map is None:
        return False

    waypoint = None
    get_waypoint = getattr(carla_map, "get_waypoint", None)
    if not callable(get_waypoint):
        return False

    try:
        import carla  # type: ignore

        waypoint = get_waypoint(location, project_to_road=True, lane_type=carla.LaneType.Any)
    except Exception:
        try:
            waypoint = get_waypoint(location, project_to_road=True)
        except Exception:
            return False

    lane_type_name = str(getattr(waypoint, "lane_type", "")).lower()
    return "parking" in lane_type_name


def _actor_location(actor: Any) -> Any | None:
    get_location = getattr(actor, "get_location", None)
    if callable(get_location):
        try:
            return get_location()
        except Exception:
            pass
    get_transform = getattr(actor, "get_transform", None)
    if callable(get_transform):
        try:
            transform = get_transform()
            return getattr(transform, "location", None)
        except Exception:
            pass
    return None


def _distance(a: Any | None, b: Any | None) -> float | None:
    if a is None or b is None:
        return None
    distance = getattr(a, "distance", None)
    if callable(distance):
        try:
            return float(distance(b))
        except Exception:
            pass
    try:
        dx = float(getattr(a, "x")) - float(getattr(b, "x"))
        dy = float(getattr(a, "y")) - float(getattr(b, "y"))
        dz = float(getattr(a, "z", 0.0)) - float(getattr(b, "z", 0.0))
    except Exception:
        return None
    return (dx * dx + dy * dy + dz * dz) ** 0.5


def _resolve_route_metadata(
    routes_path: str | None,
    route_index: int | str | None,
    routes_subset: str,
    repetitions: int,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "route_date_string": str(route_index) if route_index is not None else None,
        "evaluator_index": _extract_evaluator_index(route_index),
        "route_id": None,
        "town": None,
        "scenario_names": [],
        "scenario_types": [],
    }
    if not routes_path:
        return metadata
    route_file = Path(routes_path)
    if not route_file.exists():
        return metadata

    try:
        tree = ET.parse(route_file)
        route_elements = list(tree.iter("route"))
    except Exception:
        return metadata

    selected_routes = _select_route_elements(route_elements, routes_subset)
    if not selected_routes:
        return metadata

    evaluator_index = metadata["evaluator_index"]
    if evaluator_index is None:
        route_id_from_env = os.environ.get("ROUTE_INDEX") or routes_subset
        route_element = next((route for route in selected_routes if route.attrib.get("id") == route_id_from_env), None)
    else:
        repetition_count = max(1, int(repetitions or 1))
        route_ordinal = int(evaluator_index) // repetition_count
        metadata["repetition_index"] = int(evaluator_index) % repetition_count
        route_element = selected_routes[route_ordinal] if route_ordinal < len(selected_routes) else None

    if route_element is None:
        return metadata

    scenarios = list((route_element.find("scenarios") or []))
    metadata.update(
        {
            "route_file": str(route_file),
            "route_file_stem": route_file.stem,
            "route_id": route_element.attrib.get("id"),
            "town": route_element.attrib.get("town"),
            "scenario_names": [scenario.attrib.get("name") for scenario in scenarios if scenario.attrib.get("name")],
            "scenario_types": [scenario.attrib.get("type") for scenario in scenarios if scenario.attrib.get("type")],
        }
    )
    return metadata


def _select_route_elements(route_elements: list[Any], routes_subset: str) -> list[Any]:
    subset = (routes_subset or "").replace(" ", "")
    if not subset:
        return route_elements

    by_id = {route.attrib.get("id"): route for route in route_elements}
    selected_ids: list[str] = []
    for group in subset.split(","):
        if not group:
            continue
        if "-" not in group:
            if group in by_id and group not in selected_ids:
                selected_ids.append(group)
            continue

        start, end = group.split("-", 1)
        found_start = False
        for route in route_elements:
            route_id = route.attrib.get("id")
            if route_id == start:
                found_start = True
            if found_start and route_id not in selected_ids:
                selected_ids.append(route_id)
            if found_start and route_id == end:
                break

    selected_ids.sort(key=lambda value: int(value) if str(value).isdigit() else str(value))
    return [by_id[route_id] for route_id in selected_ids if route_id in by_id]


def _extract_evaluator_index(route_index: int | str | None) -> int | None:
    if route_index is None:
        return None
    if isinstance(route_index, int):
        return route_index
    match = re.search(r"_route(\d+)(?:_|$)", str(route_index))
    if match:
        return int(match.group(1))
    if str(route_index).isdigit():
        return int(str(route_index))
    return None


def _format_route_folder_name(metadata: dict[str, Any], fallback_route_index: int | str | None) -> str | None:
    if not metadata:
        return str(fallback_route_index) if fallback_route_index is not None else None

    town = _safe_folder_part(metadata.get("town") or os.environ.get("TOWN") or "UnknownTown")
    route_id = _safe_folder_part(metadata.get("route_id") or f"idx{metadata.get('evaluator_index', 'unknown')}")
    scenario_name = _scenario_folder_part(metadata)
    repetition = metadata.get("repetition_index")
    repetition_part = f"rep{repetition}" if repetition is not None else None
    timestamp = _timestamp_suffix(metadata.get("route_date_string"))

    parts = [town, scenario_name, f"route{route_id}"]
    if repetition_part is not None:
        parts.append(repetition_part)
    if timestamp is not None:
        parts.append(timestamp)
    return "_".join(part for part in parts if part)


def _scenario_folder_part(metadata: dict[str, Any]) -> str:
    names = [name for name in metadata.get("scenario_names", []) if name]
    types = [scenario_type for scenario_type in metadata.get("scenario_types", []) if scenario_type]
    if names:
        selected = names[:2]
        label = "_".join(selected)
        if len(names) > len(selected):
            label += f"_plus{len(names) - len(selected)}"
        return _safe_folder_part(label)
    if types:
        selected = types[:2]
        label = "_".join(selected)
        if len(types) > len(selected):
            label += f"_plus{len(types) - len(selected)}"
        return _safe_folder_part(label)
    return "NoScenario"


def _timestamp_suffix(route_date_string: Any) -> str | None:
    if route_date_string is None:
        return None
    match = re.search(r"(\d{2}_\d{2}_\d{2}_\d{2}_\d{2})$", str(route_date_string))
    return match.group(1) if match else None


def _safe_folder_part(value: Any) -> str:
    text = str(value or "").strip()
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "unknown"


_CLEAN_PDM_CONFIG_OVERRIDE_MAP: dict[str, tuple[str, str]] = {
    "default_max_distance_to_process_scenario_m": ("default_max_distance_to_process_scenario", "float"),
    "max_distance_to_overtake_two_way_scenarios_m": ("max_distance_to_overtake_two_way_scnearios", "meters_to_points"),
    "default_overtake_speed_kmh": ("default_overtake_speed", "kmh_to_mps"),
    "check_path_free_safety_distance_m": ("check_path_free_safety_distance", "float"),
    "check_path_free_safety_time_s": ("check_path_free_safety_time", "float"),
    "idm_two_way_scenarios_minimum_distance_m": ("idm_two_way_scenarios_minimum_distance", "float"),
    "idm_two_way_scenarios_time_headway_s": ("idm_two_way_scenarios_time_headway", "float"),
    "transition_length_accident_two_ways_m": ("transition_length_accident_two_ways", "meters_to_points"),
    "transition_length_construction_obstacle_two_ways_m": (
        "transition_length_construction_obstacle_two_ways",
        "meters_to_points",
    ),
    "add_before_accident_two_ways_m": ("add_before_accident_two_ways", "meters_to_points"),
    "add_after_accident_two_ways_m": ("add_after_accident_two_ways", "meters_to_points"),
    "add_before_construction_obstacle_two_ways_m": (
        "add_before_construction_obstacle_two_ways",
        "meters_to_points",
    ),
    "add_after_construction_obstacle_two_ways_m": (
        "add_after_construction_obstacle_two_ways",
        "meters_to_points",
    ),
    "factor_accident_two_ways": ("factor_accident_two_ways", "float"),
    "factor_construction_obstacle_two_ways": ("factor_construction_obstacle_two_ways", "float"),
    "force_accident_two_ways_emergency_bypass": ("force_accident_two_ways_emergency_bypass", "bool"),
    "accident_two_ways_emergency_bypass_min_oncoming_gap_m": (
        "accident_two_ways_emergency_bypass_min_oncoming_gap",
        "float",
    ),
    "accident_two_ways_emergency_bypass_extra_clearance_m": (
        "accident_two_ways_emergency_bypass_extra_clearance",
        "float",
    ),
    "idm_pedestrian_minimum_distance_m": ("idm_pedestrian_minimum_distance", "float"),
    "idm_pedestrian_desired_time_headway_s": ("idm_pedestrian_desired_time_headway", "float"),
    "force_vehicle_turning_route_pedestrian_wait_for_crossing": (
        "force_vehicle_turning_route_pedestrian_wait_for_crossing",
        "bool",
    ),
    "vehicle_turning_route_pedestrian_hold_distance_to_stop_sign_m": (
        "vehicle_turning_route_pedestrian_hold_distance_to_stop_sign",
        "float",
    ),
    "vehicle_turning_route_pedestrian_fallback_hold_distance_to_collision_m": (
        "vehicle_turning_route_pedestrian_fallback_hold_distance_to_collision",
        "float",
    ),
    "vehicle_turning_route_pedestrian_max_hold_distance_to_collision_m": (
        "vehicle_turning_route_pedestrian_max_hold_distance_to_collision",
        "float",
    ),
    "vehicle_turning_route_pedestrian_crossing_start_progress_m": (
        "vehicle_turning_route_pedestrian_crossing_start_progress_m",
        "float",
    ),
    "vehicle_turning_route_pedestrian_clearance_before_sidewalk_m": (
        "vehicle_turning_route_pedestrian_clearance_before_sidewalk_m",
        "float",
    ),
    "pedestrian_minimum_extent_m": ("pedestrian_minimum_extent", "float"),
    "default_forecast_length_s": ("default_forecast_length", "float"),
    "forecast_length_lane_change_s": ("forecast_length_lane_change", "float"),
    "min_walker_speed_mps": ("min_walker_speed", "float"),
    "max_blocked_ticks": ("max_blocked_ticks", "int"),
}


def _load_json_dict(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _deep_merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def _convert_clean_control_value(value: Any, converter_name: str, points_per_meter: float) -> Any:
    if converter_name == "int":
        return int(value)
    if converter_name == "bool":
        return bool(value)
    if converter_name == "kmh_to_mps":
        return float(value) / 3.6
    if converter_name == "meters_to_points":
        return int(round(float(value) * points_per_meter))
    return float(value)

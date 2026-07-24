from __future__ import annotations

from enum import Enum
from typing import Any

from .fault_config import FaultConfig


class InjectorState(str, Enum):
    DISABLED = "disabled"
    ARMED = "armed"
    ACTIVE = "active"
    RELEASING = "releasing"
    COMPLETED = "completed"
    ABORTED = "aborted"


class _VehicleControlFallback:
    def __init__(self) -> None:
        self.steer = 0.0
        self.throttle = 0.0
        self.brake = 0.0
        self.hand_brake = False
        self.reverse = False
        self.manual_gear_shift = False
        self.gear = 0


def copy_vehicle_control(control: Any) -> Any:
    try:
        import carla  # type: ignore
    except Exception:
        carla = None

    if carla is not None and hasattr(carla, "VehicleControl"):
        copied = carla.VehicleControl()
        copied.throttle = float(getattr(control, "throttle", 0.0))
        copied.steer = float(getattr(control, "steer", 0.0))
        copied.brake = float(getattr(control, "brake", 0.0))
        copied.hand_brake = bool(getattr(control, "hand_brake", False))
        copied.reverse = bool(getattr(control, "reverse", False))
        copied.manual_gear_shift = bool(getattr(control, "manual_gear_shift", False))
        copied.gear = int(getattr(control, "gear", 0))
        return copied

    copied = _VehicleControlFallback()
    copied.throttle = float(getattr(control, "throttle", 0.0))
    copied.steer = float(getattr(control, "steer", 0.0))
    copied.brake = float(getattr(control, "brake", 0.0))
    copied.hand_brake = bool(getattr(control, "hand_brake", False))
    copied.reverse = bool(getattr(control, "reverse", False))
    copied.manual_gear_shift = bool(getattr(control, "manual_gear_shift", False))
    copied.gear = int(getattr(control, "gear", 0))
    return copied


class FaultInjector:
    def __init__(self, config: FaultConfig, simulation_fps: float) -> None:
        self.config = config
        self.simulation_fps = float(simulation_fps or 20.0)
        if config.fault_type == "none":
            self.state = InjectorState.DISABLED
        elif config.fault_type == "inappropriate_response":
            self.state = InjectorState.ACTIVE
        else:
            self.state = InjectorState.ARMED
        self.confirmation_count = 0
        self.active_ticks = 0
        self.release_ticks = 0
        self.previous_injected_throttle = 0.0
        self.previous_injected_brake = 0.0
        self.abort_reason: str | None = None
        self.fault_triggered = config.fault_type == "inappropriate_response"
        self.fault_active = config.fault_type == "inappropriate_response"
        self.armed_step: int | None = None
        self.activated_step: int | None = None
        self.release_started_step: int | None = None
        self.completed_step: int | None = None
        self.aborted_step: int | None = None
        self._step = 0
        self._prepared_before_control_step: int | None = None
        self._prepared_before_control_result: dict[str, Any] | None = None

    def prepare_before_control(self, driving_state: dict[str, Any]) -> dict[str, Any]:
        step = int(driving_state.get("step") or self._step + 1)
        state_before = self.state.value
        if self.config.fault_type not in {"late_turn_lane_entry", "red_light_entry"} or self.state != InjectorState.ARMED:
            result = {
                "state_before": state_before,
                "state_after": self.state.value,
                "trigger_matches": False,
                "trigger_failure_reasons": [],
                "trigger_confirmation_count": self.confirmation_count,
                "activated_before_control": False,
                "aborted_before_control": False,
            }
            self._prepared_before_control_step = step
            self._prepared_before_control_result = result
            return result

        trigger_matches, trigger_failure_reasons = self._check_trigger(driving_state)
        abort_reason = self._abort_reason(driving_state) if self._should_abort(driving_state) else None
        if abort_reason in {"vehicle_hazard", "walker_hazard", "maximum_speed_exceeded"}:
            self.state = InjectorState.ABORTED
            self.aborted_step = step
            self.abort_reason = abort_reason
            self.fault_active = False
            self.fault_triggered = False
            result = {
                "state_before": state_before,
                "state_after": self.state.value,
                "trigger_matches": trigger_matches,
                "trigger_failure_reasons": trigger_failure_reasons,
                "trigger_confirmation_count": self.confirmation_count,
                "activated_before_control": False,
                "aborted_before_control": True,
                "abort_reason": abort_reason,
            }
            self._prepared_before_control_step = step
            self._prepared_before_control_result = result
            return result

        if trigger_matches:
            self.confirmation_count += 1
        else:
            self.confirmation_count = 0

        activated = self.confirmation_count >= int(self.config.trigger.get("confirmation_ticks", 1))
        if activated:
            self.state = InjectorState.ACTIVE
            self.activated_step = step
            self.armed_step = self.armed_step if self.armed_step is not None else step
            self.fault_triggered = True
            self.fault_active = True
            self.abort_reason = None

        result = {
            "state_before": state_before,
            "state_after": self.state.value,
            "trigger_matches": trigger_matches,
            "trigger_failure_reasons": trigger_failure_reasons,
            "trigger_confirmation_count": self.confirmation_count,
            "activated_before_control": activated,
            "aborted_before_control": False,
        }
        self._prepared_before_control_step = step
        self._prepared_before_control_result = result
        return result

    def apply(self, expert_control: Any, driving_state: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        self._step += 1
        step = int(driving_state.get("step") or self._step)
        timestamp_s = driving_state.get("timestamp_s")
        if timestamp_s is None:
            timestamp_s = step / self.simulation_fps

        expert_copy = copy_vehicle_control(expert_control)
        applied_control = copy_vehicle_control(expert_control)
        state_before = self.state.value

        if self.config.fault_type == "none":
            self.state = InjectorState.DISABLED
            self.confirmation_count = 0
            self.abort_reason = None
            self.fault_triggered = False
            self.fault_active = False
            record = self._build_record(
                step=step,
                timestamp_s=timestamp_s,
                state_before=state_before,
                state_after=self.state.value,
                trigger_matches=False,
                trigger_failure_reasons=[],
                trigger_confirmation_count=0,
                fault_active=False,
                fault_triggered=False,
                abort_reason=None,
                expert_control=expert_copy,
                applied_control=applied_control,
                driving_state=driving_state,
            )
            return applied_control, record

        prepared_before_control = (
            self._prepared_before_control_result if self._prepared_before_control_step == step else None
        )
        if prepared_before_control is not None:
            trigger_matches = bool(prepared_before_control.get("trigger_matches", False))
            trigger_failure_reasons = list(prepared_before_control.get("trigger_failure_reasons", []))
        else:
            trigger_matches, trigger_failure_reasons = self._check_trigger(driving_state)
        if self.state == InjectorState.ARMED:
            if prepared_before_control is not None:
                self.abort_reason = None
                self.fault_active = False
                self.fault_triggered = False
            elif self._should_abort(driving_state) and self._abort_reason(driving_state) in {"vehicle_hazard", "walker_hazard", "maximum_speed_exceeded"}:
                self.state = InjectorState.ABORTED
                self.aborted_step = step
                self.abort_reason = self._abort_reason(driving_state)
                self.fault_active = False
                self.fault_triggered = False
            else:
                if trigger_matches:
                    self.confirmation_count += 1
                else:
                    self.confirmation_count = 0

                if bool(self.config.trigger.get("trigger_on_first_red_light", False)) and self.confirmation_count == 1:
                    self.state = InjectorState.ACTIVE
                    self.activated_step = step
                    self.armed_step = self.armed_step if self.armed_step is not None else step
                    self.fault_triggered = True
                    self.fault_active = True
                    self.abort_reason = None
                elif self.confirmation_count >= int(self.config.trigger.get("confirmation_ticks", 1)):
                    self.state = InjectorState.ACTIVE
                    self.activated_step = step
                    self.armed_step = self.armed_step if self.armed_step is not None else step
                    self.fault_triggered = True
                    self.fault_active = True
                    self.abort_reason = None
                else:
                    self.abort_reason = None
                    self.fault_active = False
                    self.fault_triggered = False

        if self.state == InjectorState.ACTIVE:
            self.active_ticks += 1
            if self._should_abort(driving_state):
                self.state = InjectorState.ABORTED
                self.aborted_step = step
                self.abort_reason = self._abort_reason(driving_state)
                self.fault_active = False
                self.fault_triggered = True
                applied_control = copy_vehicle_control(expert_control)
            elif self._should_release(driving_state):
                self.state = InjectorState.RELEASING
                self.release_started_step = step
                self.release_ticks = 0
                self.fault_active = True
            else:
                applied_control = self._apply_intervention(expert_control, driving_state)
                self.previous_injected_throttle = float(getattr(applied_control, "throttle", 0.0))
                self.previous_injected_brake = float(getattr(applied_control, "brake", 0.0))

        if self.state == InjectorState.RELEASING:
            if self.config.fault_type == "late_turn_lane_entry" and self._should_abort(driving_state):
                self.state = InjectorState.ABORTED
                self.aborted_step = step
                self.abort_reason = self._abort_reason(driving_state)
                self.fault_active = False
                self.fault_triggered = True
                applied_control = copy_vehicle_control(expert_control)
            else:
                applied_control = self._apply_release(expert_control, driving_state)
                self.release_ticks += 1
                transition_ticks = max(1, int(self.config.release.get("transition_ticks", 1)))
                release_complete = self.release_ticks >= transition_ticks
                if self.config.fault_type == "late_turn_lane_entry" and self.config.release.get(
                    "require_correct_turn_lane_before_junction", False
                ):
                    release_complete = release_complete and bool((driving_state.get("route") or {}).get("current_lane_is_preferred"))
                if release_complete:
                    self.state = InjectorState.COMPLETED
                    self.completed_step = step
                    self.fault_active = False

        if self.state == InjectorState.ABORTED:
            applied_control = copy_vehicle_control(expert_control)
            self.fault_active = False

        if self.state == InjectorState.COMPLETED:
            self.fault_active = False
            self.fault_triggered = True

        record = self._build_record(
            step=step,
            timestamp_s=timestamp_s,
            state_before=state_before,
            state_after=self.state.value,
            trigger_matches=trigger_matches,
            trigger_failure_reasons=trigger_failure_reasons,
            trigger_confirmation_count=self.confirmation_count,
            fault_active=self.fault_active,
            fault_triggered=self.fault_triggered,
            abort_reason=self.abort_reason,
            expert_control=expert_copy,
            applied_control=applied_control,
            driving_state=driving_state,
        )
        return applied_control, record

    @staticmethod
    def _normalize_traffic_light_state(state: Any) -> str | None:
        if state is None:
            return None
        if isinstance(state, str):
            normalized = state.strip().lower()
            return {
                "red": "red",
                "green": "green",
                "yellow": "yellow",
                "off": "off",
                "unknown": "unknown",
            }.get(normalized, normalized)
        if isinstance(state, (int, float)):
            mapping = {0: "red", 1: "yellow", 2: "green", 3: "off", 4: "unknown"}
            int_state = int(state)
            return mapping.get(int_state, str(int_state).lower())
        return str(state).strip().lower()

    def _check_trigger(self, driving_state: dict[str, Any]) -> tuple[bool, list[str]]:
        if self.config.fault_type == "late_turn_lane_entry":
            return self._check_late_lane_trigger(driving_state)
        if self.config.fault_type != "red_light_entry":
            return False, []

        traffic_light = driving_state.get("traffic_light") or {}
        ego = driving_state.get("ego") or {}
        hazards = driving_state.get("hazards") or {}
        trigger = self.config.trigger
        reasons: list[str] = []

        expected_state = self._normalize_traffic_light_state(trigger.get("traffic_light_state"))
        actual_state = self._normalize_traffic_light_state(traffic_light.get("state"))

        if not traffic_light.get("present"):
            reasons.append("traffic_light.present is false")
        if actual_state is None:
            reasons.append("required_traffic_light_state_missing")
        elif actual_state != expected_state:
            reasons.append("traffic_light.state does not match")
        if bool(trigger.get("require_traffic_light_hazard", False)) and not bool(hazards.get("traffic_light")):
            reasons.append("traffic_light hazard is false")
        if not bool(trigger.get("ignore_traffic_light_affects_ego", False)) and bool(
            traffic_light.get("affects_ego")
        ) is not bool(trigger.get("traffic_light_affects_ego")):
            reasons.append("traffic_light.affects_ego mismatch")
        distance_m = traffic_light.get("distance_m")
        if distance_m is None:
            reasons.append("traffic_light.distance_m unavailable")
        else:
            max_distance = float(trigger.get("maximum_traffic_light_distance_m", float("inf")))
            if float(distance_m) > max_distance:
                reasons.append("traffic_light.distance_m exceeds threshold")
        if bool(trigger.get("require_not_in_junction", False)) and bool(ego.get("is_junction")):
            reasons.append("ego is in a junction")
        speed_mps = ego.get("speed_mps")
        max_speed = float(self.config.guards.get("maximum_speed_mps", float("inf")))
        if speed_mps is None:
            reasons.append("ego.speed_mps unavailable")
        elif speed_mps > max_speed:
            reasons.append("ego speed exceeds maximum")
        return not reasons, reasons

    @staticmethod
    def _normalize_route_command(command: Any) -> str | None:
        if command is None:
            return None
        value = getattr(command, "name", command)
        if isinstance(value, (int, float)):
            mapping = {
                1: "turn_left",
                2: "turn_right",
                3: "straight",
                4: "lane_follow",
                5: "change_lane_left",
                6: "change_lane_right",
            }
            return mapping.get(int(value), str(int(value)))
        normalized = str(value).strip().lower()
        normalized = normalized.replace("roadoption.", "").replace(" ", "_")
        mapping = {
            "left": "turn_left",
            "right": "turn_right",
            "straight": "straight",
            "lanefollow": "lane_follow",
            "lane_follow": "lane_follow",
            "changelaneleft": "change_lane_left",
            "change_lane_left": "change_lane_left",
            "changelaneright": "change_lane_right",
            "change_lane_right": "change_lane_right",
        }
        return mapping.get(normalized, normalized)

    @classmethod
    def _route_command_matches(cls, expected_command: Any, actual_command: Any) -> bool:
        expected_commands = expected_command if isinstance(expected_command, list) else [expected_command]
        actual = cls._normalize_route_command(actual_command)
        for expected in expected_commands:
            normalized_expected = cls._normalize_route_command(expected)
            if actual == normalized_expected:
                return True
            if normalized_expected == "turn_right" and actual == "change_lane_right":
                return True
            if normalized_expected == "turn_left" and actual == "change_lane_left":
                return True
        return False

    @staticmethod
    def _is_solid_marking(marking_type: Any) -> bool:
        if marking_type is None:
            return False
        return "solid" in str(marking_type).lower()

    def _check_late_lane_trigger(self, driving_state: dict[str, Any]) -> tuple[bool, list[str]]:
        ego = driving_state.get("ego") or {}
        route = driving_state.get("route") or {}
        lane_topology = driving_state.get("lane_topology") or {}
        trigger = self.config.trigger
        reasons: list[str] = []

        if not self._route_command_matches(trigger.get("route_command"), route.get("command")):
            reasons.append("route.command does not match")
        if bool(trigger.get("require_not_in_junction", False)) and bool(ego.get("is_junction")):
            reasons.append("ego is in a junction")

        distance_m = route.get("distance_to_junction_m")
        if distance_m is None:
            reasons.append("route.distance_to_junction_m unavailable")
        else:
            min_distance = float(trigger.get("minimum_junction_distance_m", 0.0))
            max_distance = float(trigger.get("maximum_junction_distance_m", float("inf")))
            if float(distance_m) < min_distance or float(distance_m) > max_distance:
                reasons.append("route.distance_to_junction_m outside approach interval")

        if trigger.get("require_right_lane_available", False) and lane_topology.get("right_lane_available") is not True:
            reasons.append("right driving lane unavailable")
        if lane_topology.get("right_lane_change_allowed") is not True:
            reasons.append("right lane change is not allowed")
        if bool(route.get("current_lane_is_preferred")):
            reasons.append("ego is already in preferred approach lane")
        if route.get("preferred_approach_lane_id") is None:
            reasons.append("preferred approach lane unavailable")
        if lane_topology.get("right_lane_id") is None:
            reasons.append("right lane id unavailable")

        ego_speed = ego.get("speed_mps")
        max_speed = float(self.config.intervention.get("maximum_speed_mps", float("inf")))
        if ego_speed is None:
            reasons.append("ego.speed_mps unavailable")
        elif float(ego_speed) > max_speed:
            reasons.append("ego speed exceeds maximum")
        return not reasons, reasons

    def _should_abort(self, driving_state: dict[str, Any]) -> bool:
        hazards = driving_state.get("hazards") or {}
        if bool(hazards.get("vehicle")) and bool(self.config.guards.get("abort_on_vehicle_hazard", False)):
            return True
        if bool(hazards.get("walker")) and bool(self.config.guards.get("abort_on_walker_hazard", False)):
            return True
        traffic_light = driving_state.get("traffic_light") or {}
        if self.config.guards.get("abort_if_light_not_red_before_activation", False):
            actual_state = self._normalize_traffic_light_state(traffic_light.get("state"))
            expected_state = self._normalize_traffic_light_state(self.config.trigger.get("traffic_light_state"))
            if actual_state != expected_state:
                return True
        ego = driving_state.get("ego") or {}
        maximum_speed = float(self.config.guards.get("maximum_speed_mps", float("inf")))
        if self.config.fault_type == "late_turn_lane_entry":
            maximum_speed = float(self.config.intervention.get("maximum_speed_mps", maximum_speed))
        if self.config.fault_type == "late_turn_lane_entry":
            maximum_speed = float(self.config.intervention.get("maximum_speed_mps", maximum_speed))
        speed_mps = ego.get("speed_mps")
        if speed_mps is not None and speed_mps > maximum_speed:
            return True
        if self.config.fault_type == "late_turn_lane_entry":
            route = driving_state.get("route") or {}
            lane_topology = driving_state.get("lane_topology") or {}
            if self.config.guards.get("abort_on_junction_entry_before_correction", False):
                if bool(ego.get("is_junction")) and not bool(route.get("current_lane_is_preferred")):
                    return True
            if self.config.guards.get("abort_on_solid_lane_crossing_risk", False):
                if self._is_solid_marking(lane_topology.get("right_lane_marking_type")) or self._is_solid_marking(
                    lane_topology.get("current_lane_marking_right_type")
                ):
                    return True
            if self.config.guards.get("abort_on_route_deviation_risk", False) and bool(route.get("route_deviation_risk")):
                return True
            max_lateral_offset = self.config.guards.get("maximum_lateral_offset_m")
            lateral_offset = ego.get("lateral_offset_from_lane_center_m")
            if max_lateral_offset is not None and lateral_offset is not None:
                if abs(float(lateral_offset)) > float(max_lateral_offset):
                    return True
            maximum_active_duration_s = float(self.config.intervention.get("maximum_active_duration_s", float("inf")))
            if self.active_ticks > 0 and self.active_ticks / self.simulation_fps >= maximum_active_duration_s:
                return True
        return False

    def _should_release(self, driving_state: dict[str, Any]) -> bool:
        if self.config.fault_type == "late_turn_lane_entry":
            route = driving_state.get("route") or {}
            if bool(route.get("current_lane_is_preferred")):
                return True
            distance_m = route.get("distance_to_junction_m")
            if distance_m is not None:
                return float(distance_m) <= float(self.config.release.get("release_at_junction_distance_m", 0.0))
            return False
        if self.config.release.get("release_after_junction_entry") and bool((driving_state.get("ego") or {}).get("is_junction")):
            return True
        max_active_duration_s = float(self.config.release.get("maximum_active_duration_s", float("inf")))
        timestamp_s = driving_state.get("timestamp_s")
        if timestamp_s is not None and self.activated_step is not None:
            elapsed_s = float(timestamp_s) - (self.activated_step / self.simulation_fps)
            if elapsed_s >= max_active_duration_s:
                return True
        return False

    def _apply_intervention(self, expert_control: Any, driving_state: dict[str, Any] | None = None) -> Any:
        if self.config.fault_type == "late_turn_lane_entry":
            return self._apply_late_lane_intervention(expert_control, driving_state)
        if self.config.fault_type == "red_light_entry" and bool(self.config.intervention.get("ignore_red_light", False)):
            return copy_vehicle_control(expert_control)

        applied_control = copy_vehicle_control(expert_control)
        applied_control.steer = float(getattr(expert_control, "steer", 0.0))
        applied_control.brake = float(self.config.intervention.get("brake", 0.0))
        minimum_throttle = float(self.config.intervention.get("minimum_throttle", 0.0))
        maximum_throttle = float(self.config.intervention.get("maximum_throttle", 1.0))
        expert_throttle = float(getattr(expert_control, "throttle", 0.0))
        applied_throttle = max(expert_throttle, minimum_throttle)
        applied_throttle = min(applied_throttle, maximum_throttle)
        applied_control.throttle = applied_throttle
        applied_control.hand_brake = False
        applied_control.reverse = False
        applied_control.manual_gear_shift = False
        return applied_control

    def _apply_late_lane_intervention(self, expert_control: Any, driving_state: dict[str, Any] | None = None) -> Any:
        applied_control = copy_vehicle_control(expert_control)
        if not bool(self.config.intervention.get("preserve_longitudinal_control", True)):
            return applied_control

        applied_control.throttle = float(getattr(expert_control, "throttle", 0.0))
        applied_control.brake = float(getattr(expert_control, "brake", 0.0))

        strategy = self.config.intervention.get("strategy")
        if strategy == "hold_current_lane_route":
            return applied_control
        if strategy == "bounded_steering_delay":
            right_sign = 1.0 if float(self.config.intervention.get("right_steer_sign", 1.0)) >= 0.0 else -1.0
            expert_steer = float(getattr(expert_control, "steer", 0.0))
            signed_steer = expert_steer * right_sign
            if signed_steer > 0.0:
                max_right_steer = float(self.config.intervention.get("maximum_right_steer", 0.05))
                max_suppression = float(self.config.intervention.get("maximum_steer_suppression", 0.35))
                suppressed = max(max_right_steer, signed_steer - max_suppression)
                applied_control.steer = min(signed_steer, suppressed) * right_sign
        return applied_control

    def _apply_release(self, expert_control: Any, driving_state: dict[str, Any] | None = None) -> Any:
        if self.config.fault_type == "late_turn_lane_entry":
            return copy_vehicle_control(expert_control)

        applied_control = copy_vehicle_control(expert_control)
        transition_ticks = max(1, int(self.config.release.get("transition_ticks", 1)))
        alpha = min(1.0, self.release_ticks / float(transition_ticks))
        expert_throttle = float(getattr(expert_control, "throttle", 0.0))
        expert_brake = float(getattr(expert_control, "brake", 0.0))
        applied_control.steer = float(getattr(expert_control, "steer", 0.0))
        applied_control.throttle = (1.0 - alpha) * self.previous_injected_throttle + alpha * expert_throttle
        applied_control.brake = (1.0 - alpha) * self.previous_injected_brake + alpha * expert_brake
        return applied_control

    def _abort_reason(self, driving_state: dict[str, Any]) -> str | None:
        hazards = driving_state.get("hazards") or {}
        if bool(hazards.get("vehicle")) and bool(self.config.guards.get("abort_on_vehicle_hazard", False)):
            return "vehicle_hazard"
        if bool(hazards.get("walker")) and bool(self.config.guards.get("abort_on_walker_hazard", False)):
            return "walker_hazard"
        traffic_light = driving_state.get("traffic_light") or {}
        if self.config.guards.get("abort_if_light_not_red_before_activation", False):
            if traffic_light.get("state") != self.config.trigger.get("traffic_light_state"):
                return "light_changed_before_activation"
        ego = driving_state.get("ego") or {}
        maximum_speed = float(self.config.guards.get("maximum_speed_mps", float("inf")))
        speed_mps = ego.get("speed_mps")
        if speed_mps is not None and speed_mps > maximum_speed:
            return "maximum_speed_exceeded"
        if self.config.fault_type == "late_turn_lane_entry":
            route = driving_state.get("route") or {}
            lane_topology = driving_state.get("lane_topology") or {}
            if bool(ego.get("is_junction")) and not bool(route.get("current_lane_is_preferred")):
                return "junction_entered_before_lane_correction"
            if self._is_solid_marking(lane_topology.get("right_lane_marking_type")) or self._is_solid_marking(
                lane_topology.get("current_lane_marking_right_type")
            ):
                return "solid_lane_marking_crossed"
            if bool(route.get("route_deviation_risk")):
                return "route_deviation_risk"
            max_lateral_offset = self.config.guards.get("maximum_lateral_offset_m")
            lateral_offset = ego.get("lateral_offset_from_lane_center_m")
            if max_lateral_offset is not None and lateral_offset is not None and abs(float(lateral_offset)) > float(max_lateral_offset):
                return "maximum_lateral_offset_exceeded"
            maximum_active_duration_s = float(self.config.intervention.get("maximum_active_duration_s", float("inf")))
            if self.active_ticks > 0 and self.active_ticks / self.simulation_fps >= maximum_active_duration_s:
                return "maximum_active_duration_exceeded"
        if self.config.fault_type == "red_light_entry" and traffic_light.get("state") is None:
            return "required_traffic_light_state_missing"
        return None

    def _build_lane_state(self, driving_state: dict[str, Any]) -> dict[str, Any] | None:
        if self.config.fault_type != "late_turn_lane_entry":
            return None
        ego = driving_state.get("ego") or {}
        route = driving_state.get("route") or {}
        lane_topology = driving_state.get("lane_topology") or {}
        return {
            "current_lane_id": ego.get("lane_id"),
            "preferred_approach_lane_id": route.get("preferred_approach_lane_id"),
            "current_lane_is_preferred": route.get("current_lane_is_preferred"),
            "right_lane_available": lane_topology.get("right_lane_available"),
            "right_lane_change_allowed": lane_topology.get("right_lane_change_allowed"),
            "right_lane_marking_type": lane_topology.get("right_lane_marking_type"),
            "distance_to_junction_m": route.get("distance_to_junction_m"),
            "distance_to_stop_line_m": route.get("distance_to_stop_line_m"),
            "lateral_offset_m": ego.get("lateral_offset_from_lane_center_m"),
        }

    def _build_lateral_intervention(self, expert_control: Any, applied_control: Any, driving_state: dict[str, Any]) -> dict[str, Any] | None:
        if self.config.fault_type != "late_turn_lane_entry":
            return None
        route = driving_state.get("route") or {}
        ego = driving_state.get("ego") or {}
        route_override = (driving_state.get("fault_injector") or {}).get("route_override") or {}
        steering_modified = float(getattr(expert_control, "steer", 0.0)) != float(getattr(applied_control, "steer", 0.0))
        strategy = self.config.intervention.get("strategy", "bounded_steering_delay")
        return {
            "strategy": strategy,
            "native_target_lane_id": route.get("preferred_approach_lane_id"),
            "applied_target_lane_id": ego.get("lane_id"),
            "route_modified": bool(route_override.get("route_modified", False)),
            "steering_modified": steering_modified,
            "expert_steer": float(getattr(expert_control, "steer", 0.0)),
            "applied_steer": float(getattr(applied_control, "steer", 0.0)),
            "route_override": route_override,
        }

    def _build_record(
        self,
        *,
        step: int,
        timestamp_s: float | None,
        state_before: str,
        state_after: str,
        trigger_matches: bool,
        trigger_failure_reasons: list[str],
        trigger_confirmation_count: int,
        fault_active: bool,
        fault_triggered: bool,
        abort_reason: str | None,
        expert_control: Any,
        applied_control: Any,
        driving_state: dict[str, Any],
    ) -> dict[str, Any]:
        record = {
            "step": step,
            "timestamp_s": timestamp_s,
            "config_id": self.config.config_id,
            "fault_type": self.config.fault_type,
            "injector_state": state_after,
            "state_before": state_before,
            "state_after": state_after,
            "trigger_matches": trigger_matches,
            "trigger_failure_reasons": trigger_failure_reasons,
            "trigger_confirmation_count": trigger_confirmation_count,
            "fault_active": fault_active,
            "fault_triggered": fault_triggered,
            "abort_reason": abort_reason,
            "expert_control": {
                "steer": float(getattr(expert_control, "steer", 0.0)),
                "throttle": float(getattr(expert_control, "throttle", 0.0)),
                "brake": float(getattr(expert_control, "brake", 0.0)),
            },
            "applied_control": {
                "steer": float(getattr(applied_control, "steer", 0.0)),
                "throttle": float(getattr(applied_control, "throttle", 0.0)),
                "brake": float(getattr(applied_control, "brake", 0.0)),
            },
            "driving_state": {
                "ego": driving_state.get("ego", {}),
                "route": driving_state.get("route", {}),
                "lane_topology": driving_state.get("lane_topology", {}),
                "traffic_light": driving_state.get("traffic_light", {}),
                "hazards": driving_state.get("hazards", {}),
            },
        }
        lane_state = self._build_lane_state(driving_state)
        if lane_state is not None:
            record["lane_state"] = lane_state
        lateral_intervention = self._build_lateral_intervention(expert_control, applied_control, driving_state)
        if lateral_intervention is not None:
            record["lateral_intervention"] = lateral_intervention
        prepare_before_control = (driving_state.get("fault_injector") or {}).get("prepare_before_control")
        if prepare_before_control is not None:
            record["prepare_before_control"] = prepare_before_control
        ignore_red_light = (driving_state.get("fault_injector") or {}).get("ignore_red_light")
        if ignore_red_light is not None:
            record["ignore_red_light"] = ignore_red_light
        return record

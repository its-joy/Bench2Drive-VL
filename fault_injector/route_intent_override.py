from __future__ import annotations

from typing import Any

import numpy as np

from .fault_config import FaultConfig


class LateLaneRouteOverride:
    def __init__(self, config: FaultConfig) -> None:
        self.config = config
        self.original_route_points: np.ndarray | None = None
        self.original_route_waypoints: list[Any] | None = None
        self.original_commands: Any | None = None
        self.hold_lane_id: int | None = None
        self.hold_road_id: int | None = None
        self.active = False
        self.last_status: dict[str, Any] = {"active": False, "route_modified": False}

    def apply(self, agent: Any, driving_state: dict[str, Any]) -> dict[str, Any]:
        planner = getattr(agent, "_waypoint_planner", None)
        if planner is None or not self._planner_has_route(planner):
            self.last_status = {"active": False, "route_modified": False, "reason": "planner_unavailable"}
            return self.last_status

        self._ensure_snapshot(planner)
        self.restore(planner)

        ego = driving_state.get("ego") or {}
        route = driving_state.get("route") or {}
        distance_to_junction_m = route.get("distance_to_junction_m")
        release_distance_m = float(self.config.release.get("release_at_junction_distance_m", 0.0))
        if distance_to_junction_m is not None and float(distance_to_junction_m) <= release_distance_m:
            self.active = False
            self.last_status = {
                "active": False,
                "route_modified": False,
                "reason": "release_distance_reached",
                "distance_to_junction_m": float(distance_to_junction_m),
                "release_at_junction_distance_m": release_distance_m,
            }
            return self.last_status

        if self.hold_lane_id is None:
            lane_id = ego.get("lane_id")
            self.hold_lane_id = int(lane_id) if lane_id is not None else None
            road_id = ego.get("road_id")
            self.hold_road_id = int(road_id) if road_id is not None else None

        if self.hold_lane_id is None:
            self.last_status = {"active": False, "route_modified": False, "reason": "hold_lane_unavailable"}
            return self.last_status

        route_index = int(getattr(planner, "route_index", 0) or 0)
        junction_index = self._first_junction_index(route_index)
        if junction_index is None:
            self.last_status = {"active": False, "route_modified": False, "reason": "junction_index_unavailable"}
            return self.last_status

        points_per_meter = float(getattr(planner, "points_per_meter", getattr(getattr(agent, "config", None), "points_per_meter", 10.0)) or 10.0)
        release_index = max(route_index, junction_index - int(round(release_distance_m * points_per_meter)))
        if route_index >= release_index:
            self.active = False
            self.last_status = {
                "active": False,
                "route_modified": False,
                "reason": "release_index_reached",
                "route_index": route_index,
                "release_index": release_index,
                "junction_index": junction_index,
            }
            return self.last_status

        changed_points = 0
        skipped_points = 0
        max_search_steps = int(self.config.intervention.get("max_adjacent_lane_search_steps", 3))
        lane_follow_command = self._lane_follow_command(planner)
        for index in range(route_index, release_index):
            source_wp = self.original_route_waypoints[index]
            hold_wp = self._find_lane_waypoint(source_wp, self.hold_lane_id, self.hold_road_id, max_search_steps)
            if hold_wp is None:
                skipped_points += 1
                continue
            location = hold_wp.transform.location
            planner.route_points[index] = np.array([location.x, location.y, location.z])
            planner.route_waypoints[index] = hold_wp
            if lane_follow_command is not None:
                planner.commands[index] = lane_follow_command
            changed_points += 1

        self.active = changed_points > 0
        self.last_status = {
            "active": self.active,
            "route_modified": self.active,
            "strategy": self.config.intervention.get("strategy"),
            "route_index": route_index,
            "hold_start_index": route_index,
            "release_index": release_index,
            "junction_index": junction_index,
            "hold_lane_id": self.hold_lane_id,
            "hold_road_id": self.hold_road_id,
            "changed_points": changed_points,
            "skipped_points": skipped_points,
            "distance_to_junction_m": float(distance_to_junction_m) if distance_to_junction_m is not None else None,
            "release_at_junction_distance_m": release_distance_m,
        }
        return self.last_status

    def restore(self, planner: Any) -> None:
        if self.original_route_points is not None:
            planner.route_points[:] = self.original_route_points
        if self.original_route_waypoints is not None:
            planner.route_waypoints[:] = list(self.original_route_waypoints)
        if self.original_commands is not None:
            try:
                planner.commands[:] = self.original_commands
            except TypeError:
                planner.commands = self.original_commands.copy()
        self.active = False

    def _ensure_snapshot(self, planner: Any) -> None:
        if self.original_route_points is not None:
            return
        self.original_route_points = np.copy(planner.route_points)
        self.original_route_waypoints = list(planner.route_waypoints)
        self.original_commands = planner.commands.copy()
        planner.fault_original_route_waypoints = list(planner.route_waypoints)

    @staticmethod
    def _planner_has_route(planner: Any) -> bool:
        route_points = getattr(planner, "route_points", None)
        route_waypoints = getattr(planner, "route_waypoints", None)
        commands = getattr(planner, "commands", None)
        return route_points is not None and len(route_points) > 0 and route_waypoints is not None and commands is not None

    def _first_junction_index(self, start_index: int) -> int | None:
        if self.original_route_waypoints is None:
            return None
        for index in range(start_index, len(self.original_route_waypoints)):
            waypoint = self.original_route_waypoints[index]
            if bool(getattr(waypoint, "is_junction", False)):
                return index
        return None

    @staticmethod
    def _find_lane_waypoint(source_wp: Any, lane_id: int, road_id: int | None, max_steps: int) -> Any | None:
        queue: list[tuple[Any, int]] = [(source_wp, 0)]
        seen: set[tuple[int | None, int | None, int]] = set()
        while queue:
            waypoint, depth = queue.pop(0)
            if waypoint is None:
                continue
            key = (getattr(waypoint, "road_id", None), getattr(waypoint, "lane_id", None), depth)
            if key in seen:
                continue
            seen.add(key)

            waypoint_lane_id = getattr(waypoint, "lane_id", None)
            waypoint_road_id = getattr(waypoint, "road_id", None)
            if waypoint_lane_id is not None and int(waypoint_lane_id) == int(lane_id):
                if road_id is None or waypoint_road_id is None or int(waypoint_road_id) == int(road_id):
                    return waypoint

            if depth >= max_steps:
                continue
            for accessor in ("get_left_lane", "get_right_lane"):
                get_lane = getattr(waypoint, accessor, None)
                if callable(get_lane):
                    try:
                        queue.append((get_lane(), depth + 1))
                    except Exception:
                        pass
        return None

    @staticmethod
    def _lane_follow_command(planner: Any) -> Any | None:
        commands = getattr(planner, "commands_orig", None)
        if commands is None:
            commands = getattr(planner, "commands", None)
        if commands is None:
            return None
        for command in commands:
            command_class = getattr(command, "__class__", None)
            if command_class is not None and hasattr(command_class, "LANEFOLLOW"):
                return command_class.LANEFOLLOW
            if str(command).lower().endswith("lanefollow"):
                return command
        return 4

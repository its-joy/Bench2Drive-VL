from __future__ import annotations

from typing import Any


def _enum_name(value: Any) -> str | None:
    if value is None:
        return None
    name = getattr(value, "name", None)
    if name is not None:
        return str(name)
    return str(value)


def _lane_type_name(value: Any) -> str | None:
    name = _enum_name(value)
    if name is None:
        return None
    if "." in name:
        name = name.split(".")[-1]
    return name


def _lane_change_allows_right(value: Any) -> bool | None:
    if value is None:
        return None
    text = str(value).lower()
    if "right" in text or "both" in text:
        return True
    try:
        return bool(value & 2)
    except Exception:
        return False


def _location_xyz(location: Any) -> dict[str, float | None]:
    return {
        "x": getattr(location, "x", None) if location is not None else None,
        "y": getattr(location, "y", None) if location is not None else None,
        "z": getattr(location, "z", None) if location is not None else None,
    }


def _distance_2d(a: Any, b: Any) -> float | None:
    try:
        return float(((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5)
    except Exception:
        return None


def _lateral_offset_from_lane_center(location: Any, waypoint: Any) -> float | None:
    if location is None or waypoint is None:
        return None
    try:
        wp_location = waypoint.transform.location
        forward = waypoint.transform.get_forward_vector()
        dx = float(location.x - wp_location.x)
        dy = float(location.y - wp_location.y)
        # Positive means the ego is left of the lane center in CARLA's XY plane.
        return float(-forward.y * dx + forward.x * dy)
    except Exception:
        return None


def _estimate_route_distance_to_junction(agent: Any) -> float | None:
    planner = getattr(agent, "_waypoint_planner", None)
    if planner is None:
        return getattr(agent, "distance_to_junction_m", None) if hasattr(agent, "distance_to_junction_m") else None
    route_waypoints = getattr(planner, "fault_original_route_waypoints", None) or getattr(planner, "route_waypoints", None)
    route_index = int(getattr(planner, "route_index", 0) or 0)
    if not route_waypoints:
        return getattr(agent, "distance_to_junction_m", None) if hasattr(agent, "distance_to_junction_m") else None
    points_per_meter = float(getattr(getattr(agent, "config", None), "points_per_meter", 1.0) or 1.0)
    for index in range(route_index, len(route_waypoints)):
        waypoint = route_waypoints[index]
        if bool(getattr(waypoint, "is_junction", False)):
            return max(0.0, float(index - route_index) / points_per_meter)
    return None


def _preferred_approach_lane_id(agent: Any) -> int | None:
    planner = getattr(agent, "_waypoint_planner", None)
    route_waypoints = (
        getattr(planner, "fault_original_route_waypoints", None) or getattr(planner, "route_waypoints", None)
        if planner is not None
        else None
    )
    if not route_waypoints:
        return None
    route_index = int(getattr(planner, "route_index", 0) or 0)
    previous_non_junction_lane = None
    for index in range(route_index, len(route_waypoints)):
        waypoint = route_waypoints[index]
        if bool(getattr(waypoint, "is_junction", False)):
            return previous_non_junction_lane
        previous_non_junction_lane = getattr(waypoint, "lane_id", None)
    return previous_non_junction_lane


def extract_driving_state(agent: Any, input_data: dict[str, Any] | None = None, timestamp: float | None = None) -> dict[str, Any]:
    step = getattr(agent, "step", None)
    config = getattr(agent, "config", None)
    fps = getattr(config, "fps", 20.0)
    timestamp_s = timestamp if timestamp is not None else (step / fps if step is not None else None)

    ego_speed = None
    location = None
    road_id = None
    lane_id = None
    is_junction = False
    waypoint = None
    lateral_offset = None

    vehicle = getattr(agent, "_vehicle", None)
    if vehicle is not None:
        try:
            velocity = vehicle.get_velocity()
            ego_speed = float(velocity.length()) if hasattr(velocity, "length") else None
        except Exception:
            ego_speed = None
        try:
            location = vehicle.get_location()
        except Exception:
            location = None

    if location is not None:
        try:
            world_map = getattr(agent, "world_map", None)
            if world_map is not None:
                waypoint = world_map.get_waypoint(location)
                if waypoint is not None:
                    road_id = getattr(waypoint, "road_id", None)
                    lane_id = getattr(waypoint, "lane_id", None)
                    is_junction = bool(getattr(waypoint, "is_junction", False))
                    lateral_offset = _lateral_offset_from_lane_center(location, waypoint)
        except Exception:
            road_id = None
            lane_id = None
            is_junction = False

    command = None
    commands = getattr(agent, "commands", None)
    if commands:
        try:
            command = commands[-1]
        except Exception:
            command = None

    route_distance_to_junction = _estimate_route_distance_to_junction(agent)
    preferred_lane_id = _preferred_approach_lane_id(agent)
    current_lane_is_preferred = None
    if preferred_lane_id is not None and lane_id is not None:
        current_lane_is_preferred = int(lane_id) == int(preferred_lane_id)

    right_lane = None
    right_lane_available = None
    right_lane_id = None
    right_lane_type = None
    right_lane_change_allowed = None
    right_lane_marking_type = None
    current_lane_marking_right_type = None
    if waypoint is not None:
        try:
            right_lane = waypoint.get_right_lane()
        except Exception:
            right_lane = None
        if right_lane is not None:
            right_lane_id = getattr(right_lane, "lane_id", None)
            right_lane_type = _lane_type_name(getattr(right_lane, "lane_type", None))
            right_lane_available = right_lane_type == "Driving"
            right_lane_marking_type = _lane_type_name(getattr(getattr(right_lane, "left_lane_marking", None), "type", None))
        else:
            right_lane_available = False
        current_lane_marking_right_type = _lane_type_name(getattr(getattr(waypoint, "right_lane_marking", None), "type", None))
        right_lane_change_allowed = _lane_change_allows_right(getattr(waypoint, "lane_change", None))

    traffic_light = {
        "present": False,
        "state": None,
        "affects_ego": False,
        "distance_m": None,
        "actor_id": None,
    }

    close_lights = getattr(agent, "close_traffic_lights", None) or []
    if isinstance(close_lights, list) and close_lights:
        first_light = close_lights[0]
        if isinstance(first_light, (list, tuple)) and len(first_light) >= 2:
            bbox = first_light[0]
            bbox_location = getattr(bbox, "location", None)
            distance_m = None
            if location is not None and bbox_location is not None:
                try:
                    distance_m = float(((location.x - bbox_location.x) ** 2 + (location.y - bbox_location.y) ** 2 + (location.z - bbox_location.z) ** 2) ** 0.5)
                except Exception:
                    distance_m = None
            traffic_light.update(
                {
                    "present": True,
                    "state": first_light[1],
                    "affects_ego": bool(first_light[3]) if len(first_light) > 3 else False,
                    "actor_id": first_light[2] if len(first_light) > 2 else None,
                    "distance_m": distance_m,
                }
            )
    distance_to_stop_line_m = traffic_light["distance_m"]

    if getattr(agent, "traffic_light_hazard", False) and traffic_light["present"] is False:
        traffic_light["present"] = True

    hazards = {
        "traffic_light": bool(getattr(agent, "traffic_light_hazard", False)),
        "vehicle": bool(getattr(agent, "vehicle_hazard", False)),
        "walker": bool(getattr(agent, "walker_hazard", False)),
        "stop_sign": bool(getattr(agent, "stop_sign_hazard", False)),
    }

    active_scenarios = []
    try:
        from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

        for scenario_type, scenario_data in list(getattr(CarlaDataProvider, "active_scenarios", [])):
            actor_ids = []
            actor_alive = []
            for item in list(scenario_data)[:2]:
                actor_ids.append(getattr(item, "id", None))
                actor_alive.append(getattr(item, "is_alive", None))
            active_scenarios.append(
                {
                    "type": scenario_type,
                    "actor_ids": actor_ids,
                    "actor_alive": actor_alive,
                }
            )
    except Exception:
        active_scenarios = []

    return {
        "step": step,
        "timestamp_s": timestamp_s,
        "ego": {
            "speed_mps": ego_speed,
            "location": {
                "x": getattr(location, "x", None) if location is not None else None,
                "y": getattr(location, "y", None) if location is not None else None,
                "z": getattr(location, "z", None) if location is not None else None,
            },
            "road_id": road_id,
            "lane_id": lane_id,
            "is_junction": is_junction,
            "lateral_offset_from_lane_center_m": lateral_offset,
        },
        "route": {
            "command": command,
            "distance_to_junction_m": route_distance_to_junction,
            "distance_to_stop_line_m": distance_to_stop_line_m,
            "preferred_approach_lane_id": preferred_lane_id,
            "current_lane_is_preferred": current_lane_is_preferred,
        },
        "active_scenarios": active_scenarios,
        "route_obstacle": getattr(agent, "route_obstacle_debug", {"active": False}),
        "vehicle_turning_route_pedestrian_hold": getattr(
            agent,
            "vehicle_turning_route_pedestrian_hold_debug",
            {"active": False},
        ),
        "lane_topology": {
            "right_lane_available": right_lane_available,
            "right_lane_id": right_lane_id,
            "right_lane_type": right_lane_type,
            "right_lane_change_allowed": right_lane_change_allowed,
            "right_lane_marking_type": right_lane_marking_type,
            "current_lane_marking_right_type": current_lane_marking_right_type,
        },
        "traffic_light": traffic_light,
        "hazards": hazards,
    }

from dataclasses import replace
from pathlib import Path

import numpy as np

from fault_injector.fault_config import FaultConfig
from fault_injector.fault_injector import FaultInjector, InjectorState
from fault_injector.pdm_fault_data_agent import PDMFaultDataAgent
from fault_injector.route_intent_override import LateLaneRouteOverride


class DummyControl:
    def __init__(self, steer=0.0, throttle=0.0, brake=0.0):
        self.steer = steer
        self.throttle = throttle
        self.brake = brake
        self.hand_brake = False
        self.reverse = False
        self.manual_gear_shift = False
        self.gear = 0


class DummyLocation:
    def __init__(self, x, y, z=0.0):
        self.x = x
        self.y = y
        self.z = z

    def distance(self, other):
        dx = self.x - other.x
        dy = self.y - other.y
        dz = self.z - other.z
        return (dx * dx + dy * dy + dz * dz) ** 0.5


class DummyActor:
    def __init__(self, actor_id, type_id, location, role_name=""):
        self.id = actor_id
        self.type_id = type_id
        self._location = location
        self.attributes = {"role_name": role_name}
        self.destroyed = False

    def get_location(self):
        return self._location

    def destroy(self):
        self.destroyed = True
        return True


class DummyActorList(list):
    def filter(self, pattern):
        prefix = pattern[:-1]
        return [actor for actor in self if actor.type_id.startswith(prefix)]


class DummyWorld:
    def __init__(self, actors, carla_map=None):
        self._actors = DummyActorList(actors)
        self._map = carla_map

    def get_actors(self):
        return self._actors

    def get_map(self):
        return self._map


class DummyMap:
    def __init__(self, waypoints_by_xy):
        self.waypoints_by_xy = waypoints_by_xy

    def get_waypoint(self, location, project_to_road=True, lane_type=None):
        return self.waypoints_by_xy[(location.x, location.y)]


class DummyTransform:
    def __init__(self, location):
        self.location = location


class DummyWaypoint:
    def __init__(self, road_id, lane_id, x, y, is_junction=False, lane_type="Driving"):
        self.road_id = road_id
        self.lane_id = lane_id
        self.lane_type = lane_type
        self.transform = DummyTransform(DummyLocation(x, y, 0.0))
        self.is_junction = is_junction
        self._left_lane = None
        self._right_lane = None

    def get_left_lane(self):
        return self._left_lane

    def get_right_lane(self):
        return self._right_lane


class DummyPlanner:
    def __init__(self, route_waypoints, points_per_meter=1):
        self.route_waypoints = list(route_waypoints)
        self.route_points = np.array(
            [
                [wp.transform.location.x, wp.transform.location.y, wp.transform.location.z]
                for wp in route_waypoints
            ],
            dtype=float,
        )
        self.original_route_points = np.copy(self.route_points)
        self.commands = [6 for _ in route_waypoints]
        self.commands_orig = self.commands.copy()
        self.route_index = 2
        self.points_per_meter = points_per_meter


def test_late_lane_config_loads_and_triggers():
    config = FaultConfig.from_json(Path("fault_injector/configs/signalized_right_turn/late_turn_lane_entry.json"))
    injector = FaultInjector(config, simulation_fps=20.0)
    expert = DummyControl(steer=0.2, throttle=0.7, brake=0.0)
    driving_state = {
        "step": 1,
        "timestamp_s": 0.05,
        "ego": {"speed_mps": 4.0, "is_junction": False, "lane_id": -1, "lateral_offset_from_lane_center_m": 0.2},
        "route": {"command": "turn_right", "distance_to_junction_m": 22.0, "distance_to_stop_line_m": 18.0, "preferred_approach_lane_id": -2, "current_lane_is_preferred": False},
        "lane_topology": {"right_lane_available": True, "right_lane_id": -2, "right_lane_type": "Driving", "right_lane_change_allowed": True, "right_lane_marking_type": "Broken", "current_lane_marking_right_type": "Broken"},
        "hazards": {"vehicle": False, "walker": False, "traffic_light": True, "stop_sign": False},
    }
    for _ in range(3):
        injector.apply(expert, driving_state)
    assert injector.state == InjectorState.ACTIVE


def test_late_lane_preserves_longitudinal_control():
    config = FaultConfig.from_json(Path("fault_injector/configs/signalized_right_turn/late_turn_lane_entry.json"))
    injector = FaultInjector(config, simulation_fps=20.0)
    expert = DummyControl(steer=0.3, throttle=0.4, brake=0.1)
    driving_state = {
        "step": 1,
        "timestamp_s": 0.05,
        "ego": {"speed_mps": 4.0, "is_junction": False, "lane_id": -1, "lateral_offset_from_lane_center_m": 0.2},
        "route": {"command": "turn_right", "distance_to_junction_m": 22.0, "distance_to_stop_line_m": 18.0, "preferred_approach_lane_id": -2, "current_lane_is_preferred": False},
        "lane_topology": {"right_lane_available": True, "right_lane_id": -2, "right_lane_type": "Driving", "right_lane_change_allowed": True, "right_lane_marking_type": "Broken", "current_lane_marking_right_type": "Broken"},
        "hazards": {"vehicle": False, "walker": False, "traffic_light": True, "stop_sign": False},
    }
    for _ in range(3):
        injector.apply(expert, driving_state)
    applied, record = injector.apply(expert, driving_state)
    assert applied.throttle == expert.throttle
    assert applied.brake == expert.brake
    assert record["lane_state"]["preferred_approach_lane_id"] == -2


def test_late_lane_does_not_activate_for_wrong_command():
    config = FaultConfig.from_json(Path("fault_injector/configs/signalized_right_turn/late_turn_lane_entry.json"))
    injector = FaultInjector(config, simulation_fps=20.0)
    expert = DummyControl(steer=0.3, throttle=0.4, brake=0.0)
    driving_state = {
        "step": 1,
        "timestamp_s": 0.05,
        "ego": {"speed_mps": 4.0, "is_junction": False, "lane_id": -1, "lateral_offset_from_lane_center_m": 0.2},
        "route": {"command": "straight", "distance_to_junction_m": 22.0, "preferred_approach_lane_id": -2, "current_lane_is_preferred": False},
        "lane_topology": {"right_lane_available": True, "right_lane_id": -2, "right_lane_type": "Driving", "right_lane_change_allowed": True, "right_lane_marking_type": "Broken", "current_lane_marking_right_type": "Broken"},
        "hazards": {"vehicle": False, "walker": False},
    }
    for _ in range(3):
        injector.apply(expert, driving_state)
    assert injector.state == InjectorState.ARMED


def test_late_lane_accepts_lane_change_right_as_turn_preparation():
    config = FaultConfig.from_json(Path("fault_injector/configs/signalized_right_turn/late_turn_lane_entry.json"))
    injector = FaultInjector(config, simulation_fps=20.0)
    expert = DummyControl(steer=0.3, throttle=0.4, brake=0.0)
    driving_state = {
        "step": 1,
        "timestamp_s": 0.05,
        "ego": {"speed_mps": 4.0, "is_junction": False, "lane_id": -1, "lateral_offset_from_lane_center_m": 0.2},
        "route": {"command": 6, "distance_to_junction_m": 100.0, "preferred_approach_lane_id": -2, "current_lane_is_preferred": False},
        "lane_topology": {"right_lane_available": True, "right_lane_id": -2, "right_lane_type": "Driving", "right_lane_change_allowed": True, "right_lane_marking_type": "Broken", "current_lane_marking_right_type": "Broken"},
        "hazards": {"vehicle": False, "walker": False},
    }
    for _ in range(3):
        injector.apply(expert, driving_state)
    assert injector.state == InjectorState.ACTIVE


def test_late_lane_prepare_before_control_does_not_double_count_confirmation():
    config = FaultConfig.from_json(Path("fault_injector/configs/signalized_right_turn/late_turn_lane_entry.json"))
    config = replace(config, trigger={**config.trigger, "confirmation_ticks": 3})
    injector = FaultInjector(config, simulation_fps=20.0)
    expert = DummyControl(steer=0.3, throttle=0.4, brake=0.0)

    def state(step):
        return {
            "step": step,
            "timestamp_s": step / 20.0,
            "ego": {"speed_mps": 4.0, "is_junction": False, "lane_id": -1, "lateral_offset_from_lane_center_m": 0.2},
            "route": {"command": 6, "distance_to_junction_m": 100.0, "preferred_approach_lane_id": -2, "current_lane_is_preferred": False},
            "lane_topology": {"right_lane_available": True, "right_lane_id": -2, "right_lane_type": "Driving", "right_lane_change_allowed": True, "right_lane_marking_type": "Broken", "current_lane_marking_right_type": "Broken"},
            "hazards": {"vehicle": False, "walker": False},
        }

    first = state(1)
    first["fault_injector"] = {"prepare_before_control": injector.prepare_before_control(first)}
    injector.apply(expert, first)
    assert injector.confirmation_count == 1
    assert injector.state == InjectorState.ARMED

    second = state(2)
    second["fault_injector"] = {"prepare_before_control": injector.prepare_before_control(second)}
    injector.apply(expert, second)
    assert injector.confirmation_count == 2
    assert injector.state == InjectorState.ARMED

    third = state(3)
    prepare = injector.prepare_before_control(third)
    third["fault_injector"] = {"prepare_before_control": prepare}
    injector.apply(expert, third)
    assert prepare["activated_before_control"] is True
    assert injector.confirmation_count == 3
    assert injector.state == InjectorState.ACTIVE


def test_late_lane_fallback_only_bounds_rightward_steer():
    config = FaultConfig.from_json(Path("fault_injector/configs/signalized_right_turn/late_turn_lane_entry.json"))
    config = replace(
        config,
        trigger={**config.trigger, "confirmation_ticks": 1},
        intervention={**config.intervention, "strategy": "bounded_steering_delay"},
    )
    injector = FaultInjector(config, simulation_fps=20.0)
    expert = DummyControl(steer=-0.5, throttle=0.4, brake=0.1)
    driving_state = {
        "step": 1,
        "timestamp_s": 0.05,
        "ego": {"speed_mps": 4.0, "is_junction": False, "lane_id": -1, "lateral_offset_from_lane_center_m": 0.2},
        "route": {"command": "turn_right", "distance_to_junction_m": 22.0, "preferred_approach_lane_id": -2, "current_lane_is_preferred": False},
        "lane_topology": {"right_lane_available": True, "right_lane_id": -2, "right_lane_type": "Driving", "right_lane_change_allowed": True, "right_lane_marking_type": "Broken", "current_lane_marking_right_type": "Broken"},
        "hazards": {"vehicle": False, "walker": False},
    }
    applied, record = injector.apply(expert, driving_state)
    assert applied.steer > expert.steer
    assert abs(applied.steer) < abs(expert.steer)
    assert applied.throttle == expert.throttle
    assert applied.brake == expert.brake
    assert record["lateral_intervention"]["steering_modified"] is True


def _make_two_lane_route():
    route_waypoints = []
    for index in range(20):
        left_lane = DummyWaypoint(road_id=1, lane_id=-1, x=float(index), y=0.0, is_junction=index >= 15)
        right_lane = DummyWaypoint(road_id=1, lane_id=-2, x=float(index), y=-3.5, is_junction=index >= 15)
        left_lane._right_lane = right_lane
        right_lane._left_lane = left_lane
        route_waypoints.append(left_lane if index < 5 else right_lane)
    return route_waypoints


def test_high_level_late_lane_route_override_holds_current_lane_until_release():
    config = FaultConfig.from_json(Path("fault_injector/configs/signalized_right_turn/late_turn_lane_entry.json"))
    planner = DummyPlanner(_make_two_lane_route(), points_per_meter=1)
    agent = type("Agent", (), {"_waypoint_planner": planner})()
    override = LateLaneRouteOverride(config)
    driving_state = {
        "ego": {"road_id": 1, "lane_id": -1},
        "route": {"distance_to_junction_m": 20.0},
    }

    status = override.apply(agent, driving_state)

    assert status["route_modified"] is True
    assert status["release_index"] == 10
    assert np.allclose(planner.route_points[2:10, 1], 0.0)
    assert np.allclose(planner.route_points[10:15, 1], -3.5)
    assert planner.commands[2:10] == [4] * 8


def test_high_level_late_lane_route_override_restores_at_release_distance():
    config = FaultConfig.from_json(Path("fault_injector/configs/signalized_right_turn/late_turn_lane_entry.json"))
    planner = DummyPlanner(_make_two_lane_route(), points_per_meter=1)
    agent = type("Agent", (), {"_waypoint_planner": planner})()
    override = LateLaneRouteOverride(config)

    override.apply(agent, {"ego": {"road_id": 1, "lane_id": -1}, "route": {"distance_to_junction_m": 20.0}})
    status = override.apply(agent, {"ego": {"road_id": 1, "lane_id": -1}, "route": {"distance_to_junction_m": 4.5}})

    assert status["route_modified"] is False
    assert status["reason"] == "release_distance_reached"
    assert np.allclose(planner.route_points, planner.original_route_points)


def test_late_lane_solid_marking_aborts():
    config = FaultConfig.from_json(Path("fault_injector/configs/signalized_right_turn/late_turn_lane_entry.json"))
    config = replace(config, trigger={**config.trigger, "confirmation_ticks": 1})
    injector = FaultInjector(config, simulation_fps=20.0)
    expert = DummyControl(steer=0.3, throttle=0.4, brake=0.0)
    driving_state = {
        "step": 1,
        "timestamp_s": 0.05,
        "ego": {"speed_mps": 4.0, "is_junction": False, "lane_id": -1, "lateral_offset_from_lane_center_m": 0.2},
        "route": {"command": "turn_right", "distance_to_junction_m": 22.0, "preferred_approach_lane_id": -2, "current_lane_is_preferred": False},
        "lane_topology": {"right_lane_available": True, "right_lane_id": -2, "right_lane_type": "Driving", "right_lane_change_allowed": True, "right_lane_marking_type": "Solid", "current_lane_marking_right_type": "Solid"},
        "hazards": {"vehicle": False, "walker": False},
    }
    injector.apply(expert, driving_state)
    assert injector.state == InjectorState.ABORTED


def test_actor_spawn_control_near_ego_only_removes_nearby_dynamic_actors():
    agent = object.__new__(PDMFaultDataAgent)
    ego = DummyActor(1, "vehicle.tesla.model3", DummyLocation(0.0, 0.0), role_name="hero")
    nearby_vehicle = DummyActor(2, "vehicle.audi.tt", DummyLocation(10.0, 0.0))
    distant_vehicle = DummyActor(3, "vehicle.audi.tt", DummyLocation(120.0, 0.0))
    nearby_walker = DummyActor(4, "walker.pedestrian.0001", DummyLocation(5.0, 0.0))
    world = DummyWorld([ego, nearby_vehicle, distant_vehicle, nearby_walker])
    agent._vehicle = ego
    agent._world = world
    agent.actor_spawn_mode = "near_ego"
    agent.actor_clear_radius_m = 50.0
    agent.actor_clear_vehicles = True
    agent.actor_clear_walkers = True
    agent.actor_clear_repeat = True
    agent.actor_spawn_control_applied = False
    agent.actor_spawn_control_removed = {}
    agent.fault_summary = {}

    agent._apply_actor_spawn_control()

    assert ego.destroyed is False
    assert nearby_vehicle.destroyed is True
    assert nearby_walker.destroyed is True
    assert distant_vehicle.destroyed is False
    assert agent.fault_summary["actor_spawn_control"]["removed_count"] == 2


def test_actor_spawn_control_parking_lanes_near_ego_only_removes_nearby_parking_actors():
    agent = object.__new__(PDMFaultDataAgent)
    ego = DummyActor(1, "vehicle.tesla.model3", DummyLocation(0.0, 0.0), role_name="hero")
    nearby_parked_vehicle = DummyActor(2, "vehicle.audi.tt", DummyLocation(10.0, 0.0))
    nearby_driving_vehicle = DummyActor(3, "vehicle.lincoln.mkz", DummyLocation(12.0, 0.0))
    distant_parked_vehicle = DummyActor(4, "vehicle.mercedes.coupe", DummyLocation(120.0, 0.0))
    carla_map = DummyMap({
        (10.0, 0.0): DummyWaypoint(1, 1, 10.0, 0.0, lane_type="Parking"),
        (12.0, 0.0): DummyWaypoint(1, 2, 12.0, 0.0, lane_type="Driving"),
        (120.0, 0.0): DummyWaypoint(1, 1, 120.0, 0.0, lane_type="Parking"),
    })
    world = DummyWorld([ego, nearby_parked_vehicle, nearby_driving_vehicle, distant_parked_vehicle], carla_map=carla_map)
    agent._vehicle = ego
    agent._world = world
    agent.actor_spawn_mode = "parking_lanes_near_ego"
    agent.actor_clear_radius_m = 50.0
    agent.actor_clear_vehicles = True
    agent.actor_clear_walkers = False
    agent.actor_clear_repeat = True
    agent.actor_spawn_control_applied = False
    agent.actor_spawn_control_removed = {}
    agent.fault_summary = {}

    agent._apply_actor_spawn_control()

    assert ego.destroyed is False
    assert nearby_parked_vehicle.destroyed is True
    assert nearby_driving_vehicle.destroyed is False
    assert distant_parked_vehicle.destroyed is False
    assert agent.fault_summary["actor_spawn_control"]["removed_count"] == 1


def test_actor_spawn_control_none_removes_all_non_ego_dynamic_actors():
    agent = object.__new__(PDMFaultDataAgent)
    ego = DummyActor(1, "vehicle.tesla.model3", DummyLocation(0.0, 0.0), role_name="hero")
    vehicle = DummyActor(2, "vehicle.audi.tt", DummyLocation(120.0, 0.0))
    walker = DummyActor(3, "walker.pedestrian.0001", DummyLocation(160.0, 0.0))
    world = DummyWorld([ego, vehicle, walker])
    agent._vehicle = ego
    agent._world = world
    agent.actor_spawn_mode = "none"
    agent.actor_clear_radius_m = 50.0
    agent.actor_clear_vehicles = True
    agent.actor_clear_walkers = True
    agent.actor_clear_repeat = False
    agent.actor_spawn_control_applied = False
    agent.actor_spawn_control_removed = {}
    agent.fault_summary = {}

    agent._apply_actor_spawn_control()

    assert ego.destroyed is False
    assert vehicle.destroyed is True
    assert walker.destroyed is True
    assert agent.fault_summary["actor_spawn_control"]["removed_count"] == 2

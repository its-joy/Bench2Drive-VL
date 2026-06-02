"""
Unit tests for generator_modules/post_actions.py

Run from B2DVL_Adapter/:
    python3 test_post_actions.py
"""
import sys
import types
import importlib.util
import pathlib
import unittest

# solve the import problem
# post_action.py cannot be imported normally outside CARLA
# --- Stub carla (CARLA simulator, not available outside the sim) ---
# create a fake empty python module and registered in sys.modules
# when python's import system looks up carla or io_utils, it finds the fake instead of trying to load the real thing
carla_stub = types.ModuleType("carla") 
carla_stub.Map = object
sys.modules["carla"] = carla_stub

# --- Stub io_utils ---
io_utils_stub = types.ModuleType("io_utils")
io_utils_stub.print_debug = lambda *a, **kw: None
sys.modules["io_utils"] = io_utils_stub

# --- Stub generator_modules as a proper package so relative imports work ---
gm = types.ModuleType("generator_modules")
gm.__path__ = []        # marks it as a package
gm.__package__ = "generator_modules"
sys.modules["generator_modules"] = gm

# Stub offline_map_calculations — expose get_future_measurements so the
# `from .offline_map_calculations import *` inside post_actions resolves it.
omc = types.ModuleType("generator_modules.offline_map_calculations")
omc.get_future_measurements = lambda path, k: None  # overridden per-test
sys.modules["generator_modules.offline_map_calculations"] = omc
setattr(gm, "offline_map_calculations", omc)

for submod in ("hyper_params", "graph_utils"):
    stub = types.ModuleType(f"generator_modules.{submod}")
    sys.modules[f"generator_modules.{submod}"] = stub
    setattr(gm, submod, stub)

# --- Load post_actions.py as part of the generator_modules package ---
_src = pathlib.Path(__file__).parent / "generator_modules" / "post_actions.py"
spec = importlib.util.spec_from_file_location("generator_modules.post_actions", _src)
pa = importlib.util.module_from_spec(spec)
pa.__package__ = "generator_modules"   # enables relative imports
sys.modules["generator_modules.post_actions"] = pa
spec.loader.exec_module(pa)

# Grab the pure helper functions under test
_describe_action  = pa._describe_action
_describe_reason  = pa._describe_reason
_describe_changes = pa._describe_changes


# ---------------------------------------------------------------------------
# Helpers for building fake measurement dicts
# ---------------------------------------------------------------------------

# these are fake data builders to stimulate the measurement format
def _meas(speed, x=0.0, y=0.0, throttle=0.0, brake=0.0, steer=0.0,
          cmd_near=4, road_id=1, lane_id=-1, extra_bbs=None):
    ego_bb = {
        "class": "ego_vehicle",
        "id": "ego",
        "speed": speed,
        "throttle": throttle,
        "brake": brake,
        "steer": steer,
        "road_id": road_id,
        "lane_id": lane_id,
    }
    bbs = [ego_bb] + (extra_bbs or [])
    return {
        "x": x, "y": y,
        "speed": speed,
        "throttle": throttle,
        "brake": brake,
        "steer": steer,
        "command_near": cmd_near,
        "bounding_boxes": bbs,
    }

def _red_tl():
    return {"class": "traffic_light", "state": 0, "affects_ego": True, "distance": 15.0}

def _vehicle(dist, road_id=1, lane_id=-1, speed=0.0, obj_id="v1"):
    return {"class": "vehicle", "id": obj_id, "distance": dist,
            "road_id": road_id, "lane_id": lane_id, "speed": speed}

def _pedestrian(dist):
    return {"class": "walker", "distance": dist}


# ---------------------------------------------------------------------------
# Tests for _describe_action
# ---------------------------------------------------------------------------

class TestDescribeAction(unittest.TestCase):

    def test_accelerated(self):
        cur  = _meas(speed=10.0, x=50.0)
        prev = _meas(speed=2.0,  x=0.0)
        result = _describe_action(cur, prev, cmd_near=4)
        self.assertIn("accelerated", result)
        self.assertIn("7.2", result)   # 2.0 m/s → 7.2 km/h
        self.assertIn("36.0", result)  # 10.0 m/s → 36.0 km/h

    def test_decelerated(self):
        cur  = _meas(speed=2.0,  x=50.0)
        prev = _meas(speed=10.0, x=0.0)
        result = _describe_action(cur, prev, cmd_near=4)
        self.assertIn("decelerated", result)

    def test_came_to_a_stop(self):
        cur  = _meas(speed=0.1)
        prev = _meas(speed=8.0)
        result = _describe_action(cur, prev, cmd_near=4)
        self.assertIn("stop", result)

    def test_remained_stopped(self):
        cur  = _meas(speed=0.1)
        prev = _meas(speed=0.2)
        result = _describe_action(cur, prev, cmd_near=4)
        self.assertIn("remained stopped", result)

    def test_maintained_speed(self):
        cur  = _meas(speed=8.0)
        prev = _meas(speed=8.3)
        result = _describe_action(cur, prev, cmd_near=4)
        self.assertIn("continued at approximately", result)

    def test_turn_left_at_intersection(self):
        cur  = _meas(speed=5.0)
        prev = _meas(speed=5.0)
        result = _describe_action(cur, prev, cmd_near=1)
        self.assertIn("left", result)

    def test_turn_right_at_intersection(self):
        cur  = _meas(speed=5.0)
        prev = _meas(speed=5.0)
        result = _describe_action(cur, prev, cmd_near=2)
        self.assertIn("right", result)

    def test_lane_change_left(self):
        cur  = _meas(speed=8.0, road_id=1, lane_id=-2, steer=-0.3)
        prev = _meas(speed=8.0, road_id=1, lane_id=-1, steer=-0.3)
        result = _describe_action(cur, prev, cmd_near=5)
        self.assertIn("left lane", result)

    def test_lane_change_right(self):
        cur  = _meas(speed=8.0, road_id=1, lane_id=-1, steer=0.3)
        prev = _meas(speed=8.0, road_id=1, lane_id=-2, steer=0.3)
        result = _describe_action(cur, prev, cmd_near=6)
        self.assertIn("right lane", result)


# ---------------------------------------------------------------------------
# Tests for _describe_reason
# ---------------------------------------------------------------------------

class TestDescribeReason(unittest.TestCase):

    def test_braked_for_red_light(self):
        prev = _meas(speed=10.0, extra_bbs=[_red_tl()])
        result = _describe_reason("decelerated from 36 km/h to 7 km/h", prev, prev_cmd_near=4)
        self.assertIn("red traffic light", result)

    def test_braked_for_pedestrian(self):
        prev = _meas(speed=8.0, extra_bbs=[_pedestrian(dist=10.0)])
        result = _describe_reason("decelerated from 28 km/h to 3 km/h", prev, prev_cmd_near=4)
        self.assertIn("pedestrian", result)

    def test_braked_for_vehicle(self):
        prev = _meas(speed=8.0, extra_bbs=[_vehicle(dist=12.0)])
        result = _describe_reason("came to a stop", prev, prev_cmd_near=4)
        self.assertIn("vehicle", result)

    def test_braked_no_hazard(self):
        prev = _meas(speed=8.0)
        result = _describe_reason("decelerated from 28 km/h to 5 km/h", prev, prev_cmd_near=4)
        self.assertIn("traffic situation", result)

    def test_turn_reason(self):
        prev = _meas(speed=5.0)
        result = _describe_reason("turned left at the intersection", prev, prev_cmd_near=1)
        self.assertIn("left turn", result)

    def test_lane_change_reason_nav(self):
        prev = _meas(speed=8.0)
        result = _describe_reason("changed to the left lane", prev, prev_cmd_near=5)
        self.assertIn("navigation commanded", result)

    def test_lane_change_reason_obstacle(self):
        prev = _meas(speed=8.0, extra_bbs=[_vehicle(dist=8.0)])
        result = _describe_reason("changed to the right lane", prev, prev_cmd_near=4)
        self.assertIn("obstacle", result)

    def test_follow_lane_reason(self):
        prev = _meas(speed=8.0)
        result = _describe_reason("continued at approximately 30.0 km/h", prev, prev_cmd_near=4)
        self.assertIn("clear", result)


# ---------------------------------------------------------------------------
# Tests for _describe_changes
# ---------------------------------------------------------------------------

class TestDescribeChanges(unittest.TestCase):

    def test_speed_increased(self):
        cur  = _meas(speed=10.0, x=50.0)
        prev = _meas(speed=2.0,  x=0.0)
        result = _describe_changes(cur, prev)
        self.assertIn("increased", result)

    def test_speed_decreased(self):
        cur  = _meas(speed=2.0, x=20.0)
        prev = _meas(speed=10.0, x=0.0)
        result = _describe_changes(cur, prev)
        self.assertIn("decreased", result)

    def test_speed_maintained(self):
        cur  = _meas(speed=8.0, x=80.0)
        prev = _meas(speed=8.0, x=0.0)
        result = _describe_changes(cur, prev)
        self.assertIn("maintained", result)

    def test_distance_traveled(self):
        cur  = _meas(speed=8.0, x=30.0, y=40.0)
        prev = _meas(speed=8.0, x=0.0,  y=0.0)
        result = _describe_changes(cur, prev)
        self.assertIn("50.0 m", result)  # 3-4-5 triangle

    def test_lane_change_detected(self):
        cur  = _meas(speed=8.0, road_id=1, lane_id=-2, extra_bbs=[])
        prev = _meas(speed=8.0, road_id=1, lane_id=-1, extra_bbs=[])
        result = _describe_changes(cur, prev)
        self.assertIn("different lane", result)

    def test_nearest_vehicle_distance_change(self):
        cur  = _meas(speed=8.0, x=20.0, extra_bbs=[_vehicle(dist=30.0, obj_id="v1")])
        prev = _meas(speed=8.0, x=0.0,  extra_bbs=[_vehicle(dist=10.0, obj_id="v1")])
        result = _describe_changes(cur, prev)
        self.assertIn("nearest vehicle", result)
        self.assertIn("increased", result)


if __name__ == "__main__":
    unittest.main(verbosity=2)

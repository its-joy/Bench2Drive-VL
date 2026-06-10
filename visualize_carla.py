import carla
import xml.etree.ElementTree as ET
from agents.navigation.global_route_planner import GlobalRoutePlanner
import time

# --------------------
# CONNECT
# --------------------
client = carla.Client("localhost", 2000)
client.set_timeout(10.0)

world = client.get_world()
carla_map = world.get_map()

grp = GlobalRoutePlanner(carla_map, 2.0)

# --------------------
# LOAD XML ROUTE
# --------------------
tree = ET.parse("leaderboard/data/routes_devtest_short.xml")
root = tree.getroot()

xml_points = []
for pos in root.iter("position"):
    xml_points.append(
        carla.Location( 
            x=float(pos.attrib["x"]),
            y=float(pos.attrib["y"]),
            z=float(pos.attrib["z"])
        )
    )

# --------------------
# BUILD DENSE ROUTE
# --------------------
full_route = []
for i in range(len(xml_points) - 1):
    segment = grp.trace_route(xml_points[i], xml_points[i + 1])
    full_route.extend(segment)

# --------------------
# DRAW ONLY PLANNED WAYPOINTS
# --------------------
for wp, _ in full_route:
    loc = wp.transform.location
    world.debug.draw_point(
        loc + carla.Location(z=0.3),
        size=0.1,
        color=carla.Color(0, 255, 0),
        life_time=0
    )

# --------------------
# KEEP ALIVE
# --------------------
while True:
    time.sleep(1)
import xml.etree.ElementTree as ET
import math

ET.register_namespace('', '')
tree = ET.parse('leaderboard/data/routes_devtest.xml')
route = tree.getroot().find('route')

# Keep only first N waypoints
N = 12
waypoints = route.find('waypoints')
positions = waypoints.findall('position')
for p in positions[N:]:
    waypoints.remove(p)

kept = positions[:N]

# Remove scenarios whose trigger point is far from all kept waypoints
MAX_TRIGGER_DIST = 300.0   # metres — increase to keep more scenarios

def dist(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

scenarios_elem = route.find('scenarios')
removed = 0
for s in list(scenarios_elem.findall('scenario')):
    tp = s.find('trigger_point')
    tx, ty = float(tp.attrib['x']), float(tp.attrib['y'])
    nearest = min(dist(tx, ty, float(p.attrib['x']), float(p.attrib['y'])) for p in kept)
    if nearest > MAX_TRIGGER_DIST:
        scenarios_elem.remove(s)
        removed += 1

remaining = len(scenarios_elem.findall('scenario'))
print(f"Kept {N} waypoints, removed {removed} far scenarios, {remaining} remain.")

tree.write('leaderboard/data/routes_devtest_short.xml',
           xml_declaration=True, encoding='unicode')

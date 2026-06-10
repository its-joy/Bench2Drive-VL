"""
split_scenarios.py

Reads routes_devtest.xml and creates a new XML file where each scenario
gets its own route. Each route has the trigger waypoint plus surrounding
waypoints for approach and exit.

Usage:
    python3 split_scenarios.py
    python3 split_scenarios.py --input leaderboard/data/routes_devtest.xml \
                                --output leaderboard/data/routes_post_action.xml \
                                --before 3 --after 5 --route-id 0
"""

import argparse
import math
import xml.etree.ElementTree as ET


def dist(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--input',  default='leaderboard/data/routes_devtest.xml')
    p.add_argument('--output', default='leaderboard/data/routes_post_action.xml')
    p.add_argument('--before', type=int, default=2,
                   help='Minimum waypoints to keep before trigger point')
    p.add_argument('--after',  type=int, default=4,
                   help='Waypoints to keep after trigger point')
    p.add_argument('--min-approach', type=float, default=200.0,
                   help='Minimum road distance (m) from spawn to trigger; '
                        'overrides --before if more waypoints are needed')
    p.add_argument('--max-trigger-dist', type=float, default=200.0,
                   help='Only include scenarios whose trigger is within this distance of a waypoint')
    p.add_argument('--route-id', type=int, default=0,
                   help='Only process the source route with this id (default: 0)')
    return p.parse_args()


def main():
    args = parse_args()

    tree = ET.parse(args.input)
    root = tree.getroot()

    new_root = ET.Element('routes')
    route_id = 0

    for route_elem in root.findall('route'):
        if int(route_elem.attrib.get('id', 0)) != args.route_id:
            continue
        town = route_elem.attrib['town']

        # Collect waypoints as (x, y, z) tuples
        wp_elems = route_elem.find('waypoints').findall('position')
        waypoints = [(float(p.attrib['x']), float(p.attrib['y']),
                      float(p.attrib.get('z', '0'))) for p in wp_elems]

        # Weather block (reuse same weather for all sub-routes)
        weather_elem = route_elem.find('weathers')

        for scenario_elem in route_elem.find('scenarios').findall('scenario'):
            tp = scenario_elem.find('trigger_point')
            tx, ty = float(tp.attrib['x']), float(tp.attrib['y'])
            s_type = scenario_elem.attrib['type']
            s_name = scenario_elem.attrib['name']

            # Find closest waypoint to trigger
            closest_idx = min(range(len(waypoints)),
                              key=lambda i: dist(tx, ty, waypoints[i][0], waypoints[i][1]))

            # Skip if trigger is too far from any waypoint
            min_d = dist(tx, ty, waypoints[closest_idx][0], waypoints[closest_idx][1])
            if min_d > args.max_trigger_dist:
                print(f"  Skipping {s_name}: trigger {min_d:.0f}m from nearest waypoint")
                continue

            # Walk back from closest waypoint until we have >= min_approach metres
            # of road on the approach, but always take at least --before waypoints.
            start_idx = max(0, closest_idx - args.before)
            approach_dist = 0.0
            for i in range(closest_idx, 0, -1):
                approach_dist += dist(waypoints[i][0], waypoints[i][1],
                                      waypoints[i-1][0], waypoints[i-1][1])
                if i - 1 <= start_idx:
                    start_idx = i - 1   # already satisfied --before
                if approach_dist >= args.min_approach:
                    start_idx = min(start_idx, i - 1)
                    break

            end_idx = min(len(waypoints), closest_idx + args.after + 1)
            sub_wps = waypoints[start_idx:end_idx]

            if len(sub_wps) < 2:
                print(f"  Skipping {s_name}: fewer than 2 waypoints")
                continue

            # Build new <route> element
            new_route = ET.SubElement(new_root, 'route',
                                      id=str(route_id), town=town,
                                      scenario_type=s_type)
            route_id += 1

            # Copy weather
            if weather_elem is not None:
                new_route.append(weather_elem)  # shallow copy fine for read-only use

            # Waypoints
            wps_elem = ET.SubElement(new_route, 'waypoints')
            for wx, wy, wz in sub_wps:
                ET.SubElement(wps_elem, 'position',
                              x=str(round(wx, 1)),
                              y=str(round(wy, 1)),
                              z=str(round(wz, 1)))

            # Single scenario
            scen_elem = ET.SubElement(new_route, 'scenarios')
            new_s = ET.SubElement(scen_elem, 'scenario',
                                  name=s_name, type=s_type)
            ET.SubElement(new_s, 'trigger_point',
                          x=tp.attrib['x'], y=tp.attrib['y'],
                          z=tp.attrib.get('z', '0'),
                          yaw=tp.attrib.get('yaw', '0'))

            print(f"Route {route_id-1:3d}: {s_type:<45} "
                  f"trigger=({tx:.0f},{ty:.0f})  wps={len(sub_wps)}")

    ET.indent(new_root, space='   ')
    ET.ElementTree(new_root).write(args.output, xml_declaration=True,
                                   encoding='unicode')
    print(f"\nWrote {route_id} routes to {args.output}")


if __name__ == '__main__':
    main()

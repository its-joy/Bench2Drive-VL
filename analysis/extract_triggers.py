"""
extract_triggers.py

Extracts all trigger points from routes_post_action.xml and prints them
with their route id, scenario type, and trigger coordinates.
"""

import xml.etree.ElementTree as ET
import argparse


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', default='leaderboard/data/routes_post_action.xml')
    p.add_argument('--csv', action='store_true', help='Output as CSV')
    args = p.parse_args()

    tree = ET.parse(args.input)
    root = tree.getroot()

    rows = []
    for route in root.findall('route'):
        route_id    = route.attrib.get('id', '?')
        town        = route.attrib.get('town', '?')
        scen_type   = route.attrib.get('scenario_type', '?')

        for scenario in route.findall('.//scenario'):
            s_name  = scenario.attrib.get('name', '?')
            s_type  = scenario.attrib.get('type', scen_type)
            tp      = scenario.find('trigger_point')
            if tp is None:
                continue
            tx   = tp.attrib.get('x', '?')
            ty   = tp.attrib.get('y', '?')
            tz   = tp.attrib.get('z', '?')
            yaw  = tp.attrib.get('yaw', '?')
            rows.append((route_id, town, s_type, s_name, tx, ty, tz, yaw))

    if args.csv:
        print('route_id,town,scenario_type,scenario_name,x,y,z,yaw')
        for r in rows:
            print(','.join(r))
    else:
        header = f"{'ID':>4}  {'town':<8}  {'scenario_type':<45}  {'name':<45}  {'x':>10}  {'y':>10}  {'z':>7}  {'yaw':>8}"
        print(header)
        print('-' * len(header))
        for route_id, town, s_type, s_name, tx, ty, tz, yaw in rows:
            print(f"{route_id:>4}  {town:<8}  {s_type:<45}  {s_name:<45}  {tx:>10}  {ty:>10}  {tz:>7}  {yaw:>8}")

    print(f'\nTotal: {len(rows)} trigger points')


if __name__ == '__main__':
    main()

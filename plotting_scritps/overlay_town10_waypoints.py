"""
overlay_town10_waypoints.py

Plots route_waypoints_5m.json on top of the Town10HD road network.
Road points = blue, junction points = orange (from xodr).
Route waypoints coloured by is_junction: green = road, red = junction.
Optionally overlays vehicle positions from an anno file (magenta=ego, blue=others).

Usage:
    python3 overlay_town10_waypoints.py
    python3 overlay_town10_waypoints.py --json route_waypoints_5m.json \
        --xodr carlaCache/0.9.15/Carla/Maps/OpenDrive/Town10HD_Opt.xodr \
        --out town10_route_overlay.png
    python3 overlay_town10_waypoints.py --anno anno/00425.json.gz \
        --out town10_vehicles_overlay.png
"""

import argparse
import json
import math
import gzip
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe


# ── xodr parser (Y-flipped to CARLA coords) ──────────────────────────────────

def parse_xodr(filepath, sample_interval=1.0):
    tree = ET.parse(filepath)
    root = tree.getroot()
    road_pts, junc_pts = [], []
    for road in root.findall('road'):
        is_junc   = road.attrib.get('junction', '-1') != '-1'
        plan_view = road.find('planView')
        if plan_view is None:
            continue
        for geom in plan_view.findall('geometry'):
            x      = float(geom.attrib['x'])
            y      = float(geom.attrib['y'])
            hdg    = float(geom.attrib['hdg'])
            length = float(geom.attrib['length'])
            if length <= 0:
                continue
            n   = max(2, int(length / sample_interval))
            arc = geom.find('arc')
            for i in range(n + 1):
                ds = (i / n) * length
                if arc is not None:
                    curv = float(arc.attrib.get('curvature', 0))
                    if abs(curv) > 1e-10:
                        r  = 1.0 / curv
                        cx = x - r * math.sin(hdg)
                        cy = y + r * math.cos(hdg)
                        a  = hdg - math.pi / 2 + ds * curv
                        px, py = cx + r * math.cos(a), cy + r * math.sin(a)
                    else:
                        px, py = x + ds * math.cos(hdg), y + ds * math.sin(hdg)
                else:
                    px, py = x + ds * math.cos(hdg), y + ds * math.sin(hdg)
                # Y-flip: CARLA Y increases south, xodr Y increases north
                (junc_pts if is_junc else road_pts).append((px, -py))
    return road_pts, junc_pts


# ── Drawing helpers ───────────────────────────────────────────────────────────

def draw_network(ax, road_pts, junc_pts, pt_size=6):
    rx, ry = zip(*road_pts)
    jx, jy = zip(*junc_pts)
    ax.scatter(rx, ry, s=pt_size,       c='#4a9eff', alpha=0.55, linewidths=0, rasterized=True)
    ax.scatter(jx, jy, s=pt_size * 0.7, c='#ff9f43', alpha=0.35, linewidths=0, rasterized=True)


def load_vehicles_from_anno(anno_path):
    """Load ego and vehicle positions from anno JSON file."""
    anno_path = Path(anno_path)

    # Handle both .json and .json.gz
    if str(anno_path).endswith('.gz'):
        with gzip.open(anno_path) as f:
            data = json.load(f)
    else:
        with open(anno_path) as f:
            data = json.load(f)

    # Extract ego position
    ego = {
        'x': data.get('x', 0),
        'y': data.get('y', 0),
    }

    # Extract vehicles
    vehicles = []
    for bb in data.get('bounding_boxes', []):
        if bb.get('class') != 'vehicle':
            continue

        loc = bb.get('location', [0, 0, 0])
        vehicles.append({
            'id': str(bb.get('id', '?')),
            'x': loc[0],
            'y': loc[1],
        })

    return ego, vehicles


def draw_vehicles(ax, ego, vehicles, pt_size=80, label_ids=True):
    """Draw ego and vehicles on the map."""
    # Draw ego vehicle
    ax.scatter([ego['x']], [ego['y']], s=pt_size * 2, color='#ff006e',
              marker='*', edgecolors='white', linewidths=1.0, zorder=10)
    if label_ids:
        ax.annotate('EGO', (ego['x'], ego['y']),
                   xytext=(4, 4), textcoords='offset points',
                   fontsize=7, color='#ff006e', zorder=11,
                   fontweight='bold',
                   path_effects=[pe.withStroke(linewidth=1.5, foreground='#0d1117')])

    # Draw other vehicles
    for veh in vehicles:
        ax.scatter([veh['x']], [veh['y']], s=pt_size * 0.6, color='#79c0ff',
                  edgecolors='white', linewidths=0.5, zorder=9)
        if label_ids:
            ax.annotate(f"V{veh['id']}", (veh['x'], veh['y']),
                       xytext=(2, 2), textcoords='offset points',
                       fontsize=5.5, color='#79c0ff', zorder=9.5,
                       path_effects=[pe.withStroke(linewidth=0.8, foreground='#0d1117')])


def draw_waypoints(ax, waypoints, pt_size=60, label_ids=True):
    for i, wp in enumerate(waypoints):
        color = '#ff4757' if wp['is_junction'] else '#2ed573'
        ax.scatter(wp['x'], wp['y'], s=pt_size, color=color,
                   edgecolors='white', linewidths=0.6, zorder=5)
        if label_ids:
            ax.annotate(str(i), (wp['x'], wp['y']),
                        xytext=(4, 4), textcoords='offset points',
                        fontsize=5.5, color=color, zorder=6,
                        path_effects=[pe.withStroke(linewidth=1.2,
                                                    foreground='#0d1117')])

    # Connect with a line to show trajectory order
    xs = [wp['x'] for wp in waypoints]
    ys = [wp['y'] for wp in waypoints]
    ax.plot(xs, ys, color='white', lw=0.8, alpha=0.4, zorder=4)

    # Start / end markers
    ax.scatter(xs[0],  ys[0],  s=pt_size * 3, color='#3fb950', marker='*',
               edgecolors='white', linewidths=0.7, zorder=7)
    ax.scatter(xs[-1], ys[-1], s=pt_size * 2, color='#f85149', marker='X',
               edgecolors='white', linewidths=0.7, zorder=7)


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--json', default='route_waypoints_5m.json')
    p.add_argument('--xodr',
                   default='carlaCache/0.9.15/Carla/Maps/OpenDrive/Town10HD_Opt.xodr')
    p.add_argument('--anno', default=None,
                   help='Anno JSON file to overlay vehicle positions')
    p.add_argument('--interval', type=float, default=1.0)
    p.add_argument('--out',      default='town10_route_overlay.png')
    p.add_argument('--pad',      type=float, default=0.2)
    return p.parse_args()


def main():
    args = parse_args()

    print('Parsing xodr …')
    road_pts, junc_pts = parse_xodr(args.xodr, args.interval)

    print(f'Loading waypoints from {args.json} …')
    with open(args.json) as f:
        waypoints = json.load(f)
    print(f'  {len(waypoints)} waypoints  '
          f'({sum(1 for w in waypoints if w["is_junction"])} junction, '
          f'{sum(1 for w in waypoints if not w["is_junction"])} road)')

    ego, vehicles = None, None
    if args.anno:
        print(f'Loading vehicle positions from {args.anno} …')
        ego, vehicles = load_vehicles_from_anno(args.anno)
        print(f'  Ego at ({ego["x"]:.2f}, {ego["y"]:.2f}), {len(vehicles)} vehicles')

    xs = [w['x'] for w in waypoints]
    ys = [w['y'] for w in waypoints]
    pad_x = (max(xs) - min(xs)) * args.pad
    pad_y = (max(ys) - min(ys)) * args.pad
    zx0, zx1 = min(xs) - pad_x, max(xs) + pad_x
    zy0, zy1 = min(ys) - pad_y, max(ys) + pad_y

    fig = plt.figure(figsize=(26, 13), facecolor='#0d1117')

    # ── Left: full Town10HD map ───────────────────────────────────────────────
    ax_main = fig.add_axes([0.02, 0.05, 0.44, 0.90])
    ax_main.set_facecolor('#0d1117')
    draw_network(ax_main, road_pts, junc_pts, pt_size=3)
    draw_waypoints(ax_main, waypoints, pt_size=25, label_ids=False)
    if ego and vehicles is not None:
        draw_vehicles(ax_main, ego, vehicles, pt_size=25, label_ids=False)

    # Gold rectangle marking the zoom region
    from matplotlib.patches import Rectangle
    ax_main.add_patch(Rectangle((zx0, zy0), zx1 - zx0, zy1 - zy0,
                                 linewidth=1.5, edgecolor='#ffd700',
                                 facecolor='#ffd70012', zorder=8))
    ax_main.set_aspect('equal')
    ax_main.invert_yaxis()
    ax_main.set_title('Town10HD — Full Map', color='#c9d1d9', fontsize=13, pad=10)
    ax_main.set_xlabel('X (m)  →  east',  color='#8b949e', fontsize=10)
    ax_main.set_ylabel('Y (m)  ↓  south', color='#8b949e', fontsize=10)
    ax_main.tick_params(colors='#8b949e', labelsize=8)
    for sp in ax_main.spines.values():
        sp.set_edgecolor('#30363d')
    ax_main.grid(True, alpha=0.08, color='white', lw=0.3)

    # ── Right: zoomed view ────────────────────────────────────────────────────
    ax_zoom = fig.add_axes([0.51, 0.05, 0.47, 0.90])
    ax_zoom.set_facecolor('#0d1117')
    draw_network(ax_zoom, road_pts, junc_pts, pt_size=20)
    draw_waypoints(ax_zoom, waypoints, pt_size=80, label_ids=True)
    if ego and vehicles is not None:
        draw_vehicles(ax_zoom, ego, vehicles, pt_size=80, label_ids=True)

    ax_zoom.set_xlim(zx0, zx1)
    ax_zoom.set_ylim(zy1, zy0)   # inverted Y axis
    ax_zoom.set_aspect('equal')
    ax_zoom.set_title(
        f'Route waypoints — zoomed  ({len(waypoints)} pts)',
        color='#c9d1d9', fontsize=13, pad=10)
    ax_zoom.set_xlabel('X (m)  →  east',  color='#8b949e', fontsize=10)
    ax_zoom.set_ylabel('Y (m)  ↓  south', color='#8b949e', fontsize=10)
    ax_zoom.tick_params(colors='#8b949e', labelsize=9)
    for sp in ax_zoom.spines.values():
        sp.set_edgecolor('#ffd700')
        sp.set_linewidth(1.5)
    ax_zoom.grid(True, alpha=0.1, color='white', lw=0.3)

    # Legend
    legend_els = [
        mpatches.Patch(color='#4a9eff', label='Road network'),
        mpatches.Patch(color='#ff9f43', label='Junction network'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ed573',
               markersize=9, lw=0, label='Waypoint (road)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#ff4757',
               markersize=9, lw=0, label='Waypoint (junction)'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='#3fb950',
               markersize=12, lw=0, label='Start'),
        Line2D([0], [0], marker='X', color='w', markerfacecolor='#f85149',
               markersize=10, lw=0, label='End'),
    ]
    if ego and vehicles is not None:
        legend_els.extend([
            Line2D([0], [0], marker='*', color='w', markerfacecolor='#ff006e',
                   markersize=14, lw=0, label='Ego vehicle'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#79c0ff',
                   markersize=7, lw=0, label='Other vehicles'),
        ])
    ax_zoom.legend(handles=legend_els, loc='upper right', fontsize=8,
                   facecolor='#161b22', edgecolor='#30363d',
                   labelcolor='white', framealpha=0.9)

    # Connector lines between zoom box corners and right panel edges
    from matplotlib.patches import ConnectionPatch
    for (cx, cy), yfrac in [((zx0, zy1), 1.0), ((zx0, zy0), 0.0)]:
        fig.add_artist(ConnectionPatch(
            xyA=(cx, cy), coordsA=ax_main.transData,
            xyB=(0.0, yfrac), coordsB=ax_zoom.transAxes,
            color='#ffd700', lw=0.8, alpha=0.5, linestyle='--'))

    plt.savefig(args.out, dpi=150, bbox_inches='tight', facecolor='#0d1117')
    print(f'Saved → {args.out}')


if __name__ == '__main__':
    main()

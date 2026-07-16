"""
overlay_route_on_map.py

Renders the Town12 road network from the OpenDRIVE file, overlays the
waypoints from routes_devtest.xml as a trajectory, then adds an automatic
zoomed inset panel over the waypoint bounding box.

Usage:
    python3 overlay_route_on_map.py
    python3 overlay_route_on_map.py --xodr carlaCache/0.9.15/Carla/Maps/Town12/OpenDrive/Town12.xodr
    python3 overlay_route_on_map.py --routes leaderboard/data/routes_devtest.xml --route-id 0
    python3 overlay_route_on_map.py --out overlay_map.png
"""

import argparse
import math
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as pe
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset


# ── OpenDRIVE parser (same as plot_town12_map.py) ────────────────────────────

def parse_xodr(filepath, sample_interval=5.0):
    tree = ET.parse(filepath)
    root = tree.getroot()
    road_pts, junc_pts = [], []

    for road in root.findall('road'):
        is_junction = road.attrib.get('junction', '-1') != '-1'
        plan_view   = road.find('planView')
        if plan_view is None:
            continue

        for geom in plan_view.findall('geometry'):
            x      = float(geom.attrib['x'])
            y      = float(geom.attrib['y'])
            hdg    = float(geom.attrib['hdg'])
            length = float(geom.attrib['length'])
            if length <= 0:
                continue

            n = max(2, int(length / sample_interval))
            line, arc, spiral = geom.find('line'), geom.find('arc'), geom.find('spiral')

            for i in range(n + 1):
                ds = (i / n) * length
                if arc is not None:
                    curv = float(arc.attrib.get('curvature', 0))
                    if abs(curv) > 1e-10:
                        r = 1.0 / curv
                        cx = x - r * math.sin(hdg)
                        cy = y + r * math.cos(hdg)
                        a  = hdg - math.pi / 2 + ds * curv
                        px, py = cx + r * math.cos(a), cy + r * math.sin(a)
                    else:
                        px = x + ds * math.cos(hdg)
                        py = y + ds * math.sin(hdg)
                else:
                    px = x + ds * math.cos(hdg)
                    py = y + ds * math.sin(hdg)

                (junc_pts if is_junction else road_pts).append((px, py))

    return road_pts, junc_pts


# ── Route waypoint / trigger loader ──────────────────────────────────────────

def load_waypoints(xml_path, route_id=0):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    for route in root.findall('route'):
        if int(route.attrib.get('id', 0)) == route_id:
            return [(float(p.attrib['x']), float(p.attrib['y']))
                    for p in route.find('waypoints').findall('position')]
    route = root.find('route')
    return [(float(p.attrib['x']), float(p.attrib['y']))
            for p in route.find('waypoints').findall('position')]


def load_triggers(xml_path, route_id=0):
    """Return list of (x, y, scenario_type, scenario_name) for every trigger in the route."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    for route in root.findall('route'):
        if int(route.attrib.get('id', 0)) == route_id:
            triggers = []
            for scen in route.find('scenarios').findall('scenario'):
                tp = scen.find('trigger_point')
                if tp is None:
                    continue
                triggers.append((
                    float(tp.attrib['x']),
                    float(tp.attrib['y']),
                    scen.attrib.get('type', ''),
                    scen.attrib.get('name', ''),
                ))
            return triggers
    return []


# ── Drawing helpers ───────────────────────────────────────────────────────────

def draw_triggers(ax, triggers, ms=90, label=True):
    """Plot trigger points as red diamonds with a short scenario-type label."""
    for tx, ty, stype, sname in triggers:
        ax.scatter(tx, ty, s=ms, color='#ff4757', marker='D', zorder=10,
                   edgecolors='white', linewidths=0.6, alpha=0.92)
        if label:
            short = sname.split('_')[0]          # e.g. "SignalizedJunctionLeftTurn"
            ax.annotate(short, (tx, ty),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=5, color='#ff4757', zorder=11,
                        path_effects=[pe.withStroke(linewidth=1.2,
                                                    foreground='black')])


def draw_map(ax, road_pts, junc_pts, pt_size=0.3):
    rx, ry = zip(*road_pts)
    jx, jy = zip(*junc_pts)
    ax.scatter(rx, ry, s=pt_size, c='#4a9eff', alpha=0.55,
               linewidths=0, rasterized=True)
    ax.scatter(jx, jy, s=pt_size * 0.7, c='#ff9f43', alpha=0.35,
               linewidths=0, rasterized=True)


def draw_route(ax, wps, lw=2.0, ms=60, label_ids=True):
    wx = [p[0] for p in wps]
    wy = [p[1] for p in wps]

    # trajectory line
    ax.plot(wx, wy, color='#00ff88', lw=lw, alpha=0.85, zorder=5,
            path_effects=[pe.withStroke(linewidth=lw + 1.5,
                                        foreground='#003322')])

    # direction arrows between waypoints
    for i in range(len(wps) - 1):
        mx = (wx[i] + wx[i+1]) / 2
        my = (wy[i] + wy[i+1]) / 2
        dx, dy = wx[i+1] - wx[i], wy[i+1] - wy[i]
        ax.annotate('', xy=(mx + dx * 0.01, my + dy * 0.01),
                    xytext=(mx - dx * 0.01, my - dy * 0.01),
                    arrowprops=dict(arrowstyle='->', color='#00ff88',
                                   lw=lw * 0.8, alpha=0.75),
                    zorder=6)

    # waypoint dots
    ax.scatter(wx, wy, s=ms, color='#00ff88', zorder=7,
               edgecolors='white', linewidths=0.5)

    if label_ids:
        for i, (x, y) in enumerate(wps):
            ax.annotate(str(i), (x, y), color='white', fontsize=6,
                        fontweight='bold', zorder=8,
                        xytext=(5, 4), textcoords='offset points',
                        path_effects=[pe.withStroke(linewidth=1.5,
                                                    foreground='black')])

    # start / end markers
    ax.scatter(wx[0],  wy[0],  s=ms * 3, color='#3fb950', zorder=9,
               marker='*', edgecolors='white', linewidths=0.7)
    ax.scatter(wx[-1], wy[-1], s=ms * 2, color='#f85149', zorder=9,
               marker='X', edgecolors='white', linewidths=0.7)


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--xodr',
                   default='carlaCache/0.9.15/Carla/Maps/Town12/OpenDrive/Town12.xodr')
    p.add_argument('--routes',    default='leaderboard/data/routes_devtest_short.xml')
    p.add_argument('--route-id',  type=int, default=0)
    p.add_argument('--out',       default='overlay_map.png')
    p.add_argument('--interval',  type=float, default=5.0,
                   help='xodr sampling interval in metres')
    p.add_argument('--pad',       type=float, default=0.15,
                   help='Fractional padding around waypoints for zoom box')
    return p.parse_args()


def main():
    args = parse_args()

    print('Parsing OpenDRIVE …')
    road_pts_raw, junc_pts_raw = parse_xodr(args.xodr, args.interval)

    # CARLA world coords have Y increasing south; OpenDRIVE has Y increasing north.
    # Transform road network to CARLA projection: carla_x = xodr_x, carla_y = -xodr_y
    road_pts = [(x, -y) for x, y in road_pts_raw]
    junc_pts = [(x, -y) for x, y in junc_pts_raw]

    print('Loading route waypoints and triggers …')
    wps      = load_waypoints(args.routes, args.route_id)
    triggers = load_triggers(args.routes, args.route_id)
    print(f'  {len(wps)} waypoints, {len(triggers)} triggers from route {args.route_id}')

    wx = [p[0] for p in wps]
    wy = [p[1] for p in wps]

    # Bounding box of waypoints + padding for zoom region
    pad_x = (max(wx) - min(wx)) * args.pad
    pad_y = (max(wy) - min(wy)) * args.pad
    zx0, zx1 = min(wx) - pad_x, max(wx) + pad_x
    zy0, zy1 = min(wy) - pad_y, max(wy) + pad_y

    # ── Figure: full map + zoomed inset ──────────────────────────────────────
    fig = plt.figure(figsize=(28, 14), facecolor='#0d1117')

    # Left panel — full town map
    ax_main = fig.add_axes([0.02, 0.05, 0.45, 0.90])
    ax_main.set_facecolor('#0d1117')
    draw_map(ax_main, road_pts, junc_pts, pt_size=0.3)
    draw_route(ax_main, wps, lw=1.5, ms=30, label_ids=False)
    draw_triggers(ax_main, triggers, ms=20, label=False)

    # Highlight the zoom rectangle on the full map
    rect = patches.Rectangle(
        (zx0, zy0), zx1 - zx0, zy1 - zy0,
        linewidth=1.5, edgecolor='#ffd700', facecolor='#ffd70015', zorder=10)
    ax_main.add_patch(rect)

    ax_main.set_aspect('equal')
    ax_main.invert_yaxis()   # CARLA: Y increases southward → larger Y = lower on screen
    ax_main.set_title('Town12 — Full Map (CARLA coords)', color='#c9d1d9', fontsize=14, pad=10)
    ax_main.set_xlabel('X (m)  →  east',  color='#8b949e', fontsize=10)
    ax_main.set_ylabel('Y (m)  ↓  south', color='#8b949e', fontsize=10)
    ax_main.tick_params(colors='#8b949e', labelsize=8)
    for sp in ax_main.spines.values():
        sp.set_edgecolor('#30363d')
    ax_main.grid(True, alpha=0.08, color='white', lw=0.4)

    # Right panel — zoomed view
    ax_zoom = fig.add_axes([0.52, 0.05, 0.46, 0.90])
    ax_zoom.set_facecolor('#0d1117')
    draw_map(ax_zoom, road_pts, junc_pts, pt_size=2.0)
    draw_route(ax_zoom, wps, lw=2.5, ms=80, label_ids=True)
    draw_triggers(ax_zoom, triggers, ms=90, label=True)

    ax_zoom.set_xlim(zx0, zx1)
    ax_zoom.set_ylim(zy1, zy0)   # inverted: zy1 > zy0, so south is down
    ax_zoom.set_aspect('equal')
    ax_zoom.set_title(
        f'Route {args.route_id} — Zoomed  ({len(wps)} waypoints)',
        color='#c9d1d9', fontsize=14, pad=10)
    ax_zoom.set_xlabel('X (m)  →  east',  color='#8b949e', fontsize=10)
    ax_zoom.set_ylabel('Y (m)  ↓  south', color='#8b949e', fontsize=10)
    ax_zoom.tick_params(colors='#8b949e', labelsize=9)
    for sp in ax_zoom.spines.values():
        sp.set_edgecolor('#ffd700')
        sp.set_linewidth(1.5)
    ax_zoom.grid(True, alpha=0.12, color='white', lw=0.4)

    # Legend
    from matplotlib.lines import Line2D
    legend_els = [
        Line2D([0], [0], color='#4a9eff',  lw=2, label='Roads'),
        Line2D([0], [0], color='#ff9f43',  lw=2, label='Junctions'),
        Line2D([0], [0], color='#00ff88',  lw=2, label='Route trajectory'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='#3fb950',
               markersize=12, lw=0, label='Start'),
        Line2D([0], [0], marker='X', color='w', markerfacecolor='#f85149',
               markersize=10, lw=0, label='End'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='#ff4757',
               markersize=9,  lw=0, label='Trigger points'),
    ]
    ax_zoom.legend(handles=legend_els, loc='upper right', fontsize=9,
                   facecolor='#161b22', edgecolor='#30363d',
                   labelcolor='white', framealpha=0.9)

    # Connector lines between full map zoom box and right panel border
    from matplotlib.patches import ConnectionPatch
    for (corner_x, corner_y), (side, y_frac) in [
        ((zx0, zy1), ('right', 1.0)),
        ((zx0, zy0), ('right', 0.0)),
    ]:
        con = ConnectionPatch(
            xyA=(corner_x, corner_y), coordsA=ax_main.transData,
            xyB=(0.0, y_frac),        coordsB=ax_zoom.transAxes,
            color='#ffd700', lw=1.0, alpha=0.6, linestyle='--', zorder=20)
        fig.add_artist(con)

    plt.savefig(args.out, dpi=150, bbox_inches='tight', facecolor='#0d1117')
    print(f'Saved → {args.out}')


if __name__ == '__main__':
    main()

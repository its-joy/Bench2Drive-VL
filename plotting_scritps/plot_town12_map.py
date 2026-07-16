"""
plot_town12_map.py

Parses a CARLA OpenDRIVE (.xodr) file and plots the full road network.
Optionally saves road geometry to JSON for interactive waypoint picking.

Usage:
    python3 plot_town12_map.py --xodr Town12.xodr
    python3 plot_town12_map.py --xodr Town12.xodr --interval 5
    python3 plot_town12_map.py --xodr Town12.xodr --interactive --interval 2
    python3 plot_town12_map.py --xodr Town12.xodr --save-json --interactive --interval 1
"""

import sys
import argparse

# ── Set matplotlib backend before pyplot is imported anywhere ─────────────────
_interactive_mode = '--interactive' in sys.argv
import matplotlib
if _interactive_mode:
    for _backend in ('Qt5Agg', 'Qt6Agg', 'GTK4Agg', 'GTK3Agg', 'TkAgg', 'wxAgg'):
        try:
            matplotlib.use(_backend)
            import matplotlib.pyplot as plt
            plt.figure()
            plt.close()
            print(f'Using backend: {_backend}')
            break
        except Exception:
            continue
    else:
        print('No interactive backend found. Install python3-tk or python3-pyqt5.')
        print('  sudo apt install python3-tk')
        sys.exit(1)
else:
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
# ─────────────────────────────────────────────────────────────────────────────

import xml.etree.ElementTree as ET
import math
import json

# ── Config ────────────────────────────────────────────────────────────────────
# Lower interval = denser points. Use 1.0 for small towns (Town10), 5.0 for large (Town12).
SAMPLE_INTERVAL =  0.1
OUTPUT_PNG      = 'carla_map.png'
OUTPUT_JSON     = 'carla_road_pts.json'
# ─────────────────────────────────────────────────────────────────────────────


def parse_xodr(filepath, sample_interval=0.1):
    """
    Parse OpenDRIVE file and extract road centerline geometry.
    Handles line, arc, and spiral (Euler spiral / clothoid) geometries.
    Returns two lists of (x, y) tuples: road points and junction points.
    """
    print(f"Parsing {filepath} ...")
    tree = ET.parse(filepath)
    root = tree.getroot()

    roads = root.findall('road')
    print(f"  Roads found: {len(roads)}")

    road_pts       = []
    junc_pts       = []
    junc_endpoints = []   # start + end of every junction road → blue connection dots

    for road in roads:
        junction    = road.attrib.get('junction', '-1')
        is_junction = junction != '-1'

        plan_view = road.find('planView')
        if plan_view is None:
            continue

        geoms = plan_view.findall('geometry')
        first_pt = last_pt = None

        for geom in geoms:
            x      = float(geom.attrib['x'])
            y      = float(geom.attrib['y'])
            hdg    = float(geom.attrib['hdg'])
            length = float(geom.attrib['length'])

            if length <= 0:
                continue

            n_samples = max(2, int(length / sample_interval))

            arc    = geom.find('arc')
            spiral = geom.find('spiral')

            for i in range(n_samples + 1):
                ds = (i / n_samples) * length

                if arc is not None:
                    curvature = float(arc.attrib.get('curvature', 0))
                    if abs(curvature) > 1e-10:
                        radius = 1.0 / curvature
                        dtheta = ds * curvature
                        cx    = x - radius * math.sin(hdg)
                        cy    = y + radius * math.cos(hdg)
                        angle = hdg - math.pi / 2 + dtheta
                        px = cx + radius * math.cos(angle)
                        py = cy + radius * math.sin(angle)
                    else:
                        px = x + ds * math.cos(hdg)
                        py = y + ds * math.sin(hdg)
                elif spiral is not None:
                    curv_start = float(spiral.attrib.get('curvStart', 0))
                    curv_end   = float(spiral.attrib.get('curvEnd',   0))
                    n_steps    = max(10, int(ds / 1.0))
                    px, py     = x, y
                    heading    = hdg
                    step       = ds / n_steps
                    for k in range(n_steps):
                        s_k  = (i / n_samples) * length * (k / n_steps)
                        curv = curv_start + (curv_end - curv_start) * s_k / length
                        heading += curv * step
                        px += step * math.cos(heading)
                        py += step * math.sin(heading)
                else:
                    px = x + ds * math.cos(hdg)
                    py = y + ds * math.sin(hdg)

                if is_junction:
                    junc_pts.append((px, py))
                else:
                    road_pts.append((px, py))

                if i == 0 and first_pt is None:
                    first_pt = (px, py)
                last_pt = (px, py)

        if is_junction and first_pt and last_pt:
            junc_endpoints.append(first_pt)
            junc_endpoints.append(last_pt)

    print(f"  Road points:          {len(road_pts)}")
    print(f"  Junction points:      {len(junc_pts)}")
    print(f"  Junction endpoints:   {len(junc_endpoints)}")
    return road_pts, junc_pts, junc_endpoints


def plot_map(road_pts, junc_pts, output_png,
            junc_endpoints=None, interactive=False, clicked_cb=None, dot_size=10):
    import matplotlib.patches as mpatches

    rx = [p[0] for p in road_pts]
    ry = [p[1] for p in road_pts]
    jx = [p[0] for p in junc_pts]
    jy = [p[1] for p in junc_pts]

    print(f"\nMap bounds:")
    print(f"  X: {min(rx):.0f} to {max(rx):.0f}  ({max(rx)-min(rx):.0f} m wide)")
    print(f"  Y: {min(ry):.0f} to {max(ry):.0f}  ({max(ry)-min(ry):.0f} m tall)")

    # 12×12 fits on screen without Tk downscaling; 24×24 causes Tk to shrink
    # the canvas to ~80%, making every dot appear much smaller than s= suggests.
    figsize = (12, 12) if interactive else (24, 24)
    fig, ax = plt.subplots(figsize=figsize, facecolor='#1a1a2e')
    ax.set_facecolor('#1a1a2e')

    # rasterized=True bitmaps the scatter at figure resolution then Tk scales
    # that bitmap — so changing s= has no visual effect in the live window.
    raster = not interactive

    ax.scatter(rx, ry, s=dot_size,        c='#4a9eff', alpha=0.6,
               linewidths=0, rasterized=raster, label='Roads')
    ax.scatter(jx, jy, s=dot_size * 0.7,  c='#ff9f43', alpha=0.4,
               linewidths=0, rasterized=raster, label='Junctions')

    # Blue dots at junction–road connection points
    if junc_endpoints:
        ex = [p[0] for p in junc_endpoints]
        ey = [p[1] for p in junc_endpoints]
        ax.scatter(ex, ey, s=dot_size * 2, c='#4a9eff', alpha=1.0, zorder=5,
                   linewidths=0, rasterized=raster, label='Junction connections')

    ax.set_aspect('equal')
    ax.invert_yaxis()   # CARLA: Y increases southward, so larger Y → lower on screen
    import os
    map_name = os.path.splitext(os.path.basename(output_png))[0].replace('_map', '')
    ax.set_title(f'{map_name} — Road Network (CARLA coords, Y↓=south)',
                 color='white', fontsize=20, pad=20)
    ax.set_xlabel('X (m)  →  east', color='#aaaaaa', fontsize=12)
    ax.set_ylabel('Y (m)  ↓  south', color='#aaaaaa', fontsize=12)
    ax.tick_params(colors='#aaaaaa')
    for spine in ax.spines.values():
        spine.set_edgecolor('#333355')

    from matplotlib.lines import Line2D
    legend_handles = [
        mpatches.Patch(color='#4a9eff', label='Roads'),
        mpatches.Patch(color='#ff9f43', label='Junctions'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#4a9eff',
               markersize=8, lw=0, label='Junction connections'),
    ]
    ax.legend(handles=legend_handles, loc='upper right',
              facecolor='#2a2a4e', labelcolor='white', fontsize=12)
    ax.grid(True, alpha=0.1, color='white', linewidth=0.5)

    # ── Interactive waypoint picker ──────────────────────────────────────────
    # Each entry: {'x': float, 'y': float, 'artists': [marker_line, annot]}
    clicked = []

    def _redraw_labels():
        """Refresh index labels after any add/remove so numbers stay sequential."""
        for i, wp in enumerate(clicked):
            wp['artists'][1].set_text(str(i + 1))
        fig.canvas.draw()

    def _add_waypoint(x, y):
        n = len(clicked) + 1
        (marker,) = ax.plot(x, y, 'o', color='#ff4757', markersize=4, zorder=5)
        annot = ax.annotate(str(n), (x, y), color='#ff4757',
                            fontsize=9, fontweight='bold',
                            xytext=(5, 5), textcoords='offset points',
                            zorder=6)
        clicked.append({'x': x, 'y': y, 'artists': [marker, annot]})
        fig.canvas.draw()
        print(f'  [{n}] <position x="{x:.2f}" y="{y:.2f}" z="0.0"/>')

    def _remove_nearest(x, y):
        if not clicked:
            return
        # Tolerance = 2% of the current view width so it works at any zoom level
        xlim = ax.get_xlim()
        tol = abs(xlim[1] - xlim[0]) * 0.02
        best_idx, best_dist = None, float('inf')
        for i, wp in enumerate(clicked):
            d = ((wp['x'] - x) ** 2 + (wp['y'] - y) ** 2) ** 0.5
            if d < tol and d < best_dist:
                best_dist, best_idx = d, i
        if best_idx is None:
            return
        # Remove artists from the axes
        for artist in clicked[best_idx]['artists']:
            artist.remove()
        removed = clicked.pop(best_idx)
        print(f'  Removed waypoint at ({removed["x"]:.2f}, {removed["y"]:.2f})')
        _redraw_labels()

    if interactive:
        print("\nInteractive mode:")
        print("  LEFT-CLICK        — place waypoint")
        print("  RIGHT-CLICK       — remove nearest waypoint")
        print("  Close the window when done.\n")

        def onclick(event):
            if fig.canvas.toolbar.mode != '' or event.xdata is None:
                return
            if event.button == 1:
                _add_waypoint(event.xdata, event.ydata)
            elif event.button == 3:
                _remove_nearest(event.xdata, event.ydata)

        fig.canvas.mpl_connect('button_press_event', onclick)

    plt.tight_layout()

    if not interactive:
        plt.savefig(output_png, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
        print(f"\nMap saved to: {output_png}")
    else:
        plt.show()
        if clicked:
            print(f"\n{'='*50}")
            print(f"Collected {len(clicked)} waypoints:")
            print("<waypoints>")
            for wp in clicked:
                print('   <position x="{:.2f}" y="{:.2f}" z="0.0"/>'.format(wp['x'], wp['y']))
            print("</waypoints>")
            print("\nNote: z=0.0 is approximate. CARLA's project_to_road=True")
            print("will snap these to the correct road elevation automatically.")


def save_json(road_pts, junc_pts, output_json):
    data = {
        'roads':     road_pts,
        'junctions': junc_pts
    }
    with open(output_json, 'w') as f:
        json.dump(data, f)
    print(f"Road geometry saved to: {output_json}")
    print(f"  Load with: import json; data = json.load(open('{output_json}'))")


def main():
    import os
    parser = argparse.ArgumentParser(description='Plot CARLA OpenDRIVE map')
    parser.add_argument('--xodr',        required=True,  help='Path to .xodr file')
    parser.add_argument('--output',      default=None,
                        help='Output PNG (default: <map_name>_map.png)')
    parser.add_argument('--save-json',   action='store_true', help='Save road geometry to JSON')
    parser.add_argument('--json-output', default=None,
                        help='Output JSON (default: <map_name>_road_pts.json)')
    parser.add_argument('--interactive', action='store_true',
                        help='Show interactive map for clicking waypoints (requires display). '
                             'Use with --interval to control map density.')
    parser.add_argument('--interval',    type=float, default=SAMPLE_INTERVAL,
                        help=f'Sampling interval in meters along each road — '
                             f'smaller = denser (e.g. 1.0 for Town10, 5.0 for Town12). '
                             f'Default: {SAMPLE_INTERVAL}')
    parser.add_argument('--dot-size',   type=float, default=None,
                        help='Scatter marker size (s= in matplotlib). '
                             'Defaults: 50 in interactive mode, 10 for PNG.')
    args = parser.parse_args()

    # Auto-derive output names from the xodr filename
    map_name = os.path.splitext(os.path.basename(args.xodr))[0]
    output_png  = args.output      or f'{map_name}_map.png'
    output_json = args.json_output or f'{map_name}_road_pts.json'

    road_pts_raw, junc_pts_raw, junc_ep_raw = parse_xodr(args.xodr, args.interval)

    # Transform to CARLA world projection: carla_x = xodr_x, carla_y = -xodr_y
    road_pts       = [(x, -y) for x, y in road_pts_raw]
    junc_pts       = [(x, -y) for x, y in junc_pts_raw]
    junc_endpoints = [(x, -y) for x, y in junc_ep_raw]

    if args.save_json:
        save_json(road_pts, junc_pts, output_json)

    dot_size = args.dot_size or (50 if args.interactive else 10)
    print(f'  Dot size (s=): {dot_size}')
    plot_map(road_pts, junc_pts, output_png,
             junc_endpoints=junc_endpoints,
             interactive=args.interactive,
             dot_size=dot_size)


if __name__ == '__main__':
    main()

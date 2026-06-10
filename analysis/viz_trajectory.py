"""
viz_trajectory.py

Plots the route waypoints as a trajectory and overlays all trigger points,
colour-coded by scenario type.

Usage:
    python3 viz_trajectory.py
    python3 viz_trajectory.py --points points.txt --triggers triggers.csv --out traj_map.png
"""

import argparse
import csv
import math
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import numpy as np
from matplotlib.lines import Line2D


# ── colour palette (auto-extended with tab20 if more than 20 types) ──────────
_PALETTE = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
    "#911eb4", "#42d4f4", "#f032e6", "#bfef45", "#fabed4",
    "#469990", "#dcbeff", "#9a6324", "#fffac8", "#800000",
    "#aaffc3", "#808000", "#ffd8b1", "#000075", "#a9a9a9",
]


def load_waypoints(path):
    pts = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            pts.append((float(parts[0]), float(parts[1])))
    return pts


def load_triggers(path):
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--points',   default='points.txt')
    p.add_argument('--triggers', default='triggers.csv')
    p.add_argument('--out',      default='traj_map.png')
    p.add_argument('--dpi',      type=int, default=180)
    return p.parse_args()


def main():
    args = parse_args()

    waypoints = load_waypoints(args.points)
    triggers  = load_triggers(args.triggers)

    xs = [p[0] for p in waypoints]
    ys = [p[1] for p in waypoints]

    # CARLA: +Y = south → flip Y for intuitive top-down view
    ys_flip = [-y for y in ys]

    # Collect unique scenario types and assign colours
    types = sorted({t['scenario_type'] for t in triggers})
    cmap  = plt.cm.get_cmap('tab20', len(types))
    type_colour = {t: _PALETTE[i] if i < len(_PALETTE) else cmap(i)
                   for i, t in enumerate(types)}

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(18, 14))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#161b22')

    # ── Trajectory line ───────────────────────────────────────────────────────
    ax.plot(xs, ys_flip, color='#58a6ff', linewidth=1.8, alpha=0.55,
            zorder=2, label='_nolegend_')

    # Arrows along the trajectory
    for i in range(len(xs) - 1):
        mx, my = (xs[i] + xs[i+1]) / 2, (ys_flip[i] + ys_flip[i+1]) / 2
        dx, dy = xs[i+1] - xs[i], ys_flip[i+1] - ys_flip[i]
        ax.annotate('', xy=(mx + dx * 0.01, my + dy * 0.01),
                    xytext=(mx - dx * 0.01, my - dy * 0.01),
                    arrowprops=dict(arrowstyle='->', color='#58a6ff',
                                   lw=1.2, alpha=0.6),
                    zorder=3)

    # ── Waypoint dots ─────────────────────────────────────────────────────────
    ax.scatter(xs, ys_flip, color='#58a6ff', s=40, zorder=4,
               edgecolors='white', linewidths=0.4, alpha=0.8)
    for i, (x, y) in enumerate(zip(xs, ys_flip)):
        ax.annotate(str(i), (x, y), textcoords='offset points',
                    xytext=(5, 4), fontsize=6.5, color='#8b949e', zorder=5)

    # Start / end markers
    ax.scatter(xs[0],  ys_flip[0],  color='#3fb950', s=220, zorder=6,
               marker='*', edgecolors='white', linewidths=0.8)
    ax.scatter(xs[-1], ys_flip[-1], color='#f85149', s=220, zorder=6,
               marker='X', edgecolors='white', linewidths=0.8)

    # ── Trigger points ────────────────────────────────────────────────────────
    for t in triggers:
        tx   = float(t['x'])
        ty_f = -float(t['y'])   # flip Y
        col  = type_colour[t['scenario_type']]
        rid  = t['route_id']

        ax.scatter(tx, ty_f, color=col, s=80, zorder=7,
                   edgecolors='white', linewidths=0.6, marker='D', alpha=0.9)
        ax.annotate(rid, (tx, ty_f), textcoords='offset points',
                    xytext=(4, 4), fontsize=5.5, color=col, zorder=8,
                    path_effects=[pe.withStroke(linewidth=1.5,
                                                foreground='#0d1117')])

    # ── Legend ────────────────────────────────────────────────────────────────
    wp_handle   = Line2D([0], [0], color='#58a6ff', lw=2, label='Waypoint trajectory')
    start_h     = Line2D([0], [0], marker='*', color='w', markerfacecolor='#3fb950',
                         markersize=11, label='Start', lw=0)
    end_h       = Line2D([0], [0], marker='X', color='w', markerfacecolor='#f85149',
                         markersize=10, label='End', lw=0)
    trig_handles = [
        Line2D([0], [0], marker='D', color='w',
               markerfacecolor=type_colour[t], markersize=8, lw=0, label=t)
        for t in types
    ]

    legend = ax.legend(
        handles=[wp_handle, start_h, end_h] + trig_handles,
        loc='upper left', fontsize=7, ncol=2,
        facecolor='#21262d', edgecolor='#30363d', labelcolor='white',
        framealpha=0.92, borderpad=0.8, handletextpad=0.6,
    )

    # ── Labels / styling ──────────────────────────────────────────────────────
    ax.set_xlabel('X (m)', color='#c9d1d9', fontsize=10)
    ax.set_ylabel('Y (m, north-up)', color='#c9d1d9', fontsize=10)
    ax.set_title(
        f'Route Trajectory with Trigger Points  ·  {len(waypoints)} waypoints  ·  {len(triggers)} triggers',
        color='#c9d1d9', fontsize=13, pad=14,
    )
    ax.tick_params(colors='#8b949e')
    for spine in ax.spines.values():
        spine.set_edgecolor('#30363d')
    ax.grid(True, alpha=0.12, color='#c9d1d9', linestyle='--')
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(args.out, dpi=args.dpi, bbox_inches='tight',
                facecolor='#0d1117')
    print(f'Saved → {args.out}')


if __name__ == '__main__':
    main()

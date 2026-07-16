#!/usr/bin/env python3
"""
Simple script to plot ego and agent positions from an anno JSON file.
Plots 2D top-down view with ego at origin.

Usage:
    python3 plot_agent_positions.py /path/to/anno/00425.json.gz
"""

import json
import sys
import gzip
from pathlib import Path
import matplotlib.pyplot as plt

def plot_positions(json_path):
    """Plot ego and agent positions from anno file."""
    json_path = Path(json_path)

    # Handle both .json and .json.gz
    if str(json_path).endswith('.gz'):
        with gzip.open(json_path) as f:
            data = json.load(f)
    else:
        with open(json_path) as f:
            data = json.load(f)

    # Extract ego position from top-level x, y (world coordinates)
    ego_x = data.get('x', 0)
    ego_y = data.get('y', 0)

    # Extract agent positions from bounding boxes
    agents = []
    for bb in data.get('bounding_boxes', []):
        cls = bb.get('class', '')
        if cls != 'vehicle':
            continue

        loc = bb.get('location', [0, 0, 0])
        agent_x, agent_y = loc[0], loc[1]
        agent_id = str(bb.get('id', '?'))
        agents.append({
            'x': agent_x,
            'y': agent_y,
            'id': agent_id,
        })

    # Plot
    fig, ax = plt.subplots(figsize=(12, 12))

    # Plot ego
    ax.scatter([ego_x], [ego_y], color='red', s=300, marker='*', label='Ego', zorder=5)
    ax.text(ego_x, ego_y - 3, 'EGO', fontsize=12, color='red', ha='center', fontweight='bold')

    # Plot agents with ID labels
    for agent in agents:
        ax.scatter([agent['x']], [agent['y']], color='blue', s=100, alpha=0.7, edgecolors='black', linewidth=0.5)
        ax.text(agent['x'], agent['y'] - 2, f"ID{agent['id']}", fontsize=9, ha='center')

    ax.set_xlabel('X (world coords)', fontsize=12)
    ax.set_ylabel('Y (world coords)', fontsize=12)
    ax.set_title(f'Vehicle Positions - {Path(json_path).stem}', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.axis('equal')

    plt.tight_layout()

    # Save
    out_path = Path(json_path).parent / f"{Path(json_path).stem}_positions.png"
    plt.savefig(out_path, dpi=100, bbox_inches='tight')
    print(f"✓ Saved plot to {out_path}")

    # Print summary
    print(f"\nFrame: {Path(json_path).stem}")
    print(f"Ego: ({ego_x:7.2f}, {ego_y:7.2f})")
    print(f"Vehicles ({len(agents)}):")
    for agent in agents:
        print(f"  ID {agent['id']:3s} | pos=({agent['x']:8.2f}, {agent['y']:8.2f})")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 plot_agent_positions.py /path/to/measurement.json")
        sys.exit(1)

    json_path = Path(sys.argv[1])
    if not json_path.exists():
        print(f"✗ File not found: {json_path}")
        sys.exit(1)

    plot_positions(json_path)

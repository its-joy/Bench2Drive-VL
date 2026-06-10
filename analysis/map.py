import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

positions = [
    (983.5, 5382.2, 371),
    (824.2, 5575.3, 371),
    (605.5, 5575.5, 370),
    (497.5, 5718.3, 368),
    (516.3, 5897.4, 364),
    (355.1, 6138.1, 358),
    (285.9, 6127.4, 358),
    (213.5, 6100.5, 358),
    (-221.6, 6100.7, 358),
    (-264.6, 6019.4, 360),
    (-297.9, 5816.4, 366),
    (-421.5, 5871.9, 367),
    (-374.0, 6012.7, 361),
    (-453.1, 6101.2, 360),
    (-536.9, 6071.1, 363),
    (-537.4, 6000.8, 365),
    (-620.9, 6001.5, 367),
    (-784.0, 6185.4, 365),
    (-817.1, 6063.3, 369),
    (-817.8, 5685.4, 375),
    (-819.3, 4835.2, 372),
    (-529.7, 4730.9, 375),
    (-497.3, 4777.9, 375),
    (-497.3, 4884.0, 375),
    (-496.1, 5014.6, 375),
    (-414.4, 5056.2, 375),
    (-222.5, 4944.1, 374),
    (-223.4, 4834.8, 374),
    (-586.4, 4834.7, 375),
    (-732.7, 4962.2, 376),
    (-705.4, 5204.9, 376),
    (-269.4, 5151.9, 374),
    (-218.7, 5271.0, 373),
    (-163.9, 5404.6, 372),
    (75.0, 5587.0, 367.2),
]

xs = [p[0] for p in positions]
ys = [p[1] for p in positions]
zs = [p[2] for p in positions]

fig, axes = plt.subplots(1, 2, figsize=(18, 8))
fig.patch.set_facecolor('#1a1a2e')

# ── Left: XY top-down route map ─────────────────────
ax1 = axes[0]
ax1.set_facecolor('#16213e')

ax1.plot(xs, ys, color='#4a9eff', linewidth=1.5, alpha=0.5, zorder=1)

for i in range(len(xs) - 1):
    ax1.annotate('', xy=(xs[i+1], ys[i+1]), xytext=(xs[i], ys[i]),
                arrowprops=dict(arrowstyle='->', color='#4a9eff', lw=1.2, alpha=0.6))

scatter = ax1.scatter(xs, ys, c=range(len(xs)), cmap='plasma',
                      s=80, zorder=3, edgecolors='white', linewidths=0.5)

ax1.scatter(xs[0], ys[0], c='#00ff88', s=200, zorder=4,
            edgecolors='white', linewidths=1.5, marker='*', label='Start')
ax1.scatter(xs[-1], ys[-1], c='#ff4444', s=200, zorder=4,
            edgecolors='white', linewidths=1.5, marker='X', label='End')

for i, (x, y) in enumerate(zip(xs, ys)):
    ax1.annotate(f'{i}', (x, y), textcoords='offset points',
                xytext=(6, 4), fontsize=7, color='white', alpha=0.8)

ax1.set_xlabel('X (m)', color='white')
ax1.set_ylabel('Y (m)', color='white')
ax1.set_title('Route — Top-Down View (XY)', color='white', fontsize=13, pad=12)
ax1.tick_params(colors='white')
ax1.spines[:].set_color('#444')
ax1.grid(True, alpha=0.15, color='white')
ax1.legend(facecolor='#1a1a2e', edgecolor='#444', labelcolor='white', fontsize=9)
plt.colorbar(scatter, ax=ax1, label='Waypoint order').ax.yaxis.label.set_color('white')

# ── Right: Elevation profile ────────────────────────
ax2 = axes[1]
ax2.set_facecolor('#16213e')

dists = [0]
for i in range(1, len(positions)):
    dx = xs[i] - xs[i-1]
    dy = ys[i] - ys[i-1]
    dists.append(dists[-1] + np.sqrt(dx**2 + dy**2))

ax2.fill_between(dists, zs, min(zs) - 1, alpha=0.3, color='#4a9eff')
ax2.plot(dists, zs, color='#4a9eff', linewidth=2)
ax2.scatter(dists, zs, c=range(len(dists)), cmap='plasma',
            s=60, zorder=3, edgecolors='white', linewidths=0.5)

z_range = max(zs) - min(zs)
ax2.annotate(f'Δz = {z_range:.1f}m', xy=(dists[zs.index(max(zs))], max(zs)),
            xytext=(20, 10), textcoords='offset points',
            color='#00ff88', fontsize=9,
            arrowprops=dict(arrowstyle='->', color='#00ff88'))

ax2.set_xlabel('Distance along route (m)', color='white')
ax2.set_ylabel('Elevation z (m)', color='white')
ax2.set_title('Elevation Profile', color='white', fontsize=13, pad=12)
ax2.tick_params(colors='white')
ax2.spines[:].set_color('#444')
ax2.grid(True, alpha=0.15, color='white')

total_dist = dists[-1]
fig.suptitle(
    f'CARLA Route  ·  {len(positions)} waypoints  ·  total distance ≈ {total_dist:.0f}m  ·  elevation range {z_range:.1f}m',
    color='white', fontsize=11, y=0.02
)

plt.tight_layout(rect=[0, 0.04, 1, 1])
plt.savefig('/data2/joy/Research/Bench2Drive-VL/route_map.png', 
            dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
print("Saved to route_map.png")
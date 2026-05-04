"""
Run the swarm simulation in headless mode and plot color-coded drone
trajectories from the three orthogonal perspectives (XY top, XZ front,
YZ side). Output: three PNG files in the project `doc/` folder.

This script intentionally mirrors the headless simulation loop in `main.py`
so that no matplotlib animation runs concurrently with the physics (the
interactive `plt.pause` calls in the visual path were suspected to interfere
with the simulator timing, biasing drones beneath the target volume).
"""
import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from simulator import KDTree
from drone import Drone

# Simulation parameters (match main.py)
NUM_DRONES = 5
NUM_NEIGHBORS = 3
m = 1.0
g = 9.81
Ix = 0.02
Iy = 0.02
Iz = 0.04
dt_fixed = 0.01
T_traj = 10.0
SIM_DURATION = 60.0

DOC_DIR = "/Users/mmccall/Library/CloudStorage/OneDrive-Personal/Documents/_RPI/Spring 2026/Robotics II/Robotics2_final_project/doc"


def run_simulation():
    goal_region = {
        "center": np.array([1.0, 1.0, 1.5]),
        "extents": np.array([0.5, 0.5, 0.5]),
    }

    drones = []
    b_spiral = 0.5
    for i in range(NUM_DRONES):
        theta = 2.0 * np.sqrt(i)
        r = b_spiral * theta
        x0 = r * np.cos(theta)
        y0 = r * np.sin(theta)
        z0 = 0.0
        drones.append(Drone(f"Drone{i+1}", np.array([x0, y0, z0]),
                            goal_region, i, NUM_DRONES))

    n_steps = int(SIM_DURATION / dt_fixed)
    # Record every drone's position at every physics step (100 Hz).
    trajectories = [[] for _ in drones]
    times = []

    start = time.time()
    for k in range(n_steps):
        t = k * dt_fixed
        all_pos = np.array([d.x[0:3] for d in drones])
        kdtree = KDTree(all_pos)
        for i, drone in enumerate(drones):
            indices = kdtree.query(drone.x[0:3], NUM_NEIGHBORS + 1)
            neighbor_positions = [all_pos[idx] for idx in indices if idx != i]
            drone.update(t, dt_fixed, m, g, Ix, Iy, Iz, T_traj,
                         neighbor_positions[:NUM_NEIGHBORS])
            trajectories[i].append(drone.x[0:3].copy())
        times.append(t)

    print(f"Simulated {SIM_DURATION:.1f}s in {time.time()-start:.2f}s wall time")
    return [np.array(tr) for tr in trajectories], goal_region, drones


def draw_box_2d(ax, center, extents, axes):
    """Draw the goal box projected onto a 2-axis plane (axes=(i,j))."""
    i, j = axes
    cx, cy = center[i], center[j]
    dx, dy = extents[i], extents[j]
    rect_x = [cx - dx, cx + dx, cx + dx, cx - dx, cx - dx]
    rect_y = [cy - dy, cy - dy, cy + dy, cy + dy, cy - dy]
    ax.plot(rect_x, rect_y, color="green", linestyle="--",
            linewidth=1.5, label="Goal region")


def plot_view(trajectories, goal_region, drones, axes, labels, filename, title):
    i, j = axes
    fig, ax = plt.subplots(figsize=(8, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(trajectories)))
    for k, tr in enumerate(trajectories):
        ax.plot(tr[:, i], tr[:, j], color=colors[k],
                linewidth=1.2, label=drones[k].name, alpha=0.9)
        # start and end markers
        ax.scatter(tr[0, i], tr[0, j], color=colors[k],
                   marker="o", s=60, edgecolors="black", zorder=5)
        ax.scatter(tr[-1, i], tr[-1, j], color=colors[k],
                   marker="X", s=90, edgecolors="black", zorder=5)
    draw_box_2d(ax, goal_region["center"], goal_region["extents"], axes)
    ax.set_xlabel(f"{labels[0]} (m)")
    ax.set_ylabel(f"{labels[1]} (m)")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    out_path = os.path.join(DOC_DIR, filename)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    os.makedirs(DOC_DIR, exist_ok=True)
    trajectories, goal_region, drones = run_simulation()

    # Three orthogonal views:
    #   Top   (XY): look down the +Z axis
    #   Front (XZ): look down the +Y axis
    #   Side  (YZ): look down the +X axis
    plot_view(trajectories, goal_region, drones,
              axes=(0, 1), labels=("X", "Y"),
              filename="trajectories_top_xy.png",
              title="Drone trajectories - Top view (XY plane, looking -Z)")
    plot_view(trajectories, goal_region, drones,
              axes=(0, 2), labels=("X", "Z"),
              filename="trajectories_front_xz.png",
              title="Drone trajectories - Front view (XZ plane, looking +Y)")
    plot_view(trajectories, goal_region, drones,
              axes=(1, 2), labels=("Y", "Z"),
              filename="trajectories_side_yz.png",
              title="Drone trajectories - Side view (YZ plane, looking +X)")


if __name__ == "__main__":
    main()

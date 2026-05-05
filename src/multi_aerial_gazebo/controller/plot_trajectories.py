"""
Run the swarm simulation in headless mode and plot fixed-time scatter
samples of each drone trajectory from the three orthogonal perspectives
(XY top, XZ front, YZ side). Output: one PDF per drone in the project
`doc/` folder.

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
PLOT_SAMPLE_INTERVAL = 0.25

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


def sample_trajectory_fixed_time(traj, dt, sample_interval):
    step_interval = max(1, int(round(sample_interval / dt)))
    sampled = traj[::step_interval]
    if len(sampled) == 0 or not np.array_equal(sampled[-1], traj[-1]):
        sampled = np.vstack((sampled, traj[-1]))
    return sampled


def plot_view(ax, trajectory, goal_region, axes, labels, title, color):
    i, j = axes
    ax.scatter(
        trajectory[:, i],
        trajectory[:, j],
        color=color,
        s=18,
        alpha=0.85,
        linewidths=0.0,
    )
    ax.scatter(
        trajectory[0, i],
        trajectory[0, j],
        color=color,
        marker="o",
        s=70,
        edgecolors="black",
        zorder=5,
        label="Start",
    )
    ax.scatter(
        trajectory[-1, i],
        trajectory[-1, j],
        color=color,
        marker="X",
        s=95,
        edgecolors="black",
        zorder=5,
        label="End",
    )
    draw_box_2d(ax, goal_region["center"], goal_region["extents"], axes)
    ax.set_xlabel(f"{labels[0]} (m)")
    ax.set_ylabel(f"{labels[1]} (m)")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)


def set_equal_limits(ax, all_points_2d, padding=0.05):
    mins = np.min(all_points_2d, axis=0)
    maxs = np.max(all_points_2d, axis=0)
    center = 0.5 * (mins + maxs)
    span = max(maxs - mins)
    half_range = 0.5 * span + padding
    ax.set_xlim(center[0] - half_range, center[0] + half_range)
    ax.set_ylim(center[1] - half_range, center[1] + half_range)


def plot_drone_views(drone_name, trajectory, goal_region, filename):
    views = (
        ((0, 1), ("X", "Y"), "Top view (XY plane, looking -Z)"),
        ((0, 2), ("X", "Z"), "Front view (XZ plane, looking +Y)"),
        ((1, 2), ("Y", "Z"), "Side view (YZ plane, looking +X)"),
    )
    color = plt.cm.tab10(0)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, (view_axes, labels, title) in zip(axes, views):
        plot_view(ax, trajectory, goal_region, view_axes, labels, title, color)
        projected_traj = trajectory[:, list(view_axes)]
        center = goal_region["center"][list(view_axes)]
        extents = goal_region["extents"][list(view_axes)]
        box_corners = np.array([
            [center[0] - extents[0], center[1] - extents[1]],
            [center[0] - extents[0], center[1] + extents[1]],
            [center[0] + extents[0], center[1] - extents[1]],
            [center[0] + extents[0], center[1] + extents[1]],
        ])
        all_points_2d = np.vstack((projected_traj, box_corners))
        set_equal_limits(ax, all_points_2d)
    fig.suptitle(
        f"{drone_name} trajectory samples every {PLOT_SAMPLE_INTERVAL:.2f} s",
        fontsize=14,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95), w_pad=0.6)
    out_path = os.path.join(DOC_DIR, filename)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    os.makedirs(DOC_DIR, exist_ok=True)
    trajectories, goal_region, drones = run_simulation()
    sampled_trajectories = [
        sample_trajectory_fixed_time(tr, dt_fixed, PLOT_SAMPLE_INTERVAL)
        for tr in trajectories
    ]

    for drone, traj in zip(drones, sampled_trajectories):
        filename = f"{drone.name.lower()}_trajectory_samples.pdf"
        plot_drone_views(drone.name, traj, goal_region, filename)


if __name__ == "__main__":
    main()

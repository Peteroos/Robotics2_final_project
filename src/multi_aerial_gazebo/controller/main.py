import numpy as np
import matplotlib.pyplot as plt
import argparse
import time
from simulator import (
    KDTree
)
from drone import Drone
from renderer import draw_quad, draw_box

# ====================
# Parameters
# ====================
NUM_DRONES = 5 # Number of drones in the simulation
NUM_NEIGHBORS = 3 # Number of nearest neighbors to consider for repulsion
m = 1.0
g = 9.81

# ====================
# Performance Optimization Opportunities for Large N:
# ====================
# 1. Vectorization: Use NumPy broadcasting for repulsion calculations instead of nested loops.
# 2. Spatial Partitioning: Use k-d trees or spatial hashes to limit repulsion to local neighbors.
# 3. Parallelization: Parallelize per-drone update() calls as they are mostly independent.
# 4. JIT/GPU: Use Numba or JAX to compile physics/control logic for large swarms.
# ====================

Ix = 0.02
Iy = 0.02
Iz = 0.04

dt = 0.01
Tsim = 20
N = int(Tsim / dt)


def main():
    parser = argparse.ArgumentParser(description="Quadrotor Simulation")
    parser.add_argument("--headless", action="store_true", help="Run without visualization")
    args = parser.parse_args()

    # ====================
    # Drones initialization (Volume Occupation)
    # ====================
    shared_volume = {'center': np.array([1.0, 1.0, 1.5]), 'extents': np.array([0.5, 0.5, 0.5])}
    
    drones = []
    # Procedural generation on Archimedean spiral: r = b * theta
    # Using theta = 2*sqrt(i) to maintain approx constant spacing along the spiral
    b_spiral = 0.5
    for i in range(NUM_DRONES):
        theta = 2.0 * np.sqrt(i)
        r = b_spiral * theta
        x0 = r * np.cos(theta)
        y0 = r * np.sin(theta)
        drones.append(Drone(f"Drone{i+1}", np.array([x0, y0, 0.0]), [shared_volume]))

    T_traj = 10.0 # Trajectory duration for each segment
    threshold = 0.1 # Distance threshold to advance to next waypoint

    # ====================
    # Simulation setup
    # ====================
    if not args.headless:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        # Set fixed axis limits that include all waypoints and start positions
        all_points = []
        for d in drones:
            all_points.append(d.x[0:3])
            for wp in d.waypoints:
                all_points.append(wp['center'] + wp['extents'])
                all_points.append(wp['center'] - wp['extents'])
        all_points = np.array(all_points)
        
        min_xyz = np.min(all_points, axis=0)
        max_xyz = np.max(all_points, axis=0)
        
        # Calculate center and maximum range to equalize axes
        center = (min_xyz + max_xyz) / 2
        max_range = np.max(max_xyz - min_xyz)
        margin = 0.5
        span = max_range + 2 * margin
        half_span = span / 2
        
        ax.set_xlim([center[0] - half_span, center[0] + half_span])
        ax.set_ylim([center[1] - half_span, center[1] + half_span])
        ax.set_zlim([0, span])
        
        # Equalize axis scaling
        try:
            ax.set_box_aspect((1, 1, 1))
        except AttributeError:
            # Fallback for older matplotlib versions
            pass

        # Draw volume waypoints
        for drone in drones:
            for wp in drone.waypoints:
                draw_box(ax, wp['center'], wp['extents'])

    # ====================
    # Simulation loop
    # ====================
    t = 0.0
    print_interval = 0.1 if args.headless else 1.0 
    last_print_t = -print_interval
    
    if args.headless:
        # Fixed step simulation for headless testing
        dt_fixed = 0.01
        for k in range(N - 1):
            t = k * dt_fixed
            
            # Build KD-tree for efficient nearest neighbor search
            all_pos = np.array([d.x[0:3] for d in drones])
            kdtree = KDTree(all_pos)
            
            for i, drone in enumerate(drones):
                # Query for NUM_NEIGHBORS + 1 (including self)
                indices = kdtree.query(drone.x[0:3], NUM_NEIGHBORS + 1)
                neighbor_positions = [all_pos[idx] for idx in indices if idx != i]
                drone.update(t, dt_fixed, m, g, Ix, Iy, Iz, T_traj, threshold, neighbor_positions[:NUM_NEIGHBORS])

            # Print status
            if t - last_print_t >= print_interval:
                status_str = f"t={t:.2f}"
                for drone in drones:
                    status_str += f" | {drone.name}: x={drone.x[0]:.2f}, y={drone.x[1]:.2f}, z={drone.x[2]:.2f}"
                print(status_str)
                last_print_t = t
    else:
        # Real-time simulation for visual mode
        start_time = time.time()
        last_t = 0.0
        
        while t < Tsim:
            current_real_time = time.time() - start_time
            dt_loop = current_real_time - last_t
            
            # Avoid division by zero or extremely small dt
            if dt_loop < 1e-4:
                continue
            
            # Limit dt to prevent instability
            dt_step = min(dt_loop, 0.02)
            
            t += dt_step
            last_t = current_real_time
            
            # Build KD-tree for efficient nearest neighbor search
            all_pos = np.array([d.x[0:3] for d in drones])
            kdtree = KDTree(all_pos)
            
            for i, drone in enumerate(drones):
                # Query for NUM_NEIGHBORS + 1 (including self)
                indices = kdtree.query(drone.x[0:3], NUM_NEIGHBORS + 1)
                neighbor_positions = [all_pos[idx] for idx in indices if idx != i]
                drone.update(t, dt_step, m, g, Ix, Iy, Iz, T_traj, threshold, neighbor_positions[:NUM_NEIGHBORS])

            # Print status
            if t - last_print_t >= print_interval:
                status_str = f"t={t:.2f}"
                for drone in drones:
                    status_str += f" | {drone.name}: x={drone.x[0]:.2f}, y={drone.x[1]:.2f}, z={drone.x[2]:.2f}"
                print(status_str)
                last_print_t = t

            # Visualization
            for drone in drones:
                # Delete old plot
                for h in drone.robot_plot:
                    h.remove()
                drone.robot_plot = draw_quad(ax, drone.x)

            plt.pause(0.001)

    if not args.headless:
        plt.show()

if __name__ == "__main__":
    main()

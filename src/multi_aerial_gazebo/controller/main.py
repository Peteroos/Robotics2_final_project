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


def main():
    parser = argparse.ArgumentParser(description="Quadrotor Simulation")
    parser.add_argument("--headless", action="store_true", help="Run without visualization")
    parser.add_argument("--timeout", type=float, default=60.0, help="Simulation timeout in seconds (default: 60.0)")
    args = parser.parse_args()

    # Run long enough to verify whether the swarm actually settles in the goal region.
    sim_duration = max(Tsim, args.timeout)
    n_steps = int(sim_duration / dt)

    # ====================
    # Goal region for emergent 3D distribution
    # ====================
    goal_region = {'center': np.array([1.0, 1.0, 1.5]), 'extents': np.array([0.5, 0.5, 0.5])}

    drones = []
    # Procedural generation on Archimedean spiral: r = b * theta
    # Using theta = 2*sqrt(i) to maintain approx constant spacing along the spiral
    b_spiral = 0.5
    for i in range(NUM_DRONES):
        theta = 2.0 * np.sqrt(i)
        r = b_spiral * theta
        x0 = r * np.cos(theta)
        y0 = r * np.sin(theta)
        # Initial z: staggered per drone so the swarm is not perfectly coplanar at
        # takeoff (pure symmetry would trap the boid into a horizontal plane because
        # there is nothing to break it). This is an initial condition only, not a
        # per-drone target. Start above `ground_clearance` so the ground barrier
        # does not fire at t=0 and wind up the velocity PID integrator.
        z0 = 0.35 + 0.05 * i
        drones.append(Drone(f"Drone{i+1}", np.array([x0, y0, z0]), goal_region, i, NUM_DRONES))

    T_traj = 10.0 # Trajectory duration for each segment

    def swarm_occupancy_metrics():
        positions = np.array([d.x[0:3] for d in drones])
        mins = positions.min(axis=0)
        maxs = positions.max(axis=0)
        span = maxs - mins
        return mins, maxs, span

    # ====================
    # Simulation setup
    # ====================
    if not args.headless:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        # Set fixed axis limits that include goal region and start positions
        all_points = []
        for d in drones:
            all_points.append(d.x[0:3])
        all_points.append(goal_region['center'] + goal_region['extents'])
        all_points.append(goal_region['center'] - goal_region['extents'])
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

        # Draw goal region volume
        draw_box(ax, goal_region['center'], goal_region['extents'])

    # ====================
    # Simulation loop
    # ====================
    t = 0.0
    print_interval = 0.1 if args.headless else 1.0 
    last_print_t = -print_interval
    
    if args.headless:
        # Fixed step simulation for headless testing with emergent 3D distribution
        dt_fixed = 0.01
        start_time = time.time()
        
        for k in range(n_steps - 1):
            t = k * dt_fixed
            
            # Check for timeout or instability (drone height exceeding 100m)
            max_height = max([d.x[2] for d in drones])
            elapsed_time = time.time() - start_time
            if max_height > 100.0 or elapsed_time > args.timeout:
                print(f"[TIMEOUT/INSTABILITY] Simulation stopped: max_height={max_height:.2f}m, t={t:.2f}s, elapsed={elapsed_time:.2f}s")
                break
            
            # Build KD-tree for efficient nearest neighbor search
            all_pos = np.array([d.x[0:3] for d in drones])
            kdtree = KDTree(all_pos)
            
            for i, drone in enumerate(drones):
                # Query for NUM_NEIGHBORS + 1 (including self)
                indices = kdtree.query(drone.x[0:3], NUM_NEIGHBORS + 1)
                neighbor_positions = [all_pos[idx] for idx in indices if idx != i]
                # Updated signature: no threshold parameter
                drone.update(t, dt_fixed, m, g, Ix, Iy, Iz, T_traj, neighbor_positions[:NUM_NEIGHBORS])

            # Print status
            if t - last_print_t >= print_interval:
                status_str = f"t={t:.2f}"
                drifted_drones = []
                mins, maxs, span = swarm_occupancy_metrics()
                for drone in drones:
                    pos = drone.x[0:3]
                    status_str += f" | {drone.name}: x={pos[0]:.2f}, y={pos[1]:.2f}, z={pos[2]:.2f}"
                    if drone.is_outside_goal_region():
                        drifted_drones.append(drone.name)

                if drifted_drones:
                    status_str += f" [DRIFT: {', '.join(drifted_drones)}]"

                status_str += (
                    f" [SPAN x={span[0]:.2f}, y={span[1]:.2f}, z={span[2]:.2f}]"
                    f" [MIN x={mins[0]:.2f}, y={mins[1]:.2f}, z={mins[2]:.2f}]"
                    f" [MAX x={maxs[0]:.2f}, y={maxs[1]:.2f}, z={maxs[2]:.2f}]"
                )

                print(status_str)
                last_print_t = t
    else:
        # Real-time simulation for visual mode with emergent 3D distribution
        start_time = time.time()
        last_t = 0.0
        
        while t < sim_duration:
            elapsed_time = time.time() - start_time
            if elapsed_time > args.timeout:
                print(f"[TIMEOUT] Simulation stopped by timeout: {elapsed_time:.2f}s > {args.timeout}s")
                break

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
                # Updated signature: no threshold parameter
                drone.update(t, dt_step, m, g, Ix, Iy, Iz, T_traj, neighbor_positions[:NUM_NEIGHBORS])

            # Print status
            if t - last_print_t >= print_interval:
                status_str = f"t={t:.2f}"
                drifted_drones = []
                mins, maxs, span = swarm_occupancy_metrics()
                for drone in drones:
                    pos = drone.x[0:3]
                    status_str += f" | {drone.name}: x={pos[0]:.2f}, y={pos[1]:.2f}, z={pos[2]:.2f}"
                    if drone.is_outside_goal_region():
                        drifted_drones.append(drone.name)

                if drifted_drones:
                    status_str += f" [DRIFT: {', '.join(drifted_drones)}]"

                status_str += (
                    f" [SPAN x={span[0]:.2f}, y={span[1]:.2f}, z={span[2]:.2f}]"
                    f" [MIN x={mins[0]:.2f}, y={mins[1]:.2f}, z={mins[2]:.2f}]"
                    f" [MAX x={maxs[0]:.2f}, y={maxs[1]:.2f}, z={maxs[2]:.2f}]"
                )

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

import numpy as np
import matplotlib.pyplot as plt
import argparse
from controller import (
    minimum_jerk_trajectory,
    quad_dynamics,
    CascadedController
)
from renderer import draw_quad

# ====================
# Parameters
# ====================
m = 1.0
g = 9.81

Ix = 0.02
Iy = 0.02
Iz = 0.04

dt = 0.01
Tsim = 15
N = int(Tsim / dt)

def main():
    parser = argparse.ArgumentParser(description="Quadrotor Simulation")
    parser.add_argument("--headless", action="store_true", help="Run without visualization")
    args = parser.parse_args()

    # ====================
    # Initial state
    # ====================
    x = np.zeros((12, N))
    p0 = np.array([0.0, 0.0, 0.0])
    waypoints = [
        np.array([0.0, 0.0, 1.0]),
        np.array([1.0, 0.0, 1.0]),
        np.array([1.0, 1.0, 1.0]),
        np.array([0.0, 1.0, 1.0]),
        np.array([0.0, 0.0, 1.0])
    ]
    wp_idx = 0
    pf = waypoints[wp_idx]
    T_traj = 3.0 # Trajectory duration for each segment
    threshold = 0.1 # Distance threshold to advance to next waypoint

    controller = CascadedController(dt)

    # ====================
    # Simulation setup
    # ====================
    if not args.headless:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        robot_plot = []

    # ====================
    # Simulation loop
    # ====================
    t_start = 0.0
    print_interval = 10 if args.headless else 100 # Print every 0.1s in headless, 1.0s otherwise

    for k in range(N - 1):
        t = k * dt
        t_segment = t - t_start
        
        # Check if we reached the current waypoint
        dist_to_wp = np.linalg.norm(pf - x[0:3, k])
        if dist_to_wp < threshold and wp_idx < len(waypoints) - 1:
            wp_idx += 1
            p0 = pf
            pf = waypoints[wp_idx]
            t_start = t
            t_segment = 0.0
            print(f"Reached waypoint {wp_idx-1}, moving to waypoint {wp_idx}: {pf}")

        pos_ref, vel_ref, acc_ref = minimum_jerk_trajectory(p0, pf, T_traj, t_segment)
        yaw_ref = 0.0
        
        # Cascaded Controller
        u1, u2, u3, u4 = controller.control(x[:, k], pos_ref, vel_ref, acc_ref, yaw_ref, m, g)
        
        # Dynamics update
        dx = quad_dynamics(x[:, k], u1, u2, u3, u4, m, g, Ix, Iy, Iz)
        x[:, k + 1] = x[:, k] + dt * dx

        # Print status
        if k % print_interval == 0:
            print(f"t={t:.2f}, wp={wp_idx}, x={x[0, k+1]:.2f}, y={x[1, k+1]:.2f}, z={x[2, k+1]:.2f}")

        if not args.headless:
            # Delete old plot
            for h in robot_plot:
                h.remove()
            robot_plot = draw_quad(ax, x[:, k + 1])
            plt.pause(0.001)

    if not args.headless:
        plt.show()

if __name__ == "__main__":
    main()

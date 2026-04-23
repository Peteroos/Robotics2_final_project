import numpy as np
import matplotlib.pyplot as plt
import argparse
import time
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

class Drone:
    def __init__(self, name, start_pos, waypoints):
        self.name = name
        self.x = np.zeros(12)
        self.x[0:3] = start_pos
        self.p0 = start_pos
        self.waypoints = waypoints
        self.wp_idx = 0
        self.pf = waypoints[0]
        self.t_start_segment = 0.0
        self.controller = CascadedController()
        self.robot_plot = []

    def calculate_repulsion(self, other_drones, d0=1.0, eta=0.2):
        repulsion = np.zeros(3)
        pos = self.x[0:3]
        for other in other_drones:
            if other == self:
                continue
            other_pos = other.x[0:3]
            diff = pos - other_pos
            dist = np.linalg.norm(diff)
            if dist < d0 and dist > 0.01:
                # Standard repulsive potential gradient: η * (1/d - 1/d0) * (1/d^2) * unit_vec
                mag = eta * (1.0/dist - 1.0/d0) * (1.0/dist**2)
                unit_vec = diff / dist
                # Add a lateral bias to break symmetry for head-on collisions
                # This ensures drones steer around each other
                lateral_bias = np.array([-unit_vec[1], unit_vec[0], 0.1])
                repulsion += mag * (unit_vec + lateral_bias)
        return repulsion

    def update(self, t, dt, m, g, Ix, Iy, Iz, T_traj, threshold, other_drones):
        t_segment = t - self.t_start_segment
        
        # Check if we reached the current waypoint
        dist_to_wp = np.linalg.norm(self.pf - self.x[0:3])
        if dist_to_wp < threshold and self.wp_idx < len(self.waypoints) - 1:
            self.wp_idx += 1
            self.p0 = self.pf
            self.pf = self.waypoints[self.wp_idx]
            self.t_start_segment = t
            t_segment = 0.0
            print(f"[{self.name}] Reached waypoint {self.wp_idx-1}, moving to waypoint {self.wp_idx}: {self.pf}")

        # Integrate Potential Field with Min Jerk
        # Calculate repulsive force from other drones
        rep_force = self.calculate_repulsion(other_drones)
        # Shift the min-jerk destination based on the potential field force
        # This allows the trajectory to "bend" smoothly as drones approach each other
        pf_reactive = self.pf + rep_force
        
        pos_ref, vel_ref, acc_ref = minimum_jerk_trajectory(self.p0, pf_reactive, T_traj, t_segment)
        yaw_ref = 0.0
        
        # Cascaded Controller
        u1, u2, u3, u4 = self.controller.control(self.x, pos_ref, vel_ref, acc_ref, yaw_ref, m, g, dt)
        
        # Dynamics update
        dx = quad_dynamics(self.x, u1, u2, u3, u4, m, g, Ix, Iy, Iz)
        self.x = self.x + dt * dx

def main():
    parser = argparse.ArgumentParser(description="Quadrotor Simulation")
    parser.add_argument("--headless", action="store_true", help="Run without visualization")
    args = parser.parse_args()

    # ====================
    # Drones initialization (Collision Course)
    # ====================
    drones = [
        Drone("Drone1", np.array([0.0, 0.0, 1.0]), [
            np.array([2.0, 0.0, 1.0])
        ]),
        Drone("Drone2", np.array([2.0, 0.0, 1.0]), [
            np.array([0.0, 0.0, 1.0])
        ])
    ]

    T_traj = 5.0 # Trajectory duration for each segment
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
                all_points.append(wp)
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
            
            for drone in drones:
                drone.update(t, dt_fixed, m, g, Ix, Iy, Iz, T_traj, threshold, drones)

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
            
            for drone in drones:
                drone.update(t, dt_step, m, g, Ix, Iy, Iz, T_traj, threshold, drones)

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

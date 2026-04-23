import numpy as np
from controller import CascadedController
from simulator import quad_dynamics, sd_box

class Drone:
    def __init__(self, name, start_pos, waypoints):
        self.name = name
        self.x = np.zeros(12)
        self.x[0:3] = start_pos
        self.p0 = start_pos
        self.waypoints = waypoints
        self.wp_idx = 0
        self.pf = waypoints[0]['center']
        self.t_start_segment = 0.0
        self.controller = CascadedController()
        self.robot_plot = []

    def calculate_repulsion(self, neighbor_positions, d0=1.0, eta=0.2):
        repulsion = np.zeros(3)
        pos = self.x[0:3]
        for other_pos in neighbor_positions:
            diff = pos - other_pos
            dist = np.linalg.norm(diff)
            if dist < d0 and dist > 0.01:
                # Standard repulsive potential gradient: η * (1/d - 1/d0) * (1/d^2) * unit_vec
                mag = eta * (1.0/dist - 1.0/d0) * (1.0/dist**2)
                unit_vec = diff / dist
                repulsion += mag * unit_vec
        return repulsion

    def update(self, t, dt, m, g, Ix, Iy, Iz, T_traj, threshold, neighbor_positions):
        t_segment = t - self.t_start_segment
        
        # Check if we reached the current waypoint volume
        current_wp = self.waypoints[self.wp_idx]
        sdf_val = sd_box(self.x[0:3], current_wp['center'], current_wp['extents'])
        
        if sdf_val < threshold and self.wp_idx < len(self.waypoints) - 1:
            self.wp_idx += 1
            self.p0 = self.pf
            self.pf = self.waypoints[self.wp_idx]['center']
            self.t_start_segment = t
            t_segment = 0.0
            print(f"[{self.name}] Reached waypoint {self.wp_idx-1}, moving to waypoint {self.wp_idx}: {self.pf}")

        # Integrate Potential Field with Min Jerk
        # Calculate repulsive force from nearest neighbors
        # Using tuned d0 and eta for stable occupation within the volume
        rep_force = self.calculate_repulsion(neighbor_positions, d0=0.6, eta=0.01)
        # Shift the min-jerk destination based on the potential field force
        pf_reactive = self.pf + rep_force

        # Cascaded Controller with integrated min-jerk trajectory
        u1, u2, u3, u4 = self.controller.control(self.x, self.p0, pf_reactive, T_traj, t_segment, 0.0, m, g, dt)
        
        # Dynamics update
        dx = quad_dynamics(self.x, u1, u2, u3, u4, m, g, Ix, Iy, Iz)
        self.x = self.x + dt * dx

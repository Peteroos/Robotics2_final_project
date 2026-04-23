import numpy as np
from controller import CascadedController
from simulator import quad_dynamics


class Drone:
    """
    Boid-style swarm drone.

    Every drone runs the *same* control law. There are no per-drone goal points,
    no per-drone offsets, no randomness, and no seed. The only inputs to the
    force field are:
        1. The shared goal volume (identical for every drone).
        2. Local neighbor positions (collision avoidance / separation).
        3. The ground plane z = 0.
    Even distribution inside the goal volume is therefore an *emergent*
    equilibrium between goal-volume attraction and pairwise separation.

    The velocity controller in the outer loop consumes `ext_forces` as a
    velocity-field command. The min-jerk trajectory is deliberately inert
    (p0 == pf == current position) so that it contributes zero reference and
    does not pull any drone to a pre-picked point.
    """

    def __init__(self, name, start_pos, goal_region, index=0, num_drones=1):
        self.name = name
        self.index = index
        self.num_drones = max(1, num_drones)
        self.x = np.zeros(12)
        self.x[0:3] = np.array(start_pos, dtype=float)
        self.goal_region = goal_region
        self.controller = CascadedController()
        # Re-tune the velocity PID for a pure boid / velocity-field outer loop:
        #   * Zero the derivative term. With `prev_error=0` on the very first step,
        #     kd*(err - 0)/dt is a huge transient kick that gets clipped by
        #     `max_accel` and injects a ~0.08 m/s velocity step that a 2nd-order
        #     controller cannot fully unwind each sub-step, producing a sustained
        #     0.04 m/s drift. The boid `damping` term already provides velocity
        #     damping, so kd is redundant.
        #   * Zero the z-integrator to eliminate slow wind-up/unwind drifts.
        self.controller.vel_pid.kd = np.array([0.0, 0.0, 0.0])
        self.controller.vel_pid.ki = np.array([0.2, 0.2, 0.0])
        self.controller.vel_pid.integral = np.zeros(3)
        self.controller.vel_pid.prev_error = np.zeros(3)
        self.robot_plot = []
        self.trajectory_history = [self.x[0:3].copy()]

        # ---- Separation (collision avoidance) ----
        # Soft 1/d kernel inside d0 plus a much stiffer 1/d^2 core inside
        # `close` distance so repulsion dominates as drones approach contact.
        self.sep_radius = 0.9            # neighbor influence radius (m)
        self.sep_gain = 1.2              # soft (1/d) gain
        self.sep_close = 0.35            # hard-avoidance turn-on distance (m)
        self.sep_hard_gain = 4.0         # hard (1/d^2) gain
        self.sep_max = 2.5               # saturate separation velocity command (m/s)

        # ---- Goal-volume attraction (identical for every drone) ----
        # Zero inside the goal box (the swarm is free to arrange itself) and
        # pulls proportionally to the distance outside the box when a drone is
        # outside. A mild soft wall just inside the box face prevents drones
        # from being pushed through the boundary by separation alone.
        self.goal_gain = 1.5
        self.goal_max = 1.5              # saturate goal-attraction velocity command (m/s)
        self.wall_margin_scale = 0.20    # fraction of extents used as soft-wall thickness
        self.wall_gain = 2.0

        # ---- Ground barrier (never crash into z = 0) ----
        self.ground_clearance = 0.30
        self.ground_gain = 6.0

        # Velocity damping on the commanded velocity field: kills flailing by
        # opposing motion that is not produced by one of the force terms.
        # Stronger on z to suppress any common-mode vertical drift from hover
        # thrust biases without over-damping horizontal boid motion.
        self.damping = np.array([1.0, 1.0, 3.0])

    # ------------------------------------------------------------------
    # Force-field components (all returned as velocity commands, m/s)
    # ------------------------------------------------------------------
    def calculate_repulsion(self, neighbor_positions):
        pos = self.x[0:3]
        d0 = self.sep_radius
        repulsion = np.zeros(3)

        for other_pos in neighbor_positions:
            diff = pos - other_pos
            dist = np.linalg.norm(diff)
            if dist < 1e-4 or dist >= d0:
                continue

            dir_vec = diff / dist
            # Soft 1/d potential gradient, smoothly goes to zero at d0
            mag = self.sep_gain * (1.0 / dist - 1.0 / d0)
            # Stiff 1/d^2 core that dominates as dist -> 0
            if dist < self.sep_close:
                mag += self.sep_hard_gain * (
                    1.0 / (dist * dist) - 1.0 / (self.sep_close * self.sep_close)
                )
            repulsion += mag * dir_vec

        mag = np.linalg.norm(repulsion)
        if mag > self.sep_max:
            repulsion *= self.sep_max / mag
        return repulsion

    def goal_attraction(self):
        """
        Shared goal-volume attractor (identical for every drone). The drone is
        pulled toward the nearest point inside the *margin-shrunk* goal box.
        Inside that safe interior the force is zero, so distribution is set by
        pairwise separation alone; outside it (including "just outside a face"),
        the pull is proportional to the distance to the safe interior and is
        strong enough to overcome separation, so equilibria are strictly
        inside the goal volume (not trapped on a face).
        """
        pos = self.x[0:3]
        center = self.goal_region['center']
        extents = self.goal_region['extents']
        margin = self.wall_margin_scale * extents

        lower_safe = center - extents + margin
        upper_safe = center + extents - margin

        # Nearest point inside the safe interior
        target = np.minimum(np.maximum(pos, lower_safe), upper_safe)
        force = self.goal_gain * (target - pos)

        # Saturate so takeoff or a large initial offset cannot produce huge
        # velocity commands that would destabilize the attitude loop.
        mag = np.linalg.norm(force)
        if mag > self.goal_max:
            force *= self.goal_max / mag
        return force

    def ground_avoidance(self):
        """One-sided barrier that guarantees the drone never touches z = 0."""
        z = self.x[2]
        vz = self.x[8]
        if z >= self.ground_clearance:
            return np.zeros(3)
        deficit = (self.ground_clearance - z) / self.ground_clearance
        push_up = self.ground_gain * deficit * deficit
        if vz < 0.0:
            push_up += -2.0 * vz  # damp descent
        return np.array([0.0, 0.0, push_up])

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def is_outside_goal_region(self):
        pos = self.x[0:3]
        center = self.goal_region['center']
        extents = self.goal_region['extents']
        tol = 0.1
        for i in range(3):
            if pos[i] < center[i] - extents[i] - tol or pos[i] > center[i] + extents[i] + tol:
                return True
        return False

    def distance_to_goal_center(self):
        return float(np.linalg.norm(self.x[0:3] - self.goal_region['center']))

    # ------------------------------------------------------------------
    # Main per-step update
    # ------------------------------------------------------------------
    def update(self, t, dt, m, g, Ix, Iy, Iz, T_traj, neighbor_positions):
        pos = self.x[0:3].copy()
        vel = self.x[6:9]

        # Min-jerk segment is intentionally inert: p0 == pf == current pos, so
        # pos_ref = pos, vel_ref = 0, acc_ref = 0, and the velocity controller
        # is driven entirely by ext_forces (the emergent boid force field).
        p0 = pos
        pf = pos

        rep = self.calculate_repulsion(neighbor_positions)
        goal = self.goal_attraction()
        ground = self.ground_avoidance()

        # Velocity-field command: sum of the three emergent terms, plus a small
        # linear damping on the current velocity to suppress flailing. This
        # does not bias equilibrium location; at rest it is identically zero.
        ext_forces = rep + goal + ground - self.damping * vel

        u1, u2, u3, u4 = self.controller.control(
            self.x, p0, pf, ext_forces, T_traj, 0.0, 0.0, m, g, dt
        )

        # Dynamics
        dx = quad_dynamics(self.x, u1, u2, u3, u4, m, g, Ix, Iy, Iz)
        self.x = self.x + dt * dx

        # Hard floor safety net (no ground contact model in the simulator)
        if self.x[2] < 0.0:
            self.x[2] = 0.0
            if self.x[8] < 0.0:
                self.x[8] = 0.0

        if len(self.trajectory_history) % 10 == 0:
            self.trajectory_history.append(self.x[0:3].copy())

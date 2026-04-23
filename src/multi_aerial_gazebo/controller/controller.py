# Generated with the assistance of generative AI

import numpy as np


class PID:
    def __init__(self, kp, ki, kd, ilimit=None):
        self.kp = np.array(kp)
        self.ki = np.array(ki)
        self.kd = np.array(kd)
        self.ilimit = np.array(ilimit) if ilimit is not None else None
        self.integral = np.zeros_like(self.kp, dtype=float)
        self.prev_error = np.zeros_like(self.kp, dtype=float)

    def update(self, error, dt):
        self.integral += error * dt
        if self.ilimit is not None:
            self.integral = np.clip(self.integral, -self.ilimit, self.ilimit)
        
        derivative = (error - self.prev_error) / dt
        self.prev_error = error
        return self.kp * error + self.ki * self.integral + self.kd * derivative


# ====================
# Controller functions
# ====================
class CascadedController:
    def __init__(self):
        # Increased gains to prevent overshoot and ground collisions
        self.vel_pid = PID(kp=[3.0, 3.0, 5.0], ki=[0.2, 0.2, 0.5], kd=[1.0, 1.0, 2.0], ilimit=[2.0, 2.0, 5.0])

        # Attitude controller: takes (phi, theta, psi) error -> desired (p, q, r)
        self.att_pid = PID(kp=[10.0, 10.0, 5.0], ki=[0.0, 0.0, 0.0], kd=[0.0, 0.0, 0.0], ilimit=[1.0, 1.0, 1.0])
        
        # Rate controller: takes (p, q, r) error -> desired (u2, u3, u4)
        self.rate_pid = PID(kp=[0.5, 0.5, 0.5], ki=[0.0, 0.0, 0.0], kd=[0.0, 0.0, 0.0], ilimit=[2.0, 2.0, 2.0])

        # Output limits to prevent aggressive overshoot
        self.max_accel = np.array([5.0, 5.0, 8.0], dtype=float)
        self.max_thrust_scale = 3.0

    def reset_integral_state(self):
        """Reset integral states when starting a new trajectory segment"""
        self.vel_pid.integral = np.zeros_like(self.vel_pid.kp, dtype=float)
        self.vel_pid.prev_error = np.zeros_like(self.vel_pid.kp, dtype=float)

    def control(self, x, p0, pf, ext_forces, T_traj, t_segment, yaw_ref, m, g, dt):
        # x: [x, y, z, phi, theta, psi, vx, vy, vz, p, q, r]
        pos = x[0:3]
        att = x[3:6]
        vel = x[6:9]
        rates = x[9:12]

        # 1. Trajectory Generation (Minimum Jerk)
        pos_ref, vel_ref, acc_ref = self.minimum_jerk_trajectory(p0, pf, T_traj, t_segment)

        # 2. Velocity Controller (Outer Loop)
        # Computes acceleration errors based on velocity feedback
        # More responsive to changing force fields than position control
        # Added position feedback to address overshoot for the min jerk trajectory
        Kp_pos = np.array([4.0, 4.0, 6.0])
        vel_cmd = vel_ref + Kp_pos * (pos_ref - pos)

        # Integrate external forces (repulsion + bounds) directly as a velocity disturbance command
        # This lets the drones react to forces without changing the min jerk endpoint
        vel_cmd = vel_cmd + ext_forces

        vel_error = vel_cmd - vel
        
        acc_cmd = self.vel_pid.update(vel_error, dt)
        
        # Desired accelerations: feedforward from trajectory + feedback correction
        acc_des = acc_ref + acc_cmd
        acc_des = np.clip(acc_des, -self.max_accel, self.max_accel)

        # Vertical control
        u1 = m * (g + acc_des[2])
        u1 = u1 / (np.cos(att[0]) * np.cos(att[1]))
        u1 = np.clip(u1, 0.0, self.max_thrust_scale * m * g)

        # Desired horizontal accelerations
        ax_des = acc_des[0]
        ay_des = acc_des[1]

        # Mapping horizontal acc to desired phi, theta
        phi_des = (ax_des * np.sin(yaw_ref) - ay_des * np.cos(yaw_ref)) / g
        theta_des = (ax_des * np.cos(yaw_ref) + ay_des * np.sin(yaw_ref)) / g
        
        # Clip desired angles to stay within small angle assumption
        phi_des = np.clip(phi_des, -np.pi/4, np.pi/4)
        theta_des = np.clip(theta_des, -np.pi/4, np.pi/4)

        # 2. Attitude Controller
        att_ref = np.array([phi_des, theta_des, yaw_ref])
        att_error = att_ref - att
        # Wrap yaw error
        att_error[2] = (att_error[2] + np.pi) % (2 * np.pi) - np.pi
        
        rates_des = self.att_pid.update(att_error, dt)
        
        # 3. Rate Controller
        rate_error = rates_des - rates
        u_rates = self.rate_pid.update(rate_error, dt)
        
        u2 = u_rates[0]
        u3 = u_rates[1]
        u4 = u_rates[2]
        
        return u1, u2, u3, u4

    def minimum_jerk_trajectory(self, p0, pf, T, t):
        """
        p0: initial position
        pf: final position
        T: total time
        t: current time
        """
        if t >= T:
            return pf, np.zeros(3), np.zeros(3)
        tau = t / T
        # 5th order polynomial for minimum jerk
        pos = p0 + (pf - p0) * (10 * tau**3 - 15 * tau**4 + 6 * tau**5)
        vel = (pf - p0) * (30 * tau**2 - 60 * tau**3 + 30 * tau**4) / T
        acc = (pf - p0) * (60 * tau - 180 * tau**2 + 120 * tau**3) / T**2
        return pos, vel, acc




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
        # Position controller: takes (x, y, z) error -> desired (phi, theta, u1)
        # Added integral limits for anti-windup
        self.pos_pid = PID(kp=[1.5, 1.5, 5.0], ki=[0.0, 0.0, 0.1], kd=[1.2, 1.2, 3.0], ilimit=[1.0, 1.0, 5.0])
        
        # Attitude controller: takes (phi, theta, psi) error -> desired (p, q, r)
        self.att_pid = PID(kp=[10.0, 10.0, 5.0], ki=[0.0, 0.0, 0.0], kd=[0.0, 0.0, 0.0], ilimit=[1.0, 1.0, 1.0])
        
        # Rate controller: takes (p, q, r) error -> desired (u2, u3, u4)
        self.rate_pid = PID(kp=[0.1, 0.1, 0.1], ki=[0.0, 0.0, 0.0], kd=[0.0, 0.0, 0.0], ilimit=[1.0, 1.0, 1.0])

    def control(self, x, p0, pf, T_traj, t_segment, yaw_ref, m, g, dt):
        # x: [x, y, z, phi, theta, psi, vx, vy, vz, p, q, r]
        pos = x[0:3]
        att = x[3:6]
        vel = x[6:9]
        rates = x[9:12]

        # 1. Trajectory Generation (Minimum Jerk)
        pos_ref, vel_ref, acc_ref = self.minimum_jerk_trajectory(p0, pf, T_traj, t_segment)

        # 2. Position Controller
        pos_error = pos_ref - pos
        # Use velocity error as well for better damping
        vel_error = vel_ref - vel
        
        # Adjust PID update to handle D term separately if we want to use vel_ref
        # Or just use the position error and let PID handle it. 
        # But here we have vel_ref, so let's use it.
        u_pos = self.pos_pid.update(pos_error, dt)
        
        # Vertical control
        u1 = m * (g + u_pos[2] + acc_ref[2])
        u1 = u1 / (np.cos(att[0]) * np.cos(att[1]))
        
        # Desired horizontal accelerations
        ax_des = u_pos[0] + acc_ref[0]
        ay_des = u_pos[1] + acc_ref[1]
        
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
        if t > T:
            t = T
        tau = t / T
        # 5th order polynomial for minimum jerk
        pos = p0 + (pf - p0) * (10 * tau**3 - 15 * tau**4 + 6 * tau**5)
        vel = (pf - p0) * (30 * tau**2 - 60 * tau**3 + 30 * tau**4) / T
        acc = (pf - p0) * (60 * tau - 180 * tau**2 + 120 * tau**3) / T**2
        return pos, vel, acc






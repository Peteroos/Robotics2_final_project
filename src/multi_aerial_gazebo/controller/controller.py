# Generated with the assistance of generative AI

import numpy as np

class PID:
    def __init__(self, kp, ki, kd, dt):
        self.kp = np.array(kp)
        self.ki = np.array(ki)
        self.kd = np.array(kd)
        self.dt = dt
        self.integral = np.zeros_like(self.kp, dtype=float)
        self.prev_error = np.zeros_like(self.kp, dtype=float)

    def update(self, error):
        self.integral += error * self.dt
        derivative = (error - self.prev_error) / self.dt
        self.prev_error = error
        return self.kp * error + self.ki * self.integral + self.kd * derivative

def minimum_jerk_trajectory(p0, pf, T, t):
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

def quad_dynamics(x, u1, u2, u3, u4, m, g, Ix, Iy, Iz):
    phi, theta, psi = x[3], x[4], x[5]
    vx, vy, vz = x[6], x[7], x[8]
    p, q, r = x[9], x[10], x[11]

    R = rotation_matrix(phi, theta, psi)

    dx = np.zeros(12)

    # Positions
    dx[0:3] = [vx, vy, vz]

    # Euler angles
    dx[3] = p + r * np.sin(theta)
    dx[4] = p * np.sin(theta) * np.tan(phi) + q - r * np.cos(theta) * np.tan(phi)
    dx[5] = -p * np.sin(theta) / np.cos(phi) + r * np.cos(theta) / np.cos(phi)

    # Velocities
    dx[6:9] = (1 / m) * R @ np.array([0, 0, u1]) + np.array([0, 0, -g])

    # Angular rates
    dx[9] = ((Iy - Iz) / Ix) * q * r + u2 / Ix
    dx[10] = ((Iz - Ix) / Iy) * p * r + u3 / Iy
    dx[11] = u4 / Iz

    return dx

# ====================
# Controller functions
# ====================
class CascadedController:
    def __init__(self, dt):
        self.dt = dt
        # Position controller: takes (x, y, z) error -> desired (phi, theta, u1)
        # Note: u1 is total thrust, but for simplicity here we might map it differently
        self.pos_pid = PID(kp=[1.5, 1.5, 5.0], ki=[0.0, 0.0, 0.1], kd=[1.2, 1.2, 3.0], dt=dt)
        
        # Attitude controller: takes (phi, theta, psi) error -> desired (p, q, r)
        self.att_pid = PID(kp=[10.0, 10.0, 5.0], ki=[0.0, 0.0, 0.0], kd=[0.0, 0.0, 0.0], dt=dt)
        
        # Rate controller: takes (p, q, r) error -> desired (u2, u3, u4)
        self.rate_pid = PID(kp=[0.1, 0.1, 0.1], ki=[0.0, 0.0, 0.0], kd=[0.0, 0.0, 0.0], dt=dt)

    def control(self, x, pos_ref, vel_ref, acc_ref, yaw_ref, m, g):
        # x: [x, y, z, phi, theta, psi, vx, vy, vz, p, q, r]
        pos = x[0:3]
        att = x[3:6]
        vel = x[6:9]
        rates = x[9:12]

        # 1. Position Controller
        pos_error = pos_ref - pos
        # Use velocity error as well for better damping
        vel_error = vel_ref - vel
        
        # Adjust PID update to handle D term separately if we want to use vel_ref
        # Or just use the position error and let PID handle it. 
        # But here we have vel_ref, so let's use it.
        u_pos = self.pos_pid.update(pos_error)
        
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
        
        rates_des = self.att_pid.update(att_error)
        
        # 3. Rate Controller
        rate_error = rates_des - rates
        u_rates = self.rate_pid.update(rate_error)
        
        u2 = u_rates[0]
        u3 = u_rates[1]
        u4 = u_rates[2]
        
        return u1, u2, u3, u4

# ====================
# Rotation matrix
# ====================
def rotation_matrix(phi, theta, psi):
    
    R = np.array([
        [np.cos(psi)*np.cos(theta) - np.sin(phi)*np.sin(psi)*np.sin(theta),
         -np.cos(phi)*np.sin(psi),
         np.cos(psi)*np.sin(theta) + np.cos(theta)*np.sin(phi)*np.sin(psi)],
        [np.sin(psi)*np.cos(theta) + np.cos(psi)*np.sin(phi)*np.sin(theta),
         np.cos(phi)*np.cos(psi),
         np.sin(psi)*np.sin(theta) - np.cos(psi)*np.cos(theta)*np.sin(phi)],
        [-np.cos(phi)*np.sin(theta),
         np.sin(phi),
         np.cos(phi)*np.cos(theta)]
    ])

    return R





#!/usr/bin/env python3

"""Multi-robot PID + APF obstacle-avoidance controller for the aerial demo.

Per robot input topics:
- /<robot>/state12      (std_msgs/Float64MultiArray)
- /<robot>/target_pose  (geometry_msgs/Pose)

Global takeoff trigger topic:
- /start_flight         (std_msgs/Bool)
    * data = True  -> use `default_target_positions` (fly to red sphere goals)
    * data = False -> hold current position (stay where you are)
Before the first True arrives the drones hold their spawn pose on the ground.

Per robot output topics:
- /<robot>/cmd_vel_body (geometry_msgs/Twist)

Control objective:
- PID position control on x, y, z in world frame (attractive term)
- Artificial-potential-field (APF) repulsion from static obstacles and from
  neighbouring drones (avoidance term)
- Yaw control around z
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import rclpy
from geometry_msgs.msg import Pose, Twist
from rclpy.node import Node
from rclpy.time import Time
from std_msgs.msg import Bool, Float64MultiArray


@dataclass
class RobotState:
    x: float
    y: float
    z: float
    yaw: float
    stamp: Time


@dataclass
class TargetPose:
    x: float
    y: float
    z: float
    yaw: float


@dataclass
class PIDMemory:
    ex_i: float = 0.0
    ey_i: float = 0.0
    ez_i: float = 0.0
    eyaw_i: float = 0.0
    ex_prev: float = 0.0
    ey_prev: float = 0.0
    ez_prev: float = 0.0
    eyaw_prev: float = 0.0
    last_time: Optional[Time] = None
    initialized: bool = False


@dataclass
class Obstacle:
    """Cylindrical keep-out volume for APF avoidance.

    x, y:   center of the obstacle footprint in world frame
    radius: physical XY bounding radius (in metres)
    z_top:  obstacle top height (assumed to sit on the ground z=0)
    """

    x: float
    y: float
    radius: float
    z_top: float


class MultiRobotPidController(Node):
    def __init__(self) -> None:
        super().__init__("multi_robot_pid_controller")

        self.declare_parameter(
            "robot_names", ["aerial_robot_1", "aerial_robot_2", "aerial_robot_3"]
        )
        self.declare_parameter("state_topic_suffix", "state12")
        self.declare_parameter("target_topic_suffix", "target_pose")
        self.declare_parameter("command_topic_suffix", "cmd_vel_body")
        self.declare_parameter("control_rate_hz", 20.0)

        self.declare_parameter("kp_pos", [1.2, 1.2, 1.5])
        self.declare_parameter("ki_pos", [0.0, 0.0, 0.0])
        self.declare_parameter("kd_pos", [0.3, 0.3, 0.4])

        self.declare_parameter("kp_yaw", 1.5)
        self.declare_parameter("ki_yaw", 0.0)
        self.declare_parameter("kd_yaw", 0.2)

        self.declare_parameter("int_limit_pos", 1.0)
        self.declare_parameter("int_limit_yaw", 0.8)
        self.declare_parameter("max_linear_speed", 1.0)
        self.declare_parameter("max_yaw_rate", 1.0)

        self.declare_parameter("hold_current_on_start", True)
        self.declare_parameter(
            "default_target_positions",
            [0.0, 0.0, 1.0, 2.0, 0.0, 1.0, -2.0, 0.0, 1.0],
        )
        self.declare_parameter("default_target_yaws", [0.0, 0.0, 0.0])
        self.declare_parameter("start_trigger_topic", "/start_flight")
        # If False, the robots will immediately chase `default_target_positions`
        # on startup (old behavior). If True they wait for /start_flight=True.
        self.declare_parameter("wait_for_start_signal", True)

        # --- APF obstacle avoidance ------------------------------------
        # Obstacle list as a flat array of quadruples [x, y, radius, z_top].
        # Each obstacle is modelled as a vertical cylinder sitting on the
        # ground (z = 0) with physical radius `radius` and top at `z_top`.
        # Default is a single dummy obstacle with radius=0 at the origin
        # (effectively disabled) because ROS 2 parameters cannot be declared
        # as empty lists without an explicit ParameterDescriptor.
        self.declare_parameter("obstacles", [0.0, 0.0, 0.0, 0.0])
        # Approximate drone half-span (the ~0.2 m arm tips plus 0.1 m margin).
        self.declare_parameter("drone_radius", 0.30)
        # Extra safety clearance kept between drone hull and obstacle hull.
        self.declare_parameter("avoidance_margin", 0.40)
        # Influence zone width: repulsion kicks in when clearance < this.
        self.declare_parameter("avoidance_influence", 1.8)
        # APF repulsion gain (higher = pushes harder).
        self.declare_parameter("avoidance_k_rep", 1.6)
        # Extra lift added when clearance drops below this so the drone
        # tries to climb over shorter obstacles rather than wall-sliding.
        self.declare_parameter("avoidance_lift_gain", 0.6)
        # Peer-drone repulsion (prevents pile-up at the shared target).
        self.declare_parameter("mutual_influence", 1.2)
        self.declare_parameter("mutual_k_rep", 0.8)

        self._robot_names = list(self.get_parameter("robot_names").value)
        self._state_topic_suffix = str(self.get_parameter("state_topic_suffix").value)
        self._target_topic_suffix = str(self.get_parameter("target_topic_suffix").value)
        self._command_topic_suffix = str(self.get_parameter("command_topic_suffix").value)
        self._control_rate_hz = max(1.0, float(self.get_parameter("control_rate_hz").value))

        self._kp_pos = self._read_vector_param("kp_pos", 3, [1.2, 1.2, 1.5])
        self._ki_pos = self._read_vector_param("ki_pos", 3, [0.0, 0.0, 0.0])
        self._kd_pos = self._read_vector_param("kd_pos", 3, [0.3, 0.3, 0.4])

        self._kp_yaw = float(self.get_parameter("kp_yaw").value)
        self._ki_yaw = float(self.get_parameter("ki_yaw").value)
        self._kd_yaw = float(self.get_parameter("kd_yaw").value)

        self._int_limit_pos = abs(float(self.get_parameter("int_limit_pos").value))
        self._int_limit_yaw = abs(float(self.get_parameter("int_limit_yaw").value))
        self._max_linear_speed = abs(float(self.get_parameter("max_linear_speed").value))
        self._max_yaw_rate = abs(float(self.get_parameter("max_yaw_rate").value))
        self._hold_current_on_start = bool(self.get_parameter("hold_current_on_start").value)
        self._start_trigger_topic = str(self.get_parameter("start_trigger_topic").value)
        self._wait_for_start_signal = bool(self.get_parameter("wait_for_start_signal").value)

        self._obstacles: List[Obstacle] = self._parse_obstacles(
            list(self.get_parameter("obstacles").value)
        )
        self._drone_radius = abs(float(self.get_parameter("drone_radius").value))
        self._avoidance_margin = abs(float(self.get_parameter("avoidance_margin").value))
        self._avoidance_influence = abs(float(self.get_parameter("avoidance_influence").value))
        self._avoidance_k_rep = abs(float(self.get_parameter("avoidance_k_rep").value))
        self._avoidance_lift_gain = abs(float(self.get_parameter("avoidance_lift_gain").value))
        self._mutual_influence = abs(float(self.get_parameter("mutual_influence").value))
        self._mutual_k_rep = abs(float(self.get_parameter("mutual_k_rep").value))

        self._states: Dict[str, RobotState] = {}
        self._targets: Dict[str, TargetPose] = {}
        # Flight targets loaded from parameters but not applied until the
        # start signal is received (when wait_for_start_signal=True).
        self._flight_targets: Dict[str, TargetPose] = {}
        self._flight_started: bool = not self._wait_for_start_signal
        self._pid_memories: Dict[str, PIDMemory] = {name: PIDMemory() for name in self._robot_names}
        self._cmd_publishers: Dict[str, rclpy.publisher.Publisher] = {}

        for robot in self._robot_names:
            self.create_subscription(
                Float64MultiArray,
                f"/{robot}/{self._state_topic_suffix}",
                lambda msg, robot_name=robot: self._on_state(robot_name, msg),
                20,
            )
            self.create_subscription(
                Pose,
                f"/{robot}/{self._target_topic_suffix}",
                lambda msg, robot_name=robot: self._on_target(robot_name, msg),
                20,
            )
            self._cmd_publishers[robot] = self.create_publisher(
                Twist,
                f"/{robot}/{self._command_topic_suffix}",
                20,
            )

        self._load_default_targets()

        self.create_subscription(
            Bool,
            self._start_trigger_topic,
            self._on_start_trigger,
            10,
        )

        self.create_timer(1.0 / self._control_rate_hz, self._control_step)

        if self._wait_for_start_signal:
            self.get_logger().info(
                "PID controller idle. Waiting for start signal on '%s' "
                "(publish std_msgs/Bool data:true to take off). robots=%s, rate=%.1fHz"
                % (self._start_trigger_topic, self._robot_names, self._control_rate_hz)
            )
        else:
            self.get_logger().info(
                "PID controller started. robots=%s, rate=%.1fHz"
                % (self._robot_names, self._control_rate_hz)
            )

    def _read_vector_param(self, name: str, size: int, fallback: List[float]) -> List[float]:
        raw = list(self.get_parameter(name).value)
        if len(raw) != size:
            self.get_logger().warn(
                "%s expects %d values, got %d. Using fallback %s" % (name, size, len(raw), fallback)
            )
            return fallback
        return [float(v) for v in raw]

    def _load_default_targets(self) -> None:
        positions = list(self.get_parameter("default_target_positions").value)
        yaws = list(self.get_parameter("default_target_yaws").value)
        expected_pos_len = 3 * len(self._robot_names)

        if len(positions) == expected_pos_len:
            yaw_defaults = [0.0] * len(self._robot_names)
            if len(yaws) == len(self._robot_names):
                yaw_defaults = [float(v) for v in yaws]
            for idx, robot in enumerate(self._robot_names):
                pose = TargetPose(
                    x=float(positions[3 * idx]),
                    y=float(positions[3 * idx + 1]),
                    z=float(positions[3 * idx + 2]),
                    yaw=yaw_defaults[idx],
                )
                self._flight_targets[robot] = pose
                if not self._wait_for_start_signal:
                    self._targets[robot] = pose
            if self._wait_for_start_signal:
                self.get_logger().info(
                    "Loaded flight targets (armed). Send True on '%s' to take off."
                    % self._start_trigger_topic
                )
            else:
                self.get_logger().info("Loaded default targets from parameters.")
        elif len(positions) != 0:
            self.get_logger().warn(
                "default_target_positions length must be %d, got %d. Ignoring." %
                (expected_pos_len, len(positions))
            )

    def _on_start_trigger(self, msg: Bool) -> None:
        if msg.data:
            if self._flight_started:
                self.get_logger().info("Start signal received but already flying; ignored.")
                return
            if not self._flight_targets:
                self.get_logger().warn(
                    "Start signal received but no flight targets are configured."
                )
                return
            self._flight_started = True
            for robot, pose in self._flight_targets.items():
                self._targets[robot] = pose
                memory = self._pid_memories.get(robot)
                if memory is not None:
                    memory.ex_i = 0.0
                    memory.ey_i = 0.0
                    memory.ez_i = 0.0
                    memory.eyaw_i = 0.0
                    memory.initialized = False
            self.get_logger().info("Takeoff command received. Flying to red targets.")
        else:
            if not self._flight_started:
                self.get_logger().info("Stop signal received but drones are already idle.")
                return
            self._flight_started = False
            for robot in self._robot_names:
                state = self._states.get(robot)
                if state is None:
                    continue
                self._targets[robot] = TargetPose(
                    x=state.x, y=state.y, z=state.z, yaw=state.yaw
                )
                memory = self._pid_memories.get(robot)
                if memory is not None:
                    memory.ex_i = 0.0
                    memory.ey_i = 0.0
                    memory.ez_i = 0.0
                    memory.eyaw_i = 0.0
                    memory.initialized = False
            self.get_logger().info("Stop command received. Holding current position.")

    def _on_state(self, robot: str, msg: Float64MultiArray) -> None:
        if len(msg.data) < 12:
            self.get_logger().warn("%s state12 size < 12, ignoring." % robot)
            return

        state = RobotState(
            x=float(msg.data[0]),
            y=float(msg.data[1]),
            z=float(msg.data[2]),
            yaw=float(msg.data[5]),
            stamp=self.get_clock().now(),
        )
        self._states[robot] = state

        # If no default target is set, hold where the robot starts.
        if self._hold_current_on_start and robot not in self._targets:
            self._targets[robot] = TargetPose(x=state.x, y=state.y, z=state.z, yaw=state.yaw)

    def _on_target(self, robot: str, msg: Pose) -> None:
        yaw = self._quaternion_to_yaw(
            msg.orientation.x,
            msg.orientation.y,
            msg.orientation.z,
            msg.orientation.w,
        )
        self._targets[robot] = TargetPose(
            x=msg.position.x,
            y=msg.position.y,
            z=msg.position.z,
            yaw=yaw,
        )
        self.get_logger().info(
            "Updated target for %s: [%.2f, %.2f, %.2f, yaw=%.2f]" %
            (robot, msg.position.x, msg.position.y, msg.position.z, yaw)
        )

    def _control_step(self) -> None:
        for robot in self._robot_names:
            state = self._states.get(robot)
            target = self._targets.get(robot)
            if state is None or target is None:
                continue

            memory = self._pid_memories[robot]
            now = self.get_clock().now()
            dt = 1.0 / self._control_rate_hz
            if memory.last_time is not None:
                dt_measured = (now - memory.last_time).nanoseconds / 1e9
                if 0.0 < dt_measured < 1.0:
                    dt = dt_measured
            memory.last_time = now

            ex = target.x - state.x
            ey = target.y - state.y
            ez = target.z - state.z
            eyaw = self._wrap_angle(target.yaw - state.yaw)

            if not memory.initialized:
                memory.ex_prev = ex
                memory.ey_prev = ey
                memory.ez_prev = ez
                memory.eyaw_prev = eyaw
                memory.initialized = True

            memory.ex_i = self._clamp(memory.ex_i + ex * dt, -self._int_limit_pos, self._int_limit_pos)
            memory.ey_i = self._clamp(memory.ey_i + ey * dt, -self._int_limit_pos, self._int_limit_pos)
            memory.ez_i = self._clamp(memory.ez_i + ez * dt, -self._int_limit_pos, self._int_limit_pos)
            memory.eyaw_i = self._clamp(
                memory.eyaw_i + eyaw * dt,
                -self._int_limit_yaw,
                self._int_limit_yaw,
            )

            dex = (ex - memory.ex_prev) / dt
            dey = (ey - memory.ey_prev) / dt
            dez = (ez - memory.ez_prev) / dt
            deyaw = (eyaw - memory.eyaw_prev) / dt

            memory.ex_prev = ex
            memory.ey_prev = ey
            memory.ez_prev = ez
            memory.eyaw_prev = eyaw

            vx_world = self._pid_axis(ex, memory.ex_i, dex, self._kp_pos[0], self._ki_pos[0], self._kd_pos[0])
            vy_world = self._pid_axis(ey, memory.ey_i, dey, self._kp_pos[1], self._ki_pos[1], self._kd_pos[1])
            vz_world = self._pid_axis(ez, memory.ez_i, dez, self._kp_pos[2], self._ki_pos[2], self._kd_pos[2])
            yaw_rate = self._pid_axis(eyaw, memory.eyaw_i, deyaw, self._kp_yaw, self._ki_yaw, self._kd_yaw)

            # --- APF repulsion from obstacles and other drones ---------
            rep_vx, rep_vy, rep_vz = self._compute_repulsion(robot, state)
            vx_world += rep_vx
            vy_world += rep_vy
            vz_world += rep_vz

            # Clamp horizontal velocity magnitude (preserves direction so
            # avoidance doesn't get distorted by per-axis clipping).
            speed_xy = math.hypot(vx_world, vy_world)
            if speed_xy > self._max_linear_speed and speed_xy > 1e-9:
                scale = self._max_linear_speed / speed_xy
                vx_world *= scale
                vy_world *= scale
            vz_world = self._clamp(vz_world, -self._max_linear_speed, self._max_linear_speed)
            yaw_rate = self._clamp(yaw_rate, -self._max_yaw_rate, self._max_yaw_rate)

            # Convert world velocity command to body frame using yaw.
            cy = math.cos(state.yaw)
            sy = math.sin(state.yaw)
            vx_body = cy * vx_world + sy * vy_world
            vy_body = -sy * vx_world + cy * vy_world

            cmd = Twist()
            cmd.linear.x = vx_body
            cmd.linear.y = vy_body
            cmd.linear.z = vz_world
            cmd.angular.z = yaw_rate

            self._cmd_publishers[robot].publish(cmd)

    def _parse_obstacles(self, raw: List[float]) -> List[Obstacle]:
        if not raw:
            return []
        if len(raw) % 4 != 0:
            self.get_logger().warn(
                "`obstacles` parameter must have length divisible by 4 "
                "(x, y, radius, z_top). Got %d; ignoring." % len(raw)
            )
            return []
        result: List[Obstacle] = []
        for i in range(0, len(raw), 4):
            radius = float(raw[i + 2])
            z_top = float(raw[i + 3])
            # Skip dummy zero-radius entries (used as default sentinel).
            if radius <= 0.0 or z_top <= 0.0:
                continue
            result.append(
                Obstacle(
                    x=float(raw[i]),
                    y=float(raw[i + 1]),
                    radius=radius,
                    z_top=z_top,
                )
            )
        if result:
            self.get_logger().info(
                "APF avoidance active with %d obstacle(s)." % len(result)
            )
        return result

    def _compute_repulsion(self, robot: str, state: RobotState) -> tuple:
        """Return (rep_vx, rep_vy, rep_vz) world-frame velocity additions."""
        rep_vx = 0.0
        rep_vy = 0.0
        rep_vz = 0.0
        eps = 1e-3
        drone_effective = self._drone_radius + self._avoidance_margin

        # --- Repulsion from static obstacles ---------------------------
        for obs in self._obstacles:
            # Ignore obstacles the drone is already flying safely above.
            vertical_gap = state.z - (obs.z_top + self._drone_radius)
            if vertical_gap > 0.3:
                continue

            dx = state.x - obs.x
            dy = state.y - obs.y
            dist_xy = math.hypot(dx, dy)
            clearance = dist_xy - obs.radius - drone_effective

            if dist_xy < eps:
                # Degenerate: drone center coincides with obstacle center.
                # Push in +x as a fallback.
                rep_vx += self._avoidance_k_rep * 5.0
                continue

            nx = dx / dist_xy
            ny = dy / dist_xy

            if clearance <= 0.0:
                # Already inside the keep-out shell: push out hard.
                magnitude = self._avoidance_k_rep * (
                    5.0 + abs(clearance) * 4.0
                )
                rep_vx += magnitude * nx
                rep_vy += magnitude * ny
                if obs.z_top > state.z:
                    rep_vz += self._avoidance_lift_gain * 1.5
                continue

            if clearance < self._avoidance_influence:
                # Smooth APF term, Khatib-style:
                # F = k * (1/d - 1/d0) / d^2, in direction away from obstacle.
                inv_clearance = 1.0 / max(clearance, 0.05)
                inv_influence = 1.0 / self._avoidance_influence
                magnitude = self._avoidance_k_rep * (inv_clearance - inv_influence) * inv_clearance * inv_clearance
                rep_vx += magnitude * nx
                rep_vy += magnitude * ny
                # If obstacle is taller than the drone's current z, push up.
                if obs.z_top > state.z and self._avoidance_lift_gain > 0.0:
                    lift_weight = (self._avoidance_influence - clearance) / self._avoidance_influence
                    rep_vz += self._avoidance_lift_gain * lift_weight

        # --- Repulsion from other drones -------------------------------
        for other_name, other_state in self._states.items():
            if other_name == robot:
                continue
            dx = state.x - other_state.x
            dy = state.y - other_state.y
            dz = state.z - other_state.z
            dist = math.sqrt(dx * dx + dy * dy + dz * dz)
            if dist < eps:
                rep_vx += self._mutual_k_rep * 3.0
                continue
            clearance = dist - 2.0 * self._drone_radius - 0.10
            if clearance < self._mutual_influence:
                inv_clearance = 1.0 / max(clearance, 0.05)
                inv_influence = 1.0 / self._mutual_influence
                magnitude = self._mutual_k_rep * (inv_clearance - inv_influence) * inv_clearance * inv_clearance
                rep_vx += magnitude * dx / dist
                rep_vy += magnitude * dy / dist
                # Only a weak vertical component so drones mostly separate
                # laterally, keeping roughly the same altitude.
                rep_vz += 0.3 * magnitude * dz / dist

        return rep_vx, rep_vy, rep_vz

    @staticmethod
    def _pid_axis(error: float, integral: float, derivative: float, kp: float, ki: float, kd: float) -> float:
        return kp * error + ki * integral + kd * derivative

    @staticmethod
    def _quaternion_to_yaw(x: float, y: float, z: float, w: float) -> float:
        norm = math.sqrt(x * x + y * y + z * z + w * w)
        if norm < 1e-9:
            return 0.0
        x /= norm
        y /= norm
        z /= norm
        w /= norm
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    @staticmethod
    def _clamp(value: float, minimum: float, maximum: float) -> float:
        return max(minimum, min(maximum, value))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = MultiRobotPidController()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

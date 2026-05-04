#!/usr/bin/env python3

"""Bridge Gazebo multi-robot simulation to 12-state and command topics.

State order per robot:
[x, y, z, roll, pitch, yaw, vx, vy, vz, p, q, r]

- (x, y, z): position in world frame
- (roll, pitch, yaw): attitude in world frame
- (vx, vy, vz): linear velocity in body frame
- (p, q, r): angular velocity in body frame

Command topic per robot:
/<robot_name>/cmd_vel_body  (geometry_msgs/Twist)
- linear: desired [vx, vy, vz] in body frame
- angular: desired [p, q, r] in body frame
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import rclpy
from geometry_msgs.msg import Pose, Twist
from gazebo_msgs.msg import EntityState, ModelState, ModelStates
from gazebo_msgs.srv import SetEntityState, SetModelState
from rclpy.node import Node
from rclpy.time import Time
from std_msgs.msg import Float64MultiArray, MultiArrayDimension


@dataclass
class RobotSnapshot:
    pose: Pose
    twist_world: Twist
    state12: List[float]


@dataclass
class TimedCommand:
    twist_body: Twist
    stamp: Time


class MultiRobotStateInterface(Node):
    def __init__(self) -> None:
        super().__init__("multi_robot_state_interface")

        self.declare_parameter(
            "robot_names", ["aerial_robot_1", "aerial_robot_2", "aerial_robot_3"]
        )
        self.declare_parameter("command_rate_hz", 20.0)
        self.declare_parameter("command_timeout_sec", 0.5)
        self.declare_parameter("state_topic_suffix", "state12")
        self.declare_parameter("cmd_topic_suffix", "cmd_vel_body")

        self._robot_names = list(self.get_parameter("robot_names").value)
        self._command_rate_hz = float(self.get_parameter("command_rate_hz").value)
        self._command_timeout_sec = float(self.get_parameter("command_timeout_sec").value)
        self._state_topic_suffix = str(self.get_parameter("state_topic_suffix").value)
        self._cmd_topic_suffix = str(self.get_parameter("cmd_topic_suffix").value)

        if not self._robot_names:
            raise ValueError("robot_names cannot be empty")

        self._snapshots: Dict[str, RobotSnapshot] = {}
        self._commands: Dict[str, TimedCommand] = {}
        self._state_publishers = {}
        self._missing_state_warned = False
        self._missing_service_warned = False

        for robot in self._robot_names:
            state_topic = f"/{robot}/{self._state_topic_suffix}"
            cmd_topic = f"/{robot}/{self._cmd_topic_suffix}"

            self._state_publishers[robot] = self.create_publisher(Float64MultiArray, state_topic, 10)
            self.create_subscription(
                Twist,
                cmd_topic,
                lambda msg, robot_name=robot: self._on_body_command(robot_name, msg),
                10,
            )

        self._all_state_pub = self.create_publisher(Float64MultiArray, "/multi_aerial/state12_all", 10)

        self.create_subscription(ModelStates, "/gazebo/model_states", self._on_model_states, 20)
        self._set_entity_client = self.create_client(SetEntityState, "/gazebo/set_entity_state")
        self._set_model_client = self.create_client(SetModelState, "/gazebo/set_model_state")

        self._cmd_timer = self.create_timer(1.0 / max(self._command_rate_hz, 1.0), self._apply_commands)

        self.get_logger().info(
            "State interface ready for robots=%s, command_rate=%.1fHz"
            % (self._robot_names, self._command_rate_hz)
        )

    def _on_body_command(self, robot: str, msg: Twist) -> None:
        self._commands[robot] = TimedCommand(twist_body=self._copy_twist(msg), stamp=self.get_clock().now())

    def _on_model_states(self, msg: ModelStates) -> None:
        name_to_index = {name: idx for idx, name in enumerate(msg.name)}

        aggregate: List[float] = []
        found_any = False
        missing: List[str] = []

        for robot in self._robot_names:
            idx = name_to_index.get(robot)
            if idx is None:
                missing.append(robot)
                continue

            pose = self._copy_pose(msg.pose[idx])
            twist_world = self._copy_twist(msg.twist[idx])
            state12 = self._compose_state12(pose, twist_world)

            self._snapshots[robot] = RobotSnapshot(
                pose=pose,
                twist_world=twist_world,
                state12=state12,
            )

            out = Float64MultiArray()
            out.data = state12
            self._state_publishers[robot].publish(out)

            aggregate.extend(state12)
            found_any = True

        if found_any:
            out_all = Float64MultiArray()
            out_all.layout.dim = [
                MultiArrayDimension(label="robot", size=len(self._robot_names), stride=12 * len(self._robot_names)),
                MultiArrayDimension(label="state", size=12, stride=12),
            ]
            out_all.data = aggregate
            self._all_state_pub.publish(out_all)

        if missing and not self._missing_state_warned:
            self.get_logger().warn(
                "Waiting for robot states in /gazebo/model_states. Missing: %s" % missing
            )
            self._missing_state_warned = True

    def _apply_commands(self) -> None:
        if not self._commands:
            return

        entity_ready = self._set_entity_client.service_is_ready()
        model_ready = self._set_model_client.service_is_ready()

        if not entity_ready and not model_ready:
            if not self._missing_service_warned:
                self.get_logger().warn(
                    "Neither /gazebo/set_entity_state nor /gazebo/set_model_state is ready yet."
                )
                self._missing_service_warned = True
            return

        self._missing_service_warned = False

        for robot in self._robot_names:
            snapshot = self._snapshots.get(robot)
            if snapshot is None:
                continue

            cmd_world = self._get_latest_world_command(robot, snapshot.pose)
            if cmd_world is None:
                continue

            if entity_ready:
                self._send_set_entity_state(robot, snapshot.pose, cmd_world)
            elif model_ready:
                self._send_set_model_state(robot, snapshot.pose, cmd_world)

    def _get_latest_world_command(self, robot: str, pose: Pose) -> Optional[Twist]:
        timed_cmd = self._commands.get(robot)
        if timed_cmd is None:
            return None

        age = (self.get_clock().now() - timed_cmd.stamp).nanoseconds / 1e9
        if age > self._command_timeout_sec:
            return self._zero_twist()

        q = pose.orientation
        linear_world = self._body_to_world(
            [
                timed_cmd.twist_body.linear.x,
                timed_cmd.twist_body.linear.y,
                timed_cmd.twist_body.linear.z,
            ],
            q.x,
            q.y,
            q.z,
            q.w,
        )
        angular_world = self._body_to_world(
            [
                timed_cmd.twist_body.angular.x,
                timed_cmd.twist_body.angular.y,
                timed_cmd.twist_body.angular.z,
            ],
            q.x,
            q.y,
            q.z,
            q.w,
        )

        out = Twist()
        out.linear.x = linear_world[0]
        out.linear.y = linear_world[1]
        out.linear.z = linear_world[2]
        out.angular.x = angular_world[0]
        out.angular.y = angular_world[1]
        out.angular.z = angular_world[2]
        return out

    def _send_set_entity_state(self, robot: str, pose: Pose, twist_world: Twist) -> None:
        req = SetEntityState.Request()
        state = EntityState()
        state.name = robot
        state.pose = self._copy_pose(pose)
        state.twist = self._copy_twist(twist_world)
        state.reference_frame = "world"
        req.state = state
        self._set_entity_client.call_async(req)

    def _send_set_model_state(self, robot: str, pose: Pose, twist_world: Twist) -> None:
        req = SetModelState.Request()
        state = ModelState()
        state.model_name = robot
        state.pose = self._copy_pose(pose)
        state.twist = self._copy_twist(twist_world)
        state.reference_frame = "world"
        req.model_state = state
        self._set_model_client.call_async(req)

    def _compose_state12(self, pose: Pose, twist_world: Twist) -> List[float]:
        q = pose.orientation
        roll, pitch, yaw = self._quaternion_to_euler(q.x, q.y, q.z, q.w)

        linear_body = self._world_to_body(
            [twist_world.linear.x, twist_world.linear.y, twist_world.linear.z],
            q.x,
            q.y,
            q.z,
            q.w,
        )
        angular_body = self._world_to_body(
            [twist_world.angular.x, twist_world.angular.y, twist_world.angular.z],
            q.x,
            q.y,
            q.z,
            q.w,
        )

        return [
            pose.position.x,
            pose.position.y,
            pose.position.z,
            roll,
            pitch,
            yaw,
            linear_body[0],
            linear_body[1],
            linear_body[2],
            angular_body[0],
            angular_body[1],
            angular_body[2],
        ]

    @staticmethod
    def _copy_pose(src: Pose) -> Pose:
        out = Pose()
        out.position.x = src.position.x
        out.position.y = src.position.y
        out.position.z = src.position.z
        out.orientation.x = src.orientation.x
        out.orientation.y = src.orientation.y
        out.orientation.z = src.orientation.z
        out.orientation.w = src.orientation.w
        return out

    @staticmethod
    def _copy_twist(src: Twist) -> Twist:
        out = Twist()
        out.linear.x = src.linear.x
        out.linear.y = src.linear.y
        out.linear.z = src.linear.z
        out.angular.x = src.angular.x
        out.angular.y = src.angular.y
        out.angular.z = src.angular.z
        return out

    @staticmethod
    def _zero_twist() -> Twist:
        return Twist()

    @staticmethod
    def _quaternion_to_euler(x: float, y: float, z: float, w: float) -> Tuple[float, float, float]:
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        sinp = 2.0 * (w * y - z * x)
        sinp = max(-1.0, min(1.0, sinp))
        pitch = math.asin(sinp)

        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw

    @staticmethod
    def _rotation_matrix_from_quaternion(
        x: float, y: float, z: float, w: float
    ) -> List[List[float]]:
        r00 = 1.0 - 2.0 * (y * y + z * z)
        r01 = 2.0 * (x * y - z * w)
        r02 = 2.0 * (x * z + y * w)

        r10 = 2.0 * (x * y + z * w)
        r11 = 1.0 - 2.0 * (x * x + z * z)
        r12 = 2.0 * (y * z - x * w)

        r20 = 2.0 * (x * z - y * w)
        r21 = 2.0 * (y * z + x * w)
        r22 = 1.0 - 2.0 * (x * x + y * y)

        return [
            [r00, r01, r02],
            [r10, r11, r12],
            [r20, r21, r22],
        ]

    @classmethod
    def _world_to_body(
        cls, vector_world: List[float], qx: float, qy: float, qz: float, qw: float
    ) -> List[float]:
        rotation = cls._rotation_matrix_from_quaternion(qx, qy, qz, qw)
        # body = R^T * world
        return [
            rotation[0][0] * vector_world[0]
            + rotation[1][0] * vector_world[1]
            + rotation[2][0] * vector_world[2],
            rotation[0][1] * vector_world[0]
            + rotation[1][1] * vector_world[1]
            + rotation[2][1] * vector_world[2],
            rotation[0][2] * vector_world[0]
            + rotation[1][2] * vector_world[1]
            + rotation[2][2] * vector_world[2],
        ]

    @classmethod
    def _body_to_world(
        cls, vector_body: List[float], qx: float, qy: float, qz: float, qw: float
    ) -> List[float]:
        rotation = cls._rotation_matrix_from_quaternion(qx, qy, qz, qw)
        # world = R * body
        return [
            rotation[0][0] * vector_body[0]
            + rotation[0][1] * vector_body[1]
            + rotation[0][2] * vector_body[2],
            rotation[1][0] * vector_body[0]
            + rotation[1][1] * vector_body[1]
            + rotation[1][2] * vector_body[2],
            rotation[2][0] * vector_body[0]
            + rotation[2][1] * vector_body[1]
            + rotation[2][2] * vector_body[2],
        ]


def main(args=None) -> None:
    rclpy.init(args=args)
    node = MultiRobotStateInterface()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

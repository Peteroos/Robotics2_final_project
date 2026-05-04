#!/usr/bin/env python3

"""Launch three aerial robots with random ground spawn positions.

The world file (three_aerial.world) contains a single red sphere target and
several obstacles. This launch script:

1. Randomly samples a ground-level (x, y, z=0.1) spawn position for each of
   the three drones, avoiding the fence, the obstacles, and each other.
2. Sends all three drones to the same red sphere target.
3. Holds them on the ground until the user publishes
   std_msgs/Bool(true) on /start_flight.

Reproducibility: set the environment variable DRONE_SPAWN_SEED to an integer
before launching, or pass `seed:=<int>` to this launch file.
"""

import math
import os
import random
from typing import List, Tuple

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


# ---------------------------------------------------------------------------
# Scene knowledge (must stay in sync with three_aerial.world)
# ---------------------------------------------------------------------------

# Physical obstacle footprints shared between the random spawner and the
# APF avoidance controller.
#   (x, y, physical_xy_radius, z_top)
# `physical_xy_radius` is the tight bounding radius of the obstacle itself
# (no drone margin). The drone half-span and safety margin are added in the
# PID controller via `drone_radius` and `avoidance_margin`.
PHYSICAL_OBSTACLES: List[Tuple[float, float, float, float]] = [
    (4.0, 3.5, 0.90, 3.5),   # obstacle_block_blue  (1.2 x 1.2 x 3.5)
    (-4.5, -4.0, 1.05, 4.0), # obstacle_block_orange (1.0 x 1.8 x 4.0)
    (-3.0, 4.0, 0.70, 3.0),  # obstacle_cylinder_green (r=0.7, h=3.0)
    (3.0, -4.0, 0.60, 4.5),  # obstacle_cylinder_purple (r=0.6, h=4.5)
    (6.0, -6.0, 0.60, 5.0),  # obstacle_pillar_1 (0.8 x 0.8 x 5.0)
    (-6.0, 6.0, 0.60, 5.0),  # obstacle_pillar_2 (0.8 x 0.8 x 5.0)
    (0.0, -6.0, 0.85, 2.5),  # obstacle_short_box (1.4 x 0.8 x 2.5)
]

# Spawn-time safe keep-out circles: the physical radius plus enough margin
# so a randomly chosen drone position is already well clear of the obstacle.
_SPAWN_SAFETY_MARGIN = 0.9  # ≈ drone half-span (0.3) + comfort buffer
OBSTACLES: List[Tuple[float, float, float]] = [
    (x, y, r + _SPAWN_SAFETY_MARGIN) for x, y, r, _ in PHYSICAL_OBSTACLES
]

# Fence walls live at +/-10; keep 1.5 m clearance from the inner face.
SPAWN_XY_MIN = -8.5
SPAWN_XY_MAX = 8.5

# Keep-together constraints for the random spawn.
#   MIN_DRONE_SEPARATION  - minimum XY distance between any two drones
#   CLUSTER_RADIUS        - all drones must lie within this radius of a
#                           randomly chosen cluster centre, so the team
#                           spawns close together
MIN_DRONE_SEPARATION = 1.2
CLUSTER_RADIUS = 3.0

# Single target in the sky (must match target_marker in three_aerial.world).
TARGET_XYZ: Tuple[float, float, float] = (0.0, 5.0, 3.5)

# Ground-spawn height: drone body half-height ~0.03, so 0.1 rests on the floor.
GROUND_Z = 0.1


def _resolve_seed() -> int:
    env_seed = os.environ.get("DRONE_SPAWN_SEED")
    if env_seed:
        try:
            return int(env_seed)
        except ValueError:
            pass
    # Derive from system entropy so each launch gets a different layout.
    return random.SystemRandom().randrange(1 << 31)


def _is_xy_free(x: float, y: float, extra_radius: float = 0.0) -> bool:
    """Return True if (x, y) clears every obstacle footprint."""
    for ox, oy, r in OBSTACLES:
        if math.hypot(x - ox, y - oy) < r + extra_radius:
            return False
    return True


def _sample_cluster_center(rng: random.Random, max_attempts: int = 2000) -> Tuple[float, float]:
    """Pick a random XY that leaves room for a whole cluster around it."""
    inner_min = SPAWN_XY_MIN + CLUSTER_RADIUS
    inner_max = SPAWN_XY_MAX - CLUSTER_RADIUS
    for _ in range(max_attempts):
        cx = rng.uniform(inner_min, inner_max)
        cy = rng.uniform(inner_min, inner_max)

        # Keep the cluster centre at least CLUSTER_RADIUS away from every
        # obstacle so the drones around it still have free spots to land in.
        if not _is_xy_free(cx, cy, extra_radius=CLUSTER_RADIUS * 0.5):
            continue
        # Stay away from the target's XY footprint so take-off has some travel.
        if math.hypot(cx - TARGET_XYZ[0], cy - TARGET_XYZ[1]) < 2.5:
            continue
        return cx, cy

    raise RuntimeError(
        "Could not find a cluster centre; scene may be too cluttered."
    )


def _sample_ground_positions(num_robots: int, seed: int) -> List[Tuple[float, float]]:
    rng = random.Random(seed)
    cluster_cx, cluster_cy = _sample_cluster_center(rng)

    chosen: List[Tuple[float, float]] = []
    max_attempts = 4000

    for _ in range(num_robots):
        for _attempt in range(max_attempts):
            # Uniform sampling inside the cluster disk around (cx, cy).
            r = CLUSTER_RADIUS * math.sqrt(rng.random())
            theta = rng.uniform(0.0, 2.0 * math.pi)
            x = cluster_cx + r * math.cos(theta)
            y = cluster_cy + r * math.sin(theta)

            if x < SPAWN_XY_MIN or x > SPAWN_XY_MAX:
                continue
            if y < SPAWN_XY_MIN or y > SPAWN_XY_MAX:
                continue
            if not _is_xy_free(x, y):
                continue
            if any(
                math.hypot(x - cx, y - cy) < MIN_DRONE_SEPARATION for cx, cy in chosen
            ):
                continue
            if math.hypot(x - TARGET_XYZ[0], y - TARGET_XYZ[1]) < 1.5:
                continue

            chosen.append((x, y))
            break
        else:
            raise RuntimeError(
                "Failed to find a collision-free random spawn position inside "
                "the cluster disk. Consider enlarging CLUSTER_RADIUS or "
                "reducing the number of obstacles."
            )

    return chosen


def _spawn_robot(name: str, x: float, y: float, z: float) -> Node:
    return Node(
        package="gazebo_ros",
        executable="spawn_entity.py",
        output="screen",
        arguments=[
            "-entity",
            name,
            "-file",
            PathJoinSubstitution(
                [FindPackageShare("multi_aerial_gazebo"), "urdf", "aerial_robot.urdf"]
            ),
            "-x",
            str(x),
            "-y",
            str(y),
            "-z",
            str(z),
        ],
    )


def generate_launch_description() -> LaunchDescription:
    gui_arg = DeclareLaunchArgument(
        "gui",
        default_value="false",
        description="Set to true when launching from a local graphical desktop terminal.",
    )

    enable_pid_arg = DeclareLaunchArgument(
        "enable_pid",
        default_value="true",
        description="Enable multi-robot PID position controller.",
    )

    seed = _resolve_seed()
    spawn_xy = _sample_ground_positions(3, seed)
    robot_names = ["aerial_robot_1", "aerial_robot_2", "aerial_robot_3"]

    # Log a tight summary so the user knows where the cluster landed and how
    # tightly packed the drones are.
    cx = sum(x for x, _ in spawn_xy) / len(spawn_xy)
    cy = sum(y for _, y in spawn_xy) / len(spawn_xy)
    pairwise = max(
        math.hypot(a[0] - b[0], a[1] - b[1])
        for i, a in enumerate(spawn_xy)
        for b in spawn_xy[i + 1:]
    )
    print("[three_aerial_gazebo.launch] Random spawn seed = %d" % seed)
    print(
        "[three_aerial_gazebo.launch] Cluster centre ~ (%.2f, %.2f), max pair distance = %.2fm"
        % (cx, cy, pairwise)
    )
    for name, (x, y) in zip(robot_names, spawn_xy):
        print(
            "[three_aerial_gazebo.launch] %s spawn = (%.2f, %.2f, %.2f)"
            % (name, x, y, GROUND_Z)
        )
    print(
        "[three_aerial_gazebo.launch] Shared red target = (%.2f, %.2f, %.2f)"
        % TARGET_XYZ
    )

    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare("gazebo_ros"), "launch", "gazebo.launch.py"])
        ),
        launch_arguments={
            "gui": LaunchConfiguration("gui"),
            "world": PathJoinSubstitution(
                [FindPackageShare("multi_aerial_gazebo"), "worlds", "three_aerial.world"]
            )
        }.items(),
    )

    spawn_nodes = [
        _spawn_robot(name, x, y, GROUND_Z)
        for name, (x, y) in zip(robot_names, spawn_xy)
    ]

    delayed_spawns = TimerAction(
        period=5.0,
        actions=spawn_nodes,
    )

    state_interface = Node(
        package="multi_aerial_gazebo",
        executable="multi_robot_state_interface.py",
        name="multi_robot_state_interface",
        output="screen",
        parameters=[
            {
                "robot_names": robot_names,
                "command_rate_hz": 20.0,
                "command_timeout_sec": 0.5,
            }
        ],
    )

    delayed_state_interface = TimerAction(
        period=7.0,
        actions=[state_interface],
    )

    # All three drones share the same (x, y, z) goal: the red sphere in sky.
    flight_targets = list(TARGET_XYZ) * len(robot_names)

    # Flatten PHYSICAL_OBSTACLES into [x, y, r, z_top, x, y, r, z_top, ...]
    # for the PID controller's APF avoidance module.
    obstacle_param: List[float] = []
    for x, y, r, z_top in PHYSICAL_OBSTACLES:
        obstacle_param.extend([float(x), float(y), float(r), float(z_top)])

    pid_controller = Node(
        package="multi_aerial_gazebo",
        executable="multi_robot_pid_controller.py",
        name="multi_robot_pid_controller",
        output="screen",
        condition=IfCondition(LaunchConfiguration("enable_pid")),
        parameters=[
            {
                "robot_names": robot_names,
                "default_target_positions": flight_targets,
                "default_target_yaws": [0.0] * len(robot_names),
                "control_rate_hz": 20.0,
                "kp_pos": [1.2, 1.2, 1.5],
                "ki_pos": [0.0, 0.0, 0.0],
                "kd_pos": [0.3, 0.3, 0.4],
                "kp_yaw": 1.5,
                "ki_yaw": 0.0,
                "kd_yaw": 0.2,
                "max_linear_speed": 1.0,
                "max_yaw_rate": 1.0,
                "wait_for_start_signal": True,
                "start_trigger_topic": "/start_flight",
                # APF obstacle avoidance
                "obstacles": obstacle_param,
                "drone_radius": 0.30,
                "avoidance_margin": 0.40,
                "avoidance_influence": 1.8,
                "avoidance_k_rep": 1.6,
                "avoidance_lift_gain": 0.6,
                "mutual_influence": 1.2,
                "mutual_k_rep": 0.8,
            }
        ],
    )

    delayed_pid_controller = TimerAction(
        period=8.0,
        actions=[pid_controller],
    )

    return LaunchDescription(
        [
            gui_arg,
            enable_pid_arg,
            gazebo_launch,
            delayed_spawns,
            delayed_state_interface,
            delayed_pid_controller,
        ]
    )

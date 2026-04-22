#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


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

    spawn_robot_1 = _spawn_robot("aerial_robot_1", 0.0, 0.0, 1.0)
    spawn_robot_2 = _spawn_robot("aerial_robot_2", 2.0, 0.0, 1.0)
    spawn_robot_3 = _spawn_robot("aerial_robot_3", -2.0, 0.0, 1.0)

    delayed_spawns = TimerAction(
        period=5.0,
        actions=[
            spawn_robot_1,
            spawn_robot_2,
            spawn_robot_3,
        ],
    )

    return LaunchDescription(
        [
            gui_arg,
            gazebo_launch,
            delayed_spawns,
        ]
    )

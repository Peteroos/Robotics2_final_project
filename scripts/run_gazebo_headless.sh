#!/usr/bin/env bash
set -e

cd /home/yahboom/robotics2_finalproject
source /opt/ros/humble/setup.bash
source install/setup.bash

# Safe mode for SSH terminals (no Gazebo client window).
ros2 launch multi_aerial_gazebo three_aerial_gazebo.launch.py gui:=false

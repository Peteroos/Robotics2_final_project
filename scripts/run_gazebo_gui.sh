#!/usr/bin/env bash
set -e

cd /home/yahboom/robotics2_finalproject
source /opt/ros/humble/setup.bash
source install/setup.bash

# Helps in VM/older GPU drivers and still keeps GUI enabled.
export LIBGL_ALWAYS_SOFTWARE=1

ros2 launch multi_aerial_gazebo three_aerial_gazebo.launch.py gui:=true

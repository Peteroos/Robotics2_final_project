#!/usr/bin/env bash
set -e

cd /home/yahboom/robotics2_finalproject
source /opt/ros/humble/setup.bash
colcon build --symlink-install

echo "Build done."

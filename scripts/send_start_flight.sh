#!/usr/bin/env bash
# Usage:
#   ./send_start_flight.sh            # take off (data: true)
#   ./send_start_flight.sh start      # take off
#   ./send_start_flight.sh stop       # hold in place (data: false)
set -e

cd /home/yahboom/robotics2_finalproject
source /opt/ros/humble/setup.bash
source install/setup.bash

CMD="${1:-start}"
case "$CMD" in
  start|takeoff|fly|true|1)
    DATA="true"
    ;;
  stop|hold|land|false|0)
    DATA="false"
    ;;
  *)
    echo "Unknown command: $CMD. Use 'start' or 'stop'." >&2
    exit 1
    ;;
esac

echo "Publishing /start_flight data:${DATA}"
ros2 topic pub --once /start_flight std_msgs/msg/Bool "{data: ${DATA}}"

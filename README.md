# Multi Aerial Gazebo (3 Robots)

This workspace provides a minimal Gazebo simulation with three aerial robots.

## Quick Start (Recommended)

Build:

```bash
cd /home/yahboom/robotics2_finalproject
./scripts/run_build.sh
```

Launch with Gazebo GUI (run this in a local graphical desktop terminal, not a pure SSH terminal):

```bash
cd /home/yahboom/robotics2_finalproject
./scripts/run_gazebo_gui.sh
```

Launch headless mode for SSH terminal:

```bash
cd /home/yahboom/robotics2_finalproject
./scripts/run_gazebo_headless.sh
```

## Manual Launch

```bash
cd /home/yahboom/robotics2_finalproject
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch multi_aerial_gazebo three_aerial_gazebo.launch.py gui:=true
```

### Launch Without PID

```bash
cd /home/yahboom/robotics2_finalproject
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch multi_aerial_gazebo three_aerial_gazebo.launch.py gui:=false enable_pid:=false
```

## 12-State Interface (Per Robot)

The launch file now starts a bridge node that exposes each robot's 12-state vector and accepts body-frame velocity commands.

State vector order:

```
[x, y, z, roll, pitch, yaw, vx, vy, vz, p, q, r]
```

- `x, y, z`: position in world frame
- `roll, pitch, yaw`: attitude in world frame
- `vx, vy, vz`: linear velocity in body frame
- `p, q, r`: angular velocity in body frame

Published topics:

- `/aerial_robot_1/state12` (`std_msgs/Float64MultiArray`)
- `/aerial_robot_2/state12` (`std_msgs/Float64MultiArray`)
- `/aerial_robot_3/state12` (`std_msgs/Float64MultiArray`)
- `/multi_aerial/state12_all` (`std_msgs/Float64MultiArray`, flattened by robot order)

Command topics (for upper-level controllers):

- `/aerial_robot_1/cmd_vel_body` (`geometry_msgs/Twist`)
- `/aerial_robot_2/cmd_vel_body` (`geometry_msgs/Twist`)
- `/aerial_robot_3/cmd_vel_body` (`geometry_msgs/Twist`)

`cmd_vel_body` meaning:

- `linear.x/y/z` -> desired body-frame `vx/vy/vz`
- `angular.x/y/z` -> desired body-frame `p/q/r`

Quick test:

```bash
# Observe robot-1 state12
ros2 topic echo /aerial_robot_1/state12

# Send a forward body velocity and slight yaw rate to robot-1
ros2 topic pub --rate 20 /aerial_robot_1/cmd_vel_body geometry_msgs/msg/Twist \
'{linear: {x: 0.5, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.3}}'
```

## PID Controller (Position + Yaw)

A multi-robot PID node is launched by default (`enable_pid:=true`) and publishes `/cmd_vel_body` for each robot.

- Input state: `/aerial_robot_i/state12`
- Input target: `/aerial_robot_i/target_pose` (`geometry_msgs/Pose`)
- Output command: `/aerial_robot_i/cmd_vel_body`

By default, each robot holds its spawn location:

- robot1 -> `(0, 0, 1)`
- robot2 -> `(2, 0, 1)`
- robot3 -> `(-2, 0, 1)`

### PID Test Commands (Directly Usable)

Terminal A:

```bash
cd /home/yahboom/robotics2_finalproject
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch multi_aerial_gazebo three_aerial_gazebo.launch.py gui:=false
```

Terminal B:

```bash
cd /home/yahboom/robotics2_finalproject
source /opt/ros/humble/setup.bash
source install/setup.bash

# Watch robot1 state
ros2 topic echo /aerial_robot_1/state12
```

Terminal C (send a new target to robot1):

```bash
cd /home/yahboom/robotics2_finalproject
source /opt/ros/humble/setup.bash
source install/setup.bash

ros2 topic pub --once /aerial_robot_1/target_pose geometry_msgs/msg/Pose \
'{position: {x: 1.5, y: 1.0, z: 1.4}, orientation: {x: 0.0, y: 0.0, z: 0.3826834, w: 0.9238795}}'
```

The quaternion above corresponds to yaw about `0.785` rad (~45 deg). You should see robot1 move toward the new target and settle.

## Notes

- Launch file defaults to `gui:=false` to avoid GLX/X11 crashes in SSH sessions.
- For GUI mode, start from your desktop environment terminal.

## Files

- src/multi_aerial_gazebo/launch/three_aerial_gazebo.launch.py
- src/multi_aerial_gazebo/scripts/multi_robot_pid_controller.py
- src/multi_aerial_gazebo/scripts/multi_robot_state_interface.py
- src/multi_aerial_gazebo/urdf/aerial_robot.urdf
- src/multi_aerial_gazebo/worlds/three_aerial.world
- scripts/run_build.sh
- scripts/run_gazebo_gui.sh
- scripts/run_gazebo_headless.sh

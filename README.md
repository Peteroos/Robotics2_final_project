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

## Notes

- Launch file defaults to `gui:=false` to avoid GLX/X11 crashes in SSH sessions.
- For GUI mode, start from your desktop environment terminal.

## Files

- src/multi_aerial_gazebo/launch/three_aerial_gazebo.launch.py
- src/multi_aerial_gazebo/urdf/aerial_robot.urdf
- src/multi_aerial_gazebo/worlds/three_aerial.world
- scripts/run_build.sh
- scripts/run_gazebo_gui.sh
- scripts/run_gazebo_headless.sh

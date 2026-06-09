# rgb_lidar_bev

ROS 2 BEV perception + joystick-driven safety nav for JetAuto Orin Nano.

This repository is currently focused on a single runtime entrypoint:
`bev_nav_stack` (`src/rgb_lidar_bev/bev_nav_stack.py`), which runs:

- `BevNode` (camera + LiDAR + YOLO -> sector proximity map)
- `NavNode` (Xbox Y/X control + forward obstacle e-stop)

## Primary Functionality

`bev_nav_stack` starts both nodes in one process using a `MultiThreadedExecutor`.

- `BevNode`
  - Subscribes to camera image, camera info, LiDAR scan, and TF
  - Runs YOLO person detection and BEV projection per image frame
  - Compresses scene to a forward 180 deg sector map
  - Publishes sector map on `/bev/sector_proximity` as `sensor_msgs/LaserScan`
    - `ranges`: nearest obstacle distance per sector
    - `intensities`: object class per sector (`0=clear`, `1=lidar`, `2=human`)
  - Optional live sector visualization (`ascii` / `tk` / `none`)

- `NavNode`
  - Subscribes to `/ros_robot_controller/joy` (`sensor_msgs/Joy`)
  - Publishes `/cmd_vel` (`geometry_msgs/Twist`)
  - Controller behavior:
    - `button[4]` (Y): clear manual e-stop and enable forward drive
    - `button[3]` (X): latch manual e-stop and stop immediately
  - Safety behavior:
    - Subscribes to `/bev/sector_proximity`
    - E-stops when obstacle is detected in immediate forward cone

## Runtime Architecture

```text
/camera + /camera_info + /scan + TF
               |
            BevNode
               |  publishes LaserScan-encoded sector map
               v
     /bev/sector_proximity
               |
            NavNode <----- /ros_robot_controller/joy
               |
            /cmd_vel
```

## Prerequisites

- ROS 2 Humble environment sourced
- JetAuto bringup running (publishing `/ros_robot_controller/joy`)
- Valid camera/LiDAR/TF topics for your robot configuration
- Python packages available in the same interpreter used by ROS:
  - `ultralytics`
  - `torch` (Jetson-compatible build recommended)
  - `opencv-python-headless`, `numpy`, `scipy`, `pyyaml`

## Build and Source (ament_python)

From your ROS workspace root (the folder containing `src/`):

```bash
source /opt/ros/humble/setup.bash
colcon build --packages-select rgb_lidar_bev --symlink-install
source install/setup.bash
```

Confirm executables:

```bash
ros2 pkg executables | grep rgb_lidar_bev
```

Expected:

- `rgb_lidar_bev bev_nav_stack`
- `rgb_lidar_bev bev_node`
- `rgb_lidar_bev nav_node`

## Launch and Operation

Run both perception + nav together:

```bash
 PYTHONNOUSERSITE=1 ros2 run rgb_lidar_bev bev_nav_stack
```

### Controller Controls

- Press **Y** (`button[4]`) -> enable forward driving
- Press **X** (`button[3]`) -> immediate manual e-stop

### Safety Logic

Even after Y is pressed, nav remains blocked if a close obstacle is detected in
the configured front cone from `/bev/sector_proximity`.

## Key Parameters

### BevNode Parameters

- `camera_topic` (default `/depth_cam/rgb/image_raw`)
- `camera_info_topic` (default `/depth_cam/rgb/camera_info`)
- `lidar_topic` (default `/scan`)
- `base_frame` (default `base_link`)
- `cam_optical_frame` (default `depth_cam_color_optical_frame`)
- `laser_frame` (default `lidar_frame`)
- `model` (default `yolov8n.pt`)
- `conf` (default `0.70`)
- `show` (default `false`) - OpenCV composite window
- `sector_viz` (default `ascii`) - `ascii`, `tk`, or `none`
- `num_sectors` (default `36`)
- `max_range` (default `10.0`)
- `sector_topic` (default `/bev/sector_proximity`)

### NavNode Parameters

- `joy_topic` (default `/ros_robot_controller/joy`)
- `cmd_vel_topic` (default `/cmd_vel`)
- `sector_topic` (default `/bev/sector_proximity`)
- `forward_speed` (default `0.20`)
- `publish_rate_hz` (default `20.0`)
- `btn_drive_index` (default `4`)  # Y
- `btn_estop_index` (default `3`)  # X
- `front_half_angle_deg` (default `15.0`)
- `estop_dist_static_m` (default `0.45`)
- `estop_dist_human_m` (default `0.70`)

## Quick Topic Checks

```bash
ros2 topic echo /ros_robot_controller/joy
ros2 topic echo /bev/sector_proximity
ros2 topic echo /cmd_vel
```

## Troubleshooting (Common)

- `Package 'rgb_lidar_bev' not found`
  - Build from workspace root, then `source install/setup.bash`
- `No executable found`
  - Confirm `setup.cfg` installs scripts to `lib/rgb_lidar_bev`
  - Rebuild and re-source
- `ModuleNotFoundError: ultralytics`
  - Install dependencies into the same Python environment ROS uses
- NumPy / OpenCV ABI mismatch errors
  - Align package versions in ROS runtime Python environment

## Scope Note

This README intentionally prioritizes the launch and operation path for
`bev_nav_stack`. Offline replay/demo workflows are not documented here.

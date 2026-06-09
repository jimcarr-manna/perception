"""
ROS 2 integration helpers (optional).

Install rclpy and sensor_msgs in your ROS 2 environment; this module is not imported by default.

Typical usage:
- Subclass or copy patterns: Image + CameraInfo → K, dist
- tf2: lookup_transform('base_link', 'camera_optical_frame', rclpy.time.Time)
- LaserScan → ranges, angles arrays
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as R

from rgb_lidar_bev.datatypes import SE3

def camera_info_to_K_dist(camera_info: Any) -> tuple[np.ndarray, np.ndarray | None]:
    """Build K and (optional) distortion vector from sensor_msgs/msg/CameraInfo."""
    K = np.array(camera_info.k, dtype=np.float64).reshape(3, 3)
    d = np.array(camera_info.d, dtype=np.float64)
    if d.size == 0:
        return K, None
    return K, d

def quat_to_R(x: float, y: float, z: float, w: float) -> np.ndarray:
    quat = [x, y, z, w]
    r2 = R.from_quat(quat).as_matrix()
    return r2

def laser_scan_to_ranges_angles(scan: Any) -> tuple[np.ndarray, np.ndarray]:
    """Unwrap sensor_msgs/msg/LaserScan to parallel numpy arrays."""
    n = len(scan.ranges)
    a0 = float(scan.angle_min)
    step = float(scan.angle_increment)
    angles = np.array([a0 + i * step for i in range(n)], dtype=np.float64)
    ranges = np.array(scan.ranges, dtype=np.float64)
    if scan.angle_increment < 0:
        angles = np.flip(angles)
        ranges = np.flip(ranges)
    if not np.isfinite(step):
        raise ValueError("LaserScan angle_increment not finite")
    return ranges, angles

def transform_stamped_to_se3(tf_stamped) -> SE3:
    """Convert geometry_msgs/TransformStamped → SE3."""
    tr = tf_stamped.transform.translation
    ro = tf_stamped.transform.rotation
    R = quat_to_R(ro.x, ro.y, ro.z, ro.w)
    return SE3.from_Rt(R, np.array([tr.x, tr.y, tr.z], dtype=np.float64))
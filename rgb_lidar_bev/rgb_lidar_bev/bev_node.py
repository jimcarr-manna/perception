"""
ROS 2 live node: subscribes to camera, LiDAR, and TF topics, runs
YOLO person detection + BEV pipeline on every incoming camera frame.

Run (after sourcing your ROS 2 workspace):
    ros2 run rgb_lidar_bev ros2_node
    ros2 run rgb_lidar_bev ros2_node --ros-args \
        -p camera_topic:=/camera/color/image_raw \
        -p lidar_topic:=/scan \
        -p conf:=0.5 \
        -p show:=true
"""
from __future__ import annotations

import threading
from typing import Any

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image, LaserScan
import tf2_ros

from rgb_lidar_bev.datatypes import SE3
from rgb_lidar_bev.ros2_bridge import (
    camera_info_to_K_dist,
    laser_scan_to_ranges_angles,
    quat_to_R,
)
from rgb_lidar_bev.frame_processor import _COMPOSITE_H, _COMPOSITE_W, _process_image
from rgb_lidar_bev.sector_proximity import generate_sector_proximity
from rgb_lidar_bev.sector_viz import SectorAsciiViz, SectorFanViz


def _transform_stamped_to_se3(tf_stamped) -> SE3:
    """Convert a geometry_msgs/TransformStamped to SE3."""
    tr = tf_stamped.transform.translation
    ro = tf_stamped.transform.rotation
    R = quat_to_R(ro.x, ro.y, ro.z, ro.w)
    return SE3.from_Rt(R, np.array([tr.x, tr.y, tr.z], dtype=np.float64))


def _image_msg_to_bgr(msg: Image) -> np.ndarray:
    """Manual Image → numpy BGR without cv_bridge dependency."""
    if msg.encoding in ("rgb8", "bgr8"):
        buf = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        return buf if msg.encoding == "bgr8" else buf[:, :, ::-1]
    if msg.encoding == "mono8":
        return cv2.cvtColor(
            np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width),
            cv2.COLOR_GRAY2BGR,
        )
    raise ValueError(f"Unsupported image encoding: {msg.encoding}")

class BevNode(Node):
    def __init__(self) -> None:
        super().__init__("rgb_lidar_bev")

        # ── ROS 2 parameters ──────────────────────────────────────────────
        self.declare_parameter("camera_topic",      "/depth_cam/rgb/image_raw")
        self.declare_parameter("camera_info_topic", "/depth_cam/rgb/camera_info")
        self.declare_parameter("lidar_topic",       "/scan")
        self.declare_parameter("base_frame",        "base_link")
        self.declare_parameter("cam_optical_frame", "depth_cam_color_optical_frame")
        self.declare_parameter("laser_frame",       "lidar_frame")
        self.declare_parameter("model",             "yolov8n.pt")
        self.declare_parameter("conf",              0.70)
        self.declare_parameter("show",              False)
        self.declare_parameter("camera_tf_use_latest", True)
        self.declare_parameter("sector_viz",        "ascii")   # tk | ascii | none
        self.declare_parameter("num_sectors",       36)
        self.declare_parameter("max_range",         10.0)
        self.declare_parameter("sector_topic",      "/bev/sector_proximity")

        cam_topic   = str(self.get_parameter("camera_topic").value)
        info_topic  = str(self.get_parameter("camera_info_topic").value)
        lidar_topic = str(self.get_parameter("lidar_topic").value)
        sector_topic = str(self.get_parameter("sector_topic").value)
        model_path  = str(self.get_parameter("model").value)

        # Heavy raster composite — only create the window when explicitly
        # shown. Over a remote X link (xQuartz/SSH) this avoids pushing full
        # frames; the lightweight vector fan below is the remote-friendly view.
        self._show = bool(self.get_parameter("show").value)
        if self._show:
            cv2.namedWindow("RGB-LiDAR BEV", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("RGB-LiDAR BEV", _COMPOSITE_W, _COMPOSITE_H)
            cv2.waitKey(200)

        # ── Lightweight sector-proximity view (remote-friendly) ───────────
        # 'tk'    → vector fan window (X11 draw commands, not rasters)
        # 'ascii' → in-terminal radar (text only, no X server at all)
        # 'none'  → disabled
        self._num_sectors = int(self.get_parameter("num_sectors").value or 36)
        self._max_range   = float(self.get_parameter("max_range").value or 10.0)
        mode = str(self.get_parameter("sector_viz").value).lower()
        self._sector_viz: SectorFanViz | SectorAsciiViz | None = None
        if mode == "tk":
            self._sector_viz = SectorFanViz(
                num_sectors=self._num_sectors, max_range=self._max_range
            )
        elif mode == "ascii":
            self._sector_viz = SectorAsciiViz(
                num_sectors=self._num_sectors, max_range=self._max_range
            )

        # ── YOLO model ────────────────────────────────────────────────────
        from ultralytics.models import YOLO
        self._model: Any = YOLO(str(model_path))

        # ── TF buffer ─────────────────────────────────────────────────────
        self._tf_buffer   = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        # ── Shared state (written by subscriber callbacks) ─────────────────
        self._lock            = threading.Lock()
        self._K: np.ndarray | None           = None
        self._dist: np.ndarray | None        = None
        self._latest_scan: tuple | None      = None   # (ranges, angles, stamp)
        self._T_laser_cached: SE3 | None     = None   # static, computed once
        self._T_cam_cached: SE3 | None       = None   # dynamic, computed on image timestamp
        self._frame_idx = 0

        # ── Subscriptions ─────────────────────────────────────────────────
        # CameraInfo: latched-style — keep QoS history=1, reliability=RELIABLE
        from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
        latching_qos = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        self.create_subscription(CameraInfo, info_topic,  self._camera_info_cb, latching_qos)
        self.create_subscription(LaserScan,  lidar_topic, self._scan_cb,         10)
        self.create_subscription(Image,      cam_topic,   self._image_cb,        10)
        self._sector_pub = self.create_publisher(LaserScan, sector_topic, 10)

        self.get_logger().info(
            f"BevNode ready — camera: {cam_topic}, lidar: {lidar_topic}, sector: {sector_topic}"
        )

    # ── Callbacks ─────────────────────────────────────────────────────────

    def _camera_info_cb(self, msg: CameraInfo) -> None:
        with self._lock:
            if self._K is None:
                self._K, self._dist = camera_info_to_K_dist(msg)
                self.get_logger().info("Camera intrinsics received and cached.")

    def _scan_cb(self, msg: LaserScan) -> None:
        ranges, angles = laser_scan_to_ranges_angles(msg)
        with self._lock:
            self._latest_scan = (ranges, angles, msg.header.stamp)

    def _image_cb(self, msg: Image) -> None:
        with self._lock:
            K    = self._K
            dist = self._dist
            scan = self._latest_scan

        if K is None:
            self.get_logger().warn("Waiting for CameraInfo…", throttle_duration_sec=5.0)
            return
        if scan is None:
            self.get_logger().warn("Waiting for first LiDAR scan…", throttle_duration_sec=5.0)
            return

        ranges, angles, _ = scan
        base_frame        = str(self.get_parameter("base_frame").value)
        cam_optical_frame = str(self.get_parameter("cam_optical_frame").value)
        laser_frame       = str(self.get_parameter("laser_frame").value)
        conf              = float(self.get_parameter("conf").value or 0.7)
        
        # ── Laser transform (static — look up once, then cache) ────────────
        T_laser = self._T_laser_cached
        if T_laser is None:
            try:
                tf_stamped = self._tf_buffer.lookup_transform(
                    base_frame, laser_frame, Time()
                )
                T_laser = _transform_stamped_to_se3(tf_stamped)
                self._T_laser_cached = T_laser
            except Exception as e:
                self.get_logger().warn(f"Laser TF not yet available: {e}", throttle_duration_sec=5.0)
                return

        # ── Camera transform (fixed mount — latest TF, cached) ─────────────
        T_cam = self._T_cam_cached
        if T_cam is None:
            try:
                tf_stamped = self._tf_buffer.lookup_transform(
                    base_frame,
                    cam_optical_frame,
                    Time(),
                )
                T_cam = _transform_stamped_to_se3(tf_stamped)
                self._T_cam_cached = T_cam
            except Exception as e:
                self.get_logger().warn(
                    f"Camera TF not yet available: {e}",
                    throttle_duration_sec=5.0,
                )
                return
            
        # ── Run detection + BEV ────────────────────────────────────────────
        try:
            frame_bgr = _image_msg_to_bgr(msg)
        except ValueError as e:
            self.get_logger().error(str(e))
            return

        frame_idx = self._frame_idx
        self._frame_idx += 1

        feet_bev, lidar_xy, composite, canvas, annotated = _process_image(
            frame_bgr, ranges, angles, T_cam, T_laser,
            K, dist, self._model, conf,
            frame_idx=frame_idx,
        )

        # ── Sector-proximity compression + publish + lightweight render ─────
        sector_map = generate_sector_proximity(
            lidar_xy, feet_bev,
            num_sectors=self._num_sectors, max_range=self._max_range,
        )
        
        sector_scan = LaserScan()
        sector_scan.header = msg.header
        sector_scan.header.frame_id = base_frame
        sector_scan.angle_min = float(sector_map["angle_min"])
        sector_scan.angle_max = float(sector_map["angle_max"])
        sector_scan.angle_increment = float(sector_map["angle_increment"])
        sector_scan.range_min = 0.0
        sector_scan.range_max = float(self._max_range)
        sector_scan.ranges = [float(r) for r in sector_map["ranges"]]
        # Reuse LaserScan intensities to encode object type classes:
        # 0 = CLEAR, 1 = LIDAR_STATIC, 2 = HUMAN_DANGEROUS.
        sector_scan.intensities = [float(t) for t in sector_map["object_types"]]
        self._sector_pub.publish(sector_scan)

        if self._sector_viz is not None:
            self._sector_viz.update(sector_map)

        if self._show:
            cv2.imshow("RGB-LiDAR BEV", composite)
            cv2.waitKey(1)

def main(args=None) -> None:
    rclpy.init(args=args)
    node = BevNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node._sector_viz is not None:
            node._sector_viz.close()
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

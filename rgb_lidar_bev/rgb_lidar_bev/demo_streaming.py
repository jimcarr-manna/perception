"""
Streaming demo: replay offline2 flat-file captures in frame order,
running YOLO person detection + BEV pipeline on every camera frame.

Simulates a live stream without requiring a running ROS 2 environment.

Data sources (files/offline2/):
    clip_rgb.mp4          — RGB video
    lidar.npz             — per-scan LiDAR arrays with nanosecond timestamps
    tf.json               — TF static + dynamic transforms with nanosecond timestamps
    camera_info_rgb.yaml  — camera calibration

Run:
    python -m rgb_lidar_bev.demo_streaming
    python -m rgb_lidar_bev.demo_streaming --show
    python -m rgb_lidar_bev.demo_streaming --out-dir ./frames --conf 0.4

Requires:
    pip install 'rgb-lidar-bev[yolo]'
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from rgb_lidar_bev.datatypes import SE3
from rgb_lidar_bev.detection import yolo_boxes_to_detections
from rgb_lidar_bev.lidar_fusion import points_to_xy, scan_to_points_base
from rgb_lidar_bev.pipeline import process_frame_boxes
from rgb_lidar_bev.ros2_bridge import quat_to_R
from rgb_lidar_bev.viz import draw_bev
from rgb_lidar_bev.sector_proximity import generate_sector_proximity   


FILES = Path(__file__).parent / "files"
OFFLINE2 = FILES / "offline2"

DEFAULT_VIDEO = OFFLINE2 / "clip_rgb.mp4"
DEFAULT_TF_JSON = OFFLINE2 / "tf.json"
DEFAULT_LIDAR_NPZ = OFFLINE2 / "lidar.npz"
DEFAULT_CAMERA_INFO = OFFLINE2 / "camera_info_rgb.yaml"

# ---------------------------------------------------------------------------
# Offline data container
# ---------------------------------------------------------------------------


@dataclass
class OfflineData:
    """Pre-loaded sensor data from offline2 flat files."""

    # Static transforms (never change)
    static_tfs: dict[tuple[str, str], SE3] = field(default_factory=dict)

    # Full 4-link dynamic camera TF chain (outermost first):
    #   base_link → depth_cam_link → depth_cam_depth_frame
    #            → depth_cam_color_frame → depth_cam_color_optical_frame
    # Each element is (sorted_timestamps, list[SE3]).
    cam_tf_chain: list[tuple[np.ndarray, list[SE3]]] = field(default_factory=list)

    # LiDAR scans
    lidar_stamp_ns: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    lidar_ranges: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.float32))
    lidar_angle_min: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    lidar_angle_increment: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))

    # Cached laser-in-base transform (static, computed once)
    _T_laser_cached: SE3 | None = field(default=None, repr=False)

    @property
    def T_laser_in_base(self) -> SE3:
        if self._T_laser_cached is None:
            T_bl_ll = self.static_tfs[("base_link", "lidar_link")]
            T_ll_lf = self.static_tfs[("lidar_link", "lidar_frame")]
            self._T_laser_cached = SE3.from_Rt(
                T_bl_ll.R @ T_ll_lf.R,
                T_bl_ll.R @ T_ll_lf.t + T_bl_ll.t,
            )
        return self._T_laser_cached

    def T_cam_in_base(self, ts_ns: int) -> SE3:
        """
        Compose the full 4-link camera TF chain at ts_ns.

        Chain links are stored outermost→innermost, so the correct composition
        iterates in reverse (innermost first):
            p_base = T1.R @ T2.R @ T3.R @ T4.R @ p_optical + …
        """
        R = np.eye(3, dtype=np.float64)
        t = np.zeros(3, dtype=np.float64)
        for chain_idx in reversed(range(len(self.cam_tf_chain))):
            times, se3s = self.cam_tf_chain[chain_idx]
            idx = min(int(np.searchsorted(times, ts_ns)), len(times) - 1)
            link = se3s[idx]
            t = link.R @ t + link.t
            R = link.R @ R
        return SE3.from_Rt(R, t)

    def get_scan(self, ts_ns: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (ranges, angles) arrays for the scan closest to ts_ns."""
        idx = int(np.searchsorted(self.lidar_stamp_ns, ts_ns))
        idx = min(idx, len(self.lidar_stamp_ns) - 1)
        ranges = self.lidar_ranges[idx].astype(np.float64)
        a0 = float(self.lidar_angle_min[idx])
        step = float(self.lidar_angle_increment[idx])
        angles = np.arange(len(ranges), dtype=np.float64) * step + a0
        return ranges, angles


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------


def _json_transform_to_se3(t: dict) -> SE3:
    tr = t["transform"]["translation"]
    ro = t["transform"]["rotation"]
    R = quat_to_R(ro["x"], ro["y"], ro["z"], ro["w"])
    S = SE3.from_Rt(R, np.array([tr["x"], tr["y"], tr["z"]], dtype=np.float64))
    return S


def load_offline_data(tf_json: Path, lidar_npz: Path) -> OfflineData:
    _CAM_TF_CHAIN_LINKS = [
        ("base_link", "depth_cam_link"),
        ("depth_cam_link", "depth_cam_depth_frame"),
        ("depth_cam_depth_frame", "depth_cam_color_frame"),
        ("depth_cam_color_frame", "depth_cam_color_optical_frame"),
    ]

    """Load TF and LiDAR data from offline2 flat files."""
    with open(tf_json) as f:
        tf_data = json.load(f)

    data = OfflineData()

    # Static TFs
    for entry in tf_data.get("tf_static", []):
        for t in entry["transforms"]:
            parent = t["header"]["frame_id"]
            child = t["child_frame_id"]
            data.static_tfs[(parent, child)] = _json_transform_to_se3(t)

    # Dynamic camera TF chain: all 4 links (outermost first)
    raw: dict[tuple[str, str], list[tuple[int, SE3]]] = {k: [] for k in _CAM_TF_CHAIN_LINKS}
    for entry in tf_data.get("tf", []):
        ts = int(entry["log_time_ns"])
        for t in entry["transforms"]:
            pair = (t["header"]["frame_id"], t["child_frame_id"])
            if pair in raw:
                raw[pair].append((ts, _json_transform_to_se3(t)))

    for link in _CAM_TF_CHAIN_LINKS:
        pairs = sorted(raw[link], key=lambda x: x[0])
        if not pairs:
            raise ValueError(
                f"No dynamic TF entries found for camera link {link[0]} → {link[1]}"
            )
        times = np.array([p[0] for p in pairs], dtype=np.int64)
        se3s = [p[1] for p in pairs]
        data.cam_tf_chain.append((times, se3s))

    # LiDAR
    lidar = np.load(lidar_npz, allow_pickle=True)
    data.lidar_stamp_ns = lidar["stamp_ns"].astype(np.int64)
    data.lidar_ranges = lidar["ranges"]
    data.lidar_angle_min = lidar["angle_min"]
    data.lidar_angle_increment = lidar["angle_increment"]

    return data


# ---------------------------------------------------------------------------
# Camera calibration
# ---------------------------------------------------------------------------


def _load_camera_info(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    with open(path) as f:
        docs = [d for d in yaml.safe_load_all(f) if d]
    ci = docs[0]
    K = np.array(ci["k"], dtype=np.float64).reshape(3, 3)
    d = np.array(ci["d"], dtype=np.float64)
    dist: np.ndarray | None = d if d.size > 0 and not np.allclose(d, 0) else None
    return K, dist
    

# ---------------------------------------------------------------------------
# Per-frame processing
# ---------------------------------------------------------------------------


def _annotate_image(
    image_bgr: np.ndarray,
    boxes_xyxy: np.ndarray,
    scores: np.ndarray,
    feet_bev: list,
    frame_idx: int = 0,
) -> np.ndarray:
    """Return a copy of image_bgr with bounding boxes, confidence labels, and foot dots."""
    import cv2

    from rgb_lidar_bev.viz import class_color_bgr

    out = image_bgr.copy()

    cv2.putText(
        out,
        f"frame {frame_idx:04d}",
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    for box, score in zip(boxes_xyxy, scores):
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        col = class_color_bgr(0)
        cv2.rectangle(out, (x1, y1), (x2, y2), col, 2)
        cv2.putText(
            out,
            f"{score:.2f}",
            (x1, max(y1 - 6, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            col,
            1,
            cv2.LINE_AA,
        )
        foot_u = int((x1 + x2) / 2)
        foot_v = y2
        cv2.circle(out, (foot_u, foot_v), 5, (0, 255, 255), -1)

    for fb in feet_bev:
        import math
        dist_m = math.hypot(fb.x, fb.y)
        tag = f"{dist_m:.1f}m{'*' if fb.lidar_refined else ''}"
        cv2.putText(
            out,
            tag,
            (10, out.shape[0] - 10 - feet_bev.index(fb) * 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return out


_COMPOSITE_H = 240
_COMPOSITE_W = 500
#_COMPOSITE_H = 480
#_COMPOSITE_W = 1000
_CAM_W = int(_COMPOSITE_W * 0.6)   # 300 px — 60 % for camera
_BEV_W = _COMPOSITE_W - _CAM_W     # 200 px — 40 % for BEV


def _process_image(
    image_bgr: np.ndarray,
    ranges: np.ndarray,
    angles: np.ndarray,
    T_cam_in_base: SE3,
    T_laser_in_base: SE3,
    K: np.ndarray,
    dist: np.ndarray | None,
    model: Any,
    conf: float,
    frame_idx: int = 0,
) -> tuple[list, list, np.ndarray, np.ndarray, np.ndarray]:
    """
    Run detection + BEV pipeline.

    Returns (feet_bev, composite_frame, bev_canvas, annotated_camera).
    composite_frame is a side-by-side view: annotated camera (60 %) | BEV (40 %).
    """
    import cv2

    res = model(image_bgr, classes=[0], conf=conf)[0]
    if res.boxes is not None and len(res.boxes) > 0:
        raw_boxes = res.boxes.xyxy.cpu().numpy()
        raw_scores = res.boxes.conf.cpu().numpy()
        raw_classes = res.boxes.cls.cpu().numpy()
    else:
        raw_boxes = np.zeros((0, 4), dtype=np.float32)
        raw_scores = np.zeros(0, dtype=np.float32)
        raw_classes = np.zeros(0, dtype=np.float32)

    detections = yolo_boxes_to_detections(raw_boxes, raw_scores, raw_classes, conf_thresh=conf)

    if frame_idx in [250, 308, 341, 367]:
         print("Frame")
        
    feet_bev = process_frame_boxes(
        boxes=detections,
        K=K,
        dist=dist,
        T_cam_optical_in_base=T_cam_in_base,
        z_plane=0.0,
        lidar_ranges=ranges,
        lidar_angles=angles,
        T_laser_in_base=T_laser_in_base,
    )

    pts3 = scan_to_points_base(ranges, angles, T_laser_in_base, 0.05, 30.0)
    lidar_xy = points_to_xy(pts3)

    canvas = np.zeros((_COMPOSITE_H, _BEV_W, 3), dtype=np.uint8)
    draw_bev(canvas, feet_bev, lidar_xy=lidar_xy, scale_px_per_m=18.0, left_sign=-1.0)

    annotated = _annotate_image(image_bgr, raw_boxes, raw_scores, feet_bev, frame_idx=frame_idx)

    cam_panel = cv2.resize(annotated, (_CAM_W, _COMPOSITE_H))
    bev_panel = cv2.resize(canvas, (_BEV_W, _COMPOSITE_H))
    composite = np.hstack([cam_panel, bev_panel])
    print("====composite is not none===" + str(composite is not None))

    return feet_bev, lidar_xy, composite, canvas, annotated


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description="Streaming BEV demo (offline2 flat-file replay)")
    p.add_argument("--video", default=str(DEFAULT_VIDEO), help="Path to RGB video (clip_rgb.mp4)")
    p.add_argument("--tf-json", default=str(DEFAULT_TF_JSON), help="Path to tf.json")
    p.add_argument("--lidar-npz", default=str(DEFAULT_LIDAR_NPZ), help="Path to lidar.npz")
    p.add_argument("--camera-info", default=str(DEFAULT_CAMERA_INFO), help="Camera calibration YAML")
    p.add_argument("--model", default="yolov8n.pt", help="Ultralytics YOLO weights")
    p.add_argument("--conf", type=float, default=0.70, help="Detection confidence threshold")
    p.add_argument("--out-dir", default="", help="Directory to save per-frame BEV PNGs")
    p.add_argument("--show", action="store_true", help="Display live BEV window (requires GUI)")
    p.add_argument("--fps-limit", type=float, default=0.0, help="Max playback rate (0 = unlimited)")
    p.add_argument("--start-frame", type=int, default=0, help="Frame index to begin playback from (0-based)")
    p.add_argument("--step", type=int, default=2, help="Process every Nth frame (e.g. --step 3 = 3× faster). Default: 1")
    p.add_argument(
        "--foot-frac", type=float, default=1.0,
        help=(
            "Fraction of box height at which to sample the foot position "
            "(0=top, 1=bottom edge). Reduce below 1.0 to avoid including "
            "floor pixels when the camera is mounted low. Default: 0.85"
        ),
    )
    args = p.parse_args()
        
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "ultralytics is not installed.\nRun: pip install 'rgb-lidar-bev[yolo]'"
        ) from exc

    import cv2

    if args.show:
        print("\nControls:  SPACE = pause / resume    R = restart    Q = quit")
        cv2.namedWindow("RGB-LiDAR BEV", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("RGB-LiDAR BEV", _COMPOSITE_W, _COMPOSITE_H)
        cv2.waitKey(200) 
        
    # Load sensor data
    print("Loading offline data...")

    # static transforms and dynamic camera TF chain
    offline = load_offline_data(Path(args.tf_json), Path(args.lidar_npz))

    # camera intrinsics and distortion coefficients
    K, dist = _load_camera_info(Path(args.camera_info))

    model = YOLO(args.model)

    # Validate required static TFs are present
    required_static = [
        ("base_link", "lidar_link"),
        ("lidar_link", "lidar_frame"),
    ]

    missing = [k for k in required_static if k not in offline.static_tfs]
    if missing:
        raise SystemExit(f"Missing static TFs in tf.json: {missing}")

    if len(offline.cam_tf_chain) == 0 or any(len(se3s) == 0 for _, se3s in offline.cam_tf_chain):
        raise SystemExit("Incomplete dynamic camera TF chain entries found in tf.json")

    # Derive the video start timestamp from the TF data start time
    # Earliest timestamp present in ALL camera chain links.
    # tf_start_ns: int = int(max(times[0] for times, _ in offline.cam_tf_chain))

    # Simplified variant
    start_times = []
    for times, _ in offline.cam_tf_chain:
        first_time = times.flatten()[0]
        start_times.append(first_time)
    tf_start_ns = int(max(start_times))
    # 755929003392

    out_dir: Path | None = None
    if args.out_dir:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    min_frame_interval = 1.0 / args.fps_limit if args.fps_limit > 0 else 0.0

    if args.show:
        print("\nControls:  SPACE = pause / resume    R = restart    Q = quit")

    quit_requested = False

    while not quit_requested:
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            raise SystemExit(f"Cannot open video: {args.video}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        ns_per_frame = int(1_000_000_000 / fps)

        # go to start frame index
        frame_idx = args.start_frame
        if frame_idx > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        last_frame_wall = 0.0
        paused = False
        restart_requested = False

        print(f"Replaying: {args.video}  ({fps:.2f} fps, starting at frame {frame_idx})")

        while True:
            # Handle pause: spin on waitKey until spacebar or quit/restart
            if paused and args.show:
                while True:
                    key = cv2.waitKey(50) & 0xFF
                    if key == ord(" "):
                        paused = False
                        print("Resumed.")
                        break
                    if key == ord("r"):
                        restart_requested = True
                        break
                    if key == ord("q"):
                        quit_requested = True
                        break

            if quit_requested or restart_requested:
                break

            for _ in range(args.step - 1):
                cap.read()          # burn N-1 frames to advance position
            ret, frame_bgr = cap.read()
            if not ret:
                break
            
            # Map frame index to nanosecond timestamp
            ts_ns = tf_start_ns + frame_idx * ns_per_frame

            # Look up closest lidar scan and TF's
            ranges, angles = offline.get_scan(ts_ns)
            T_cam = offline.T_cam_in_base(ts_ns)
            T_laser = offline.T_laser_in_base

            # Optional FPS throttle
            if min_frame_interval > 0:
                now = time.monotonic()
                gap = min_frame_interval - (now - last_frame_wall)
                if gap > 0:
                    time.sleep(gap)
                last_frame_wall = time.monotonic()

            feet_bev, lidar_xy, composite, canvas, annotated = _process_image(
                frame_bgr, ranges, angles, T_cam, T_laser, K, dist, model, args.conf,
                frame_idx=frame_idx,
            )
            
            polar_coordinate_map = generate_sector_proximity(lidar_xy, feet_bev);

            print(
                f"Frame {frame_idx:04d} | "
                f"{len(feet_bev)} person(s) detected"
                + (
                    " | " + ", ".join(
                        f"({fb.x:+.2f},{fb.y:+.2f}){'*' if fb.lidar_refined else ''}"
                        for fb in feet_bev
                    )
                    if feet_bev else ""
                )
            )
            
            

            if args.show:
                cv2.imshow("RGB-LiDAR BEV", composite)
                key = cv2.waitKey(1) & 0xFF
                if key == ord(" "):
                    paused = True
                    print("Paused.  SPACE to resume, R to restart, Q to quit.")
                elif key == ord("r"):
                    restart_requested = True
                elif key == ord("q"):
                    quit_requested = True

            if quit_requested or restart_requested:
                break

            frame_idx += args.step

        cap.release()

        if restart_requested:
            print("Restarting...")
        elif not quit_requested:
            if args.show:
                print("End of video.  SPACE or R to restart, Q to quit.")
                while True:
                    key = cv2.waitKey(50) & 0xFF
                    if key in (ord(" "), ord("r")):
                        print("Restarting...")
                        break
                    if key == ord("q"):
                        quit_requested = True
                        break
            else:
                break  # non-interactive: exit after one pass

    if args.show:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

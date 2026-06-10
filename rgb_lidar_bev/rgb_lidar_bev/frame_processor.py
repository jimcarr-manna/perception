from __future__ import annotations

import math
from typing import Any

import numpy as np

from rgb_lidar_bev.datatypes import SE3
from rgb_lidar_bev.detection import yolo_boxes_to_detections
from rgb_lidar_bev.lidar_fusion import points_to_xy, scan_to_points_base
from rgb_lidar_bev.pipeline import process_frame_boxes
from rgb_lidar_bev.viz import class_color_bgr, draw_bev

_COMPOSITE_H = 240
_COMPOSITE_W = 500
_CAM_W = int(_COMPOSITE_W * 0.6)
_BEV_W = _COMPOSITE_W - _CAM_W


def _annotate_image(
    image_bgr: np.ndarray,
    boxes_xyxy: np.ndarray,
    scores: np.ndarray,
    feet_bev: list,
    frame_idx: int = 0,
) -> np.ndarray:
    import cv2

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

    for idx, fb in enumerate(feet_bev):
        dist_m = math.hypot(fb.x, fb.y)
        tag = f"{dist_m:.1f}m{'*' if fb.lidar_refined else ''}"
        cv2.putText(
            out,
            tag,
            (10, out.shape[0] - 10 - idx * 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return out


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

    return feet_bev, lidar_xy, composite, canvas, annotated

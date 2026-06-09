"""Detection and bbox → image-space foot points."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from rgb_lidar_bev.datatypes import DetBox2D, DetFoot2D


class YoloBackend(Protocol):
    """Pluggable detector (ultralytics, ONNX, ROS-perception, etc.)."""

def yolo_boxes_to_detections(
    boxes_xyxy: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    *,
    conf_thresh: float = 0.5,
) -> list[DetBox2D]:
    """
    Convert axis-aligned boxes to DetBox2D objects.

    boxes_xyxy: shape (N, 4) as x1, y1, x2, y2 in pixel coordinates.
    """
    detections: list[DetBox2D] = []
    if boxes_xyxy.size == 0:
        return detections
    for i in range(len(boxes_xyxy)):
        if float(scores[i]) < conf_thresh:
            continue
        x1, y1, x2, y2 = boxes_xyxy[i]
        detections.append(
            DetBox2D(
                x1=float(x1),
                y1=float(y1),
                x2=float(x2),
                y2=float(y2),
                class_id=int(class_ids[i]),
                score=float(scores[i]),
            )
        )
    return detections

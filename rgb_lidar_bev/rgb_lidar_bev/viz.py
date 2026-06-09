"""Draw BEV canvas (OpenCV): robot at centre, optional lidar scatter + foot circles."""

from __future__ import annotations

import numpy as np

from rgb_lidar_bev.datatypes import FootBEV


def class_color_bgr(class_id: int) -> tuple[int, int, int]:
    """Deterministic pseudo colour per class id."""
    rng = np.random.default_rng(int(class_id) * 9973 + 42)
    return tuple(int(x) for x in rng.integers(64, 255, size=3))  # type: ignore[return-value]


def draw_bev(
    canvas_bgr: np.ndarray,
    feet: list[FootBEV],
    lidar_xy: list[tuple[float, float]] | None = None,
    *,
    scale_px_per_m: float = 50.0,
    forward_is_up: bool = True,
    forward_sign: float = 1.0,
    left_sign: float = 1.0,
    robot_radius_px: int = 8,
    foot_radius_px: int = 6,
) -> np.ndarray:
    """
    Robot at image centre. Default: +x forward draws upward, +y left draws left.

    Adjust forward_sign / left_sign to match your `base_link` convention vs. ROS std.
    """
    import cv2

    h, w = canvas_bgr.shape[:2]
    cx, cy = w // 2, h // 2

    # Robot glyph
    cv2.circle(canvas_bgr, (cx, cy), robot_radius_px, (0, 255, 0), 2)
    tip = (cx, cy - robot_radius_px - 6) if forward_is_up else (cx + robot_radius_px + 6, cy)
    cv2.line(canvas_bgr, (cx, cy), tip, (0, 255, 0), 2)

    def world_to_px(x: float, y: float) -> tuple[int, int]:
        # Forward (x) → screen up if forward_is_up
        if forward_is_up:
            px = int(cx + left_sign * scale_px_per_m * y)
            py = int(cy - forward_sign * scale_px_per_m * x)
        else:
            px = int(cx + forward_sign * scale_px_per_m * x)
            py = int(cy - left_sign * scale_px_per_m * y)
        return px, py

    if lidar_xy:
        for x, y in lidar_xy:
            px, py = world_to_px(x, y)
            cv2.circle(canvas_bgr, (px, py), 1, (200, 200, 200), -1)

    # Orange (BGR) used for the camera-only initial ground-plane position.
    _INIT_COLOUR = (0, 128, 255)

    for ft in feet:
        import math

        col = class_color_bgr(ft.class_id)

        # Draw initial (camera-only) position when LiDAR refinement shifted it.
        if ft.lidar_refined and ft.x_init is not None and ft.y_init is not None:
            ix, iy = world_to_px(ft.x_init, ft.y_init)
            px, py = world_to_px(ft.x, ft.y)
            cv2.circle(canvas_bgr, (ix, iy), foot_radius_px - 2, _INIT_COLOUR, 1)
            cv2.putText(canvas_bgr, f"{ft.x_init:.1f}m", (ix + foot_radius_px + 3, iy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, _INIT_COLOUR, 1, cv2.LINE_AA)
            cv2.line(canvas_bgr, (ix, iy), (px, py), _INIT_COLOUR, 1, cv2.LINE_AA)
        else:
            px, py = world_to_px(ft.x, ft.y)

        cv2.circle(canvas_bgr, (px, py), foot_radius_px, col, 2)

        refined_marker = "*" if ft.lidar_refined else ""
        label = f"{ft.x:.1f}m{refined_marker}"
        cv2.putText(
            canvas_bgr,
            label,
            (px + foot_radius_px + 3, py + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            col,
            1,
            cv2.LINE_AA,
        )

    return canvas_bgr

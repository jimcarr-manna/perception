"""2D lidar scan helpers and optional bearing-sector range refinement."""

from __future__ import annotations

import numpy as np

from rgb_lidar_bev.datatypes import SE3


def scan_to_points_base(
    ranges: np.ndarray,
    angles: np.ndarray,
    T_laser_in_base: SE3,
    range_min: float,
    range_max: float,
) -> list[np.ndarray]:
    """
    Convert a 2D lidar scan (in the scanner's xy plane, z=0 in laser frame) to 3D points in base_link.

    angles: radians, same convention as sensor_msgs/LaserScan (angle_min + i * angle_increment).
    """
    R, t = T_laser_in_base.R, T_laser_in_base.t
    out: list[np.ndarray] = []
    for r, a in zip(np.asarray(ranges).ravel(), np.asarray(angles).ravel()):
        if not (range_min < r < range_max) or not np.isfinite(r):
            continue
        p_l = np.array([r * np.cos(a), r * np.sin(a), 0.0], dtype=np.float64)
        out.append(R @ p_l + t)
    return out


def points_to_xy(points: list[np.ndarray]) -> list[tuple[float, float]]:
    return [(float(p[0]), float(p[1])) for p in points]


def azimuth_from_xy(x: float, y: float) -> float:
    """Bearing in base xy plane: angle from +x axis toward +y (atan2 y,x)."""
    return float(np.arctan2(y, x))


def angle_diff(a: float, b: float) -> float:
    return float(np.arctan2(np.sin(a - b), np.cos(a - b)))


def refine_xy_with_lidar_sector(
    x: float,
    y: float,
    lidar_xy: list[tuple[float, float]], 
    sector_half_width_rad: float,
) -> tuple[float, float, bool]:
    """
    Scale (x, y) radially to nearest lidar return in the same bearing sector.

    Returns (x_new, y_new, fused_ok).
    """
    if not lidar_xy:
        return x, y, False

    phi0 = azimuth_from_xy(x, y)
    candidates: list[float] = []
    for px, py in lidar_xy:
        phi = azimuth_from_xy(px, py)
        if abs(angle_diff(phi, phi0)) <= sector_half_width_rad:
            candidates.append(float(np.hypot(px, py)))

    if not candidates:
        return x, y, False

    r_new = min(candidates)
    r_old = float(np.hypot(x, y))
    if r_old < 1e-6:
        return x, y, True

    scale = r_new / r_old
    return x * scale, y * scale, True

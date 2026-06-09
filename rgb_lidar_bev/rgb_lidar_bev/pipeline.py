"""End-to-end processing: feet in image → feet in BEV."""

from __future__ import annotations

import numpy as np

from rgb_lidar_bev.datatypes import DetBox2D, DetFoot2D, FootBEV, SE3
from rgb_lidar_bev.geometry import (
    cam_ray_to_base,
    intersect_ray_ground_plane_xy,
    pixel_to_ray_cam,
    undistort_uv,
    pixel_to_bev,
)
from rgb_lidar_bev.lidar_fusion import points_to_xy, refine_xy_with_lidar_sector, scan_to_points_base


def _project_pixel(
    u: float,
    v: float,
    K: np.ndarray,
    dist: np.ndarray | None,
    T_cam_optical_in_base: SE3,
    z_plane: float,
    max_ray_range_m: float,
) -> list[float]:
    """Project a single pixel to the ground plane. Returns [x, y, z] or None."""
    # u, v = undistort_uv(np.array([u, v]), K, dist)
    # o_c, d_c = pixel_to_ray_cam(K, u, v)
    # O_b, D_b = cam_ray_to_base(T_cam_optical_in_base, o_c, d_c)
    # bev = intersect_ray_ground_plane_xy(O_b, D_b, z_plane=z_plane, max_range=max_ray_range_m)
    # return bev
    bev = pixel_to_bev(u, v, K, dist,T_cam_optical_in_base)
    return bev


def _build_lidar_xy(
    lidar_ranges: np.ndarray | None,
    lidar_angles: np.ndarray | None,
    T_laser_in_base: SE3 | None,
    range_min: float,
    range_max: float,
) -> list[tuple[float, float]] | None:
    if lidar_ranges is None or lidar_angles is None or T_laser_in_base is None:
        return None
    pts3 = scan_to_points_base(lidar_ranges, lidar_angles, T_laser_in_base, range_min, range_max)
    return points_to_xy(pts3)

def process_frame_boxes(
    *,
    boxes: list[DetBox2D],
    K: np.ndarray,
    dist: np.ndarray | None,
    T_cam_optical_in_base: SE3,
    z_plane: float = 0.0,
    max_ray_range_m: float = 30.0,
    lidar_ranges: np.ndarray | None = None,
    lidar_angles: np.ndarray | None = None,
    T_laser_in_base: SE3 | None = None,
    lidar_range_min_m: float = 0.05,
    lidar_range_max_m: float = 30.0,
    lidar_sector_half_width_deg: float = 5.0,
) -> list[FootBEV]:
    """
    Project three sample pixels per box to the ground plane and return the
    centroid of the resulting triangle as the BEV foot position.

    All valid intersections are averaged; if fewer than three succeed the centroid
    is taken over whichever points did intersect. The centroid is then optionally
    refined with the nearest lidar return in the same bearing sector.

    Supply lidar_* and T_laser_in_base together for fusion; omit for camera-only BEV.
    """
    K = np.asarray(K, dtype=np.float64)
    lidar_xy = _build_lidar_xy(
        lidar_ranges, lidar_angles, T_laser_in_base, lidar_range_min_m, lidar_range_max_m
    )
    sector_half_width_rad = float(np.deg2rad(lidar_sector_half_width_deg))
    out: list[FootBEV] = []

    for box in boxes:
        v_foot = box.y1 + (box.y2 - box.y1)
        u_centre = 0.5 * (box.x1 + box.x2)
        bev_pts: list[np.ndarray] = []

        X = _project_pixel(u_centre, v_foot, K, dist, T_cam_optical_in_base, z_plane, max_ray_range_m)
        if X is not None:
            bev_pts.append(X)

        if not bev_pts:
            continue

        centroid = np.mean(bev_pts, axis=0)
        x_init, y_init = float(centroid[0]), float(centroid[1])
        z = float(centroid[2])
        x, y = x_init, y_init

        fused = False
        # JC-UNDO
        # if lidar_xy is not None:
        #     x, y, fused = refine_xy_with_lidar_sector(
        #         x, y, lidar_xy, sector_half_width_rad
        #     )

        out.append(FootBEV(
            x=x, y=y, z=z,
            class_id=box.class_id, score=box.score, lidar_refined=fused,
            x_init=x_init, y_init=y_init,
        ))

    return out

import numpy as np

from rgb_lidar_bev.datatypes import SE3
from rgb_lidar_bev.geometry import (
    cam_ray_to_base,
    intersect_ray_ground_plane_xy,
    pixel_to_ray_cam,
)


def test_pixel_ray_z_forward():
    K = np.array([[400.0, 0.0, 320.0], [0.0, 400.0, 240.0], [0.0, 0.0, 1.0]])
    _, d = pixel_to_ray_cam(K, 320.0, 240.0)
    assert abs(d[2] - 1.0) < 1e-9


def test_ground_intersection_forward():
    O = np.array([0.0, 0.0, 1.0])
    D = np.array([0.0, 0.0, -1.0])
    X = intersect_ray_ground_plane_xy(O, D, z_plane=0.0, max_range=10.0)
    assert X is not None
    assert abs(X[2]) < 1e-9


def test_cam_ray_identity_no_offset():
    T = SE3.from_Rt(np.eye(3), np.zeros(3))
    o_c = np.zeros(3)
    d_c = np.array([0.0, 0.0, 1.0])
    O_b, D_b = cam_ray_to_base(T, o_c, d_c)
    assert np.allclose(O_b, o_c)
    assert np.allclose(D_b, d_c)

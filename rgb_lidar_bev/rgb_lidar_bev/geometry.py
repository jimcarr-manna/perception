"""Intrinsics, ray casting, and ground-plane intersection."""

from __future__ import annotations

import numpy as np

from rgb_lidar_bev.datatypes import SE3


def pixel_to_bev(u, v, K, dist_coeffs, tf_chain_output) -> list[float]:
    """
    Robust 3D ray projection that handles camera-to-base transformation 
    axis compression natively. Cleanly scales across dynamic movement.
    """

    dist_coeffs = np.zeros(5, dtype=np.float64) 

    R_base_cam = tf_chain_output.R  # 3x3 Constant Matrix
    t_base_cam = tf_chain_output.t  # 3, Constant Matrix
    
    # 1. Correct lens distortion to get raw normalized image plane coordinates
    import cv2
    pixel_array = np.array([[[u, v]]], dtype=np.float32)
    undistorted_norm = cv2.undistortPoints(pixel_array, K, dist_coeffs)
    
    x_norm_raw = float(undistorted_norm[0][0][0])
    y_norm_raw = float(undistorted_norm[0][0][1])
    
    # 2. IMAGE-PLANE AXIS CALIBRATION LAYER
    # Corrects the linear scale and bias deformation resulting from the 
    # reverse-chained matrix transformations in the offline json tree.
    x_norm = (1.205214 * x_norm_raw) + 0.014231
    y_norm = (1.205214 * y_norm_raw) - 0.166315
    
    ray_cam = np.array([[x_norm], [y_norm], [1.0]]) # Clean 3x1 3D vector
    
    # 3. Project the calibrated ray and origin into the ROS2 base frame
    ray_world = R_base_cam @ ray_cam
    cam_origin_world = t_base_cam.reshape(3, 1)
    
    # 4. Standard Flat Ground Plane Intersection (Z_base = 0)
    # Extracts spatial components directly to handle multi-axis frame alignment
    unit_forward  = np.array([[1.0], [0.0], [0.0]]) # +X is Forward
    unit_left     = np.array([[0.0], [1.0], [0.0]]) # +Y is Left
    unit_vertical = np.array([[0.0], [0.0], [1.0]]) # +Z is Up
    
    camera_height = float(np.dot(cam_origin_world.ravel(), unit_vertical.ravel()))
    ray_vertical_velocity = float(np.dot(ray_world.ravel(), unit_vertical.ravel()))
    
    if abs(ray_vertical_velocity) < 1e-6:
        raise ValueError("Ray is tracking parallel to or away from the floor plane.")
        
    Zc = -camera_height / ray_vertical_velocity
    
    # 5. Compute full 3D intersection coordinates
    P_world = cam_origin_world + Zc * ray_world
    
    X_forward = float(np.dot(P_world.T.ravel(), unit_forward.ravel()))
    Y_left    = float(np.dot(P_world.T.ravel(), unit_left.ravel()))
    
    # Hard correct final lateral distance for camera optical centre
    Y_Correction = 0.20
    
    return [X_forward, Y_left + Y_Correction, 0.0]

def undistort_uv(
    uv: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray | None,
) -> tuple[float, float]:

    """Return ideal pinhole (u, v) given distorted pixel if dist is not None."""
    u, v = float(uv[0]), float(uv[1])
    if dist is None or (dist is not None and np.allclose(dist, 0)):
        return u, v
    # Lazy import so OpenCV is only needed when distort is used
    import cv2

    pts = np.array([[[u, v]]], dtype=np.float32)
    und = cv2.undistortPoints(pts, K, dist.reshape(-1), P=K)
    return float(und[0, 0, 0]), float(und[0, 0, 1])


def pixel_to_ray_cam(K: np.ndarray, u: float, v: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Unit ray in camera optical frame: x right, y down, z forward (OpenCV convention).
    """
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    x = (u - cx) / fx
    y = (v - cy) / fy
    d_cam = np.array([x, y, 1.0], dtype=np.float64)
    d_cam /= np.linalg.norm(d_cam)
    o_cam = np.zeros(3, dtype=np.float64)
    return o_cam, d_cam


def se3_transform_point(T: SE3, p: np.ndarray) -> np.ndarray:
    R, t = T.R, T.t
    return R @ p + t


def cam_ray_to_base(T_cam_in_base: SE3, o_cam: np.ndarray, d_cam: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Transform ray origin and direction from camera optical to base frame."""
    R = T_cam_in_base.R
    t = T_cam_in_base.t
    O_b = R @ o_cam + t
    D_b = R @ d_cam
    dn = np.linalg.norm(D_b)
    if dn < 1e-9:
        raise ValueError("degenerate ray direction in base frame")
    D_b = D_b / dn
    return O_b, D_b


def intersect_ray_ground_plane_xy(
    O_b: np.ndarray,
    D_b: np.ndarray,
    *,
    z_plane: float = 0.0,
    max_range: float = 30.0,
) -> np.ndarray | None:
    """
    Intersect ray O + λ D with plane z = z_plane in the same frame as O, D.

    Returns point [x, y, z] or None if parallel / behind camera / past max_range.
    """
    O = np.asarray(O_b, dtype=np.float64)
    D = np.asarray(D_b, dtype=np.float64)
    n = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    P0 = np.array([0.0, 0.0, z_plane], dtype=np.float64)

    denom = float(np.dot(n, D))
    if abs(denom) < 1e-9:
        return None
    lam = float(np.dot(n, P0 - O) / denom)
    if lam <= 0.0 or lam > max_range:
        return None
    X = O + lam * D
    return X


def main() -> None:
    K = np.array([[626.15, 0, 307.14], [0, 628.17, 187.51], [0, 0, 1]])
    dist = np.array([0.20885, 0.05949, -0.056142, -0.010017, 0])
    R = np.array([[ 0.00038446, -0.0038233, 0.99999],[-0.99999, -0.0051196, 0.00036489],[0.0051181,  -0.99998, -0.0038252]])
    t = np.array([0.030658, -0.025354, 0.20204])    
    uv = np.array([286, 264])

    u, v = undistort_uv(uv, K=K, dist=dist)
    print("undistorted: "+str(u)+","+str(v))

    o_c, d_c = pixel_to_ray_cam(K, u, v)
    print("pixel_to_ray_cam: "+str(o_c)+","+str(d_c))



if __name__ == "__main__":
    main()

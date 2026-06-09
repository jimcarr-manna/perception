import math
import numpy as np

from rgb_lidar_bev.datatypes import FootBEV

def generate_sector_proximity(lidar_pts, human_pts: list[FootBEV], num_sectors=36, max_range=10.0):
    """
    Compresses LiDAR and YOLO human coordinate arrays into fixed angular sectors.
    
    :param lidar_pts: List or NumPy array of (x, y) coordinates from LiDAR
    :param human_pts: List or NumPy array of (x, y) center coordinates of humans
    :param num_sectors: Number of angular slices (36 sectors = 10 degrees each)
    :param max_range: Maximum tracking distance in meters
    :return: dict containing arrays for 'ranges' and 'object_types'
    """
    
    # 1. Initialize arrays with max range and 'CLEAR' status (0)
    # CHANGED: Shift from 360-degree FOV to 180-degree forward-only FOV (-90 to +90 degrees)
    angle_min = -math.pi / 2.0  # -1.5708 radians (-90° / Direct Left)
    angle_max = math.pi / 2.0   #  1.5708 radians (+90° / Direct Right)
    angle_increment = (angle_max - angle_min) / num_sectors

    ranges = [float(max_range)] * num_sectors
    object_types = [0] * num_sectors 
    
    # Constants for Human Geometric Expansion
    HUMAN_RADIUS = 0.25  # 20cm shoulders + 5cm safety buffer zone
    
    # Helper to map any angle (-pi to pi) to a sector index
    def get_sector_index(angle):
        # Normalize angle to [angle_min, angle_max] safely
        if angle < angle_min: angle += 2 * math.pi
        if angle > angle_max: angle -= 2 * math.pi
        idx = int((angle - angle_min) / angle_increment)
        return max(0, min(idx, num_sectors - 1))

    # 2. Process Raw LiDAR Point Array (Static Obstacles)
    for x, y in lidar_pts:
        r = math.sqrt(x**2 + y**2)
        if r > max_range or r == 0:
            continue
            
        angle = math.atan2(y, x)
        idx = get_sector_index(angle)
        
        # Keep the closest obstacle in that sector
        if r < ranges[idx]:
            ranges[idx] = r
            object_types[idx] = 1  # 1 = LIDAR_STATIC

    # 3. Process YOLO Human Array (With Geometric Radius Expansion)
    for foot_bev in human_pts:
        x = foot_bev.x
        y = foot_bev.y
        r_center = math.sqrt(x**2 + y**2)
        if r_center > max_range or r_center <= HUMAN_RADIUS:
            continue
            
        angle_center = math.atan2(y, x)
        
        # Calculate how much angular space the human's radius occupies
        # Safe arc-sine protection if human is mathematically inside the robot
        try:
            delta_theta = math.asin(HUMAN_RADIUS / r_center)
        except ValueError:
            delta_theta = math.pi / 4  # Fallback large angle if extremely close
            
        angle_start = angle_center - delta_theta
        angle_end = angle_center + delta_theta
        
        # Find start and end sector indices
        idx_start = get_sector_index(angle_start)
        idx_end = get_sector_index(angle_end)
        
        # Calculate actual edge distance to the person's shoulder
        r_edge = r_center - HUMAN_RADIUS
        
        # Handle wrap-around cases for sectors across the -pi/pi boundary
        sectors_to_update = []
        if idx_start <= idx_end:
            sectors_to_update = list(range(idx_start, idx_end + 1))
        else:
            # Wrap-around case (e.g., spans across the rear-facing boundary)
            sectors_to_update = list(range(idx_start, num_sectors)) + list(range(0, idx_end + 1))
            
        # Dilute the person across all overlapping sectors
        for idx in sectors_to_update:
            # Humans overwrite LiDAR because they are safety-critical (higher priority)
            if r_edge < ranges[idx] or object_types[idx] != 2:
                ranges[idx] = r_edge
                object_types[idx] = 2  # 2 = HUMAN_DANGEROUS

    return {
        "angle_min": angle_min,
        "angle_max": angle_max,
        "angle_increment": angle_increment,
        "ranges": ranges,
        "object_types": object_types
    }
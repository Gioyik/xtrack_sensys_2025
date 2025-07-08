import os
import numpy as np
import open3d as o3d
from pathlib import Path

def find_closest_lidar_file(timestamp, lidar_timestamps, lidar_files):
    """
    Finds the path to the LiDAR .pcd file with the closest timestamp.
    
    Args:
        timestamp (float): The RGB camera frame timestamp.
        lidar_timestamps (list): A list of timestamps for the LiDAR scans.
        lidar_files (list): A list of paths to the LiDAR .pcd files.
        
    Returns:
        Path object to the closest LiDAR file or None if not found.
    """
    if not lidar_timestamps or not lidar_files:
        return None
    
    # Find the index of the closest timestamp
    closest_index = np.argmin(np.abs(np.array(lidar_timestamps) - timestamp))
    
    return lidar_files[closest_index]

def load_point_cloud(pcd_path):
    """
    Loads a point cloud from a .pcd file.
    """
    if not pcd_path or not pcd_path.is_file():
        return None
    try:
        pcd = o3d.io.read_point_cloud(str(pcd_path))
        return np.asarray(pcd.points)
    except Exception as e:
        print(f"Error loading point cloud {pcd_path}: {e}")
        return None

def get_3d_position_from_lidar(box_2d, point_cloud, K, T_cam_lidar):
    """
    Calculates the 3D position of an object using the LiDAR point cloud by
    projecting the cloud into the camera frame and finding points within the 2D box.

    Args:
        box_2d: The 2D bounding box (x1, y1, x2, y2) from the camera image.
        point_cloud: The LiDAR point cloud (Nx3 numpy array) in the LiDAR frame.
        K: The camera's intrinsic matrix.
        T_cam_lidar: The 4x4 transformation matrix from LiDAR to camera frame.

    Returns:
        A tuple (x, y, z) representing the 3D position in the camera frame,
        or (0, 0, 0) if no valid points are found.
    """
    if point_cloud is None or point_cloud.shape[0] == 0:
        return 0, 0, 0

    # 1. Transform LiDAR points to Camera Frame
    # Add a homogeneous coordinate (w=1) to the points
    points_h = np.hstack((point_cloud, np.ones((point_cloud.shape[0], 1))))
    # Transform points to camera coordinates
    points_cam_h = (T_cam_lidar @ points_h.T).T
    points_cam = points_cam_h[:, :3]

    # Filter points that are behind the camera
    points_in_front_of_cam = points_cam[points_cam[:, 2] > 0]
    if points_in_front_of_cam.shape[0] == 0:
        return 0, 0, 0

    # 2. Project 3D points to 2D Image Plane
    # (u, v, 1) * z = K * (x, y, z)
    points_proj_h = (K @ points_in_front_of_cam.T).T
    # Normalize by the z-coordinate to get pixel coordinates
    points_uv = points_proj_h[:, :2] / points_proj_h[:, 2, np.newaxis]

    # 3. Frustum Culling: Filter points within the 2D bounding box
    x1, y1, x2, y2 = box_2d
    in_box_indices = (
        (points_uv[:, 0] >= x1)
        & (points_uv[:, 0] <= x2)
        & (points_uv[:, 1] >= y1)
        & (points_uv[:, 1] <= y2)
    )

    points_in_box = points_in_front_of_cam[in_box_indices]
    if points_in_box.shape[0] == 0:
        return 0, 0, 0

    # 4. Depth Estimation: Use the median for robustness
    # The z-coordinate in the camera frame is the depth
    median_depth = np.median(points_in_box[:, 2])

    # 5. 3D Position Calculation (Deprojection)
    # Get the center of the bounding box
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    # Camera intrinsics
    fx = K[0, 0]
    fy = K[1, 1]
    px = K[0, 2]
    py = K[1, 2]

    # Deproject the center of the box using the robust median depth
    x_cam = (cx - px) * median_depth / fx
    y_cam = (cy - py) * median_depth / fy
    z_cam = median_depth

    return x_cam, y_cam, z_cam

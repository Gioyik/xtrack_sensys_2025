import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN


def find_closest_lidar_file(timestamp, lidar_timestamps, lidar_files):
    """
    Find the LiDAR .pcd file with the closest timestamp to the given RGB timestamp.
    
    Args:
        timestamp (float): RGB camera frame timestamp
        lidar_timestamps (list): List of LiDAR scan timestamps
        lidar_files (list): List of paths to LiDAR .pcd files
        
    Returns:
        pathlib.Path or None: Path to closest LiDAR file or None if not found
        
    Process:
        1. Calculate absolute differences between RGB timestamp and all LiDAR timestamps
        2. Find index of minimum difference
        3. Return corresponding LiDAR file path
        
    Note:
        Uses timestamp-based matching for sensor synchronization.
        Handles missing or corrupted timestamp files gracefully.
    """
    if not lidar_timestamps or not lidar_files:
        return None

    # Find the index of the closest timestamp
    closest_index = np.argmin(np.abs(np.array(lidar_timestamps) - timestamp))

    return lidar_files[closest_index]


def load_point_cloud(pcd_path):
    """
    Load a point cloud from a .pcd file.
    
    Args:
        pcd_path (pathlib.Path): Path to the .pcd file
        
    Returns:
        numpy.ndarray or None: Point cloud (Nx3) or None if loading fails
        
    File Format Support:
        - ASCII PCD format
        - Binary PCD format
        - Handles missing or corrupted files gracefully
        
    Error Handling:
        - Returns None for missing files
        - Returns None for empty point clouds
        - Prints error messages for debugging
    """
    if not pcd_path or not pcd_path.is_file():
        return None
    try:
        pcd = o3d.io.read_point_cloud(str(pcd_path))
        points = np.asarray(pcd.points)
        if points.shape[0] == 0:
            return None
        return points
    except Exception as e:
        print(f"Error loading point cloud {pcd_path}: {e}")
        return None


def filter_point_cloud(point_cloud, max_distance=50.0, min_distance=0.5):
    """
    Filter point cloud to remove noise and distant points.
    
    Args:
        point_cloud: Input point cloud (Nx3 numpy array)
        max_distance: Maximum distance from origin to keep points
        min_distance: Minimum distance from origin to keep points
        
    Returns:
        Filtered point cloud
    """
    if point_cloud is None or point_cloud.shape[0] == 0:
        return None
        
    # Calculate distances from origin
    distances = np.linalg.norm(point_cloud, axis=1)
    
    # Filter by distance
    valid_indices = (distances >= min_distance) & (distances <= max_distance)
    filtered_points = point_cloud[valid_indices]
    
    # Remove statistical outliers using Open3D
    if filtered_points.shape[0] > 10:  # Need sufficient points for statistical filtering
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(filtered_points)
        
        # Statistical outlier removal
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        filtered_points = np.asarray(pcd.points)
    
    return filtered_points


def remove_ground_plane(point_cloud, distance_threshold=0.3, ransac_n=3, num_iterations=1000):
    """
    Remove ground plane from point cloud using RANSAC.
    
    Args:
        point_cloud: Input point cloud (Nx3 numpy array)
        distance_threshold: Maximum distance to plane for inliers
        ransac_n: Number of points to sample for RANSAC
        num_iterations: Number of RANSAC iterations
        
    Returns:
        Point cloud with ground plane removed
    """
    if point_cloud is None or point_cloud.shape[0] < 10:
        return point_cloud
        
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    
    # Use RANSAC to find the ground plane
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations
    )
    
    # Remove inliers (ground points) to get non-ground points
    non_ground_pcd = pcd.select_by_index(inliers, invert=True)
    
    return np.asarray(non_ground_pcd.points)


def cluster_points_in_box(points_in_box, eps=0.5, min_samples=3):
    """
    Cluster points within a bounding box to get robust depth estimation.
    
    Args:
        points_in_box: Points within the 2D bounding box
        eps: Maximum distance between samples in a cluster
        min_samples: Minimum number of samples in a cluster
        
    Returns:
        List of clusters, each containing point indices
    """
    if points_in_box.shape[0] < min_samples:
        return []
        
    # Use DBSCAN clustering on 3D points
    clustering = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_labels = clustering.fit_predict(points_in_box)
    
    # Group points by cluster
    clusters = []
    unique_labels = np.unique(cluster_labels)
    
    for label in unique_labels:
        if label == -1:  # Skip noise points
            continue
        cluster_indices = np.where(cluster_labels == label)[0]
        clusters.append(cluster_indices)
    
    return clusters


def get_3d_position_from_lidar(box_2d, point_cloud, K, T_cam_lidar):
    """
    Enhanced 3D position calculation using LiDAR point cloud with advanced filtering and clustering.
    
    This function implements a robust pipeline for 3D localization:
    1. Point cloud filtering and noise removal
    2. Ground plane removal using RANSAC
    3. Coordinate transformation from LiDAR to camera frame
    4. Frustum culling to find points within 2D bounding box
    5. Clustering-based depth estimation for robustness
    6. Fallback to statistical methods if clustering fails

    Args:
        box_2d: The 2D bounding box (x1, y1, x2, y2) from the camera image.
        point_cloud: The LiDAR point cloud (Nx3 numpy array) in the LiDAR frame.
        K: The camera's intrinsic matrix (3x3).
        T_cam_lidar: The 4x4 transformation matrix from LiDAR to camera frame.

    Returns:
        A tuple (x, y, z) representing the 3D position in the camera frame,
        or (0, 0, 0) if no valid points are found.
    """
    if point_cloud is None or point_cloud.shape[0] == 0:
        return 0, 0, 0

    # Filter noise and distant points
    filtered_cloud = filter_point_cloud(point_cloud)
    if filtered_cloud is None or filtered_cloud.shape[0] == 0:
        return 0, 0, 0
    
    # Remove ground plane to focus on objects above ground
    non_ground_cloud = remove_ground_plane(filtered_cloud)
    if non_ground_cloud is None or non_ground_cloud.shape[0] == 0:
        # Fallback to filtered cloud if ground removal fails
        non_ground_cloud = filtered_cloud

    # Add homogeneous coordinate (w=1) to the points
    points_h = np.hstack((non_ground_cloud, np.ones((non_ground_cloud.shape[0], 1))))
    # Transform points to camera coordinates
    points_cam_h = (T_cam_lidar @ points_h.T).T
    points_cam = points_cam_h[:, :3]

    # Filter points that are behind the camera (z > 0 in camera frame)
    points_in_front_of_cam = points_cam[points_cam[:, 2] > 0]
    if points_in_front_of_cam.shape[0] == 0:
        return 0, 0, 0

    # Apply camera intrinsics: (u, v, 1) * z = K * (x, y, z)
    points_proj_h = (K @ points_in_front_of_cam.T).T
    # Normalize by the z-coordinate to get pixel coordinates
    points_uv = points_proj_h[:, :2] / points_proj_h[:, 2, np.newaxis]

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

    final_depth = None
    
    # Try clustering-based approach for robust depth estimation
    if points_in_box.shape[0] >= 3:
        clusters = cluster_points_in_box(points_in_box)
        
        if clusters:
            # Find the largest cluster (most reliable)
            largest_cluster_idx = max(clusters, key=len)
            cluster_points = points_in_box[largest_cluster_idx]
            
            # Use median depth of the largest cluster
            final_depth = np.median(cluster_points[:, 2])
        else:
            # Fallback: Remove outliers and use median
            depths = points_in_box[:, 2]
            q1, q3 = np.percentile(depths, [25, 75])
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            # Filter outliers
            inlier_depths = depths[(depths >= lower_bound) & (depths <= upper_bound)]
            if len(inlier_depths) > 0:
                final_depth = np.median(inlier_depths)
            else:
                final_depth = np.median(depths)  # Last resort
    else:
        # For very few points, just use median
        final_depth = np.median(points_in_box[:, 2])

    if final_depth is None or final_depth <= 0:
        return 0, 0, 0

    # Get the center of the bounding box
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    # Camera intrinsics
    fx = K[0, 0]
    fy = K[1, 1]
    px = K[0, 2]
    py = K[1, 2]

    # Deproject the center of the box using the robust depth estimate
    x_cam = (cx - px) * final_depth / fx
    y_cam = (cy - py) * final_depth / fy
    z_cam = final_depth

    return x_cam, y_cam, z_cam


def fuse_depth_and_lidar(box_2d, depth_image, point_cloud, K_depth, K_rgb, T_cam_lidar):
    """
    Fuse depth camera and LiDAR data for more robust 3D localization.
    
    Args:
        box_2d: 2D bounding box (x1, y1, x2, y2)
        depth_image: Depth image from RGB-D camera
        point_cloud: LiDAR point cloud  
        K_depth: Depth camera intrinsic matrix
        K_rgb: RGB camera intrinsic matrix
        T_cam_lidar: Transformation from LiDAR to camera frame
        
    Returns:
        Fused 3D position (x, y, z) or (0, 0, 0) if fusion fails
    """
    # Get depth camera estimate
    from localization import get_3d_position
    depth_pos = get_3d_position(box_2d, depth_image, K_depth)
    
    # Get LiDAR estimate  
    lidar_pos = get_3d_position_from_lidar(box_2d, point_cloud, K_rgb, T_cam_lidar)
    
    # If both are valid, use weighted average (prefer depth camera for close objects)
    if (depth_pos != (0, 0, 0)) and (lidar_pos != (0, 0, 0)):
        depth_distance = depth_pos[2]
        lidar_distance = lidar_pos[2]
        
        # Weight based on distance - depth camera is more accurate for close objects
        if depth_distance < 5.0:  # < 5 meters, prefer depth camera
            weight_depth = 0.7
            weight_lidar = 0.3
        else:  # >= 5 meters, prefer LiDAR
            weight_depth = 0.3
            weight_lidar = 0.7
            
        fused_x = weight_depth * depth_pos[0] + weight_lidar * lidar_pos[0]
        fused_y = weight_depth * depth_pos[1] + weight_lidar * lidar_pos[1]
        fused_z = weight_depth * depth_pos[2] + weight_lidar * lidar_pos[2]
        
        return fused_x, fused_y, fused_z
    
    # If only one is valid, use that one
    elif depth_pos != (0, 0, 0):
        return depth_pos
    elif lidar_pos != (0, 0, 0):
        return lidar_pos
    else:
        return 0, 0, 0

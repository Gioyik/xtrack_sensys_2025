**********************
LiDAR Utilities Module
**********************

Advanced LiDAR processing and sensor fusion capabilities for robust 3D localization.

.. automodule:: lidar_utils
   :members:
   :undoc-members:
   :show-inheritance:

Main Functions
==============

get_3d_position_from_lidar(box_2d, point_cloud, K, T_cam_lidar)
---------------------------------------------------------------

Enhanced 3D position calculation using LiDAR point cloud with advanced filtering and clustering.

**Parameters**:
   * ``box_2d`` (`tuple`): 2D bounding box (x1, y1, x2, y2) from camera image
   * ``point_cloud`` (`numpy.ndarray`): LiDAR point cloud (Nx3) in LiDAR frame
   * ``K`` (`numpy.ndarray`): Camera intrinsic matrix (3x3)
   * ``T_cam_lidar`` (`numpy.ndarray`): 4x4 transformation matrix from LiDAR to camera frame

**Returns**:
   * ``tuple``: 3D position (x, y, z) in camera frame or (0, 0, 0) if failed

**Processing Pipeline**:
   1. **Point Cloud Filtering**: Remove noise and distant points
   2. **Ground Plane Removal**: Use RANSAC to detect and remove ground points
   3. **Coordinate Transformation**: Transform LiDAR points to camera frame
   4. **Frustum Culling**: Filter points within 2D bounding box
   5. **Clustering**: Use DBSCAN to group points for robust depth estimation
   6. **Depth Calculation**: Use median depth of largest cluster

fuse_depth_and_lidar(box_2d, depth_image, point_cloud, K_depth, K_rgb, T_cam_lidar)
------------------------------------------------------------------------------------

Fuse depth camera and LiDAR data for more robust 3D localization.

**Parameters**:
   * ``box_2d`` (`tuple`): 2D bounding box (x1, y1, x2, y2)
   * ``depth_image`` (`numpy.ndarray`): Depth image from RGB-D camera
   * ``point_cloud`` (`numpy.ndarray`): LiDAR point cloud
   * ``K_depth`` (`numpy.ndarray`): Depth camera intrinsic matrix
   * ``K_rgb`` (`numpy.ndarray`): RGB camera intrinsic matrix
   * ``T_cam_lidar`` (`numpy.ndarray`): LiDAR to camera transformation

**Returns**:
   * ``tuple``: Fused 3D position (x, y, z) or (0, 0, 0) if fusion fails

**Fusion Strategy**:
   * **Distance-based Weighting**:
     * Objects < 5m: 70% depth camera, 30% LiDAR
     * Objects ≥ 5m: 30% depth camera, 70% LiDAR
   * **Automatic Fallback**: Use individual sensors if fusion fails
   * **Validation**: Cross-validate measurements between sensors

load_point_cloud(pcd_path)
--------------------------

Load a point cloud from a .pcd file.

**Parameters**:
   * ``pcd_path`` (`pathlib.Path`): Path to the .pcd file

**Returns**:
   * ``numpy.ndarray`` or ``None``: Point cloud (Nx3) or None if loading fails

**File Format Support**:
   * ASCII PCD format
   * Binary PCD format
   * Handles missing or corrupted files gracefully

find_closest_lidar_file(timestamp, lidar_timestamps, lidar_files)
-----------------------------------------------------------------

Find the LiDAR .pcd file with the closest timestamp to the given RGB timestamp.

**Parameters**:
   * ``timestamp`` (`float`): RGB camera frame timestamp
   * ``lidar_timestamps`` (`list`): List of LiDAR scan timestamps
   * ``lidar_files`` (`list`): List of paths to LiDAR .pcd files

**Returns**:
   * ``pathlib.Path`` or ``None``: Path to closest LiDAR file or None if not found

**Synchronization**:
   * Uses timestamp-based matching
   * Handles missing or corrupted timestamp files
   * Returns closest available scan

Point Cloud Processing
======================

filter_point_cloud(point_cloud, max_distance=50.0, min_distance=0.5)
--------------------------------------------------------------------

Filter point cloud to remove noise and distant points.

**Parameters**:
   * ``point_cloud`` (`numpy.ndarray`): Input point cloud (Nx3)
   * ``max_distance`` (`float`): Maximum distance from origin (default: 50.0m)
   * ``min_distance`` (`float`): Minimum distance from origin (default: 0.5m)

**Returns**:
   * ``numpy.ndarray`` or ``None``: Filtered point cloud

**Filtering Process**:
   1. Calculate distances from origin
   2. Filter by distance range
   3. Remove statistical outliers using Open3D
   4. Return cleaned point cloud

remove_ground_plane(point_cloud, distance_threshold=0.3, ransac_n=3, num_iterations=1000)
-----------------------------------------------------------------------------------------

Remove ground plane from point cloud using RANSAC algorithm.

**Parameters**:
   * ``point_cloud`` (`numpy.ndarray`): Input point cloud (Nx3)
   * ``distance_threshold`` (`float`): Maximum distance to plane for inliers (default: 0.3m)
   * ``ransac_n`` (`int`): Number of points to sample for RANSAC (default: 3)
   * ``num_iterations`` (`int`): Number of RANSAC iterations (default: 1000)

**Returns**:
   * ``numpy.ndarray``: Point cloud with ground plane removed

**RANSAC Process**:
   1. Randomly sample points to define plane
   2. Count inliers within distance threshold
   3. Repeat for specified iterations
   4. Select best plane (most inliers)
   5. Remove inlier points (ground)

cluster_points_in_box(points_in_box, eps=0.5, min_samples=3)
------------------------------------------------------------

Cluster points within a bounding box using DBSCAN for robust depth estimation.

**Parameters**:
   * ``points_in_box`` (`numpy.ndarray`): Points within 2D bounding box
   * ``eps`` (`float`): Maximum distance between samples in cluster (default: 0.5m)
   * ``min_samples`` (`int`): Minimum samples in cluster (default: 3)

**Returns**:
   * ``list``: List of clusters, each containing point indices

**Clustering Process**:
   1. Apply DBSCAN clustering on 3D points
   2. Group points by cluster labels
   3. Filter out noise points (label = -1)
   4. Return cluster indices for analysis

Algorithm Details
=================

LiDAR Processing Pipeline
-------------------------

1. **Point Cloud Loading**: Load .pcd files with error handling
2. **Noise Filtering**: Remove statistical outliers and distant points
3. **Ground Removal**: Use RANSAC to detect and remove ground plane
4. **Coordinate Transformation**: Transform from LiDAR to camera frame
5. **Frustum Culling**: Project 3D points to 2D and filter by bounding box
6. **Clustering**: Group points using DBSCAN for robust depth estimation
7. **Depth Calculation**: Use median depth of largest cluster

Sensor Fusion Strategy
----------------------

**Distance-based Weighting**:
   * Close objects (< 5m): Prefer depth camera (more accurate at close range)
   * Distant objects (≥ 5m): Prefer LiDAR (better long-range performance)

**Fallback Mechanisms**:
   * If fusion fails, use individual sensor results
   * If one sensor fails, use the other
   * Cross-validation between sensors

**Quality Metrics**:
   * Depth consistency between sensors
   * Point cloud density in bounding box
   * Clustering quality metrics

Performance Optimizations
=========================

**Efficient Processing**:
   * Vectorized numpy operations
   * Open3D optimized algorithms
   * Early termination for failed cases

**Memory Management**:
   * In-place operations where possible
   * Efficient point cloud filtering
   * Automatic cleanup of temporary data

**Caching Strategies**:
   * Timestamp-based file caching
   * Reuse of transformation matrices
   * Efficient point cloud loading

Error Handling
==============

**File Operations**:
   * Graceful handling of missing .pcd files
   * Validation of file formats
   * Recovery from corrupted data

**Processing Errors**:
   * Empty point cloud handling
   * Invalid transformation matrices
   * Clustering failure recovery

**Numerical Stability**:
   * Division by zero protection
   * Matrix inversion error handling
   * Robust statistical calculations

Integration Notes
=================

**Coordinate Systems**:
   * LiDAR frame: Original sensor coordinate system
   * Camera frame: RGB camera coordinate system
   * Base frame: Vehicle coordinate system

**Transformation Pipeline**:
   1. LiDAR points → Camera frame (T_cam_lidar)
   2. Camera frame → Base frame (T_base_camera)
   3. 3D projection → 2D pixels (camera intrinsics)

**Synchronization**:
   * Timestamp-based sensor synchronization
   * Closest timestamp matching
   * Temporal alignment validation

Usage Examples
==============

**Basic LiDAR Localization**:
   .. code:: python

      from lidar_utils import load_point_cloud, get_3d_position_from_lidar
      from coordinate_frames import K_rgb, T_camera_lidar
   
      # Load point cloud
      point_cloud = load_point_cloud(lidar_file)
   
      if point_cloud is not None:
       # Calculate 3D position
       x, y, z = get_3d_position_from_lidar(
           bounding_box, point_cloud, K_rgb, T_camera_lidar
       )
       
       if x != 0 or y != 0 or z != 0:
           print(f"3D Position: ({x:.2f}, {y:.2f}, {z:.2f})")

**Sensor Fusion**:
   .. code:: python

      from lidar_utils import fuse_depth_and_lidar
   
      # Fuse depth and LiDAR data
      x, y, z = fuse_depth_and_lidar(
       bounding_box, depth_image, point_cloud, 
       K_depth, K_rgb, T_camera_lidar
      )
   
      print(f"Fused Position: ({x:.2f}, {y:.2f}, {z:.2f})")

**Point Cloud Processing**:
   .. code:: python

      from lidar_utils import filter_point_cloud, remove_ground_plane
   
      # Load and process point cloud
      point_cloud = load_point_cloud(pcd_file)
      filtered_cloud = filter_point_cloud(point_cloud, max_distance=30.0)
      non_ground_cloud = remove_ground_plane(filtered_cloud)
   
      print(f"Original points: {len(point_cloud)}")
      print(f"Filtered points: {len(filtered_cloud)}")
      print(f"Non-ground points: {len(non_ground_cloud)}")

Limitations and Considerations
==============================

**Point Cloud Quality**:
   * Performance depends on point cloud density
   * Noisy or sparse data may affect accuracy
   * Ground plane detection may fail in complex environments

**Computational Requirements**:
   * Requires Open3D and scikit-learn
   * RANSAC and clustering can be computationally intensive
   * Memory usage scales with point cloud size

**Calibration Dependencies**:
   * Requires accurate camera-LiDAR calibration
   * Transformation matrix errors affect accuracy
   * Intrinsic parameter accuracy is critical

**Environmental Factors**:
   * Performance may vary in different environments
   * Weather conditions can affect LiDAR data quality
   * Reflective surfaces may cause issues

Future Improvements
===================

**Algorithm Enhancements**:
   * Adaptive clustering parameters
   * Multi-scale point cloud processing
   * Advanced outlier detection methods

**Fusion Improvements**:
   * Dynamic weighting based on sensor confidence
   * Temporal consistency constraints
   * Uncertainty quantification

**Performance Optimizations**:
   * GPU-accelerated point cloud processing
   * Parallel processing for multiple detections
   * Efficient data structures for large point clouds

*******************
Localization Module
*******************

3D position estimation from depth camera data.

.. automodule:: localization
   :members:
   :undoc-members:
   :show-inheritance:

Main Functions
==============

get_3d_position(box, depth_image, K)
------------------------------------

Calculate the 3D position of an object given its bounding box, depth image, and camera intrinsic matrix.

**Parameters**:
   * ``box`` (`tuple`): Bounding box (x1, y1, x2, y2) in pixel coordinates
   * ``depth_image`` (`numpy.ndarray`): Depth image from RGB-D camera
   * ``K`` (`numpy.ndarray`): Camera intrinsic matrix (3x3)

**Returns**:
   * ``tuple``: 3D position (x, y, z) in camera frame or (0, 0, 0) if failed

**Process**:
   1. Calculate center of bounding box
   2. Extract depth value at center pixel
   3. Convert depth from millimeters to meters
   4. Use camera intrinsics to deproject 2D point to 3D
   5. Return 3D coordinates in camera frame

**Camera Model**:
   .. math::
      \begin{align}
      x_{cam} &= \frac{(u - c_x) \cdot z}{f_x} \\
      y_{cam} &= \frac{(v - c_y) \cdot z}{f_y} \\
      z_{cam} &= z
      \end{align}

   Where:
   * (u, v): 2D pixel coordinates
   * (x_cam, y_cam, z_cam): 3D camera coordinates
   * (f_x, f_y): Focal lengths
   * (c_x, c_y): Principal point

get_closest_depth_frame(rgb_timestamp, depth_timestamps, depth_folder_path)
---------------------------------------------------------------------------

Find the depth frame with the closest timestamp to the given RGB timestamp.

**Parameters**:
   * ``rgb_timestamp`` (`float`): RGB camera frame timestamp
   * ``depth_timestamps`` (`list`): List of depth frame timestamps
   * ``depth_folder_path`` (`str`): Path to depth frames directory

**Returns**:
   * ``str`` or ``None``: Path to closest depth frame or None if not found

**Synchronization Process**:
   1. Calculate absolute differences between RGB timestamp and all depth timestamps
   2. Find index of minimum difference
   3. Construct filename using index (formatted as 6-digit number)
   4. Check if file exists
   5. Return file path or None

**File Naming Convention**:
   * Depth frames: ``000000.png``, ``000001.png``, etc.
   * Index corresponds to timestamp order
   * 6-digit zero-padded format

Algorithm Details
=================

Depth Processing
----------------

**Depth Image Format**:
   * 16-bit depth values in millimeters
   * 0 values indicate invalid/no depth data
   * Depth range typically 0.1m to 10m for RGB-D cameras

**Center Point Selection**:
   * Uses bounding box center for depth sampling
   * Assumes person is centered in bounding box
   * Single depth value used for entire person

**Coordinate System**:
   * Camera frame: Right-handed coordinate system
   * X: Right (positive)
   * Y: Down (positive)
   * Z: Forward (positive)

**Depth Validation**:
   * Checks for zero depth values (invalid data)
   * Returns (0, 0, 0) for invalid depth
   * Assumes depth camera and RGB camera are aligned

Performance Considerations
===========================

**Computational Efficiency**:
   * Single depth value lookup per detection
   * Simple mathematical operations
   * No complex image processing required

**Memory Usage**:
   * Minimal memory footprint
   * No temporary data structures
   * Efficient numpy operations

**Accuracy Limitations**:
   * Single point depth sampling
   * No depth filtering or validation
   * Assumes person fills bounding box

Error Handling
==============

**Input Validation**:
   * Check for valid bounding box coordinates
   * Validate depth image dimensions
   * Handle missing depth files gracefully

**Depth Data Issues**:
   * Zero depth values (invalid data)
   * Out-of-range depth values
   * Missing depth frames

**File System Errors**:
   * Missing depth frame files
   * Corrupted depth images
   * Permission issues

Integration Notes
=================

**Camera Calibration**:
   * Requires accurate intrinsic parameters
   * Depth and RGB cameras must be calibrated
   * Coordinate frame alignment is critical

**Data Synchronization**:
   * Timestamp-based frame matching
   * Handles temporal misalignment
   * Closest timestamp selection

**Coordinate Transformations**:
   * Output in camera frame coordinates
   * Requires additional transformation to base frame
   * Uses transformation matrices from coordinate_frames module

Usage Examples
==============

**Basic 3D Localization**:
   .. code:: python

      from localization import get_3d_position, get_closest_depth_frame
      from coordinate_frames import K_rgb
   
      # Find closest depth frame
      depth_path = get_closest_depth_frame(
       rgb_timestamp, depth_timestamps, depth_folder_path
      )
   
      if depth_path:
       # Load depth image
       depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
       
       # Calculate 3D position
       x, y, z = get_3d_position(bounding_box, depth_image, K_rgb)
       
       if x != 0 or y != 0 or z != 0:
           print(f"3D Position: ({x:.2f}, {y:.2f}, {z:.2f})")

**Batch Processing**:
   .. code:: python

      # Process multiple detections
      for box in bounding_boxes:
       x, y, z = get_3d_position(box, depth_image, K_rgb)
       if x != 0 or y != 0 or z != 0:
           positions.append((x, y, z))

**Error Handling**:
   .. code:: python

      try:
       x, y, z = get_3d_position(box, depth_image, K_rgb)
       if x == 0 and y == 0 and z == 0:
           print("Warning: Invalid depth data")
      except Exception as e:
       print(f"Error in 3D localization: {e}")

Limitations and Considerations
==============================

**Single Point Sampling**:
   * Uses only center point of bounding box
   * May not represent actual person position
   * Sensitive to bounding box accuracy

**Depth Camera Limitations**:
   * Limited depth range (typically 0.1-10m)
   * Poor performance in bright sunlight
   * Reflective surfaces may cause issues

**Synchronization Issues**:
   * Temporal misalignment between RGB and depth
   * Frame rate differences
   * Clock synchronization problems

**Accuracy Dependencies**:
   * Camera calibration accuracy
   * Depth sensor quality
   * Environmental conditions

Future Improvements
===================

**Multi-Point Sampling**:
   * Sample multiple points within bounding box
   * Use median or mean depth for robustness
   * Weighted sampling based on person detection confidence

**Depth Filtering**:
   * Remove outliers in depth data
   * Temporal consistency filtering
   * Spatial smoothing

**Advanced Validation**:
   * Depth consistency checks
   * Geometric validation
   * Confidence scoring

**Performance Optimizations**:
   * Vectorized operations for multiple detections
   * Efficient depth image loading
   * Caching strategies for repeated lookups

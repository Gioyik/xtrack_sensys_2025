************************
Coordinate Frames Module
************************

Coordinate transformation matrices and camera intrinsic parameters.

.. automodule:: coordinate_frames
   :members:
   :undoc-members:
   :show-inheritance:

Constants
=========

Transformation Matrices
-----------------------

T_base_lidar
~~~~~~~~~~~~

Transformation from xTrack base frame to Ouster LiDAR frame.

**Type**: `numpy.ndarray`
**Shape**: (4, 4)
**Default**: Identity matrix

**Description**: 4x4 homogeneous transformation matrix representing the pose of the LiDAR sensor relative to the vehicle base frame.

T_base_rgb_camera
~~~~~~~~~~~~~~~~~

Transformation from xTrack base frame to Realsense RGB camera frame.

**Type**: `numpy.ndarray`
**Shape**: (4, 4)
**Default**: Identity matrix

**Description**: 4x4 homogeneous transformation matrix representing the pose of the RGB camera relative to the vehicle base frame.

T_base_depth_camera
~~~~~~~~~~~~~~~~~~~

Transformation from xTrack base frame to Realsense depth camera frame.

**Type**: `numpy.ndarray`
**Shape**: (4, 4)
**Default**: Identity matrix

**Description**: 4x4 homogeneous transformation matrix representing the pose of the depth camera relative to the vehicle base frame.

T_camera_lidar
~~~~~~~~~~~~~~

Transformation from camera frame to LiDAR frame.

**Type**: `numpy.ndarray`
**Shape**: (4, 4)
**Calculation**: T_camera_base @ T_base_lidar

**Description**: Computed transformation matrix from camera to LiDAR coordinate system.

Camera Intrinsic Parameters
===========================

K_rgb
-----

RGB camera intrinsic matrix.

**Type**: `numpy.ndarray`
**Shape**: (3, 3)

**Parameters**:
   * ``f_x``: Focal length in x-direction (pixels)
   * ``f_y``: Focal length in y-direction (pixels)
   * ``c_x``: Principal point x-coordinate (pixels)
   * ``c_y``: Principal point y-coordinate (pixels)

**Calculation**:
   .. math::
      K_{rgb} = \begin{bmatrix}
      f_{xy} & 0 & c_x \\
      0 & f_{xy} & c_y \\
      0 & 0 & 1
      \end{bmatrix}

**Where**:
   * ``f_xy = f_d435i / pixel_size``
   * ``c_x = video_resolution[0] / 2.0``
   * ``c_y = video_resolution[1] / 2.0``

K_depth
-------

Depth camera intrinsic matrix.

**Type**: `numpy.ndarray`
**Shape**: (3, 3)

**Description**: Currently set to the same values as K_rgb, assuming aligned cameras.

Camera Parameters
=================

f_d435i
-------

Focal length of D435i camera.

**Type**: `float`
**Value**: 1.93 mm

**Description**: Physical focal length of the Intel RealSense D435i camera.

pixel_size
----------

Pixel size of camera sensor.

**Type**: `float`
**Value**: 0.003 mm

**Description**: Physical size of each pixel on the OmniVision OV9282 sensor.

video_resolution
----------------

Video resolution in pixels.

**Type**: `tuple`
**Value**: (640, 480)

**Description**: Width and height of the video frames in pixels.

Coordinate Systems
==================

Base Frame
----------

**Origin**: Vehicle reference point

**Axes**:
   * X: Forward direction
   * Y: Left direction
   * Z: Up direction

**Description**: The vehicle's base coordinate frame, typically located at the center of the vehicle.

Camera Frame
------------

**Origin**: Camera optical center

**Axes**:
   * X: Right direction
   * Y: Down direction
   * Z: Forward direction

**Description**: Standard computer vision camera coordinate system.

LiDAR Frame
-----------

**Origin**: LiDAR sensor center

**Axes**:
   * X: Forward direction
   * Y: Left direction
   * Z: Up direction

**Description**: Ouster LiDAR sensor coordinate system.

Transformation Pipeline
=======================

3D Point Transformation
-----------------------

**From LiDAR to Camera**:
   1. LiDAR points → Camera frame (T_camera_lidar)
   2. Camera frame → Base frame (T_base_rgb_camera)

**From Camera to Base**:
   1. Camera points → Base frame (T_base_rgb_camera)

**Projection to 2D**:
   1. 3D camera points → 2D pixels (K_rgb)

Matrix Operations
=================

Transformation Composition
--------------------------

**Camera to LiDAR**:
   .. math::
      T_{camera\_lidar} = T_{camera\_base} \cdot T_{base\_lidar}

**Where**:
   .. math::
      T_{camera\_base} = T_{base\_rgb\_camera}^{-1}

**Error Handling**:
   * Matrix inversion with error checking
   * Fallback to identity matrix if inversion fails
   * Warning messages for calibration issues

Matrix Validation
-----------------

**Properties Checked**:
   * Matrix dimensions (4x4)
   * Determinant validation
   * Orthogonality of rotation part

**Error Recovery**:
   * Graceful handling of singular matrices
   * Default to identity transformation
   * Warning messages for invalid matrices

Usage Examples
==============

**Basic Transformation**:
   .. code:: python

      from coordinate_frames import T_camera_lidar, K_rgb
      import numpy as np
   
      # Transform LiDAR point to camera frame
      lidar_point = np.array([1.0, 2.0, 3.0, 1.0])  # Homogeneous coordinates
      camera_point = T_camera_lidar @ lidar_point
   
      # Project to 2D
      if camera_point[2] > 0:  # Check if in front of camera
       pixel_coords = K_rgb @ camera_point[:3]
       pixel_coords = pixel_coords[:2] / pixel_coords[2]
       print(f"Pixel coordinates: {pixel_coords}")

**Camera Intrinsics Usage**:
   .. code:: python

      from coordinate_frames import K_rgb
   
      # Extract intrinsic parameters
      fx = K_rgb[0, 0]
      fy = K_rgb[1, 1]
      cx = K_rgb[0, 2]
      cy = K_rgb[1, 2]
   
      print(f"Focal length: ({fx:.2f}, {fy:.2f})")
      print(f"Principal point: ({cx:.2f}, {cy:.2f})")

**3D to 2D Projection**:
   .. code:: python

      from coordinate_frames import K_rgb
   
      def project_3d_to_2d(point_3d):
       """Project 3D point to 2D pixel coordinates."""
       # Add homogeneous coordinate
       point_h = np.append(point_3d, 1.0)
       
       # Project using camera intrinsics
       pixel_h = K_rgb @ point_h
       
       # Normalize by depth
       if pixel_h[2] > 0:
           pixel_coords = pixel_h[:2] / pixel_h[2]
           return pixel_coords
       else:
           return None

**Coordinate Frame Validation**:
   .. code:: python

      from coordinate_frames import T_camera_lidar
   
      # Check if transformation matrix is valid
      def validate_transformation(T):
       """Validate transformation matrix."""
       if T.shape != (4, 4):
           return False
       
       # Check if rotation part is orthogonal
       R = T[:3, :3]
       should_be_identity = R @ R.T
       if not np.allclose(should_be_identity, np.eye(3), atol=1e-6):
           return False
       
       return True
   
      if validate_transformation(T_camera_lidar):
       print("Transformation matrix is valid")
      else:
       print("Warning: Invalid transformation matrix")

Calibration Requirements
========================

Camera Calibration
------------------

**Intrinsic Parameters**:
   * Focal lengths (fx, fy)
   * Principal point (cx, cy)
   * Distortion coefficients (if needed)

**Extrinsic Parameters**:
   * Camera pose relative to base frame
   * Rotation and translation matrices

LiDAR Calibration
-----------------

**Sensor Parameters**:
   * LiDAR pose relative to base frame
   * Sensor mounting orientation
   * Coordinate system alignment

**Data Requirements**:
   * Calibration targets
   * Synchronized sensor data
   * Ground truth measurements

Error Handling
==============

**Matrix Inversion**:
   * Check for singular matrices
   * Use pseudo-inverse if needed
   * Fallback to identity matrix

**Validation Errors**:
   * Warning messages for invalid matrices
   * Graceful degradation
   * Continue operation with defaults

**Calibration Issues**:
   * Detect poorly calibrated systems
   * Provide diagnostic information
   * Suggest recalibration procedures

Integration Notes
=================

**Module Dependencies**:
   * numpy: Matrix operations
   * No external dependencies

**System Integration**:
   * Used by localization modules
   * Referenced by LiDAR processing
   * Essential for coordinate transformations

**Performance Considerations**:
   * Matrix operations are efficient
   * Cached transformations
   * Minimal computational overhead

Future Improvements
===================

**Dynamic Calibration**:
   * Runtime calibration updates
   * Automatic calibration detection
   * Online calibration refinement

**Enhanced Validation**:
   * Comprehensive matrix validation
   * Calibration quality metrics
   * Diagnostic tools

**Multi-Sensor Support**:
   * Additional sensor types
   * Complex sensor networks
   * Advanced fusion algorithms

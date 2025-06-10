import numpy as np

# Placeholder for the transformation from the xTrack base frame to the Ouster LIDAR frame
# This will be a 4x4 homogeneous transformation matrix
T_base_lidar = np.identity(4)

# Placeholder for the transformation from the xTrack base frame to the Realsense RGB camera frame
T_base_rgb_camera = np.identity(4)

# Placeholder for the transformation from the xTrack base frame to the Realsense depth camera frame
T_base_depth_camera = np.identity(4)

# Placeholder for camera intrinsic parameters
# These will be 3x3 matrices
K_rgb = np.identity(3)
K_depth = np.identity(3)

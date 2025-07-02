import numpy as np

# Placeholder for the transformation from the xTrack base frame to the Ouster LIDAR frame
# This will be a 4x4 homogeneous transformation matrix
T_base_lidar = np.identity(4)

# Placeholder for the transformation from the xTrack base frame to the Realsense RGB camera frame
T_base_rgb_camera = np.identity(4)

# Placeholder for the transformation from the xTrack base frame to the Realsense depth camera frame
T_base_depth_camera = np.identity(4)

# Camera intrinsic parameters
# These will be 3x3 matrices
f_d435i = 1.93  # focal length in mm
pixel_size = 0.003  # pixel size in mm (Camera sensor: OmniVision OV9282)
video_resolution = (640, 480)  # video resolution in pixels
f_xy = f_d435i / pixel_size  # focal length in pixels
c_x = video_resolution[0] / 2.0  # principal points (assuming at the image center)
c_y = video_resolution[1] / 2.0  # principal points (assuming at the image center)
K_rgb = np.array([[f_xy, 0.0, c_x], [0.0, f_xy, c_y], [0.0, 0.0, 1.0]])
K_depth = K_rgb

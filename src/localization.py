import os

import numpy as np


def get_closest_depth_frame(rgb_timestamp, depth_timestamps, depth_folder_path):
    """
    Find the depth frame with the closest timestamp to the given RGB timestamp.
    
    Args:
        rgb_timestamp (float): RGB camera frame timestamp in seconds
        depth_timestamps (list): List of depth frame timestamps
        depth_folder_path (str): Path to depth frames directory
        
    Returns:
        str or None: Path to closest depth frame or None if not found
        
    Process:
        1. Calculate absolute differences between RGB timestamp and all depth timestamps
        2. Find index of minimum difference
        3. Construct filename using index (formatted as 6-digit number)
        4. Check if file exists
        5. Return file path or None
    """
    # Find the index of the closest timestamp
    closest_index = np.argmin(np.abs(np.array(depth_timestamps) - rgb_timestamp))

    # Construct the path to the depth frame
    depth_frame_filename = f"{closest_index:06d}.png"
    depth_frame_path = os.path.join(depth_folder_path, depth_frame_filename)

    if os.path.exists(depth_frame_path):
        return depth_frame_path
    else:
        return None


def get_3d_position(box, depth_image, K):
    """
    Calculate the 3D position of an object given its bounding box, depth image, and camera intrinsics.
    
    Args:
        box (tuple): Bounding box (x1, y1, x2, y2) in pixel coordinates
        depth_image (numpy.ndarray): Depth image from RGB-D camera (16-bit, millimeters)
        K (numpy.ndarray): Camera intrinsic matrix (3x3)
        
    Returns:
        tuple: 3D position (x, y, z) in camera frame or (0, 0, 0) if failed
        
    Process:
        1. Calculate center of bounding box
        2. Extract depth value at center pixel
        3. Convert depth from millimeters to meters
        4. Use camera intrinsics to deproject 2D point to 3D
        5. Return 3D coordinates in camera frame
        
    Note:
        Depth images from RealSense are typically in millimeters and need to be
        converted to meters. Zero depth values indicate invalid/no depth data.
    """
    # Get the center of the bounding box
    x1, y1, x2, y2 = box
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)

    # Get the depth value at the center of the bounding box
    # Note: Depth images from Realsense are often in millimeters.
    # We assume the depth is in millimeters and convert to meters.
    depth_in_mm = depth_image[cy, cx]
    if depth_in_mm == 0:
        return 0, 0, 0  # No depth information

    depth_in_m = depth_in_mm / 1000.0

    # Camera intrinsics
    fx = K[0, 0]
    fy = K[1, 1]
    px = K[0, 2]
    py = K[1, 2]

    # Deproject 2D point to 3D
    x_cam = (cx - px) * depth_in_m / fx
    y_cam = (cy - py) * depth_in_m / fy
    z_cam = depth_in_m

    return x_cam, y_cam, z_cam

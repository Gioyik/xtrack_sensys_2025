import cv2
import numpy as np


def extract_upper_body_roi(person_image):
    """
    Extract upper body region (top 50% of image, torso area) for vest detection.
    
    Safety vests are typically worn on the torso, so focusing on the upper body
    improves detection accuracy and reduces false positives from other yellow objects.
    
    Args:
        person_image (numpy.ndarray): Full person image (BGR format)
        
    Returns:
        numpy.ndarray: Upper half of the person image
        
    Note:
        The function takes the top 50% of the image height, which corresponds
        to the torso area where safety vests are typically worn.
    """
    height, width = person_image.shape[:2]
    upper_height = height // 2
    upper_half = person_image[:upper_height, :]

    return upper_half


def clean_mask_morphological(roi):
    """
    Clean up mask using morphological operations to remove noise and fill gaps.
    
    Args:
        roi (numpy.ndarray): Input binary mask
        
    Returns:
        numpy.ndarray: Cleaned mask after morphological operations
        
    Operations:
        1. Erosion (3x3 kernel, 1 iteration) to remove small noise
        2. Dilation (3x3 kernel, 3 iterations) to fill holes and gaps
    """
    # Erode the mask to remove small noise
    kernel = np.ones((3, 3), np.uint8)
    mask_cleaned = cv2.erode(roi, kernel, iterations=1)

    # Dilate the mask to fill in holes
    mask_cleaned = cv2.dilate(mask_cleaned, kernel, iterations=3)

    return mask_cleaned


def validate_hsv_color(roi):
    """
    Validate yellow color in HSV space for vest detection
    """
    # Convert the image to the HSV color space
    roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Define range for color yellow
    lower_yellow = np.array([25, 75, 100])
    upper_yellow = np.array([45, 255, 255])

    # Clean mask for better visualization
    roi_hsv = clean_mask_morphological(roi_hsv)

    # Create a mask for the yellow color
    mask = cv2.inRange(roi_hsv, lower_yellow, upper_yellow)

    # Calculate yellow pixel percentage based on upper region only
    number_yellow_pixels = np.sum(mask > 0)
    yellow_percentage = number_yellow_pixels / roi.size

    return mask, yellow_percentage


def adaptive_gamma_correction(roi):
    """
    Apply adaptive gamma correction to enhance yellow visibility
    """
    target_mean = 128.0  # 256.0 / 2.0

    # Calculate current mean brightness
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    current_mean = np.mean(gray)

    # Calculate adaptive gamma
    if current_mean > 0:
        gamma = np.log(target_mean / 255.0) / np.log(current_mean / 255.0)
        gamma = np.clip(gamma, 0.8, 2.5)  # limit range of gamma
    else:
        gamma = 1.0

    # Apply gamma correction (see https://docs.opencv.org/3.4/d3/dc1/tutorial_basic_linear_transform.html)
    table = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype(
        np.uint8
    )

    enhanced_upper_half = cv2.LUT(roi, table)

    return enhanced_upper_half


def enhance_yellow_with_clahe(roi, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Enhance yellow visibility using CLAHE on specific color channels
    """
    # Convert to LAB color space for better color preservation
    roi_lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)

    # Apply CLAHE to L channel (lightness)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    roi_lab[:, :, 0] = clahe.apply(roi_lab[:, :, 0])

    # Convert back to BGR
    enhanced_roi = cv2.cvtColor(roi_lab, cv2.COLOR_LAB2BGR)

    # Additional enhancement in HSV space
    roi_hsv = cv2.cvtColor(enhanced_roi, cv2.COLOR_BGR2HSV)

    # Apply CLAHE
    roi_hsv[:, :, 2] = clahe.apply(roi_hsv[:, :, 2])

    # Convert back to BGR
    mask_enhanced = cv2.cvtColor(roi_hsv, cv2.COLOR_HSV2BGR)

    return mask_enhanced


def detect_yellow_vest(person_image, threshold=5.0):
    """
    Enhanced vest detection with configurable threshold.
    
    Args:
        person_image: Cropped person image
        threshold: Yellow percentage threshold (0-100 scale)
        
    Returns:
        tuple: (is_vest_detected, mask, yellow_percentage)
    """
    # Apply gamma correction
    upper_half = extract_upper_body_roi(person_image)
    upper_half = adaptive_gamma_correction(upper_half)
    upper_half = enhance_yellow_with_clahe(upper_half)

    # HSV validation
    mask_hsv, yellow_percentage_hsv = validate_hsv_color(upper_half)
    
    # Convert to percentage (0-100 scale) to match threshold parameter
    yellow_percentage = yellow_percentage_hsv * 100

    return yellow_percentage > threshold, mask_hsv, yellow_percentage_hsv

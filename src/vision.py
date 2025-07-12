import cv2
import numpy as np


def extract_upper_body_roi(person_image):
    """
    Extract upper body region (top 50% of image, torso area)
    """
    height, width = person_image.shape[:2]
    upper_height = height // 2
    upper_half = person_image[:upper_height, :]

    return upper_half, upper_half.size


def clean_mask_morphological(mask):
    """
    Clean up mask using Morphological Operations
    """
    # Erode the mask to remove small noise
    kernel = np.ones((3, 3), np.uint8)
    mask_cleaned = cv2.erode(mask, kernel, iterations=1)
    # Dilate the mask to fill in holes
    mask_cleaned = cv2.dilate(mask_cleaned, kernel, iterations=3)

    return mask_cleaned


def adapt_yellow_range(mask_hsv):
    """
    Adaptive color segmentation based on lighting conditions for HSV validation
    """
    # Analyze lighting conditions
    avg_saturation = np.mean(mask_hsv[:, :, 1])
    avg_value = np.mean(mask_hsv[:, :, 2])

    # Adjust threshold based on lighting analysis
    if avg_value < 100:  # low light conditions
        threshold_saturation = max(60, avg_saturation * 0.7)
        threshold_value = max(60, avg_value * 0.8)
    elif avg_value > 200:  # bright conditions
        threshold_saturation = max(100, avg_saturation * 0.8)
        threshold_value = min(255, avg_value * 1.2)
    else:  # normal conditions
        threshold_saturation = 80
        threshold_value = 80

    # Apply adaptive yellow detection
    lower_yellow = np.array([15, int(threshold_saturation), int(threshold_value)])
    # lower_yellow = np.array([15, 80, 80])
    upper_yellow = np.array([35, 255, 255])

    return lower_yellow, upper_yellow


def validate_hsv_color(roi):
    """
    Validate yellow color in HSV space for vest detection
    """
    # upper_half, total_number_pixels = extract_upper_body_roi(person_image)
    upper_half = roi
    total_number_pixels = roi.size

    # Convert the image to the HSV color space
    hsv_image = cv2.cvtColor(upper_half, cv2.COLOR_BGR2HSV)

    # Boost saturation
    hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1] * 1.3, 0, 255)

    # Calculate yellow range
    lower_yellow, upper_yellow = adapt_yellow_range(hsv_image)

    # Create a mask for the yellow color
    mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow)

    # Calculate yellow pixel percentage based on upper region only
    number_yellow_pixels = np.sum(mask > 0)
    yellow_percentage = number_yellow_pixels / total_number_pixels

    return mask, yellow_percentage


def validate_lab_color(roi):
    """
    Validate yellow color in LAB  space for vest detection
    """
    # upper_half, total_number_pixels = extract_upper_body_roi(person_image)
    upper_half = roi
    total_number_pixels = roi.size

    # Convert the image to the LAB color space
    lab_image = cv2.cvtColor(upper_half, cv2.COLOR_BGR2LAB)

    # Calculate distances to reference colors (CIE 1994 formula)
    min_distances = calculate_color_distance(lab_image.astype(np.float32))

    # Create mask for pixels within tolerance
    h, w = lab_image.shape[:2]
    # TODO: define tolerance for LAB color difference
    tolerance = 40  # max_distance
    valid_pixels = min_distances < tolerance
    mask = valid_pixels.reshape(h, w).astype(np.uint8) * 255

    # Calculate yellow pixel percentage based on upper region only
    number_yellow_pixels = np.sum(mask > 0)
    yellow_percentage = number_yellow_pixels / total_number_pixels

    return mask, yellow_percentage


def calculate_color_distance(roi):
    r"""
    Calculate color distance, or color difference, $\Delta E$ between a color of ROI in
    LAB color space and a reference color (yellow) using CIE 1994 formula

    [Source](http://www.brucelindbloom.com/index.html?Eqn_DeltaE_CIE94.html)
    """
    # Define reference LAB value for color yellow
    reference_lab_values = [
        [80, 5, 70],  # bright yellow
        [75, 10, 65],  # standard yellow
        [85, 0, 75],  # light yellow
        [70, 15, 60],  # darker yellow
    ]

    # Define variables for CIE 1994
    L1, a1, b1 = roi[:, :, 0], roi[:, :, 1], roi[:, :, 2]
    C1 = np.sqrt(a1**2 + b1**2)
    KL, KC, KH = 1, 1, 1
    K1, K2 = 0.048, 0.014
    SL = 1
    SC = 1 + K1 * C1
    SH = 1 + K2 * C1

    min_distances = np.full(roi.shape[:-1], np.inf)
    for ref in reference_lab_values:
        L2, a2, b2 = ref
        dL = L1 - L2
        C2 = np.sqrt(a2**2 + b2**2)
        dC = C1 - C2
        da = a1 - a2
        db = b1 - b2
        dH = np.sqrt(da**2 + db**2 - dC**2)

        distances = np.sqrt(
            (dL / (KL * SL)) ** 2 + (dC / (KC * SC)) ** 2 + (dH / (KH * SH)) ** 2
        )
        min_distances = np.minimum(min_distances, distances)

    return min_distances


def adaptive_gamma_correction(person_image):
    """
    Apply adaptive gamma correction to enhance yellow visibility
    """
    target_mean = 128.0  # 256.0 / 2.0

    # Extract upper half for ROI
    upper_half, _ = extract_upper_body_roi(person_image)

    # Calculate current mean brightness
    gray = cv2.cvtColor(upper_half, cv2.COLOR_BGR2GRAY)
    current_mean = np.mean(gray)

    # Calculate adaptive gamma
    if current_mean > 0:
        gamma = np.log(target_mean / 255.0) / np.log(current_mean / 255.0)
        gamma = np.clip(gamma, 0.5, 2.5)  # limit range of gamma
    else:
        gamma = 1.0

    # Apply gamma correction (see https://docs.opencv.org/3.4/d3/dc1/tutorial_basic_linear_transform.html)
    table = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype(
        np.uint8
    )

    enhanced_upper_half = cv2.LUT(upper_half, table)

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


def detect_yellow_vest(person_image):
    """
    Weighted fusion of HSV and LAB validation results
    """
    # Apply gamma correction
    upper_half = adaptive_gamma_correction(person_image)
    upper_half = enhance_yellow_with_clahe(upper_half)

    # HSV validation
    mask_hsv, yellow_percentage_hsv = validate_hsv_color(upper_half)

    # LAB validation
    mask_lab, yellow_percentage_lab = validate_lab_color(upper_half)

    # Define weights
    weight_hsv = 0.37
    weight_lab = 0.63

    # Weighted combination
    combined_score = (
        weight_hsv * yellow_percentage_hsv + weight_lab * yellow_percentage_lab
    )

    # Adaptive threshold based on individual scores
    if yellow_percentage_hsv > 0.15 and yellow_percentage_lab > 0.08:
        threshold = 0.04  # lower threshold when both agree
    elif max(yellow_percentage_hsv, yellow_percentage_lab) > 0.08:
        threshold = 0.1  # medium threshold for strong single evidence
    else:
        threshold = 0.12  # higher threshold for weak evidence

    # Create combined mask
    if yellow_percentage_hsv > 0.1 and yellow_percentage_lab > 0.05:
        mask_combined = cv2.bitwise_or(mask_hsv, mask_lab)
    else:
        mask_combined = (
            mask_hsv if yellow_percentage_hsv > yellow_percentage_lab else mask_lab
        )

    return combined_score > threshold, mask_combined, combined_score

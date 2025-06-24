import cv2
import numpy as np


def detect_yellow_vest(person_image):
    """
    Detects if a person is wearing a yellow vest using color segmentation.
    """
    # Convert the image to the HSV color space
    hsv_image = cv2.cvtColor(person_image, cv2.COLOR_BGR2HSV)

    # Define the range for yellow color in HSV
    # These values might need tuning
    lower_yellow = np.array([15, 100, 100])
    upper_yellow = np.array([35, 255, 255])

    # Create a mask for the yellow color
    mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow)

    # --- Morphological Operations to clean up the mask ---
    # Erode the mask to remove small noise
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    # Dilate the mask to fill in holes
    mask = cv2.dilate(mask, kernel, iterations=3)

    # Calculate the percentage of yellow pixels in the cleaned mask
    # If the percentage is above a threshold, we assume a vest is present.
    if cv2.countNonZero(mask) > 0:
        yellow_percentage = (
            cv2.countNonZero(mask) / (person_image.shape[0] * person_image.shape[1])
        ) * 100
    else:
        yellow_percentage = 0

    # This threshold might need tuning
    return yellow_percentage > 10.0, mask, yellow_percentage

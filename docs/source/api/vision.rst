*************
Vision Module
*************

Computer vision functions for safety vest detection using color-based methods.

.. automodule:: vision
   :members:
   :undoc-members:
   :show-inheritance:

Main Functions
==============

detect_yellow_vest(person_image, threshold=5.0)
-----------------------------------------------

Enhanced vest detection with configurable threshold using HSV color segmentation.

**Parameters**:
   * ``person_image`` (`numpy.ndarray`): Cropped person image (BGR format)
   * ``threshold`` (`float`): Yellow percentage threshold (0-100 scale)

**Returns**:
   * ``tuple`): (is_vest_detected, mask, yellow_percentage)
     * ``is_vest_detected`` (`bool`): Whether vest is detected
     * ``mask`` (`numpy.ndarray`): Binary mask of yellow regions
     * ``yellow_percentage`` (`float`): Percentage of yellow pixels (0-1 scale)

**Process**:
   1. Extract upper body region (top 50% of image)
   2. Apply adaptive gamma correction for lighting normalization
   3. Enhance yellow visibility using CLAHE
   4. Convert to HSV color space
   5. Create mask for yellow color range
   6. Calculate yellow pixel percentage
   7. Compare against threshold

**Color Range**:
   * Lower yellow: HSV(25, 75, 100)
   * Upper yellow: HSV(45, 255, 255)

extract_upper_body_roi(person_image)
------------------------------------

Extract upper body region (top 50% of image, torso area) for vest detection.

**Parameters**:
   * ``person_image`` (`numpy.ndarray`): Full person image

**Returns**:
   * ``numpy.ndarray``: Upper half of the person image

**Rationale**: Safety vests are typically worn on the torso, so focusing on the upper body improves detection accuracy and reduces false positives from other yellow objects.

validate_hsv_color(roi)
-----------------------

Validate yellow color in HSV space for vest detection.

**Parameters**:
   * ``roi`` (`numpy.ndarray`): Region of interest (upper body)

**Returns**:
   * ``tuple`): (mask, yellow_percentage)
     * ``mask`` (`numpy.ndarray`): Binary mask of yellow regions
     * ``yellow_percentage`` (`float`): Percentage of yellow pixels

**Process**:
   1. Convert image to HSV color space
   2. Apply morphological operations to clean mask
   3. Create mask using yellow color range
   4. Calculate yellow pixel percentage

clean_mask_morphological(roi)
-----------------------------

Clean up mask using morphological operations to remove noise.

**Parameters**:
   * ``roi`` (`numpy.ndarray`): Input mask

**Returns**:
   * ``numpy.ndarray``: Cleaned mask

**Operations**:
   1. Erosion (3x3 kernel, 1 iteration) to remove small noise
   2. Dilation (3x3 kernel, 3 iterations) to fill holes

adaptive_gamma_correction(roi)
------------------------------

Apply adaptive gamma correction to enhance yellow visibility under different lighting conditions.

**Parameters**:
   * ``roi`` (`numpy.ndarray`): Region of interest

**Returns**:
   * ``numpy.ndarray``: Gamma-corrected image

**Process**:
   1. Calculate current mean brightness
   2. Compute adaptive gamma value (target mean: 128)
   3. Apply gamma correction with clipping (0.8-2.5 range)
   4. Use lookup table for efficient processing

enhance_yellow_with_clahe(roi, clip_limit=2.0, tile_grid_size=(8, 8))
---------------------------------------------------------------------

Enhance yellow visibility using CLAHE (Contrast Limited Adaptive Histogram Equalization).

**Parameters**:
   * ``roi`` (`numpy.ndarray`): Region of interest
   * ``clip_limit`` (`float`): CLAHE clip limit (default: 2.0)
   * ``tile_grid_size`` (`tuple`): Grid size for CLAHE (default: (8, 8))

**Returns**:
   * ``numpy.ndarray``: Enhanced image

**Process**:
   1. Convert to LAB color space for better color preservation
   2. Apply CLAHE to L channel (lightness)
   3. Convert back to BGR
   4. Additional enhancement in HSV space
   5. Apply CLAHE to V channel (value)
   6. Convert back to BGR

Algorithm Details
=================

Color Space Selection
---------------------

**HSV Color Space**: Chosen for vest detection because:
* Hue represents color information (yellow detection)
* Saturation represents color purity
* Value represents brightness (lighting independence)

**Yellow Color Range**:
* Hue: 25-45 degrees (yellow range)
* Saturation: 75-255 (sufficient color intensity)
* Value: 100-255 (visible under various lighting)

Image Preprocessing Pipeline
----------------------------

1. **Upper Body Extraction**: Focus on torso area where vests are worn
2. **Gamma Correction**: Normalize lighting conditions
3. **CLAHE Enhancement**: Improve contrast and visibility
4. **Morphological Operations**: Clean up noise and fill gaps

Threshold Tuning
================

**Recommended Thresholds**:
* ``3.0%``: Very sensitive (high false positives)
* ``5.0%``: Default (balanced)
* ``6.0%``: Recommended (good balance)
* ``8.0%``: Conservative (fewer false positives)
* ``12.0%``: Very conservative (may miss some vests)

**Factors Affecting Threshold**:
* Lighting conditions
* Camera angle and distance
* Vest condition and color
* Background interference

Performance Considerations
===========================

**Optimization Techniques**:
* Upper body ROI extraction reduces processing area
* Efficient HSV conversion and masking
* Morphological operations with small kernels
* Lookup table for gamma correction

**Memory Usage**:
* Minimal memory footprint
* In-place operations where possible
* Efficient numpy array operations

**Processing Speed**:
* Fast HSV color space operations
* Optimized morphological operations
* Vectorized numpy operations

Error Handling
==============

**Input Validation**:
* Check for empty images
* Validate image dimensions
* Handle invalid color spaces

**Robust Processing**:
* Graceful handling of edge cases
* Fallback values for failed operations
* Exception handling for image processing errors

Integration Notes
=================

**Input Requirements**:
* BGR color format (OpenCV standard)
* Valid numpy array with proper dimensions
* Non-empty image data

**Output Format**:
* Boolean detection result
* Binary mask for visualization
* Percentage value for analysis

**Dependencies**:
* OpenCV for image processing
* NumPy for array operations
* No external ML models required

Usage Examples
==============

**Basic Vest Detection**:
   .. code:: python

      import cv2
      from vision import detect_yellow_vest
   
      # Load person image
      person_img = cv2.imread('person.jpg')
   
      # Detect vest with default threshold
      is_vest, mask, percentage = detect_yellow_vest(person_img)
   
      print(f"Vest detected: {is_vest}")
      print(f"Yellow percentage: {percentage * 100:.2f}%")

**Custom Threshold**:
   .. code:: python

      # More sensitive detection
      is_vest, mask, percentage = detect_yellow_vest(person_img, threshold=3.0)
   
      # More conservative detection
      is_vest, mask, percentage = detect_yellow_vest(person_img, threshold=8.0)

**Visualization**:
   .. code:: python

      # Show detection mask
      cv2.imshow('Vest Mask', mask)
      cv2.waitKey(0)
      cv2.destroyAllWindows()

Limitations and Considerations
==============================

**Lighting Sensitivity**:
* Performance may vary under extreme lighting conditions
* Gamma correction helps but may not handle all cases

**Color Similarity**:
* May detect other yellow objects (shirts, bags, etc.)
* Upper body ROI helps reduce false positives

**Resolution Dependency**:
* Performance may vary with image resolution
* Smaller images may have less accurate detection

**Background Interference**:
* Yellow backgrounds may cause false positives
* Morphological operations help but may not eliminate all issues

**********************
Vest Classifier Module
**********************

Model-based safety vest detection using PyTorch and MobileNetV2.

.. automodule:: vest_classifier
   :members:
   :undoc-members:
   :show-inheritance:

VestClassifier Class
====================

Model-based vest detection using a pre-trained MobileNetV2 model for binary classification.

.. autoclass:: VestClassifier
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: __init__

   .. automethod:: predict

   .. automethod:: _validate_device

   .. automethod:: _load_model

Class Attributes
================

**device** (`torch.device`)
   The compute device (CPU, CUDA, or MPS) for model inference.

**model** (`torch.nn.Module`)
   MobileNetV2 model modified for binary vest classification.

**transform** (`torchvision.transforms.Compose`)
   Image preprocessing pipeline for model input.

Key Methods
===========

__init__(model_path="vest_model.pth", device="cpu")
---------------------------------------------------

Initialize the VestClassifier with model path and device.

**Parameters**:
   * ``model_path`` (`str`): Path to trained PyTorch model file (default: "vest_model.pth")
   * ``device`` (`str`): Compute device ("cpu", "cuda", or "mps")

**Initialization Process**:
   1. Validate and set compute device
   2. Load MobileNetV2 model with custom classifier
   3. Load trained weights if available
   4. Set model to evaluation mode
   5. Move model to specified device
   6. Define image preprocessing pipeline

**Model Architecture**:
   * Base: MobileNetV2 (pre-trained on ImageNet)
   * Classifier: Modified final layer for binary classification
   * Output: 2 classes (0: no_vest, 1: vest)

predict(person_image)
---------------------

Predict if a person is wearing a safety vest using the loaded model.

**Parameters**:
   * ``person_image`` (`numpy.ndarray`): Cropped BGR image of person from OpenCV

**Returns**:
   * ``tuple`): (is_vest, confidence_score)
     * ``is_vest`` (`bool`): Whether person is wearing a vest
     * ``confidence_score`` (`float`): Confidence score (0.0 to 1.0)

**Process**:
   1. Validate input image (non-empty, proper format)
   2. Convert BGR to RGB format
   3. Apply preprocessing transformations
   4. Run model inference
   5. Apply softmax to get probabilities
   6. Return prediction and confidence

**Preprocessing Pipeline**:
   1. Convert to PIL Image
   2. Resize to 224x224 pixels
   3. Convert to tensor
   4. Normalize with ImageNet statistics

_validate_device(device_str)
----------------------------

Validate and return a torch device, falling back if necessary.

**Parameters**:
   * ``device_str`` (`str`): Device string to validate

**Returns**:
   * ``torch.device``: Validated device

**Validation Process**:
   1. Check device availability
   2. Fall back to CPU if requested device unavailable
   3. Print warnings for fallbacks

**Supported Devices**:
   * "cpu": CPU processing (always available)
   * "cuda": NVIDIA GPU (requires CUDA-compatible GPU)
   * "mps": Apple Silicon GPU (macOS only)

_load_model(model_path)
-----------------------

Load the MobileNetV2 model with custom binary classifier.

**Parameters**:
   * ``model_path`` (`str`): Path to model file

**Returns**:
   * ``torch.nn.Module``: Loaded model

**Model Loading Process**:
   1. Create MobileNetV2 with no pre-trained weights
   2. Modify classifier for binary classification (2 output classes)
   3. Load trained weights if file exists
   4. Print warnings for missing model files

**Model Structure**:
   * Feature extractor: MobileNetV2 backbone
   * Classifier: Linear layer (1280 → 2)
   * Activation: Softmax for probability output

Algorithm Details
=================

Model Architecture
------------------

**MobileNetV2 Backbone**:
   * Efficient architecture for mobile/edge deployment
   * Depthwise separable convolutions
   * Inverted residual blocks
   * 1280-dimensional feature vector

**Binary Classification**:
   * Final linear layer: 1280 → 2
   * Classes: [no_vest, vest]
   * Softmax activation for probabilities

**Input Requirements**:
   * Image size: 224x224 pixels
   * Color format: RGB
   * Normalization: ImageNet statistics

Preprocessing Pipeline
----------------------

**Image Transformations**:
   1. **Resize**: Scale to 224x224 (model input size)
   2. **ToTensor**: Convert PIL Image to tensor
   3. **Normalize**: Apply ImageNet normalization
      * Mean: [0.485, 0.456, 0.406]
      * Std: [0.229, 0.224, 0.225]

**Color Space Conversion**:
   * Input: BGR (OpenCV format)
   * Output: RGB (PyTorch format)
   * Handles format conversion automatically

Inference Process
-----------------

**Forward Pass**:
   1. Preprocess input image
   2. Add batch dimension
   3. Move to device
   4. Run model inference
   5. Apply softmax for probabilities
   6. Extract prediction and confidence

**Output Interpretation**:
   * Class 0 (no_vest): Person not wearing vest
   * Class 1 (vest): Person wearing vest
   * Confidence: Maximum probability value

Performance Considerations
===========================

**Model Efficiency**:
   * MobileNetV2: Lightweight architecture
   * Optimized for mobile/edge deployment
   * Fast inference on CPU and GPU

**Memory Usage**:
   * Efficient model loading
   * Minimal memory footprint
   * Automatic cleanup of temporary tensors

**Processing Speed**:
   * GPU acceleration support
   * Optimized preprocessing pipeline
   * Batch processing capability

**Accuracy vs Speed Trade-offs**:
   * MobileNetV2: Good balance of accuracy and speed
   * Larger models: Higher accuracy, slower inference
   * Smaller models: Lower accuracy, faster inference

Error Handling
==============

**Input Validation**:
   * Check for empty or invalid images
   * Validate image dimensions and format
   * Handle conversion errors gracefully

**Model Loading**:
   * Graceful handling of missing model files
   * Fallback to untrained model with warnings
   * Device compatibility checking

**Inference Errors**:
   * Exception handling for model inference
   * Return default values for failed predictions
   * Continue processing other detections

Integration Notes
=================

**Model Training Requirements**:
   * Requires trained model file (vest_model.pth)
   * Model should be trained on vest/no-vest dataset
   * Binary classification with 2 output classes

**Dependencies**:
   * PyTorch for model inference
   * Torchvision for model architecture
   * PIL for image processing
   * OpenCV for image format conversion

**Device Compatibility**:
   * Automatic device detection
   * Fallback mechanisms for unavailable devices
   * Cross-platform GPU support

Usage Examples
==============

**Basic Vest Classification**:
   .. code:: python

      from vest_classifier import VestClassifier
   
      # Initialize classifier
      classifier = VestClassifier("vest_model.pth", device="cpu")
   
      # Predict vest
      is_vest, confidence = classifier.predict(person_image)
   
      print(f"Vest detected: {is_vest}")
      print(f"Confidence: {confidence:.3f}")

**GPU Acceleration**:
   .. code:: python

      # Use CUDA if available
      classifier = VestClassifier("vest_model.pth", device="cuda")
   
      # Use MPS on Apple Silicon
      classifier = VestClassifier("vest_model.pth", device="mps")

**Batch Processing**:
   .. code:: python

      # Process multiple person images
      results = []
      for person_img in person_images:
       is_vest, confidence = classifier.predict(person_img)
       results.append((is_vest, confidence))

**Error Handling**:
   .. code:: python

      try:
       classifier = VestClassifier("vest_model.pth")
       is_vest, confidence = classifier.predict(person_image)
      except Exception as e:
       print(f"Error in vest classification: {e}")
       # Fallback to color-based detection

Limitations and Considerations
==============================

**Model Dependencies**:
   * Requires trained model file
   * Model quality affects detection accuracy
   * Training data bias may affect performance

**Computational Requirements**:
   * Requires PyTorch and torchvision
   * GPU memory usage for model storage
   * Processing time for each person detection

**Accuracy Limitations**:
   * Performance depends on training data quality
   * May not generalize to all vest types
   * Lighting and angle variations may affect accuracy

**Model Size**:
   * MobileNetV2: ~14MB model file
   * Memory usage during inference
   * Storage requirements for deployment

Future Improvements
===================

**Model Enhancements**:
   * Larger model architectures for higher accuracy
   * Person-specific pre-training
   * Multi-scale feature extraction

**Training Improvements**:
   * Data augmentation strategies
   * Hard negative mining
   * Transfer learning from person detection models

**Performance Optimizations**:
   * Model quantization for edge deployment
   * TensorRT optimization for NVIDIA GPUs
   * CoreML conversion for Apple devices

**Algorithm Improvements**:
   * Temporal consistency constraints
   * Multi-frame fusion
   * Confidence-based filtering

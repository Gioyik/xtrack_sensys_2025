************************
Re-Identification Module
************************

Appearance-based re-identification system for maintaining consistent track IDs across occlusions and re-entries.

.. automodule:: reid
   :members:
   :undoc-members:
   :show-inheritance:

Main Functions
==============

get_appearance_embedding(person_image)
--------------------------------------

Calculates an appearance embedding for a given person image using a pre-trained ResNet model.

**Parameters**:
   * ``person_image`` (`numpy.ndarray`): Cropped person image (BGR format from OpenCV)

**Returns**:
   * ``numpy.ndarray`` or ``None``: Feature embedding vector or None if processing fails

**Process**:
   1. Validate input image (non-empty, proper dimensions)
   2. Convert BGR to RGB format
   3. Convert to PIL Image
   4. Apply preprocessing transformations
   5. Extract features using ResNet model
   6. Return flattened embedding vector

**Model Architecture**:
   * ResNet-18 pre-trained on ImageNet
   * Features extracted from penultimate layer
   * 512-dimensional embedding vector

**Preprocessing**:
   * Resize to 256x256
   * Center crop to 224x224
   * Normalize with ImageNet statistics
   * Convert to tensor and add batch dimension

cosine_similarity(embedding1, embedding2)
-----------------------------------------

Calculates the cosine similarity between two appearance embeddings.

**Parameters**:
   * ``embedding1`` (`numpy.ndarray`): First embedding vector
   * ``embedding2`` (`numpy.ndarray`): Second embedding vector

**Returns**:
   * ``float``: Cosine similarity score (-1 to 1, higher = more similar)

**Formula**:
   .. math::
      \text{similarity} = \frac{\vec{a} \cdot \vec{b}}{|\vec{a}| \cdot |\vec{b}|}

**Interpretation**:
   * 1.0: Identical embeddings
   * 0.0: Orthogonal (no similarity)
   * -1.0: Opposite embeddings

initialize_reid_model(device_str="auto")
----------------------------------------

Initialize the ReID model with the specified device.

**Parameters**:
   * ``device_str`` (`str`): Device specification ("auto", "cpu", "cuda", "mps")

**Returns**:
   * ``tuple`): (model, device)
     * ``model`` (`torch.nn.Module`): Initialized ResNet model
     * ``device`` (`torch.device`): Selected device

**Process**:
   1. Validate and select device
   2. Load pre-trained ResNet-18
   3. Remove final classification layer
   4. Set model to evaluation mode
   5. Move model to selected device

**Device Selection**:
   * "auto": Automatically select best available device
   * "cpu": Force CPU processing
   * "cuda": NVIDIA GPU (if available)
   * "mps": Apple Silicon GPU (if available)

Device Management
=================

get_best_device()
-----------------

Automatically select the best available compute device.

**Returns**:
   * ``torch.device``: Best available device

**Priority Order**:
   1. CUDA (NVIDIA GPU)
   2. MPS (Apple Silicon)
   3. CPU (fallback)

validate_device(device_str)
---------------------------

Validate and return a torch device, falling back if necessary.

**Parameters**:
   * ``device_str`` (`str`): Device string to validate

**Returns**:
   * ``torch.device``: Validated device

**Validation Process**:
   1. Check device availability
   2. Fall back to best available device if requested device unavailable
   3. Print warnings for fallbacks

Global State Management
=======================

The module uses global variables for efficient model management:

**_model** (`torch.nn.Module` or `None`)
   Global ResNet model instance (initialized on first use)

**_device** (`torch.device` or `None`)
   Global device instance (set during initialization)

**Lazy Initialization**:
   * Model is loaded only when first needed
   * Device is selected automatically
   * Subsequent calls reuse the same model instance

Re-Identification Pipeline
==========================

The re-identification system works as follows:

1. **Embedding Extraction**: Extract appearance features for each detected person
2. **Track Management**: Maintain embeddings for active tracks
3. **Lost Track Storage**: Store embeddings when tracks are lost
4. **Similarity Calculation**: Compare new detections with lost track embeddings
5. **Re-identification**: Assign consistent IDs based on similarity threshold

**Similarity Threshold Guidelines**:
   * 0.5-0.6: Very aggressive (high false merges)
   * 0.7: Moderate (some false merges)
   * 0.75: Default (balanced approach)
   * 0.8: Conservative (fewer false merges)
   * 0.85+: Very conservative (minimal re-ID)

Performance Considerations
===========================

**Model Efficiency**:
   * ResNet-18: Lightweight compared to larger models
   * Feature extraction only (no classification)
   * Batch processing support

**Memory Management**:
   * Global model instance (loaded once)
   * Efficient tensor operations with torch.no_grad()
   * Automatic cleanup of temporary tensors

**Processing Speed**:
   * GPU acceleration support
   * Optimized preprocessing pipeline
   * Vectorized similarity calculations

**Accuracy vs Speed Trade-offs**:
   * Larger models: Higher accuracy, slower processing
   * Smaller models: Lower accuracy, faster processing
   * Current choice: ResNet-18 provides good balance

Error Handling
==============

**Input Validation**:
   * Check for empty or invalid images
   * Validate image dimensions and format
   * Handle conversion errors gracefully

**Model Loading**:
   * Graceful fallback for missing dependencies
   * Device compatibility checking
   * Automatic device selection

**Processing Errors**:
   * Exception handling for image processing
   * Return None for failed operations
   * Continue processing other detections

Integration with Tracking System
================================

**Data Flow**:
   1. Person detection provides bounding boxes
   2. Extract person images from bounding boxes
   3. Generate appearance embeddings
   4. Store embeddings for active tracks
   5. Compare with lost track embeddings
   6. Re-assign consistent track IDs

**State Management**:
   * Track embeddings dictionary
   * Lost tracks storage
   * ID mapping between systems

**Configuration**:
   * Similarity threshold (configurable)
   * Maximum lost frames (configurable)
   * Device selection (configurable)

Usage Examples
==============

**Basic Re-ID Setup**:
   .. code:: python

      from reid import initialize_reid_model, get_appearance_embedding
   
      # Initialize model
      model, device = initialize_reid_model("auto")
   
      # Extract embedding
      embedding = get_appearance_embedding(person_image)
   
      if embedding is not None:
       print(f"Embedding shape: {embedding.shape}")

**Similarity Calculation**:
   .. code:: python

      from reid import cosine_similarity
   
      # Calculate similarity between two embeddings
      similarity = cosine_similarity(embedding1, embedding2)
   
      if similarity > 0.75:  # Threshold
       print("Likely same person")
      else:
       print("Different person")

**Device-Specific Initialization**:
   .. code:: python

      # Force CPU usage
      model, device = initialize_reid_model("cpu")
   
      # Use CUDA if available
      model, device = initialize_reid_model("cuda")
   
      # Use MPS on Apple Silicon
      model, device = initialize_reid_model("mps")

Limitations and Considerations
==============================

**Appearance Changes**:
   * Clothing changes may affect similarity
   * Lighting variations can impact embeddings
   * Viewing angle changes may reduce similarity

**Computational Requirements**:
   * Requires PyTorch and torchvision
   * GPU memory usage for model storage
   * Processing time for each person detection

**Similarity Threshold Tuning**:
   * Requires experimentation for optimal values
   * Scene-specific tuning may be necessary
   * Balance between false positives and false negatives

**Model Limitations**:
   * Pre-trained on general images (not person-specific)
   * May not capture person-specific features optimally
   * Limited by ResNet-18 capacity

Future Improvements
===================

**Model Enhancements**:
   * Person-specific pre-training
   * Larger model architectures
   * Multi-scale feature extraction

**Algorithm Improvements**:
   * Temporal consistency constraints
   * Multi-modal features (color, shape, motion)
   * Adaptive threshold selection

**Performance Optimizations**:
   * Model quantization
   * Batch processing
   * Caching strategies

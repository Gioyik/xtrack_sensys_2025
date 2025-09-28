***********
User Guide
***********

This guide provides detailed information about configuring and using the xTrack person tracking system. The main script `src/person_tracker.py` accepts the following command-line arguments:

Dataset Selection
=================

``--dataset``
   Specifies the dataset to process.
   
   **Options**: ``indoor`` | ``outdoor``
   
   **Default**: ``outdoor``
   
   **Description**: Selects between indoor and outdoor datasets with different lighting and environmental conditions.

Tracking Configuration
----------------------

``--tracker``
   Specifies the tracking algorithm to use.
   
   **Options**: ``bytetrack`` | ``botsort``
   
   **Default**: ``bytetrack``
   
   **Description**: 
   
   * ``bytetrack``: Fast, lightweight tracker suitable for real-time applications
   * ``botsort``: More accurate tracker with built-in re-identification capabilities

``--reid_method``
   Specifies the re-identification method to use.
   
   **Options**: ``custom`` | ``botsort``
   
   **Default**: ``custom``
   
   **Description**:
   
   * ``custom``: Uses appearance-based embeddings from a ResNet model
   * ``botsort``: Uses the native BoTSORT re-identification (requires ``--tracker botsort``)

Vest Detection
--------------

``--vest_detection``
   Specifies the vest detection method to use.
   
   **Options**: ``color`` | ``model``
   
   **Default**: ``color``
   
   **Description**:
   
   * ``color``: Fast HSV color segmentation algorithm
   * ``model``: More robust PyTorch model-based classification (requires trained model)

``--vest_model_path``
   Path to the trained vest classifier model.
   
   **Default**: ``vest_model.pth``
   
   **Description**: Required when using ``--vest_detection model``. The model should be a trained MobileNetV2 model for binary classification.

``--vest_threshold``
   Yellow percentage threshold for vest detection.
   
   **Default**: ``5.0``
   
   **Range**: ``0.1`` to ``20.0``
   
   **Description**: Lower values = more sensitive detection (higher false positives), higher values = more conservative detection.

``--vest_persistence``
   Number of consecutive frames vest must be detected before confirming.
   
   **Default**: ``1``
   
   **Range**: ``1`` to ``10``
   
   **Description**: Higher values reduce false positives but may delay detection.

3D Localization
---------------

``--localization_method``
   Specifies the localization method to use.
   
   **Options**: ``depth`` | ``lidar`` | ``fusion``
   
   **Default**: ``depth``
   
   **Description**:
   
   * ``depth``: Uses RGB-D camera for 3D localization
   * ``lidar``: Uses LiDAR point cloud for 3D localization
   * ``fusion``: Intelligently combines both sensors for optimal accuracy

Performance Optimization
------------------------

``--device``
   Specifies the device to run the models on.
   
   **Options**: ``cpu`` | ``cuda`` | ``mps``
   
   **Default**: ``cpu``
   
   **Description**:
   
   * ``cpu``: CPU processing (compatible with all systems)
   * ``cuda``: NVIDIA GPU acceleration (requires CUDA-compatible GPU)
   * ``mps``: Apple Silicon GPU acceleration (macOS only)

``--jump_frames``
   Number of frames to skip between processed frames.
   
   **Default**: ``0``
   
   **Description**: Higher values improve processing speed but may reduce tracking accuracy. Example: ``--jump_frames 2`` processes every 3rd frame.

Re-Identification Tuning
------------------------

``--reid_threshold``
   Similarity threshold for re-identifying lost tracks.
   
   **Default**: ``0.75``
   
   **Range**: ``0.5`` to ``0.95``
   
   **Description**: 
   
   * Lower values = more re-identifications but higher chance of incorrect merging
   * Higher values = fewer re-identifications but better accuracy

``--max_lost_frames``
   Maximum frames a track can be lost before permanent deletion.
   
   **Default**: ``90``
   
   **Range**: ``30`` to ``300``
   
   **Description**: Higher values = longer memory but more computational overhead.

Debugging and Monitoring
------------------------

``--debug``
   Sets the level of debugging output.
   
   **Options**: ``0`` | ``1`` | ``2``
   
   **Default**: ``0``
   
   **Description**:
   
   * ``0``: No debugging output
   * ``1``: Basic tracking and re-ID information
   * ``2``: Detailed information and vest detection mask visualization

``--benchmark``
   Enables performance benchmarking.
   
   **Description**: When enabled, prints comprehensive performance metrics upon completion, including FPS, component latency, and detection statistics.

Usage Examples
==============

Basic Examples
--------------

**Default Configuration**:
   .. code:: bash

      python3 src/person_tracker.py --dataset outdoor

**Indoor Dataset with Debug Output**:
   .. code:: bash

      python3 src/person_tracker.py --dataset indoor --debug 1

**BoTSORT with Native Re-ID**:
   .. code:: bash

      python3 src/person_tracker.py --dataset outdoor --tracker botsort --reid_method botsort

Performance Optimization Examples
---------------------------------

**Real-time Processing**:
   .. code:: bash

      python3 src/person_tracker.py --dataset outdoor --jump_frames 2 --device mps

**Maximum Accuracy**:
   .. code:: bash

      python3 src/person_tracker.py --dataset outdoor --tracker botsort --reid_method botsort --localization_method fusion --jump_frames 0

**GPU Acceleration**:
   .. code:: bash

      python3 src/person_tracker.py --dataset outdoor --device cuda --benchmark

Advanced Configuration Examples
-------------------------------

**Model-based Vest Detection**:
   .. code:: bash

      python3 src/person_tracker.py --dataset outdoor --vest_detection model --vest_model_path /path/to/vest_model.pth

**LiDAR-only Localization**:
   .. code:: bash

      python3 src/person_tracker.py --dataset outdoor --localization_method lidar

**Sensor Fusion with Optimized Settings**:
   .. code:: bash

      python3 src/person_tracker.py --dataset outdoor --localization_method fusion --vest_threshold 6.0 --vest_persistence 3 --reid_threshold 0.8

**Conservative Re-ID Settings**:
   .. code:: bash

      python3 src/person_tracker.py --dataset outdoor --reid_threshold 0.85 --max_lost_frames 60 --debug 1

Configuration Recommendations
=============================

For Different Use Cases
-----------------------

**Real-time Applications**:
   - Use ``--tracker bytetrack`` for speed
   - Enable frame skipping: ``--jump_frames 2``
   - Use ``--localization_method depth`` for simplicity
   - Enable GPU acceleration if available

**High Accuracy Requirements**:
   - Use ``--tracker botsort --reid_method botsort``
   - Use ``--localization_method fusion``
   - Set ``--jump_frames 0`` for maximum frame processing
   - Use model-based vest detection if available

**Resource-Constrained Environments**:
   - Use ``--device cpu``
   - Increase frame skipping: ``--jump_frames 4``
   - Use ``--localization_method depth``
   - Reduce vest persistence: ``--vest_persistence 1``

**Long-range Tracking**:
   - Use ``--localization_method lidar`` or ``fusion``
   - Increase ``--max_lost_frames`` for longer memory
   - Use conservative Re-ID settings: ``--reid_threshold 0.8``

Frame Skipping Guidelines
-------------------------

* ``--jump_frames 0``: Maximum accuracy, lowest FPS
* ``--jump_frames 1``: Good balance (processes every 2nd frame)
* ``--jump_frames 2``: Real-time performance (processes every 3rd frame)
* ``--jump_frames 4``: High-speed processing (processes every 5th frame)

Vest Detection Tuning
---------------------

**For High Sensitivity** (more detections, more false positives):
   - Lower threshold: ``--vest_threshold 3.0``
   - No persistence: ``--vest_persistence 1``

**For High Precision** (fewer false positives):
   - Higher threshold: ``--vest_threshold 8.0``
   - Require persistence: ``--vest_persistence 3``

**Balanced Approach** (recommended):
   - Moderate threshold: ``--vest_threshold 6.0``
   - Light persistence: ``--vest_persistence 2``

Re-ID Threshold Guidelines
--------------------------

* ``0.5-0.6``: Very aggressive (high false merges, not recommended)
* ``0.7``: Moderate (some false merges, good for dense crowds)
* ``0.75``: Default (balanced approach)
* ``0.8``: Conservative (fewer false merges, recommended for accuracy)
* ``0.85+``: Very conservative (minimal re-ID, high track fragmentation)

Output Files
============

The system generates a CSV file at ``output/tracking_log_[dataset].csv`` with the following columns:

* ``timestamp``: Frame timestamp in seconds
* ``frame_id``: Sequential frame number
* ``object_id``: Unique track identifier
* ``x_position``: X coordinate in base frame (meters)
* ``y_position``: Y coordinate in base frame (meters)
* ``z_position``: Z coordinate in base frame (meters)

Benchmark Results
-----------------

When using ``--benchmark``, the system prints comprehensive performance metrics including:

* Average FPS
* Component-wise latency (YOLO, ReID, vest detection, localization)
* Detection statistics (total detections, vest detections, unique track IDs)
* Re-ID analysis (re-identification events, similarity distribution)
* Localization performance (success rate)
* Vest detection analysis (yellow percentage statistics)

Next Steps
==========

* Learn about the system architecture in the :doc:`api_reference`
* Run comprehensive tests using the :doc:`testing_guide`
* Troubleshoot issues using the :doc:`troubleshooting` guide

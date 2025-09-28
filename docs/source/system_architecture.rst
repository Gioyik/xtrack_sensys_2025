*******************
System Architecture
*******************

The xTrack system is organized into several key modules:

* **Core Tracking Module** (`person_tracker.py`): The main tracking system that orchestrates all components.
* **Configuration Module** (`config.py`): Manages dataset paths and system configuration.
* **Computer Vision Module** (`vision.py`): Handles vest detection using color-based methods.
* **Re-Identification Module** (`reid.py`): Manages appearance-based re-identification of lost tracks.
* **Localization Module** (`localization.py`): Provides 3D position estimation from depth data.
* **LiDAR Utilities** (`lidar_utils.py`): Advanced LiDAR processing and sensor fusion capabilities.
* **Coordinate Frames** (`coordinate_frames.py`): Manages coordinate transformations between sensors.
* **Vest Classifier** (`vest_classifier.py`): Model-based vest detection using PyTorch.
* **Utilities** (`utils.py`): Command-line argument parsing and logging utilities.

Data Flow
=========
1. **Input**: RGB video frames and depth/LiDAR data
2. **Detection**: YOLOv11 person detection
3. **Tracking**: Multi-object tracking (ByteTrack/BoTSORT)
4. **Re-ID**: Appearance-based re-identification
5. **Vest Detection**: Color-based or model-based classification
6. **3D Localization**: Depth camera, LiDAR, or sensor fusion
7. **Output**: CSV logging and live visualization

Data Structures
===============
* **Track Embeddings** (`dict`): Maps track IDs to appearance embeddings for re-identification.
* **Lost Tracks** (`dict`): Stores information about temporarily lost tracks for potential re-identification.
* **ID Links** (`dict`): Maps ultralytics track IDs to display IDs for consistent tracking.
* **Vest Detection History** (`dict`): Tracks vest detection history for temporal filtering.
* **Performance Data** (`dict`): Stores timing information for benchmarking.

Modules
=======
.. toctree::
   :maxdepth: 1
   :caption: Core Application:

   api/person_tracker

.. toctree::
   :maxdepth: 1
   :caption: Configuration & Utilities:

   api/config
   api/utils
   api/coordinate_frames

.. toctree::
   :maxdepth: 1
   :caption: Computer Vision:

   api/vision
   api/vest_classifier

.. toctree::
   :maxdepth: 1
   :caption: Tracking & Re-Identification:

   api/reid

.. toctree::
   :maxdepth: 1
   :caption: 3D Localization:

   api/localization
   api/lidar_utils

.. toctree::
   :maxdepth: 1
   :caption: Data Processing Scripts:

   api/scripts

Configuration Parameters
========================
The system can be configured through command-line arguments:

* **Dataset Selection**:
   * ``--dataset``: Choose between indoor/outdoor datasets
* **Tracking Configuration**:
   * ``--tracker``: Select tracking algorithm (bytetrack/botsort)
   * ``--reid_method``: Choose re-identification method (custom/botsort)
* **Vest Detection**:
   * ``--vest_detection``: Select detection method (color/model)
   * ``--vest_threshold``: Yellow percentage threshold
   * ``--vest_persistence``: Temporal filtering frames
* **3D Localization**:
   * ``--localization_method``: Choose localization method (depth/lidar/fusion)
* **Performance**:
   * ``--device``: Select compute device (cpu/cuda/mps)
   * ``--jump_frames``: Frame skipping for performance
* **Re-ID Tuning**:
   * ``--reid_threshold``: Similarity threshold for re-identification
   * ``--max_lost_frames``: Maximum frames to remember lost tracks

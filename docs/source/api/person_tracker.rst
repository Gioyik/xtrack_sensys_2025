*********************
Person Tracker Module
*********************

The main tracking system that orchestrates all components of the xTrack person tracking pipeline.

.. currentmodule:: person_tracker

.. automodule:: person_tracker
   :members:
   :undoc-members:
   :show-inheritance:

PersonTracker Class
===================

The main class that coordinates person detection, tracking, re-identification, vest detection, and 3D localization.

.. autoclass:: PersonTracker
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: __init__

   .. automethod:: run

   .. automethod:: _process_frame

   .. automethod:: _handle_custom_reid

   .. automethod:: _process_detection

   .. automethod:: _apply_vest_persistence

   .. automethod:: _get_3d_position

   .. automethod:: _draw_on_frame

   .. automethod:: _print_benchmark_results

   .. automethod:: _load_timestamps

   .. automethod:: _load_models

Main Function
=============

.. autofunction:: main

Class Attributes
================

**args** (`argparse.Namespace`)
   Command-line arguments and configuration parameters.

**paths** (`dict`)
   Dictionary containing paths to dataset files (video, depth, LiDAR).

**output_csv_path** (`pathlib.Path`)
   Path to the output CSV file for tracking data.

**model** (`ultralytics.YOLO`)
   YOLOv11 model for person detection.

**vest_classifier** (`VestClassifier` or `None`)
   Model-based vest classifier (if enabled).

**track_embeddings** (`dict`)
   Maps track IDs to appearance embeddings for re-identification.

**lost_tracks** (`dict`)
   Stores information about temporarily lost tracks.

**id_links** (`dict`)
   Maps ultralytics track IDs to display IDs.

**next_person_id** (`int`)
   Counter for generating unique display IDs.

**previous_track_ids** (`set`)
   Set of track IDs from the previous frame.

**vest_detection_history** (`dict`)
   Tracks vest detection history for temporal filtering.

**perf_data** (`dict`)
   Performance timing data for benchmarking.

**detection_stats** (`dict`)
   Detection and tracking statistics.

Key Methods
===========

__init__(args)
--------------

Initialize the PersonTracker with configuration arguments.

**Parameters**:
   * ``args`` (`argparse.Namespace`): Command-line arguments

**Initialization Process**:
   1. Store configuration arguments
   2. Set up dataset paths
   3. Initialize output CSV file
   4. Load timestamp data
   5. Set up logging
   6. Load ML models
   7. Initialize tracking data structures
   8. Set up performance monitoring (if benchmarking enabled)

run()
-----

Main execution loop that processes video frames and performs tracking.

**Process Flow**:
   1. Open video capture
   2. Initialize frame counters
   3. Start performance timing (if benchmarking)
   4. Process each frame:
      * Skip frames based on jump_frames setting
      * Run person detection and tracking
      * Perform re-identification
      * Detect safety vests
      * Calculate 3D positions
      * Draw visualization
      * Log results
   5. Clean up and print results

_process_frame(frame, frame_id, timestamp)
------------------------------------------

Process a single video frame through the complete tracking pipeline.

**Parameters**:
   * ``frame`` (`numpy.ndarray`): RGB video frame
   * ``frame_id`` (`int`): Sequential frame number
   * ``timestamp`` (`float`): Frame timestamp

**Process Steps**:
   1. Run YOLO person detection with tracking
   2. Handle custom re-identification (if enabled)
   3. Process each detected person:
      * Detect safety vest
      * Apply temporal persistence filtering
      * Calculate 3D position
      * Draw visualization
      * Log results

_handle_custom_reid(frame, boxes, ultralytics_ids, frame_id)
------------------------------------------------------------

Handle custom appearance-based re-identification for lost tracks.

**Parameters**:
   * ``frame`` (`numpy.ndarray`): RGB video frame
   * ``boxes`` (`numpy.ndarray`): Bounding boxes from YOLO
   * ``ultralytics_ids`` (`numpy.ndarray`): Track IDs from YOLO
   * ``frame_id`` (`int`): Current frame number

**Returns**:
   * ``list``: List of (box, display_id) tuples

**Process**:
   1. Update embeddings for active tracks
   2. Identify lost tracks and store their embeddings
   3. For each current track:
      * Check if it's a re-identified lost track
      * Calculate similarity with lost track embeddings
      * Assign display ID based on similarity threshold
      * Create new ID if no match found

_process_detection(frame, box, display_id, frame_id, timestamp)
---------------------------------------------------------------

Process a single person detection for vest detection and 3D localization.

**Parameters**:
   * ``frame`` (`numpy.ndarray`): RGB video frame
   * ``box`` (`tuple`): Bounding box (x1, y1, x2, y2)
   * ``display_id`` (`int`): Display track ID
   * ``frame_id`` (`int`): Current frame number
   * ``timestamp`` (`float`): Frame timestamp

**Process**:
   1. Extract person image from bounding box
   2. Detect safety vest (color-based or model-based)
   3. Apply temporal persistence filtering
   4. Calculate 3D position (if vest detected)
   5. Draw visualization with appropriate colors
   6. Log results to CSV

_apply_vest_persistence(track_id, is_vest_detected)
---------------------------------------------------

Apply temporal filtering to vest detection to reduce false positives.

**Parameters**:
   * ``track_id`` (`int`): Track ID
   * ``is_vest_detected`` (`bool`): Current frame vest detection result

**Returns**:
   * ``bool``: Confirmed vest detection after persistence filtering

**Process**:
   1. Maintain sliding window of vest detection history
   2. Check if vest detected for required consecutive frames
   3. Return confirmed detection based on persistence setting

_get_3d_position(box, timestamp)
--------------------------------

Calculate 3D position using the specified localization method.

**Parameters**:
   * ``box`` (`tuple`): Bounding box (x1, y1, x2, y2)
   * ``timestamp`` (`float`): Frame timestamp

**Returns**:
   * ``tuple``: 3D position (x, y, z) in base frame or (0, 0, 0) if failed

**Localization Methods**:
   * ``depth``: RGB-D camera only
   * ``lidar``: LiDAR point cloud only
   * ``fusion``: Intelligent combination of both sensors

_draw_on_frame(frame, box, display_id, position, is_vest)
---------------------------------------------------------

Draw tracking visualization on the frame.

**Parameters**:
   * ``frame`` (`numpy.ndarray`): RGB video frame
   * ``box`` (`tuple`): Bounding box (x1, y1, x2, y2)
   * ``display_id`` (`int`): Display track ID
   * ``position`` (`tuple` or `None`): 3D position (x, y, z)
   * ``is_vest`` (`bool`): Whether person is wearing a vest

**Visualization**:
   * Yellow box + thick border: Person with vest
   * Red box + thin border: Person without vest
   * Track ID label above box
   * 3D position below box (if available)

_print_benchmark_results()
--------------------------

Print comprehensive performance and accuracy metrics.

**Metrics Included**:
   * Performance: FPS, component latency, total processing time
   * Detection: Total detections, vest detections, unique track IDs
   * Re-ID: Re-identification events, similarity distribution, quality indicators
   * Localization: Success rate, failed localizations
   * Vest Detection: Yellow percentage statistics, threshold analysis
   * Configuration: All parameters used

Performance Monitoring
======================

The system includes comprehensive performance monitoring when ``--benchmark`` is enabled:

**Timing Data**:
   * Total frame processing time
   * YOLO inference time
   * Re-ID processing time
   * Vest detection and depth processing time
   * 3D localization time

**Detection Statistics**:
   * Total detections across all frames
   * Vest detections count
   * Successful vs failed localizations
   * Unique track IDs seen
   * Yellow percentage distribution
   * Re-ID events and similarity scores

**Quality Metrics**:
   * Track ID efficiency
   * Re-ID quality indicators
   * Localization success rate
   * Vest detection threshold analysis

Error Handling
==============

The system includes robust error handling:

**Model Loading**:
   * Graceful fallback if models fail to load
   * Device compatibility checking
   * Missing file warnings

**Data Processing**:
   * Empty image validation
   * Point cloud loading errors
   * Timestamp synchronization issues

**Visualization**:
   * Safe drawing operations
   * Bounds checking for text placement

Thread Safety
=============

The current implementation is single-threaded and not thread-safe. For multi-threaded applications, consider:

* Thread-safe data structures for shared state
* Synchronization for model access
* Atomic operations for counters and statistics

Memory Management
=================

The system manages memory efficiently:

**Embedding Storage**:
   * Automatic cleanup of lost track embeddings
   * Configurable maximum lost frames
   * Sliding window for vest detection history

**Performance Data**:
   * Bounded collections (deque with maxlen)
   * Automatic cleanup of old timing data

**Model Memory**:
   * Single model instances shared across frames
   * Efficient tensor operations with torch.no_grad()

Integration Notes
=================

**External Dependencies**:
   * ultralytics.YOLO for person detection
   * OpenCV for image processing
   * PyTorch for model-based vest detection
   * Open3D for LiDAR processing

**Configuration**:
   * Command-line argument parsing
   * Configurable parameters for all components
   * Environment-specific settings

**Output Formats**:
   * CSV logging for analysis
   * Real-time visualization
   * Comprehensive benchmark reports

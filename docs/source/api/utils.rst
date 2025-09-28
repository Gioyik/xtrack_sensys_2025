****************
Utilities Module
****************

Command-line argument parsing, timestamp loading, and logging utilities.

.. automodule:: utils
   :members:
   :undoc-members:
   :show-inheritance:

Main Functions
==============

parse_arguments()
-----------------

Parse command-line arguments for the xTrack person tracker.

**Returns**:
   * ``argparse.Namespace``: Parsed command-line arguments

**Arguments**:

**Dataset Selection**:
   * ``--dataset``: Dataset to process ("indoor" | "outdoor", default: "outdoor")

**Tracking Configuration**:
   * ``--tracker``: Tracker algorithm ("bytetrack" | "botsort", default: "bytetrack")
   * ``--reid_method``: Re-ID method ("custom" | "botsort", default: "custom")

**Vest Detection**:
   * ``--vest_detection``: Vest detection method ("color" | "model", default: "color")
   * ``--vest_model_path``: Path to vest model file (default: "vest_model.pth")
   * ``--vest_threshold``: Yellow percentage threshold (default: 5.0, range: 0.1-20.0)
   * ``--vest_persistence``: Consecutive frames for vest confirmation (default: 1, range: 1-10)

**3D Localization**:
   * ``--localization_method``: Localization method ("depth" | "lidar" | "fusion", default: "depth")

**Performance**:
   * ``--device``: Compute device ("cpu" | "cuda" | "mps", default: "cpu")
   * ``--jump_frames``: Frames to skip (default: 0)
   * ``--benchmark``: Enable performance benchmarking (flag)

**Re-ID Tuning**:
   * ``--reid_threshold``: ReID similarity threshold (default: 0.75, range: 0.5-0.95)
   * ``--max_lost_frames``: Max frames to remember lost tracks (default: 90, range: 30-300)

**Debugging**:
   * ``--debug``: Debug level (0 | 1 | 2, default: 0)

load_timestamps(path, scale_factor=1.0)
---------------------------------------

Load timestamps from a text file.

**Parameters**:
   * ``path`` (`pathlib.Path`): Path to timestamp file
   * ``scale_factor`` (`float`): Scale factor to apply to timestamps (default: 1.0)

**Returns**:
   * ``list``: List of timestamps as floats

**Process**:
   1. Open file and read line by line
   2. Convert each line to float
   3. Apply scale factor
   4. Skip invalid lines
   5. Return list of valid timestamps

**Error Handling**:
   * FileNotFoundError: Returns empty list with warning
   * ValueError: Skips invalid lines
   * Continues processing valid timestamps

**Scale Factor Usage**:
   * 1.0: Timestamps in seconds
   * 1e-9: Convert nanoseconds to seconds
   * 1e-6: Convert microseconds to seconds

setup_logging(output_csv_path)
------------------------------

Initialize the CSV log file with headers.

**Parameters**:
   * ``output_csv_path`` (`pathlib.Path`): Path to output CSV file

**Process**:
   1. Create CSV file with headers
   2. Set up pandas DataFrame structure
   3. Write header row to file

**CSV Structure**:
   * timestamp: Frame timestamp
   * frame_id: Sequential frame number
   * object_id: Unique track identifier
   * x_position: X coordinate in base frame (meters)
   * y_position: Y coordinate in base frame (meters)
   * z_position: Z coordinate in base frame (meters)

log_result(output_csv_path, timestamp, frame_id, display_id, x, y, z)
---------------------------------------------------------------------

Log a tracking result to the CSV file.

**Parameters**:
   * ``output_csv_path`` (`pathlib.Path`): Path to output CSV file
   * ``timestamp`` (`float`): Frame timestamp
   * ``frame_id`` (`int`): Frame number
   * ``display_id`` (`int`): Track ID
   * ``x`` (`float`): X coordinate
   * ``y`` (`float`): Y coordinate
   * ``z`` (`float`): Z coordinate

**Process**:
   1. Create DataFrame with tracking data
   2. Append to existing CSV file
   3. Maintain file format consistency

**Data Format**:
   * All coordinates in meters
   * Timestamps in seconds
   * Integer frame and object IDs

Command-Line Interface
======================

Argument Categories
-------------------

**Dataset and Input**:
   * Dataset selection (indoor/outdoor)
   * File path specifications
   * Data format options

**Algorithm Configuration**:
   * Tracker selection (ByteTrack/BoTSORT)
   * Re-ID method (custom/BotSORT)
   * Vest detection method (color/model)

**Performance Tuning**:
   * Device selection (CPU/CUDA/MPS)
   * Frame skipping for speed
   * Benchmarking options

**Quality Control**:
   * Threshold adjustments
   * Persistence settings
   * Debug output levels

Argument Validation
-------------------

**Range Validation**:
   * Vest threshold: 0.1 to 20.0
   * Vest persistence: 1 to 10
   * ReID threshold: 0.5 to 0.95
   * Max lost frames: 30 to 300

**Choice Validation**:
   * Dataset: indoor, outdoor
   * Tracker: bytetrack, botsort
   * Device: cpu, cuda, mps
   * Debug level: 0, 1, 2

**Dependency Validation**:
   * BoTSORT Re-ID requires BoTSORT tracker
   * Model-based vest detection requires model file

Data Management
===============

Timestamp Handling
------------------

**File Format**:
   * Plain text files
   * One timestamp per line
   * Floating-point precision

**Synchronization**:
   * RGB camera timestamps
   * Depth camera timestamps
   * LiDAR timestamps (with scale factor)

**Error Recovery**:
   * Skip invalid timestamps
   * Continue processing valid data
   * Warning messages for issues

CSV Logging
-----------

**File Structure**:
   * Comma-separated values
   * Header row with column names
   * Append-only operation

**Data Integrity**:
   * Consistent column order
   * Proper data types
   * No duplicate headers

**Performance**:
   * Efficient pandas operations
   * Minimal file I/O
   * Memory-efficient processing

Error Handling
==============

**File Operations**:
   * Graceful handling of missing files
   * Permission error handling
   * Disk space considerations

**Data Validation**:
   * Type checking for arguments
   * Range validation for parameters
   * Format validation for files

**User Input**:
   * Clear error messages
   * Helpful suggestions
   * Graceful degradation

Usage Examples
==============

**Basic Argument Parsing**:
   .. code:: python

      from utils import parse_arguments
   
      # Parse command-line arguments
      args = parse_arguments()
   
      print(f"Dataset: {args.dataset}")
      print(f"Tracker: {args.tracker}")
      print(f"Device: {args.device}")

**Timestamp Loading**:
   .. code:: python

      from utils import load_timestamps
   
      # Load RGB timestamps
      rgb_timestamps = load_timestamps("rgb_timestamps.txt")
   
      # Load LiDAR timestamps (convert from nanoseconds)
      lidar_timestamps = load_timestamps("lidar_timestamps.txt", scale_factor=1e-9)
   
      print(f"Loaded {len(rgb_timestamps)} RGB timestamps")
      print(f"Loaded {len(lidar_timestamps)} LiDAR timestamps")

**CSV Logging Setup**:
   .. code:: python

      from utils import setup_logging, log_result
   
      # Initialize CSV file
      output_path = Path("tracking_log.csv")
      setup_logging(output_path)
   
      # Log tracking results
      log_result(output_path, 1.234, 1, 1, 2.5, -1.2, 0.8)
      log_result(output_path, 1.235, 2, 1, 2.6, -1.1, 0.9)

**Error Handling**:
   .. code:: python

      from utils import load_timestamps
   
      try:
       timestamps = load_timestamps("missing_file.txt")
       if not timestamps:
           print("Warning: No timestamps loaded")
      except Exception as e:
       print(f"Error loading timestamps: {e}")

Integration Notes
=================

**Module Dependencies**:
   * argparse: Command-line argument parsing
   * pandas: CSV file operations
   * pathlib: Cross-platform path handling

**System Integration**:
   * Used by PersonTracker for initialization
   * Referenced by all modules for configuration
   * Centralized utility functions

**Performance Considerations**:
   * Efficient timestamp loading
   * Minimal CSV I/O overhead
   * Memory-efficient data structures

Future Improvements
===================

**Configuration Management**:
   * Configuration file support
   * Environment variable integration
   * Runtime configuration updates

**Enhanced Logging**:
   * Structured logging with levels
   * Log rotation and compression
   * Performance metrics logging

**Validation Enhancements**:
   * Comprehensive input validation
   * Configuration consistency checking
   * Automatic parameter optimization

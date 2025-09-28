********************
Configuration Module
********************

Dataset path management and system configuration.

.. automodule:: config
   :members:
   :undoc-members:
   :show-inheritance:

Constants
=========

PROJECT_DIR
-----------

Project root directory path.

**Type**: `pathlib.Path`

**Description**: Automatically determined path to the project root directory (parent of src/ directory).

DATA_DIR
--------

Data directory path.

**Type**: `pathlib.Path`

**Description**: Path to the data directory containing datasets.

OUTPUT_DIR
----------

Output directory path.

**Type**: `pathlib.Path`

**Description**: Path to the output directory for generated files.

REID_SIMILARITY_THRESHOLD
-------------------------

Default similarity threshold for re-identification.

**Type**: `float`

**Default**: `0.4`

**Description**: Cosine similarity threshold for re-identifying lost tracks.

MAX_LOST_FRAMES
---------------

Default maximum frames to remember lost tracks.

**Type**: `int`

**Default**: `90`

**Description**: Maximum number of frames a track can be lost before permanent deletion.

Main Functions
==============

get_dataset_paths(dataset_name)
--------------------------------

Get file paths for a specific dataset.

**Parameters**:
   * ``dataset_name`` (`str`): Dataset name ("indoor" or "outdoor")

**Returns**:
   * ``dict``: Dictionary containing paths to dataset files

**Returned Dictionary Keys**:
   * ``video_path``: Path to RGB video file
   * ``video_timestamps_path``: Path to video timestamps file
   * ``depth_folder_path``: Path to depth frames directory
   * ``depth_timestamps_path``: Path to depth timestamps file
   * ``lidar_folder_path``: Path to LiDAR .pcd files (outdoor only)
   * ``lidar_timestamps_path``: Path to LiDAR timestamps file (outdoor only)

**Dataset Structure**:

**Indoor Dataset**:
   * Video: `data/data_indoor/camera/d435i/color/cam_2025_06_04_09_24_42.avi`
   * Video timestamps: `data/data_indoor/camera/d435i/color/cam_2025_06_04_09_24_42_timestamps.txt`
   * Depth frames: `data/data_indoor/camera/d435i/depth/cam_2025_06_04_09_24_42_depth_frames/`
   * Depth timestamps: `data/data_indoor/camera/d435i/depth/cam_2025_06_04_09_24_42_timestamps.txt`

**Outdoor Dataset**:
   * Video: `data/data_outdoor/camera/d435i/color/cam_2025_06_04_09_41_51.avi`
   * Video timestamps: `data/data_outdoor/camera/d435i/color/cam_2025_06_04_09_41_51_timestamps.txt`
   * Depth frames: `data/data_outdoor/camera/d435i/depth/cam_2025_06_04_09_41_51_depth_frames/`
   * Depth timestamps: `data/data_outdoor/camera/d435i/depth/cam_2025_06_04_09_41_51_timestamps.txt`
   * LiDAR files: `output/ouster_20250604074152/`
   * LiDAR timestamps: `output/ouster_20250604074152/timestamps.txt`

File Structure Requirements
===========================

The system expects data to be organized in the following structure:


   .. code:: text

      data/
      ├── data_indoor/
      │   └── camera/
      │       └── d435i/
      │           ├── color/
      │           │   ├── cam_2025_06_04_09_24_42.avi
      │           │   └── cam_2025_06_04_09_24_42_timestamps.txt
      │           └── depth/
      │               ├── cam_2025_06_04_09_24_42_depth_frames/
      │               │   ├── 000000.png
      │               │   ├── 000001.png
      │               │   └── ...
      │               └── cam_2025_06_04_09_24_42_timestamps.txt
      └── data_outdoor/
       └── camera/
           └── d435i/
               ├── color/
               │   ├── cam_2025_06_04_09_41_51.avi
               │   └── cam_2025_06_04_09_41_51_timestamps.txt
               └── depth/
                   ├── cam_2025_06_04_09_41_51_depth_frames/
                   │   ├── 000000.png
                   │   ├── 000001.png
                   │   └── ...
                   └── cam_2025_06_04_09_41_51_timestamps.txt

      output/
      └── ouster_20250604074152/
       ├── 1234567890123456789.pcd
       ├── 1234567890123456790.pcd
       ├── ...
       └── timestamps.txt

File Format Specifications
==========================

Video Files
-----------

**Format**: AVI
**Codec**: Compatible with OpenCV
**Resolution**: 640x480 pixels
**Frame Rate**: Variable (typically 30 FPS)

Timestamp Files
---------------

**Format**: Text file with one timestamp per line
**Units**: Seconds (floating point)
**Precision**: Microsecond precision
**Example**:

   .. code:: text

      1.234567890
      1.234567891
      1.234567892

Depth Images
------------

**Format**: PNG (16-bit)
**Resolution**: 640x480 pixels
**Units**: Millimeters
**Naming**: 6-digit zero-padded (000000.png, 000001.png, etc.)

LiDAR Files
-----------

**Format**: PCD (Point Cloud Data)
**Fields**: x, y, z, intensity, reflectivity, nearir
**Naming**: Timestamp-based (nanoseconds)
**Associated**: timestamps.txt file with corresponding timestamps

Configuration Management
========================

Path Resolution
---------------

**Automatic Path Detection**:
   * PROJECT_DIR: Automatically determined from module location
   * DATA_DIR: Relative to project root
   * OUTPUT_DIR: Relative to project root

**Cross-Platform Compatibility**:
   * Uses pathlib.Path for cross-platform path handling
   * Handles different path separators automatically
   * Works on Windows, macOS, and Linux

Dataset Selection
-----------------

**Supported Datasets**:
   * "indoor": Controlled environment dataset
   * "outdoor": Natural environment dataset

**Dataset Differences**:
   * Indoor: RGB-D camera only
   * Outdoor: RGB-D camera + LiDAR data

**Validation**:
   * Function validates dataset name
   * Returns appropriate paths for each dataset
   * Handles missing LiDAR data gracefully

Usage Examples
==============

**Basic Usage**:
   .. code:: python

      from config import get_dataset_paths
   
      # Get indoor dataset paths
      indoor_paths = get_dataset_paths("indoor")
      print(f"Video path: {indoor_paths['video_path']}")
   
      # Get outdoor dataset paths
      outdoor_paths = get_dataset_paths("outdoor")
      print(f"LiDAR path: {outdoor_paths['lidar_folder_path']}")

**Path Validation**:
   .. code:: python

      from pathlib import Path
      from config import get_dataset_paths
   
      # Check if files exist
      paths = get_dataset_paths("outdoor")
   
      if paths['video_path'].exists():
       print("Video file found")
      else:
       print("Video file missing")
   
      if paths['lidar_folder_path'].exists():
       print("LiDAR data found")
      else:
       print("LiDAR data missing")

**Integration with Tracking System**:
   .. code:: python

      from config import get_dataset_paths
      from person_tracker import PersonTracker
   
      # Get dataset paths
      paths = get_dataset_paths("outdoor")
   
      # Initialize tracker with paths
      tracker = PersonTracker(args)
      tracker.paths = paths

Error Handling
==============

**Missing Datasets**:
   * Function returns paths even if files don't exist
   * Validation is performed at runtime
   * Graceful handling of missing LiDAR data

**Invalid Dataset Names**:
   * Function accepts only "indoor" or "outdoor"
   * Default behavior for invalid names
   * Clear error messages for debugging

**Path Resolution Issues**:
   * Automatic path resolution from module location
   * Handles different installation scenarios
   * Cross-platform path compatibility

Integration Notes
=================

**Module Dependencies**:
   * pathlib: For cross-platform path handling
   * No external dependencies required

**System Integration**:
   * Used by PersonTracker for dataset initialization
   * Referenced by utility functions for file operations
   * Centralized configuration management

**Extensibility**:
   * Easy to add new datasets
   * Configurable path structure
   * Support for additional sensor data

Future Improvements
===================

**Configuration File Support**:
   * YAML or JSON configuration files
   * Runtime configuration updates
   * Environment-specific settings

**Dynamic Path Resolution**:
   * Automatic dataset discovery
   * Multiple dataset support
   * Flexible path configuration

**Validation Enhancements**:
   * File existence checking
   * Format validation
   * Integrity verification

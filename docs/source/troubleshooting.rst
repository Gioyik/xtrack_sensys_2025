**********************
Troubleshooting Guide
**********************

This guide helps you diagnose and resolve common issues with the xTrack person tracking system.

Common Issues and Solutions
============================

Poor LiDAR Localization Performance
-----------------------------------

**Symptoms**:
   * LiDAR localization returns (0, 0, 0) positions
   * Inaccurate 3D positions
   * No LiDAR data being processed

**Solutions**:

1. **Verify LiDAR Data Processing**:

   .. code:: bash

      # Ensure LiDAR data has been converted
      python3 scripts/os_pcap_to_pcd_csv.py data/data_outdoor/os_pcaps/ouster_20250604074152.pcap -o output
      
      # Check that output files exist
      ls output/ouster_20250604074152/
      # Should show: \*.pcd files, timestamps.txt, \*_imu.csv

2. **Check Timestamp File**:

   .. code:: bash

      # Verify timestamps.txt exists and has content
      head -5 output/ouster_20250604074152/timestamps.txt
      
      # Check file format (should be one timestamp per line)
      wc -l output/ouster_20250604074152/timestamps.txt

3. **Verify Coordinate Frame Transformations**:

   * Check that `coordinate_frames.py` has proper calibration
   * Ensure transformation matrices are correctly set
   * Verify camera-LiDAR calibration accuracy

4. **Use Sensor Fusion**:

   .. code:: bash

      # Try sensor fusion for more robust results
      python3 src/person_tracker.py --dataset outdoor --localization_method fusion

5. **Adjust Filtering Parameters**:

   * Modify parameters in `lidar_utils.py` for your environment
   * Adjust distance filtering ranges
   * Tune clustering parameters

Performance Issues
------------------

**Symptoms**:
   * Low FPS (< 5 FPS)
   * High CPU/GPU usage
   * System lag or freezing

**Solutions**:

1. **Enable Frame Skipping**:

   .. code:: bash

      # Skip frames for faster processing
      python3 src/person_tracker.py --dataset outdoor --jump_frames 2
      
      # More aggressive skipping
      python3 src/person_tracker.py --dataset outdoor --jump_frames 4

2. **Enable GPU Acceleration**:

   .. code:: bash

      # Use CUDA if available
      python3 src/person_tracker.py --dataset outdoor --device cuda
      
      # Use MPS on Apple Silicon
      python3 src/person_tracker.py --dataset outdoor --device mps

3. **Use Faster Tracking Algorithm**:

   .. code:: bash

      # ByteTrack is faster than BoTSORT
      python3 src/person_tracker.py --dataset outdoor --tracker bytetrack

4. **Disable Debug Output**:

   .. code:: bash

      # Minimize console output
      python3 src/person_tracker.py --dataset outdoor --debug 0

5. **Use Depth-Only Localization**:

   .. code:: bash

      # Skip LiDAR processing for speed
      python3 src/person_tracker.py --dataset outdoor --localization_method depth

Multiple People Getting Same Track ID
-------------------------------------

**Problem**: Different people are incorrectly merged into the same track ID.

**Symptoms**:
   * Frequent "Re-identified track X as Y" messages
   * Multiple distinct people sharing the same ID
   * Track IDs not increasing properly

**Solutions**:

1. **Increase ReID Threshold**:

   .. code:: bash

      # More conservative re-identification
      python3 src/person_tracker.py --dataset outdoor --reid_threshold 0.8
      
      # Very conservative
      python3 src/person_tracker.py --dataset outdoor --reid_threshold 0.85

2. **Reduce Lost Frame Memory**:

   .. code:: bash

      # Shorter memory for lost tracks
      python3 src/person_tracker.py --dataset outdoor --max_lost_frames 60
      
      # Even shorter
      python3 src/person_tracker.py --dataset outdoor --max_lost_frames 30

3. **Use BoTSORT ReID**:

   .. code:: bash

      # Use built-in BoTSORT re-identification
      python3 src/person_tracker.py --dataset outdoor --tracker botsort --reid_method botsort

4. **Test Conservative Settings**:

   .. code:: bash

      # Very conservative configuration
      python3 src/person_tracker.py --dataset outdoor --reid_threshold 0.85 --max_lost_frames 45

**Diagnostic Commands**:

   .. code:: bash

      # Monitor re-ID events
      python3 src/person_tracker.py --dataset outdoor --debug 1
      
      # Check track ID behavior
      python3 src/person_tracker.py --dataset outdoor --benchmark

Device Compatibility Issues
---------------------------

**CUDA Issues**:

   * **Problem**: CUDA not available or fails to initialize
   * **Solution**: System automatically falls back to CPU with warning
   * **Check**: Verify CUDA installation and GPU compatibility

**MPS Issues**:

   * **Problem**: MPS not available on older macOS versions
   * **Solution**: System automatically falls back to CPU
   * **Check**: Ensure macOS version supports MPS

**Memory Issues**:

   * **Problem**: Out of memory errors with large point clouds
   * **Solution**: Reduce max_distance parameter in filtering functions
   * **Check**: Monitor system memory usage

**Diagnostic Commands**:

   .. code:: python

      # Check device availability
      import torch
      print(f"CUDA available: {torch.cuda.is_available()}")
      print(f"MPS available: {torch.backends.mps.is_available()}")
      
      # Test device selection
      python3 src/person_tracker.py --dataset outdoor --device cpu --benchmark

Vest Detection Issues
---------------------

**Symptoms**:

   * Too many false positives (non-vest people marked as wearing vests)
   * Too many false negatives (vest-wearing people not detected)
   * Inconsistent vest detection

**Solutions**:

1. **Adjust Vest Threshold**:

   .. code:: bash

      # More sensitive detection
      python3 src/person_tracker.py --dataset outdoor --vest_threshold 3.0
      
      # More conservative detection
      python3 src/person_tracker.py --dataset outdoor --vest_threshold 8.0

2. **Use Temporal Persistence**:

   .. code:: bash

      # Require multiple consecutive detections
      python3 src/person_tracker.py --dataset outdoor --vest_persistence 3
      
      # Very conservative
      python3 src/person_tracker.py --dataset outdoor --vest_persistence 5

3. **Enable Debug Visualization**:

   .. code:: bash

      # See vest detection masks
      python3 src/person_tracker.py --dataset outdoor --debug 2

4. **Fine-tune Color Range**:

   * Modify `lower_yellow` and `upper_yellow` in `vision.py`
   * Adjust HSV color range for your lighting conditions
   * Test different color thresholds

**Diagnostic Commands**:

   .. code:: bash

      # Monitor vest detection percentages
      python3 src/person_tracker.py --dataset outdoor --debug 2 --benchmark
      
      # Test different thresholds
      python3 src/person_tracker.py --dataset outdoor --vest_threshold 6.0 --vest_persistence 2

Data Loading Issues
-------------------

**Symptoms**:

   * "File not found" errors
   * Empty tracking results
   * System fails to start

**Solutions**:

1. **Verify Data Structure**:

   .. code:: bash

      # Check data directory structure
      ls -la data/data_outdoor/camera/d435i/color/
      ls -la data/data_outdoor/camera/d435i/depth/
      
      # Verify required files exist
      ls data/data_outdoor/camera/d435i/color/\*.avi
      ls data/data_outdoor/camera/d435i/color/\*_timestamps.txt

2. **Check File Permissions**:

   .. code:: bash

      # Ensure files are readable
      chmod 644 data/data_outdoor/camera/d435i/color/\*
      chmod 644 data/data_outdoor/camera/d435i/depth/\*

3. **Validate Timestamp Files**:

   .. code:: bash

      # Check timestamp file format
      head -5 data/data_outdoor/camera/d435i/color/\*_timestamps.txt
      
      # Verify file is not empty
      wc -l data/data_outdoor/camera/d435i/color/\*_timestamps.txt

4. **Test with Indoor Dataset**:

   .. code:: bash

      # Try indoor dataset if outdoor fails
      python3 src/person_tracker.py --dataset indoor

Model Loading Issues
--------------------

**Symptoms**:

   * "Model not found" errors
   * Poor detection performance
   * System crashes during initialization

**Solutions**:

1. **Verify YOLO Model**:

   * YOLOv11 model should download automatically
   * Check internet connection for model download
   * Verify model file exists: `yolo11n.pt`

2. **Check Vest Model**:

   .. code:: bash

      # Verify vest model exists
      ls -la vest_model.pth
      
      # Use color-based detection if model missing
      python3 src/person_tracker.py --dataset outdoor --vest_detection color

3. **Test Device Compatibility**:

   .. code:: bash

      # Force CPU usage
      python3 src/person_tracker.py --dataset outdoor --device cpu
      
      # Test different devices
      python3 src/person_tracker.py --dataset outdoor --device cuda
      python3 src/person_tracker.py --dataset outdoor --device mps

4. **Check Dependencies**:

   .. code:: bash

      # Verify all packages installed
      pip install -r requirements.txt
      
      # Test imports
      python3 -c "import ultralytics, cv2, torch, open3d; print('All dependencies OK')"

Synchronization Issues
----------------------

**Symptoms**:
   * Poor 3D localization accuracy
   * Misaligned sensor data
   * Inconsistent tracking results

**Solutions**:

1. **Check Timestamp Alignment**:

   .. code:: bash

      # Compare timestamp ranges
      head -1 data/data_outdoor/camera/d435i/color/\*_timestamps.txt
      tail -1 data/data_outdoor/camera/d435i/color/\*_timestamps.txt
      
      head -1 data/data_outdoor/camera/d435i/depth/\*_timestamps.txt
      tail -1 data/data_outdoor/camera/d435i/depth/\*_timestamps.txt

2. **Verify LiDAR Timestamps**:

   .. code:: bash

      # Check LiDAR timestamp format
      head -5 output/ouster_20250604074152/timestamps.txt
      
      # Verify timestamp scale (should be nanoseconds)
      python3 -c "
      import numpy as np
      timestamps = np.loadtxt('output/ouster_20250604074152/timestamps.txt')
      print(f'Timestamp range: {timestamps[0]} to {timestamps[-1]}')
      print(f'Scale factor needed: {1e-9 if timestamps[0] > 1e12 else 1.0}')
      "

3. **Use Sensor Fusion**:

   .. code:: bash

      # Fusion handles synchronization better
      python3 src/person_tracker.py --dataset outdoor --localization_method fusion

4. **Check Calibration**:
   * Verify camera intrinsic parameters
   * Check LiDAR-camera transformation matrices
   * Ensure proper coordinate frame alignment

Debugging Techniques
====================

Enable Debug Output
-------------------

**Level 1 - Basic Information**:

   .. code:: bash

      python3 src/person_tracker.py --dataset outdoor --debug 1

**Level 2 - Detailed Information**:

   .. code:: bash

      python3 src/person_tracker.py --dataset outdoor --debug 2

**Benchmarking Mode**:

   .. code:: bash

      python3 src/person_tracker.py --dataset outdoor --benchmark

Monitor System Resources
------------------------

**CPU Usage**:

   .. code:: bash

      # Monitor CPU usage
      top -p $(pgrep -f person_tracker.py)

**Memory Usage**:

   .. code:: bash

      # Monitor memory usage
      ps aux | grep person_tracker.py

**GPU Usage** (if using CUDA):

   .. code:: bash

      # Monitor GPU usage
      nvidia-smi

Check Log Files
---------------

**CSV Output**:

   .. code:: bash

      # Check tracking results
      tail -f output/tracking_log_outdoor.csv
      
      # Count successful localizations
      wc -l output/tracking_log_outdoor.csv

**Console Output**:
   * Monitor re-identification messages
   * Check for error messages
   * Verify configuration parameters

Performance Profiling
=====================

Run Benchmark Suite
-------------------

**Complete Benchmark**:

   .. code:: bash

      python3 scripts/run_benchmark_suite.py --output_dir benchmark_results/

**Individual Tests**:

   .. code:: bash

      # Test specific configurations
      python3 src/person_tracker.py --dataset outdoor --tracker bytetrack --benchmark
      python3 src/person_tracker.py --dataset outdoor --tracker botsort --benchmark

**Performance Analysis**:
   * Compare FPS across configurations
   * Analyze component latency
   * Identify bottlenecks

Configuration Optimization
==========================

Parameter Tuning
----------------

**Vest Detection Tuning**:

   .. code:: bash

      # Test different thresholds
      for threshold in 3.0 5.0 6.0 8.0 12.0; do
          python3 src/person_tracker.py --dataset outdoor --vest_threshold $threshold --benchmark
      done

**Re-ID Tuning**:

   .. code:: bash

      # Test different Re-ID thresholds
      for threshold in 0.6 0.7 0.75 0.8 0.85; do
          python3 src/person_tracker.py --dataset outdoor --reid_threshold $threshold --benchmark
      done

**Frame Skipping Optimization**:

   .. code:: bash

      # Test different frame skipping values
      for skip in 0 1 2 4 9; do
          python3 src/person_tracker.py --dataset outdoor --jump_frames $skip --benchmark
      done

Optimal Configuration Selection
-------------------------------

**For Real-time Applications**:

   .. code:: bash

      python3 src/person_tracker.py --dataset outdoor --tracker bytetrack --localization_method depth --jump_frames 2 --device mps

**For Maximum Accuracy**:

   .. code:: bash

      python3 src/person_tracker.py --dataset outdoor --tracker botsort --reid_method botsort --localization_method fusion --jump_frames 0

**For Balanced Performance**:

   .. code:: bash

      python3 src/person_tracker.py --dataset outdoor --tracker bytetrack --localization_method fusion --vest_threshold 6.0 --reid_threshold 0.8 --jump_frames 1

Getting Help
============

**Check Documentation**:
   * Review the :doc:`user_guide` for detailed usage instructions
   * Consult the :doc:`api_reference` for technical details
   * See the :doc:`testing_guide` for performance optimization

**Run Diagnostics**:

   .. code:: bash

      # Check system status
      python3 -c "
      import sys
      print(f'Python version: {sys.version}')
      
      try:
          import ultralytics
          print('✓ ultralytics installed')
      except ImportError:
          print('✗ ultralytics missing')
      
      try:
          import cv2
          print('✓ opencv-python installed')
      except ImportError:
          print('✗ opencv-python missing')
      
      try:
          import torch
          print(f'✓ PyTorch installed (CUDA: {torch.cuda.is_available()})')
      except ImportError:
          print('✗ PyTorch missing')
      
      try:
          import open3d
          print('✓ open3d installed')
      except ImportError:
          print('✗ open3d missing')
      "

**Verify Data Integrity**:

   .. code:: bash

      # Check data files
      ls -la data/data_outdoor/camera/d435i/color/
      ls -la data/data_outdoor/camera/d435i/depth/
      ls -la output/ouster_20250604074152/ 2>/dev/null || echo "LiDAR data not processed"

**Test Basic Functionality**:

   .. code:: bash

      # Run minimal test
      python3 src/person_tracker.py --dataset outdoor --debug 0 --jump_frames 9

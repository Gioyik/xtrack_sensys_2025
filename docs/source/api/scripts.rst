**********************
Scripts Documentation
**********************

Utility scripts for data processing and system benchmarking.

Scripts Overview
================

Overview
========

The scripts directory contains utility programs for:

* **Data Processing**: Converting raw sensor data to usable formats
* **Benchmarking**: Automated performance testing and evaluation
* **Data Conversion**: Transforming between different data formats
* **System Validation**: Testing system components and configurations

Data Processing Scripts
=======================

os_pcap_to_pcd_csv.py
---------------------

Convert Ouster LiDAR PCAP files to PCD format and extract IMU data.

**Purpose**: Process raw LiDAR data from PCAP files into point cloud data (PCD) format for use with the tracking system.

**Usage**:
   .. code:: bash

      python3 scripts/os_pcap_to_pcd_csv.py <pcap_path> [-o <output_dir>]

**Parameters**:
   * ``pcap_path``: Path to input PCAP file
   * ``-o, --out_dir``: Output directory (optional, defaults to base name)

**Output Files**:
   * ``*.pcd``: Point cloud data files (one per scan)
   * ``timestamps.txt``: Scan timestamps in nanoseconds
   * ``*_imu.csv``: IMU data with timestamps

**Process**:
   1. Parse PCAP file to extract LiDAR and IMU packets
   2. Convert LiDAR ranges to XYZ coordinates
   3. Extract additional LiDAR fields (intensity, reflectivity, near-IR)
   4. Generate PCD files for each complete scan
   5. Create timestamp file for synchronization
   6. Export IMU data to CSV format

**File Format**:
   * PCD files: ASCII format with x, y, z, intensity, reflectivity, nearir fields
   * Timestamps: One timestamp per line (nanoseconds)
   * IMU CSV: Columns for timestamp, linear acceleration, angular velocity

**Dependencies**:
   * ouster-sdk: For LiDAR data processing
   * dpkt: For PCAP file parsing
   * open3d: For PCD file writing

Benchmarking Scripts
====================

run_benchmark_suite.py
----------------------

Automated benchmark suite for comprehensive system testing.

**Purpose**: Run systematic tests across different configurations to evaluate performance and identify optimal settings.

**Usage**:
   .. code:: bash

      python3 scripts/run_benchmark_suite.py [--output_dir <dir>]

**Parameters**:
   * ``--output_dir``: Directory for benchmark results (optional)

**Test Categories**:
   1. **Localization Methods**: depth, lidar, fusion
   2. **Tracking Algorithms**: bytetrack, botsort
   3. **Vest Detection**: color, model
   4. **Device Performance**: cpu, cuda, mps
   5. **Frame Skipping**: 0, 1, 2, 4, 9 frames
   6. **ReID Thresholds**: 0.6, 0.7, 0.75, 0.8, 0.85
   7. **Vest Thresholds**: 3.0, 5.0, 6.0, 8.0, 12.0

**Output**:
   * Individual test results (JSON format)
   * Summary report with pass/fail status
   * Performance metrics and timing data
   * Configuration recommendations

**Automated Tests**:
   * Baseline configurations
   * Performance optimization tests
   * Accuracy evaluation tests
   * Device compatibility tests
   * Parameter sensitivity analysis

**Results Analysis**:
   * FPS performance comparison
   * Component latency analysis
   * Detection accuracy metrics
   * Re-ID quality assessment
   * Configuration optimization

Script Details
==============

os_pcap_to_pcd_csv.py
---------------------

**Input Processing**:
   * PCAP file parsing with dpkt
   * Ouster SDK integration for LiDAR data
   * Packet format validation
   * Scan reconstruction from packets

**Data Conversion**:
   * Range-to-XYZ conversion using calibration
   * Coordinate system transformation
   * Field extraction (intensity, reflectivity, near-IR)
   * Timestamp synchronization

**Output Generation**:
   * PCD file creation with proper headers
   * Timestamp file for synchronization
   * IMU data export to CSV
   * Error handling and validation

**Error Handling**:
   * Missing file validation
   * Corrupted data recovery
   * Format validation
   * Progress reporting

run_benchmark_suite.py
----------------------

**Test Configuration**:
   * Predefined test matrices
   * Configurable parameters
   * Device availability detection
   * Timeout handling

**Execution Management**:
   * Sequential test execution
   * Progress tracking
   * Error recovery
   * Result collection

**Results Processing**:
   * JSON serialization
   * Summary generation
   * Performance analysis
   * Recommendation generation

**Reporting**:
   * Console output with progress
   * Detailed result files
   * Summary statistics
   * Pass/fail status

Usage Examples
==============

**LiDAR Data Processing**:
   .. code:: bash

      # Convert outdoor dataset LiDAR data
      python3 scripts/os_pcap_to_pcd_csv.py data/data_outdoor/os_pcaps/ouster_20250604074152.pcap -o output
   
      # Check output files
      ls output/ouster_20250604074152/
      # Should show: *.pcd files, timestamps.txt, *_imu.csv

**Benchmark Suite Execution**:
   .. code:: bash

      # Run complete benchmark suite
      python3 scripts/run_benchmark_suite.py --output_dir benchmark_results/
   
      # Check results
      ls benchmark_results/
      # Should show: individual test results, summary.json

**Custom Benchmark Configuration**:
   .. code:: python

      # Modify test configurations in the script
      test_configs = [
       {
           "name": "Custom_Test",
           "params": {
               "dataset": "outdoor",
               "tracker": "bytetrack",
               "localization_method": "fusion",
               "vest_threshold": 6.0,
               "reid_threshold": 0.8
           }
       }
      ]

Integration with Main System
============================

**Data Flow**:
   1. Raw LiDAR data (PCAP) → Processing script → PCD files
   2. PCD files + timestamps → Main tracking system
   3. Benchmark results → Performance analysis

**File Dependencies**:
   * LiDAR processing creates files used by tracking system
   * Benchmark suite tests the complete system
   * Results inform configuration optimization

**Error Handling**:
   * Scripts validate input data
   * Main system handles missing processed data
   * Benchmark suite reports test failures

Performance Considerations
===========================

**LiDAR Processing**:
   * Memory usage scales with point cloud size
   * Processing time depends on PCAP file size
   * Disk I/O for large datasets

**Benchmark Suite**:
   * Total execution time: 30-60 minutes
   * Memory usage: Moderate (per test)
   * Disk usage: Results and logs

**Optimization**:
   * Parallel processing where possible
   * Efficient data structures
   * Progress reporting for long operations

Error Handling
==============

**LiDAR Processing Errors**:
   * Missing PCAP files
   * Corrupted data packets
   * Insufficient disk space
   * Permission issues

**Benchmark Suite Errors**:
   * Test execution failures
   * Timeout handling
   * Resource constraints
   * Configuration errors

**Recovery Strategies**:
   * Graceful error handling
   * Partial result saving
   * Retry mechanisms
   * Clear error messages

Future Improvements
===================

**Enhanced Data Processing**:
   * Support for additional LiDAR formats
   * Real-time processing capabilities
   * Advanced filtering options

**Benchmark Enhancements**:
   * Parallel test execution
   * Advanced result analysis
   * Automated optimization
   * Performance regression detection

**Integration Improvements**:
   * Automated data pipeline
   * Continuous integration support
   * Performance monitoring
   * Alert systems

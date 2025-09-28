xTrack Sensys - Person Tracking System
============================================

.. image:: ../final_report_2025/logo.jpg
   :alt: xTrack Logo
   :width: 200px
   :align: center
   :class: logo-spacing

Welcome to the xTrack Sensys documentation! This project focuses on the identification, localization, and tracking of people using sensor data from the xTrack vehicle. The primary goal is to detect and track individuals, with a special focus on those wearing yellow safety vests, using data from an RGB-D camera and LiDAR.

.. toctree::
   :hidden:

   getting_started
   user_guide
   system_architecture
   api_reference
   testing_guide
   troubleshooting
   license

Key Features
============

* **Person Detection and Tracking**: Utilizes the YOLOv11 model from `ultralytics` to detect and track people in a video stream
* **Advanced Re-Identification (Re-ID)**: Implements two methods to ensure consistent tracking IDs
* **3D Localization**: Calculates the 3D position of detected individuals relative to the xTrack vehicle
* **Safety Vest Detection**: Implements both color-based and model-based vest detection methods
* **Selectable Trackers**: Supports ByteTrack and BoTSORT tracking algorithms
* **Advanced LiDAR Processing**: Enhanced point cloud processing with noise filtering and ground plane removal
* **Sensor Fusion**: Intelligent fusion of RGB-D camera and LiDAR data for improved accuracy
* **Cross-Platform GPU Support**: Full support for CUDA (NVIDIA), MPS (Apple Silicon), and CPU processing
* **Performance Benchmarking**: Comprehensive tools to measure and log system performance
* **Real-time Data Logging**: Logs all tracking data for people with vests to CSV files

Quick Start
===========

To get started with xTrack, follow these steps:

1. **Install Dependencies**:

   
   .. code:: bash

      pip install -r requirements.txt

2. **Run Basic Tracking**:

   
   .. code:: bash

      python3 src/person_tracker.py --dataset outdoor

3. **View Results**: Check the `output/tracking_log_outdoor.csv` file for tracking data and watch the live visualization window.

For more detailed instructions, see the :doc:`getting_started` guide.

System Architecture
===================

The xTrack system consists of several key components:

* **Person Detection**: YOLOv11-based person detection
* **Tracking**: ByteTrack or BoTSORT multi-object tracking
* **Re-Identification**: Custom appearance-based or BoTSORT native Re-ID
* **Vest Detection**: Color-based HSV segmentation or model-based classification
* **3D Localization**: Depth camera, LiDAR, or sensor fusion
* **Data Logging**: Real-time CSV output for analysis

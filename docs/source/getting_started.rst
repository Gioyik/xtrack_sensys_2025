****************
Getting Started
****************
This guide will help you set up and run the xTrack person tracking system.

System Requirements
====================

* Python 3.8 or higher
* At least 4GB RAM (8GB recommended)
* Optional: NVIDIA GPU with CUDA support for acceleration
* Optional: Apple Silicon Mac with MPS support

Data Preparation
================
The system expects data in the following structure:

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
      │               └── cam_2025_06_04_09_24_42_timestamps.txt
      └── data_outdoor/
          └── camera/
              └── d435i/
                  ├── color/
                  │   ├── cam_2025_06_04_09_41_51.avi
                  │   └── cam_2025_06_04_09_41_51_timestamps.txt
                  └── depth/
                      ├── cam_2025_06_04_09_41_51_depth_frames/
                      └── cam_2025_06_04_09_41_51_timestamps.txt

For LiDAR-based localization, you'll also need to process the raw LiDAR data:

   .. code:: bash

      python3 scripts/os_pcap_to_pcd_csv.py data/data_outdoor/os_pcaps/ouster_20250604074152.pcap -o output

This creates the necessary `.pcd` files and timestamps for LiDAR synchronization.

Installation
============

1. **Clone the Repository**:

   .. code:: bash

      git clone <repository-url>
      cd xtrack_sensys_2025

2. **Create Virtual Environment** (Recommended):
   
   .. code:: bash

      python3 -m venv xtrack_env
      source xtrack_env/bin/activate  # On Windows: xtrack_env\Scripts\activate

3. **Install Dependencies**:
   
   .. code:: bash

      pip install -r requirements.txt

4. **Verify Installation**:
   
   .. code:: bash

      python3 -c "import ultralytics, cv2, torch, open3d; print('All dependencies installed successfully!')"

Running the Person Tracker
==========================

The main script is `src/person_tracker.py`. Here are some basic usage examples:

**Basic Tracking (Outdoor Dataset)**:

   .. code:: bash

      python3 src/person_tracker.py --dataset outdoor

**Indoor Dataset with Debug Output**:

   .. code:: bash

      python3 src/person_tracker.py --dataset indoor --debug 1

**Using LiDAR Localization**:

   .. code:: bash

      python3 src/person_tracker.py --dataset outdoor --localization_method lidar

**Sensor Fusion (Recommended)**:

   .. code:: bash

      python3 src/person_tracker.py --dataset outdoor --localization_method fusion

**Performance Benchmarking**:

   .. code:: bash

      python3 src/person_tracker.py --dataset outdoor --benchmark

Understanding the Output
=========================  
When running the tracker, you'll see a live visualization window showing:

* **Yellow bounding boxes**: People detected wearing safety vests
* **Red bounding boxes**: People detected without safety vests
* **Track IDs**: Unique identifiers for each person
* **3D positions**: Real-time position data (when available)

Raw Data Output
---------------
The system generates a CSV file at `output/tracking_log_[dataset].csv` containing:

* `timestamp`: Frame timestamp
* `frame_id`: Sequential frame number
* `object_id`: Unique track identifier
* `x_position`: X coordinate in base frame (meters)
* `y_position`: Y coordinate in base frame (meters)
* `z_position`: Z coordinate in base frame (meters)

Example CSV Output:

   .. code:: text

      timestamp,frame_id,object_id,x_position,y_position,z_position
      1.234567890,1,1,2.45,-1.23,0.85
      1.234567891,2,1,2.46,-1.22,0.84
      1.234567892,3,2,3.12,0.45,1.02

Next Steps
==========
Now that you have the basic system running, you can:

1. **Explore Advanced Features**: See the :doc:`user_guide` for detailed configuration options
2. **Run Performance Tests**: Use the :doc:`testing_guide` to benchmark different configurations
3. **Understand the API**: Check the :doc:`api_reference` for detailed function documentation
4. **Troubleshoot Issues**: Refer to :doc:`troubleshooting` for common problems and solutions

Common Issues
=============
**No Video Window Appears**:
   - Ensure you have a display available (X11 forwarding for SSH)
   - Check that the video file exists and is readable

**LiDAR Localization Returns (0,0,0)**:
   - Verify LiDAR data has been processed using the conversion script
   - Check that `timestamps.txt` exists in the LiDAR output directory

**Poor Performance**:
   - Try using frame skipping: `--jump_frames 2`
   - Enable GPU acceleration: `--device cuda` or `--device mps`
   - Use faster tracking: `--tracker bytetrack`

For more detailed troubleshooting, see the :doc:`troubleshooting` guide.

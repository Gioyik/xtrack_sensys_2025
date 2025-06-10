# Ouster PCAP Conversion Tools

Two Python scripts are provided to convert Ouster LiDAR PCAP files into:

1. **PCD + CSV**: Extract each LiDAR scan (with XYZ, intensity, reflectivity, near‐IR) as `.pcd` and IMU packets as a CSV.
2. **ROS 2 Bag**: Package all LiDAR scans (as `sensor_msgs/PointCloud2`) and IMU measurements (as `sensor_msgs/Imu`) into a ROS 2 bag.

## Scripts

- `os_pcap_to_pcd_csv.py`  
  - Reads an Ouster PCAP, writes one ASCII PCD per scan (fields: x y z intensity reflectivity nearir), named by the midpoint of the first/last packet capture time.  
  - Writes IMU measurements (ax, ay, az, wx, wy, wz) to `<pcap_basename>_imu.csv`, with PCAP-level timestamps.

- `os_pcap_to_ros2bag.py`  
  - Reads an Ouster PCAP, publishes each scan as `sensor_msgs/PointCloud2` (fields: x y z intensity reflectivity nearir) and each IMU packet as `sensor_msgs/Imu`, stamped by PCAP capture times.  
  - Bundles all messages into a ROS 2 bag called `pcap_ros2bag` (SQLite3).  

## Requirements

Both scripts require:

- Python 3.8+  
- `ouster-sdk` (Ouster Python SDK)  
- `numpy`  
- `dpkt`  
- `open3d` (for PCD writing)  

The ROS 2 bag script additionally requires:

- A ROS 2 installation (Humble or later) with:  
  - `rclpy`  
  - `rosbag2_py`  
  - `sensor_msgs`  

### Install Python Packages

```bash
pip install ouster-sdk numpy dpkt open3d
```

Ensure you have a ROS 2 environment sourced before running the second script (e.g., `source /opt/ros/humble/setup.bash`).

## Usage

### 1. Convert PCAP → PCD + CSV

```bash
python3 os_pcap_to_pcd_csv.py /path/to/your.pcap
```

* This creates a folder named after the PCAP (e.g. `your/`), containing:

  * One `.pcd` file per LiDAR scan (fields: x y z intensity reflectivity nearir), named by the midpoint of that scan’s first/last packet PCAP timestamps.
  * A CSV called `your_imu.csv` with columns:

    ```
    pcap_ts_ns, ax, ay, az, wx, wy, wz
    ```

### 2. Convert PCAP → ROS 2 Bag

```bash
source /opt/ros/humble/setup.bash
python3 os_pcap_to_ros2bag.py /path/to/your.pcap
```

* Produces a ROS 2 bag named `pcap_ros2bag` (SQLite3) in the current directory, containing two topics:

  * `/pointcloud` (`sensor_msgs/PointCloud2`, fields: x y z intensity reflectivity nearir), timestamped at the midpoint of each scan’s first/last packet.
  * `/imu` (`sensor_msgs/Imu`), timestamped by each packet’s PCAP capture time.

You can playback and inspect the bag in any ROS 2 tool (e.g. `ros2 bag play pcap_ros2bag`).

---

**Example:**

```bash
# 1) Generate PCD + IMU CSV
python3 os_pcap_to_pcd_csv.py ouster_20250531141639_short.pcap

# 2) Generate ROS 2 bag (after sourcing ROS 2)
source /opt/ros/humble/setup.bash
python3 os_pcap_to_ros2bag.py ouster_20250531141639_short.pcap
```

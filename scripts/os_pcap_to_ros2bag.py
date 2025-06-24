import argparse
import struct

import dpkt
import rclpy
import rosbag2_py
from ouster.sdk import client, pcap
from rclpy.serialization import serialize_message
from sensor_msgs.msg import Imu, PointCloud2, PointField
from std_msgs.msg import Header


def create_pointcloud2(
    points, intensity, reflectivity, nearir, timestamp_ns, frame_id="lidar"
):
    """
    Build a sensor_msgs/PointCloud2 message containing fields:
      x y z intensity reflectivity nearir
    points: (N,3) float32
    intensity: (N,) uint16
    reflectivity: (N,) uint8
    nearir: (N,) uint16
    """
    N = points.shape[0]
    fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name="intensity", offset=12, datatype=PointField.UINT16, count=1),
        PointField(name="reflectivity", offset=14, datatype=PointField.UINT8, count=1),
        PointField(name="nearir", offset=15, datatype=PointField.UINT16, count=1),
    ]
    point_step = 17
    row_step = point_step * N

    buffer = bytearray(N * point_step)
    for i in range(N):
        x, y, z = points[i]
        it = int(intensity[i])
        rf = int(reflectivity[i])
        ni = int(nearir[i])
        struct.pack_into("<fffHBH", buffer, i * point_step, x, y, z, it, rf, ni)

    sec = int(timestamp_ns // 1_000_000_000)
    nanosec = int(timestamp_ns % 1_000_000_000)
    header = Header()
    header.stamp.sec = sec
    header.stamp.nanosec = nanosec
    header.frame_id = frame_id

    pc2 = PointCloud2()
    pc2.header = header
    pc2.height = 1
    pc2.width = N
    pc2.fields = fields
    pc2.is_bigendian = False
    pc2.point_step = point_step
    pc2.row_step = row_step
    pc2.is_dense = True
    pc2.data = bytes(buffer)
    return pc2


def create_imu_msg(ax, ay, az, wx, wy, wz, timestamp_ns, frame_id="imu"):
    sec = int(timestamp_ns // 1_000_000_000)
    nanosec = int(timestamp_ns % 1_000_000_000)
    header = Header()
    header.stamp.sec = sec
    header.stamp.nanosec = nanosec
    header.frame_id = frame_id

    imu = Imu()
    imu.header = header
    imu.orientation_covariance = [0.0] * 9
    imu.orientation.x = 0.0
    imu.orientation.y = 0.0
    imu.orientation.z = 0.0
    imu.orientation.w = 0.0
    imu.angular_velocity.x = wx
    imu.angular_velocity.y = wy
    imu.angular_velocity.z = wz
    imu.angular_velocity_covariance = [0.0] * 9
    imu.linear_acceleration.x = ax
    imu.linear_acceleration.y = ay
    imu.linear_acceleration.z = az
    imu.linear_acceleration_covariance = [0.0] * 9
    return imu


def main():
    parser = argparse.ArgumentParser(
        description="Convert PCAP → ROS 2 bag with PointCloud2 and Imu topics."
    )
    parser.add_argument("pcap_path", help="Path to input PCAP file")
    args = parser.parse_args()
    pcap_path = args.pcap_path

    rclpy.init()

    bag_name = "pcap_ros2bag"
    storage_opts = rosbag2_py._storage.StorageOptions(
        uri=bag_name, storage_id="sqlite3"
    )
    converter_opts = rosbag2_py._storage.ConverterOptions(
        input_serialization_format="cdr", output_serialization_format="cdr"
    )
    writer = rosbag2_py.SequentialWriter()
    writer.open(storage_opts, converter_opts)

    writer.create_topic(
        rosbag2_py._storage.TopicMetadata(
            name="pointcloud",
            type="sensor_msgs/msg/PointCloud2",
            serialization_format="cdr",
        )
    )
    writer.create_topic(
        rosbag2_py._storage.TopicMetadata(
            name="imu", type="sensor_msgs/msg/Imu", serialization_format="cdr"
        )
    )

    scan_source = pcap.PcapScanSource(pcap_path).single_source(0)
    metadata = scan_source.metadata
    xyz_lut = client.XYZLut(metadata)
    pkt_fmt = client.PacketFormat(metadata)

    cols_frame = metadata.format.columns_per_frame
    cols_packet = metadata.format.columns_per_packet
    packets_per_scan = cols_frame // cols_packet

    lidar_timestamps = []

    with open(pcap_path, "rb") as f_pcap:
        dpkt_reader = dpkt.pcap.Reader(f_pcap)
        ouster_reader = pcap.PcapMultiPacketReader(pcap_path).single_source(0)

        for (ts_sec, _), packet in zip(dpkt_reader, ouster_reader):
            pcap_ts_ns = int(ts_sec * 1e9)

            if isinstance(packet, client.LidarPacket):
                lidar_timestamps.append(pcap_ts_ns)

            elif isinstance(packet, client.ImuPacket):
                ax = pkt_fmt.imu_la_x(packet.buf)
                ay = pkt_fmt.imu_la_y(packet.buf)
                az = pkt_fmt.imu_la_z(packet.buf)
                wx = pkt_fmt.imu_av_x(packet.buf)
                wy = pkt_fmt.imu_av_y(packet.buf)
                wz = pkt_fmt.imu_av_z(packet.buf)

                imu_msg = create_imu_msg(ax, ay, az, wx, wy, wz, pcap_ts_ns)
                serialized_imu = serialize_message(imu_msg)
                writer.write("imu", serialized_imu, pcap_ts_ns)

    total_lidar_packets = len(lidar_timestamps)
    if total_lidar_packets < packets_per_scan:
        raise RuntimeError("Not enough LidarPackets for a single scan.")

    num_scans = total_lidar_packets // packets_per_scan

    scan_source = pcap.PcapScanSource(pcap_path).single_source(0)

    for idx, scan in enumerate(scan_source):
        if idx >= num_scans:
            break

        start_idx = idx * packets_per_scan
        last_idx = start_idx + packets_per_scan - 1
        first_ts = lidar_timestamps[start_idx]
        last_ts = lidar_timestamps[last_idx]
        mid_ts = (first_ts + last_ts) // 2

        xyz = xyz_lut(scan.field(client.ChanField.RANGE))
        points = xyz.reshape(-1, 3)

        refl = scan.field(client.ChanField.REFLECTIVITY).flatten()
        intensity = scan.field(client.ChanField.SIGNAL).flatten()
        nearir = scan.field(client.ChanField.NEAR_IR).flatten()

        pc2_msg = create_pointcloud2(points, intensity, refl, nearir, mid_ts)
        serialized_pc2 = serialize_message(pc2_msg)
        writer.write("pointcloud", serialized_pc2, mid_ts)

        print(f"[{idx}] Wrote PointCloud2 at timestamp {mid_ts}")

    print(f"Finished writing bag: {bag_name}")
    rclpy.shutdown()


if __name__ == "__main__":
    main()

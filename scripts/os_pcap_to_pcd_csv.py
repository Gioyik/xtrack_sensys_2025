import argparse
import csv
import os

import dpkt
from ouster.sdk import client, pcap

try:
    import open3d as o3d  # type: ignore
except ModuleNotFoundError:
    print("Open3D is required to write PCD files. Install via `pip install open3d`.")
    exit(1)


def write_pcd_with_extra_fields(path, points, intensity, reflectivity, nearir):
    """
    Write a PCD file in ASCII with fields:
      x y z intensity reflectivity nearir
    points: (N,3) float32
    intensity: (N,) uint16
    reflectivity: (N,) uint8
    nearir: (N,) uint16
    """
    N = points.shape[0]
    header = [
        "# .PCD v0.7 - Point Cloud Data file format",
        "VERSION 0.7",
        "FIELDS x y z intensity reflectivity nearir",
        "SIZE 4 4 4 2 1 2",
        "TYPE F F F U U U",
        "COUNT 1 1 1 1 1 1",
        f"WIDTH {N}",
        "HEIGHT 1",
        "VIEWPOINT 0 0 0 1 0 0 0",
        f"POINTS {N}",
        "DATA ascii",
    ]
    with open(path, "w") as f:
        f.write("\n".join(header) + "\n")
        for i in range(N):
            x, y, z = points[i]
            it = int(intensity[i])
            rf = int(reflectivity[i])
            ni = int(nearir[i])
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {it} {rf} {ni}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Convert every scan in a PCAP to PCD (named by midpoint of first/last packet capture time) "
        "and extract IMU packets (using pcap‐level timestamps) to CSV."
    )
    parser.add_argument("pcap_path", help="Path to input PCAP file")
    parser.add_argument("-o", "--out_dir", help="Path to output directory")
    args = parser.parse_args()
    pcap_path = args.pcap_path

    base_name = os.path.splitext(os.path.basename(pcap_path))[0]

    if args.out_dir:
        out_dir = args.out_dir
    else:
        out_dir = base_name
    os.makedirs(out_dir, exist_ok=True)

    # 1) Open scan source to grab metadata
    scan_source = pcap.PcapScanSource(pcap_path).single_source(0)
    metadata = scan_source.metadata
    xyz_lut = client.XYZLut(metadata)
    pkt_fmt = client.PacketFormat(metadata)

    columns_per_frame = metadata.format.columns_per_frame  # e.g., 2048
    columns_per_packet = metadata.format.columns_per_packet  # e.g., 16
    packets_per_scan = columns_per_frame // columns_per_packet  # e.g., 128

    # 2) First pass: iterate dpkt + Ouster to record pcap Ts of each LidarPacket,
    #    and write IMU CSV.
    lidar_timestamps = []
    imu_csv_path = os.path.join(out_dir, f"{base_name}_imu.csv")
    with open(imu_csv_path, "w", newline="") as csvfile:
        imu_writer = csv.writer(csvfile)
        imu_writer.writerow(["pcap_ts_ns", "ax", "ay", "az", "wx", "wy", "wz"])

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

                    imu_writer.writerow(
                        [
                            pcap_ts_ns,
                            float(ax),
                            float(ay),
                            float(az),
                            float(wx),
                            float(wy),
                            float(wz),
                        ]
                    )

    total_lidar_packets = len(lidar_timestamps)
    if total_lidar_packets < packets_per_scan:
        raise RuntimeError("Not enough LidarPackets for even one scan.")

    num_scans = total_lidar_packets // packets_per_scan

    # 3) Reset scan_source to iterate scans from the beginning
    scan_source = pcap.PcapScanSource(pcap_path).single_source(0)

    # 4) For each scan index, compute midpoint of first/last packet pcap timestamps
    for idx, scan in enumerate(scan_source):
        if idx >= num_scans:
            break

        start_idx = idx * packets_per_scan
        last_idx = start_idx + packets_per_scan - 1

        first_pkt_ts = lidar_timestamps[start_idx]
        last_pkt_ts = lidar_timestamps[last_idx]
        mid_ts = (first_pkt_ts + last_pkt_ts) // 2

        # 4a) Convert ranges → XYZ and flatten to (N,3)
        xyz = xyz_lut(scan.field(client.ChanField.RANGE))
        points = xyz.reshape(-1, 3)

        # 4b) Extract reflectivity, intensity (SIGNAL), nearir
        refl = scan.field(client.ChanField.REFLECTIVITY).flatten()
        intensity = scan.field(client.ChanField.SIGNAL).flatten()
        nearir = scan.field(client.ChanField.NEAR_IR).flatten()

        # 4c) Write PCD named by mid_ts
        out_path = os.path.join(out_dir, f"{mid_ts}.pcd")
        write_pcd_with_extra_fields(out_path, points, intensity, refl, nearir)
        print(f"[{idx}] Wrote PCD: {out_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
cameras_to_ros2bag.py

Scan a “camera” directory tree for every *_timestamps.txt, then
reconstruct ALL image messages (depth or color) and write them
into a single rosbag2 (SQLite3) under the original topic names.

Usage:
    python3 cameras_to_ros2bag.py <input_folder> <output_bag>

Example:
    python3 cameras_to_ros2bag.py /home/alice/camera /home/alice/camera_bag
"""

import os
import glob
import argparse

import rclpy
from rclpy.serialization import serialize_message
from rclpy.time import Time

from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import cv2

# Ensure you have rosbag2_py installed in your ROS 2 environment:
try:
    import rosbag2_py
except ImportError:
    raise ImportError(
        "rosbag2_py not found. Make sure you sourced your ROS 2 environment "
        "and installed rosbag2_py."
    )


def parse_timestamps_file(txt_path):
    """
    Parse a *_timestamps.txt file that has this format:
      # metadata
      topic_name /camera/image_raw
      encoding bgr8
      qos sensor_data
      # timestamps
      1618871230000000000
      1618871230333333333
      ...

    Returns:
      (topic_name:str, encoding:str, qos_label:str, timestamps: List[int])
    """
    topic_name = None
    encoding = None
    qos_label = None
    timestamps = []

    with open(txt_path, "r") as f:
        lines = f.readlines()

    in_timestamps = False
    for line in lines:
        line = line.strip()
        if not line:
            continue

        if line.startswith("#"):
            if line.strip() == "# timestamps":
                in_timestamps = True
            continue

        if not in_timestamps:
            parts = line.split()
            if len(parts) < 2:
                continue
            key = parts[0]
            if key == "topic_name":
                topic_name = parts[1]
            elif key == "encoding":
                encoding = parts[1]
            elif key == "qos":
                qos_label = parts[1]
        else:
            try:
                ts = int(line)
                timestamps.append(ts)
            except ValueError:
                raise ValueError(f"Invalid timestamp line in {txt_path}: '{line}'")

    if topic_name is None or encoding is None:
        raise RuntimeError(f"Metadata missing in {txt_path}")

    return topic_name, encoding, qos_label, timestamps


def find_matching_media(parent_dir):
    """
    Given the folder containing cam_…_timestamps.txt, decide:
      - If there is exactly one .avi, return ("video", <path_to_avi>)
      - Else if there is a subfolder ending with "_depth_frames", return ("depth", <path_to_folder>)
      - Otherwise raise.
    """
    # 1) look for ANY *.avi in parent_dir
    avi_list = glob.glob(os.path.join(parent_dir, "*.avi"))
    if len(avi_list) == 1:
        return "video", avi_list[0]
    elif len(avi_list) > 1:
        raise RuntimeError(
            f"Multiple .avi files in {parent_dir}, cannot decide which one to use: {avi_list}"
        )

    # 2) look for a subfolder ending with "_depth_frames"
    for d in os.listdir(parent_dir):
        full = os.path.join(parent_dir, d)
        if os.path.isdir(full) and d.endswith("_depth_frames"):
            return "depth", full

    raise RuntimeError(f"No .avi and no *_depth_frames folder in {parent_dir}")


def create_or_get_topic(writer, topics_map, topic_name):
    """
    Ensure we have created this topic in the bag already. If not,
    create it (with type="sensor_msgs/msg/Image" and empty QoS string).
    """
    if topic_name in topics_map:
        return

    # rosbag2_py.TopicMetadata signature (per your ROS 2) is:
    #    TopicMetadata(name: str, type: str, serialization_format: str, offered_qos_profiles: str = "")
    topic_meta = rosbag2_py.TopicMetadata(
        name=topic_name,
        type="sensor_msgs/msg/Image",
        serialization_format="cdr",
        offered_qos_profiles=""  # we leave QoS string empty
    )
    writer.create_topic(topic_meta)
    topics_map[topic_name] = topic_meta


def main():
    parser = argparse.ArgumentParser(
        description="Convert a folder of *_timestamps.txt + media → one rosbag2 (SQLite3)."
    )
    parser.add_argument(
        "input_folder",
        help="Path to the top-level camera folder (e.g. .../camera)."
    )
    parser.add_argument(
        "output_bag",
        help="Name/path for the output rosbag2 (no file extension). "
             "This will create a folder <output_bag>/ containing SQLite files."
    )
    args = parser.parse_args()

    input_folder = os.path.abspath(args.input_folder)
    output_bag = os.path.abspath(args.output_bag)

    # 1) Initialize rclpy (needed for Time, serialize_message, etc.)
    rclpy.init()

    # 2) Prepare rosbag2 writer
    writer = rosbag2_py.SequentialWriter()
    storage_opts = rosbag2_py.StorageOptions(uri=output_bag, storage_id="sqlite3")
    converter_opts = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr"
    )
    writer.open(storage_opts, converter_opts)

    # Keep track of which topic_names we've already created in the bag
    topics_map = {}

    bridge = CvBridge()

    # 3) Recursively find all *_timestamps.txt under input_folder
    for root, _, files in os.walk(input_folder):
        for fn in files:
            if not fn.endswith("_timestamps.txt"):
                continue

            timestamps_path = os.path.join(root, fn)
            parent_dir = os.path.dirname(timestamps_path)

            # Parse metadata + timestamps
            try:
                topic_name, encoding, qos_label, timestamps = parse_timestamps_file(timestamps_path)
            except Exception as e:
                print(f"[ERROR] Could not parse {timestamps_path}: {e}")
                continue

            # Decide if this is video vs depth
            try:
                media_type, media_path = find_matching_media(parent_dir)
            except RuntimeError as e:
                print(f"[WARNING] Skipping {timestamps_path}: {e}")
                continue

            # Make sure the topic is registered in the bag
            create_or_get_topic(writer, topics_map, topic_name)

            # Now loop over timestamps + frames
            if media_type == "video":
                cap = cv2.VideoCapture(media_path)
                if not cap.isOpened():
                    print(f"[ERROR] Could not open video {media_path}, skipping.")
                    continue

                idx = 0
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if len(timestamps) != total_frames:
                    print(
                        f"[WARNING] #timestamps ({len(timestamps)}) != #frames ({total_frames}) "
                        f"in video {media_path}. Using min()."
                    )

                n_valid = min(len(timestamps), total_frames)
                while idx < n_valid:
                    ret, frame = cap.read()
                    if not ret:
                        print(f"[ERROR] Failed to read frame {idx} from {media_path}.")
                        break

                    # Build Image message using the encoding from timestamps file
                    img_msg = bridge.cv2_to_imgmsg(frame, encoding=encoding)
                    nanosec = timestamps[idx]
                    img_msg.header.stamp = Time(nanoseconds=nanosec).to_msg()

                    serialized = serialize_message(img_msg)
                    writer.write(topic_name, serialized, nanosec)
                    idx += 1

                cap.release()

            else:  # media_type == "depth"
                png_folder = media_path
                all_pngs = sorted(glob.glob(os.path.join(png_folder, "*.png")))
                if len(all_pngs) != len(timestamps):
                    print(
                        f"[WARNING] #timestamps ({len(timestamps)}) != #PNGs ({len(all_pngs)}) "
                        f"in folder {png_folder}. Using min()."
                    )
                n_valid = min(len(timestamps), len(all_pngs))
                for idx in range(n_valid):
                    png_file = all_pngs[idx]
                    frame = cv2.imread(png_file, cv2.IMREAD_UNCHANGED)
                    if frame is None:
                        print(f"[ERROR] Could not read depth PNG {png_file}.")
                        continue

                    img_msg = bridge.cv2_to_imgmsg(frame, encoding=encoding)
                    nanosec = timestamps[idx]
                    img_msg.header.stamp = Time(nanoseconds=nanosec).to_msg()

                    serialized = serialize_message(img_msg)
                    writer.write(topic_name, serialized, nanosec)

            print(f"[OK] Wrote {media_type} → topic '{topic_name}' ({len(timestamps)} messages).")

    # 4) Close bag & shutdown
    print(f"[DONE] All done. Bag written to: {output_bag}")
    rclpy.shutdown()


if __name__ == "__main__":
    main()

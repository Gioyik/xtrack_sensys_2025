#!/usr/bin/env python3
import os
import sys
import rclpy
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import rosbag2_py

def extract_header_stamp(serialized_msg, msg_type):
    """
    Deserialize a serialized ROS message (serialized_msg) of type msg_type,
    and return its header.stamp in nanoseconds (sec * 1e9 + nanosec).
    If deserialization fails, return None.
    """
    try:
        cls = get_message(msg_type)
        msg = deserialize_message(serialized_msg, cls)
        ts = msg.header.stamp
        return ts.sec * 1_000_000_000 + ts.nanosec
    except Exception:
        return None

def infer_storage_id(ext_token):
    """
    Given an extension token (".db3"/"db3" or ".mcap"/"mcap"),
    return the corresponding storage_id ("sqlite3" or "mcap").
    Raise ValueError otherwise.
    """
    t = ext_token.lower().lstrip(".")
    if t == "db3":
        return "sqlite3"
    if t == "mcap":
        return "mcap"
    raise ValueError(f"Unknown format '{ext_token}'. Use .db3 or .mcap.")

def find_bag_start_end(bag_path, converter_options):
    """
    Open bag_path, iterate all messages, extract header.stamp, and return a tuple:
      (earliest_stamp, latest_stamp) in nanoseconds.
    If no valid stamps found, raise RuntimeError.
    """
    storage_id = infer_storage_id(os.path.splitext(bag_path)[1])
    reader = rosbag2_py.SequentialReader()
    storage_opts = rosbag2_py.StorageOptions(uri=bag_path, storage_id=storage_id)
    reader.open(storage_opts, converter_options)

    # Build a local topic → type map
    local_topic_types = {}
    for info in reader.get_all_topics_and_types():
        local_topic_types[info.name] = info.type

    earliest = None
    latest = None
    while reader.has_next():
        topic, data, _ = reader.read_next()
        msg_type = local_topic_types[topic]
        raw_ts = extract_header_stamp(data, msg_type)
        if raw_ts is not None:
            if earliest is None or raw_ts < earliest:
                earliest = raw_ts
            if latest is None or raw_ts > latest:
                latest = raw_ts

    del reader

    if earliest is None or latest is None:
        raise RuntimeError(f"No valid header.stamp found in '{bag_path}'.")
    return earliest, latest

def merge_bags_start_at_latest_limited(input_bags, storage_id, output_uri):
    """
    Merge multiple ROS 2 bags into a single bag named output_uri,
    but only include messages whose header.stamp satisfies:
      common_start ≤ stamp ≤ common_start + max_dur
    where
      common_start = max(all bag_earliest)
      max_dur      = min(all bag_durations = bag_latest - bag_earliest)

    Each retained message is then written with its original header.stamp (UTC nanoseconds).
    """
    rclpy.init()

    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr",
    )

    # 1) Compute each bag's earliest & latest stamps
    bag_stamps = {}  # bag_path → (earliest, latest)
    for bag_path in input_bags:
        if not os.path.exists(bag_path):
            raise FileNotFoundError(f"Cannot find input bag: {bag_path}")

    for bag_path in input_bags:
        start, end = find_bag_start_end(bag_path, converter_options)
        bag_stamps[bag_path] = (start, end)

    # 2) Compute common_start = max(earliest_i) and durations
    earliest_list = [start for (start, end) in bag_stamps.values()]
    latest_list   = [end   for (start, end) in bag_stamps.values()]
    common_start  = max(earliest_list)
    durations     = [end - start for (start, end) in bag_stamps.values()]
    max_dur       = min(durations)

    # 3) Collect all messages that lie within [common_start, common_start + max_dur]
    cutoff = common_start + max_dur
    all_msgs   = []      # list of (orig_ts, topic_name, serialized_data)
    topic_types = {}     # global map: topic_name → msg_type

    for bag_path in input_bags:
        storage_id_in = infer_storage_id(os.path.splitext(bag_path)[1])
        reader = rosbag2_py.SequentialReader()
        storage_opts = rosbag2_py.StorageOptions(uri=bag_path, storage_id=storage_id_in)
        reader.open(storage_opts, converter_options)

        local_map = {}
        for info in reader.get_all_topics_and_types():
            local_map[info.name] = info.type
            if info.name not in topic_types:
                topic_types[info.name] = info.type

        while reader.has_next():
            topic, data, _ = reader.read_next()
            msg_type = local_map[topic]
            raw_ts = extract_header_stamp(data, msg_type)
            if raw_ts is None:
                continue
            if raw_ts < common_start:
                continue
            if raw_ts > cutoff:
                continue
            all_msgs.append((raw_ts, topic, data))

        del reader

    # 4) Sort by original timestamp
    all_msgs.sort(key=lambda x: x[0])

    # 5) Open the output bag (URI "merged_bag") for writing
    writer = rosbag2_py.SequentialWriter()
    out_opts = rosbag2_py.StorageOptions(uri=output_uri, storage_id=storage_id)
    writer.open(out_opts, converter_options)

    # 6) Create each topic exactly once
    created = set()
    for topic, msg_type in topic_types.items():
        if topic not in created:
            tm = rosbag2_py.TopicMetadata(
                name=topic,
                type=msg_type,
                serialization_format="cdr",
                offered_qos_profiles="",
            )
            writer.create_topic(tm)
            created.add(topic)

    # 7) Write all filtered messages with their original stamps
    for orig_ts, topic, data in all_msgs:
        writer.write(topic, data, orig_ts)

    del writer
    rclpy.shutdown()

if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        print(
            "Usage: merge_bags_by_header_stamp.py <bag1> [<bag2> ...] [extension]\n"
            "  <bagX>      : path to an existing .mcap or .db3 bag\n"
            "  [extension] : optional; '.db3' or '.mcap' to choose output format\n\n"
            "Examples:\n"
            "  python3 merge_bags_by_header_stamp.py bag1.mcap bag2.db3\n"
            "     → merges between common_start and common_start + min(duration),\n"
            "       writes directory 'merged_bag/' (sqlite3).\n"
            "  python3 merge_bags_by_header_stamp.py bag1.mcap bag2.db3 .mcap\n"
            "     → same, but writes single file 'merged_bag' (mcap).\n"
        )
        sys.exit(1)

    # Check if last argument is an extension
    last = args[-1]
    try:
        storage_id = infer_storage_id(last)
        input_bags = args[:-1]
    except ValueError:
        # Default to sqlite3 if not specified
        storage_id = "sqlite3"
        input_bags = args

    if not input_bags:
        print("Error: no input bags provided.")
        sys.exit(1)

    # Always write to URI "merged_bag" (no suffix on disk)
    output_uri = "merged_bag"

    try:
        merge_bags_start_at_latest_limited(input_bags, storage_id, output_uri)
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    print(f"[DONE] Merged {len(input_bags)} bag(s) → '{output_uri}'")

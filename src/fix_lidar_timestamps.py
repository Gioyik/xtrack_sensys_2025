import argparse
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lidar_path",
        type=str,
        help="Path to the lidar data",
    )
    args = parser.parse_args()
    lidar_path = args.lidar_path
    if not lidar_path.endswith(".txt"):
        lidar_path = os.path.join(lidar_path, "timestamps.txt")
    with open(lidar_path, "r") as f:
        timestamps = f.read().splitlines()
    fixed_timestamps = []
    for timestamp in timestamps:
        timestamp = float(timestamp)
        if timestamp < 1e17:
            timestamp *= 1e9
        fixed_timestamps.append(str(timestamp))
    with open(lidar_path, "w") as f:
        f.write("\n".join(fixed_timestamps))


if __name__ == "__main__":
    main()

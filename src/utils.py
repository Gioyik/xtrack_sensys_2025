import argparse

import pandas as pd


def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="xTrack Person Tracker")
    parser.add_argument(
        "--dataset",
        type=str,
        default="outdoor",
        choices=["indoor", "outdoor"],
        help="Dataset to process",
    )
    parser.add_argument(
        "--tracker",
        type=str,
        default="bytetrack",
        choices=["bytetrack", "botsort"],
        help="Tracker to use",
    )
    parser.add_argument(
        "--debug", type=int, default=0, choices=[0, 1, 2], help="Debug level"
    )
    parser.add_argument(
        "--reid_method",
        type=str,
        default="custom",
        choices=["custom", "botsort"],
        help="Re-ID method to use",
    )
    parser.add_argument(
        "--vest_detection",
        type=str,
        default="color",
        choices=["color", "model"],
        help="Vest detection method to use.",
    )
    parser.add_argument(
        "--vest_model_path",
        type=str,
        default="vest_model.pth",
        help="Path to the vest detection model file.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run the models on.",
    )
    parser.add_argument(
        "--localization_method",
        type=str,
        default="depth",
        choices=["depth", "lidar"],
        help="Localization method to use.",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Enable performance benchmarking.",
    )
    return parser.parse_args()


def load_timestamps(path, scale_factor=1.0):
    """Loads timestamps from a text file."""
    timestamps = []
    try:
        with open(path, "r") as f:
            for line in f:
                try:
                    timestamps.append(float(line.strip()) * scale_factor)
                except ValueError:
                    continue
    except FileNotFoundError:
        print(f"Warning: Timestamp file not found at {path}")
    return timestamps


def setup_logging(output_csv_path):
    """Initializes the CSV log file with a header."""
    with open(output_csv_path, "w", newline="") as f:
        pd.DataFrame(
            columns=[
                "timestamp",
                "frame_id",
                "object_id",
                "x_position",
                "y_position",
                "z_position",
            ]
        ).to_csv(f, index=False)


def log_result(output_csv_path, timestamp, frame_id, display_id, x, y, z):
    """Logs a tracking result to the CSV file."""
    log_entry = pd.DataFrame(
        [[timestamp, frame_id, display_id, x, y, z]],
        columns=[
            "timestamp",
            "frame_id",
            "object_id",
            "x_position",
            "y_position",
            "z_position",
        ],
    )
    with open(output_csv_path, "a", newline="") as f:
        log_entry.to_csv(f, header=False, index=False)

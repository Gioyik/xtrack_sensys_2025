import argparse

import pandas as pd


def parse_arguments():
    """
    Parse command-line arguments for the xTrack person tracker.
    
    Returns:
        argparse.Namespace: Parsed command-line arguments
        
    Arguments:
        Dataset Selection:
            --dataset: Dataset to process ("indoor" | "outdoor", default: "outdoor")
            
        Tracking Configuration:
            --tracker: Tracker algorithm ("bytetrack" | "botsort", default: "bytetrack")
            --reid_method: Re-ID method ("custom" | "botsort", default: "custom")
            
        Vest Detection:
            --vest_detection: Vest detection method ("color" | "model", default: "color")
            --vest_model_path: Path to vest model file (default: "vest_model.pth")
            --vest_threshold: Yellow percentage threshold (default: 5.0, range: 0.1-20.0)
            --vest_persistence: Consecutive frames for vest confirmation (default: 1, range: 1-10)
            
        3D Localization:
            --localization_method: Localization method ("depth" | "lidar" | "fusion", default: "depth")
            
        Performance:
            --device: Compute device ("cpu" | "cuda" | "mps", default: "cpu")
            --jump_frames: Frames to skip (default: 0)
            --benchmark: Enable performance benchmarking (flag)
            
        Re-ID Tuning:
            --reid_threshold: ReID similarity threshold (default: 0.75, range: 0.5-0.95)
            --max_lost_frames: Max frames to remember lost tracks (default: 90, range: 30-300)
            
        Debugging:
            --debug: Debug level (0 | 1 | 2, default: 0)
    """
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
        choices=["cpu", "cuda", "mps"],
        help="Device to run the models on (cpu, cuda for NVIDIA GPUs, mps for Apple Silicon).",
    )
    parser.add_argument(
        "--localization_method",
        type=str,
        default="depth",
        choices=["depth", "lidar", "fusion"],
        help="Localization method to use (depth: RGB-D only, lidar: LiDAR only, fusion: combine both sensors).",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Enable performance benchmarking.",
    )
    parser.add_argument(
        "--jump_frames",
        type=int,
        default=0,
        help="Number of frames to skip between processed frames (0 = no skipping).",
    )
    parser.add_argument(
        "--vest_threshold",
        type=float,
        default=5.0,
        help="Yellow percentage threshold for vest detection (default: 5.0%).",
    )
    parser.add_argument(
        "--vest_persistence",
        type=int,
        default=1,
        help="Number of consecutive frames vest must be detected before confirming (default: 1).",
    )
    parser.add_argument(
        "--reid_threshold",
        type=float,
        default=0.75,
        help="ReID similarity threshold for re-identifying lost tracks (default: 0.75, range: 0.5-0.95).",
    )
    parser.add_argument(
        "--max_lost_frames",
        type=int,
        default=90,
        help="Maximum frames a track can be lost before being permanently deleted (default: 90).",
    )
    return parser.parse_args()


def load_timestamps(path, scale_factor=1.0):
    """
    Load timestamps from a text file.
    
    Args:
        path (pathlib.Path): Path to timestamp file
        scale_factor (float): Scale factor to apply to timestamps (default: 1.0)
        
    Returns:
        list: List of timestamps as floats
        
    Process:
        1. Open file and read line by line
        2. Convert each line to float
        3. Apply scale factor
        4. Skip invalid lines
        5. Return list of valid timestamps
        
    Error Handling:
        - FileNotFoundError: Returns empty list with warning
        - ValueError: Skips invalid lines
        - Continues processing valid timestamps
        
    Scale Factor Usage:
        - 1.0: Timestamps in seconds
        - 1e-9: Convert nanoseconds to seconds
        - 1e-6: Convert microseconds to seconds
    """
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

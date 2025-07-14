from pathlib import Path

# --- Project Directory ---
PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "output"


# --- Re-ID Parameters ---
REID_SIMILARITY_THRESHOLD = 0.4
MAX_LOST_FRAMES = 90


# --- Dataset Paths ---
def get_dataset_paths(dataset_name):
    if dataset_name == "indoor":
        return {
            "video_path": DATA_DIR
            / "data_indoor/camera/d435i/color/cam_2025_06_04_09_24_42.avi",
            "video_timestamps_path": DATA_DIR
            / "data_indoor/camera/d435i/color/cam_2025_06_04_09_24_42_timestamps.txt",
            "depth_folder_path": DATA_DIR
            / "data_indoor/camera/d435i/depth/cam_2025_06_04_09_24_42_depth_frames/",
            "depth_timestamps_path": DATA_DIR
            / "data_indoor/camera/d435i/depth/cam_2025_06_04_09_24_42_timestamps.txt",
        }
    else:  # outdoor
        return {
            "video_path": DATA_DIR
            / "data_outdoor/camera/d435i/color/cam_2025_06_04_09_41_51.avi",
            "video_timestamps_path": DATA_DIR
            / "data_outdoor/camera/d435i/color/cam_2025_06_04_09_41_51_timestamps.txt",
            "depth_folder_path": DATA_DIR
            / "data_outdoor/camera/d435i/depth/cam_2025_06_04_09_41_51_depth_frames/",
            "depth_timestamps_path": DATA_DIR
            / "data_outdoor/camera/d435i/depth/cam_2025_06_04_09_41_51_timestamps.txt",
            "lidar_folder_path": OUTPUT_DIR / "ouster_20250604074152",
            "lidar_timestamps_path": OUTPUT_DIR
            / "ouster_20250604074152/timestamps.txt",
        }

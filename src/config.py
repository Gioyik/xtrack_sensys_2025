from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "output"

REID_SIMILARITY_THRESHOLD = 0.4
MAX_LOST_FRAMES = 90

def get_dataset_paths(dataset_name):
    """
    Get file paths for a specific dataset.
    
    Args:
        dataset_name (str): Dataset name ("indoor" or "outdoor")
        
    Returns:
        dict: Dictionary containing paths to dataset files
        
    Returned Dictionary Keys:
        - video_path: Path to RGB video file
        - video_timestamps_path: Path to video timestamps file
        - depth_folder_path: Path to depth frames directory
        - depth_timestamps_path: Path to depth timestamps file
        - lidar_folder_path: Path to LiDAR .pcd files (outdoor only)
        - lidar_timestamps_path: Path to LiDAR timestamps file (outdoor only)
        
    Dataset Structure:
        Indoor Dataset:
            - Video: data/data_indoor/camera/d435i/color/cam_2025_06_04_09_24_42.avi
            - Video timestamps: data/data_indoor/camera/d435i/color/cam_2025_06_04_09_24_42_timestamps.txt
            - Depth frames: data/data_indoor/camera/d435i/depth/cam_2025_06_04_09_24_42_depth_frames/
            - Depth timestamps: data/data_indoor/camera/d435i/depth/cam_2025_06_04_09_24_42_timestamps.txt
            
        Outdoor Dataset:
            - Video: data/data_outdoor/camera/d435i/color/cam_2025_06_04_09_41_51.avi
            - Video timestamps: data/data_outdoor/camera/d435i/color/cam_2025_06_04_09_41_51_timestamps.txt
            - Depth frames: data/data_outdoor/camera/d435i/depth/cam_2025_06_04_09_41_51_depth_frames/
            - Depth timestamps: data/data_outdoor/camera/d435i/depth/cam_2025_06_04_09_41_51_timestamps.txt
            - LiDAR files: output/ouster_20250604074152/
            - LiDAR timestamps: output/ouster_20250604074152/timestamps.txt
    """
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

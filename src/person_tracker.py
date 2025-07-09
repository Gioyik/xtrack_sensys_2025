import argparse
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

from coordinate_frames import K_rgb, T_base_rgb_camera, T_camera_lidar
from lidar_utils import (
    find_closest_lidar_file,
    get_3d_position_from_lidar,
    load_point_cloud,
)
from localization import get_3d_position, get_closest_depth_frame
from reid import cosine_similarity, get_appearance_embedding
from vest_classifier import VestClassifier
from vision import detect_yellow_vest


def main():
    # --- Argument Parsing ---
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
    args = parser.parse_args()

    if args.reid_method == "botsort" and args.tracker != "botsort":
        print(
            "Error: BoTSORT Re-ID can only be used with the BoTSORT tracker. Please set --tracker to 'botsort'."
        )
        return

    # --- Configuration ---
    # REID_SIMILARITY_THRESHOLD = 0.80  # Adjusted for better matching
    REID_SIMILARITY_THRESHOLD = 0.4  # Adjusted for better matching
    MAX_LOST_FRAMES = 90  # Shorter buffer for lost tracks

    # --- Path Setup ---
    current_file = Path(__file__).resolve()
    project_dir = current_file.parents[1]
    data_dir = project_dir / "data"
    if args.dataset == "indoor":
        video_path = (
            data_dir / "data_indoor/camera/d435i/color/cam_2025_06_04_09_24_42.avi"
        )
        video_timestamps_path = (
            data_dir
            / "data_indoor/camera/d435i/color/cam_2025_06_04_09_24_42_timestamps.txt"
        )
        depth_folder_path = (
            data_dir
            / "data_indoor/camera/d435i/depth/cam_2025_06_04_09_24_42_depth_frames/"
        )
        depth_timestamps_path = (
            data_dir
            / "data_indoor/camera/d435i/depth/cam_2025_06_04_09_24_42_timestamps.txt"
        )
    else:  # outdoor
        video_path = (
            data_dir / "data_outdoor/camera/d435i/color/cam_2025_06_04_09_41_51.avi"
        )
        video_timestamps_path = (
            data_dir
            / "data_outdoor/camera/d435i/color/cam_2025_06_04_09_41_51_timestamps.txt"
        )
        depth_folder_path = (
            data_dir
            / "data_outdoor/camera/d435i/depth/cam_2025_06_04_09_41_51_depth_frames/"
        )
        depth_timestamps_path = (
            data_dir
            / "data_outdoor/camera/d435i/depth/cam_2025_06_04_09_41_51_timestamps.txt"
        )
        lidar_folder_path = (
            project_dir / "output/ouster_20250604074152"
        )  # Corrected path
        lidar_timestamps_path = lidar_folder_path / "timestamps.txt"

    output_csv_path = project_dir / f"output/tracking_log_{args.dataset}.csv"

    # --- Load Timestamps ---
    def load_timestamps(path, scale_factor=1.0):
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

    video_timestamps = load_timestamps(video_timestamps_path)
    depth_timestamps = load_timestamps(depth_timestamps_path)
    if args.localization_method == "lidar":
        # LiDAR timestamps are in nanoseconds, convert to seconds
        lidar_timestamps = load_timestamps(lidar_timestamps_path, scale_factor=1e-9)
        # The pcap conversion script names files by timestamp, so we can find them
        # without a simple glob. We will construct the path from the timestamp.
        # This is a placeholder for that logic, as find_closest_lidar_file
        # will handle the matching.
        lidar_pcd_files = list(lidar_folder_path.glob("*.pcd"))
        if not lidar_pcd_files:
            print(
                f"Warning: No LiDAR .pcd files found in {lidar_folder_path}. LiDAR localization will not work."
            )

    # --- Video Processing ---
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    # --- Model and Classifier Loading ---
    print("Loading models...")
    model = YOLO("yolo11n.pt")
    model.to(args.device)

    vest_classifier = None
    if args.vest_detection == "model":
        vest_classifier = VestClassifier(
            model_path=args.vest_model_path, device=args.device
        )
    print("Models loaded.")

    # --- Setup CSV Logging ---
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

    # --- Re-ID Data Structures (for custom re-ID) ---
    track_embeddings = {}
    lost_tracks = {}
    id_links = {}
    next_person_id = 1
    previous_track_ids = set()

    # --- Performance Benchmarking Setup ---
    if args.benchmark:
        perf_data = {
            "total_frame_time": deque(maxlen=100),
            "yolo_time": deque(maxlen=100),
            "reid_time": deque(maxlen=100),
            "vest_depth_time": deque(maxlen=100),
        }

    frame_id = 0
    while cap.isOpened():
        if frame_id >= len(video_timestamps):
            break

        start_time = time.time()

        ret, frame = cap.read()
        if not ret:
            break

        timestamp = video_timestamps[frame_id]

        # --- Run Ultralytics Tracker ---
        yolo_start = time.time()
        tracker_config_path = project_dir / f"src/trackers/{args.tracker}.yaml"
        results = model.track(
            frame, persist=True, classes=[0], conf=0.5, tracker=str(tracker_config_path)
        )
        if args.benchmark:
            perf_data["yolo_time"].append(time.time() - yolo_start)

        current_ultralytics_ids = set()
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            ultralytics_ids = results[0].boxes.id.cpu().numpy().astype(int)

            # --- Custom Re-ID Logic ---
            reid_start = time.time()
            if args.reid_method == "custom":
                current_ultralytics_ids = set(ultralytics_ids)

                # Update embeddings for active tracks
                for box, track_id in zip(boxes, ultralytics_ids):
                    person_image = frame[box[1] : box[3], box[0] : box[2]]
                    embedding = get_appearance_embedding(person_image)
                    if embedding is not None:
                        track_embeddings[track_id] = embedding

                # Identify and handle lost tracks
                lost_ids = previous_track_ids - current_ultralytics_ids
                for track_id in lost_ids:
                    original_id = id_links.get(track_id, track_id)
                    if track_embeddings.get(track_id) is not None:
                        lost_tracks[original_id] = {
                            "embedding": track_embeddings[track_id],
                            "last_seen": frame_id,
                        }
                    if track_id in track_embeddings:
                        del track_embeddings[track_id]

                # Process all current tracks
                processed_ids = []
                for box, ultralytics_id in zip(boxes, ultralytics_ids):
                    display_id = id_links.get(ultralytics_id)

                    if display_id is None:  # New or re-identified track
                        embedding = track_embeddings.get(ultralytics_id)
                        if embedding is not None:
                            best_match_id, best_similarity = -1, -1
                            for lost_id, data in list(lost_tracks.items()):
                                if frame_id - data["last_seen"] > MAX_LOST_FRAMES:
                                    del lost_tracks[lost_id]
                                    continue
                                similarity = cosine_similarity(
                                    embedding, data["embedding"]
                                )
                                if similarity > best_similarity:
                                    best_similarity, best_match_id = similarity, lost_id

                            if best_similarity > REID_SIMILARITY_THRESHOLD:
                                display_id = best_match_id
                                id_links[ultralytics_id] = display_id
                                if args.debug >= 1:
                                    print(
                                        f"Re-identified track {ultralytics_id} as {display_id} with similarity {best_similarity:.2f}"
                                    )
                                if best_match_id in lost_tracks:
                                    del lost_tracks[best_match_id]
                            else:
                                display_id = next_person_id
                                id_links[ultralytics_id] = display_id
                                next_person_id += 1

                    processed_ids.append((box, display_id))
                previous_track_ids = current_ultralytics_ids
            else:  # botsort re-id
                processed_ids = list(zip(boxes, ultralytics_ids))
            if args.benchmark:
                perf_data["reid_time"].append(time.time() - reid_start)

            # --- Main Processing Loop ---
            vest_depth_start = time.time()
            for box, display_id in processed_ids:
                if display_id is None:
                    continue

                # --- Depth and Vest Detection ---
                x1, y1, x2, y2 = box
                person_image = frame[y1:y2, x1:x2]

                if person_image.size > 0:
                    if args.vest_detection == "model":
                        is_vest, confidence = vest_classifier.predict(person_image)
                        if args.debug >= 2:
                            print(
                                f"Frame {frame_id}, Track ID {display_id}: Vest Confidence = {confidence:.2f}"
                            )
                    else:  # Original color-based detection
                        is_vest, vest_mask, yellow_percentage = detect_yellow_vest(
                            person_image
                        )
                        if args.debug >= 2:
                            cv2.imshow(f"Vest Mask ID: {display_id}", vest_mask)
                            print(
                                f"Frame {frame_id}, Track ID {display_id}: Yellow Percentage = {yellow_percentage:.2f}%"
                            )

                    if is_vest:
                        x, y, z = 0, 0, 0
                        if args.localization_method == "lidar":
                            lidar_file = find_closest_lidar_file(
                                timestamp, lidar_timestamps, lidar_pcd_files
                            )
                            point_cloud = load_point_cloud(lidar_file)
                            if point_cloud is not None:
                                x_cam, y_cam, z_cam = get_3d_position_from_lidar(
                                    box, point_cloud, K_rgb, T_camera_lidar
                                )
                                pos_camera_frame = np.array([x_cam, y_cam, z_cam, 1])
                                pos_base_frame = T_base_rgb_camera @ pos_camera_frame
                                x, y, z = pos_base_frame[:3]
                        else:  # Default to depth camera
                            depth_frame_path = get_closest_depth_frame(
                                timestamp, depth_timestamps, depth_folder_path
                            )
                            if depth_frame_path:
                                depth_image = cv2.imread(
                                    str(depth_frame_path), cv2.IMREAD_UNCHANGED
                                )
                                if depth_image is not None:
                                    x_cam, y_cam, z_cam = get_3d_position(
                                        box, depth_image, K_rgb
                                    )
                                    pos_camera_frame = np.array(
                                        [x_cam, y_cam, z_cam, 1]
                                    )
                                    pos_base_frame = (
                                        T_base_rgb_camera @ pos_camera_frame
                                    )
                                    x, y, z = pos_base_frame[:3]

                        if x != 0 or y != 0 or z != 0:
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

                            cv2.rectangle(
                                    frame, (x1, y1), (x2, y2), (0, 255, 255), 2
                                )
                            cv2.putText(
                                frame,
                                f"ID: {display_id} (Vest)",
                                (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0, 255, 255),
                                2,
                            )
                            cv2.putText(
                                frame,
                                f"Pos: ({x:.2f}, {y:.2f}, {z:.2f})m",
                                (x1, y2 + 20),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0, 255, 255),
                                2,
                            )
                    else:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
                        cv2.putText(
                            frame,
                            f"ID: {display_id}",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 0, 255),
                            1,
                        )
            if args.benchmark:
                perf_data["vest_depth_time"].append(time.time() - vest_depth_start)

        if args.benchmark:
            perf_data["total_frame_time"].append(time.time() - start_time)

        cv2.imshow("xTrack Person Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        frame_id += 1

    cap.release()
    cv2.destroyAllWindows()
    print(f"Tracking data saved to {output_csv_path}")

    # --- Print Benchmark Results ---
    if args.benchmark:
        print("\n--- Performance Benchmark Results ---")
        total_time = sum(perf_data["total_frame_time"])
        avg_fps = len(perf_data["total_frame_time"]) / total_time
        print(f"Average FPS: {avg_fps:.2f}")

        for name, data in perf_data.items():
            if len(data) > 0:
                avg_time = sum(data) / len(data)
                print(f"Average {name.replace('_', ' ')}: {avg_time * 1000:.2f} ms")
        print("------------------------------------")


if __name__ == "__main__":
    main()

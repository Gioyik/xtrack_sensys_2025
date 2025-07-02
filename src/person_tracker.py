import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

from coordinate_frames import K_rgb, T_base_rgb_camera
from localization import get_3d_position, get_closest_depth_frame
from reid import cosine_similarity, get_appearance_embedding
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

    # Load the YOLO model
    model = YOLO("yolo11n.pt")

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

    output_csv_path = project_dir / f"output/tracking_log_{args.dataset}.csv"

    # --- Load Timestamps ---
    def load_timestamps(path):
        timestamps = []
        with open(path, "r") as f:
            for line in f:
                try:
                    timestamps.append(float(line.strip()))
                except ValueError:
                    continue
        return timestamps

    video_timestamps = load_timestamps(video_timestamps_path)
    depth_timestamps = load_timestamps(depth_timestamps_path)

    # --- Video Processing ---
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

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

    frame_id = 0
    while cap.isOpened():
        if frame_id >= len(video_timestamps):
            break
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = video_timestamps[frame_id]

        # --- Run Ultralytics Tracker ---
        tracker_config_path = project_dir / f"src/trackers/{args.tracker}.yaml"
        results = model.track(
            frame, persist=True, classes=[0], conf=0.5, tracker=str(tracker_config_path)
        )

        current_ultralytics_ids = set()
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            ultralytics_ids = results[0].boxes.id.cpu().numpy().astype(int)

            # --- Custom Re-ID Logic ---
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

            # --- Main Processing Loop ---
            for box, display_id in processed_ids:
                if display_id is None:
                    continue

                # --- Depth and Vest Detection ---
                x1, y1, x2, y2 = box
                person_image = frame[y1:y2, x1:x2]

                if person_image.size > 0:
                    is_vest, vest_mask, yellow_percentage = detect_yellow_vest(
                        person_image
                    )

                    if args.debug >= 2:
                        cv2.imshow(f"Vest Mask ID: {display_id}", vest_mask)
                        print(
                            f"Frame {frame_id}, Track ID {display_id}: Yellow Percentage = {yellow_percentage:.2f}%"
                        )

                    if is_vest:
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
                                pos_camera_frame = np.array([x_cam, y_cam, z_cam, 1])
                                pos_base_frame = T_base_rgb_camera @ pos_camera_frame
                                x, y, z = pos_base_frame[:3]

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

        cv2.imshow("xTrack Person Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        frame_id += 1

    cap.release()
    cv2.destroyAllWindows()
    print(f"Tracking data saved to {output_csv_path}")


if __name__ == "__main__":
    main()

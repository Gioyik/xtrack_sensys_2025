import cv2
from ultralytics import YOLO
import pandas as pd
import numpy as np
import os
import argparse

from coordinate_frames import K_rgb, T_base_rgb_camera
from localization import get_closest_depth_frame, get_3d_position
from vision import detect_yellow_vest

def main():
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description='xTrack Person Tracker')
    parser.add_argument('--dataset', type=str, default='outdoor', choices=['indoor', 'outdoor'],
                        help='Dataset to process (indoor or outdoor)')
    parser.add_argument('--tracker', type=str, default='bytetrack', choices=['bytetrack', 'botsort'],
                        help='Tracker to use (bytetrack or botsort)')
    parser.add_argument('--debug', type=int, default=0, choices=[0, 1, 2],
                        help='Debug level: 0=None, 1=Basic, 2=Full Visualization')
    args = parser.parse_args()

    # Load the YOLOv11 model
    model = YOLO('yolo11n.pt')

    # --- Configuration ---
    if args.dataset == 'indoor':
        video_path = 'data/data_indoor/camera/d435i/color/cam_2025_06_04_09_24_42.avi'
        video_timestamps_path = 'data/data_indoor/camera/d435i/color/cam_2025_06_04_09_24_42_timestamps.txt'
        depth_folder_path = 'data/data_indoor/camera/d435i/depth/cam_2025_06_04_09_24_42_depth_frames/'
        depth_timestamps_path = 'data/data_indoor/camera/d435i/depth/cam_2025_06_04_09_24_42_timestamps.txt'
    else: # outdoor
        video_path = 'data/data_outdoor/camera/d435i/color/cam_2025_06_04_09_41_51.avi'
        video_timestamps_path = 'data/data_outdoor/camera/d435i/color/cam_2025_06_04_09_41_51_timestamps.txt'
        depth_folder_path = 'data/data_outdoor/camera/d435i/depth/cam_2025_06_04_09_41_51_depth_frames/'
        depth_timestamps_path = 'data/data_outdoor/camera/d435i/depth/cam_2025_06_04_09_41_51_timestamps.txt'

    output_csv_path = f'output/tracking_log_{args.dataset}.csv'

    # --- Load Timestamps ---
    def load_timestamps(path):
        timestamps = []
        with open(path, 'r') as f:
            for line in f:
                try:
                    timestamps.append(float(line.strip()))
                except ValueError:
                    continue # Skip non-float lines
        return timestamps

    video_timestamps = load_timestamps(video_timestamps_path)
    depth_timestamps = load_timestamps(depth_timestamps_path)

    # --- Video Processing ---
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    frame_id = 0
    # --- Setup CSV Logging ---
    with open(output_csv_path, 'w', newline='') as f:
        writer = pd.DataFrame(columns=['timestamp', 'frame_id', 'object_id', 'x_position', 'y_position', 'z_position']).to_csv(f, index=False)

    while cap.isOpened():
        # Check if there are more timestamps to process
        if frame_id >= len(video_timestamps):
            break

        ret, frame = cap.read()
        if not ret:
            break

        # Get the timestamp for the current frame
        timestamp = video_timestamps[frame_id]

        # Perform tracking with a higher confidence threshold and selected tracker
        tracker_config_path = f'src/trackers/{args.tracker}.yaml'
        results = model.track(frame, persist=True, classes=[0], conf=0.5, tracker=tracker_config_path) # class 0 is 'person'

        # Process results
        if results[0].boxes.id is not None:
            if args.debug >= 1:
                print(f"Frame {frame_id}: Found {len(results[0].boxes.id)} tracks.")
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)

            # Get corresponding depth frame
            depth_frame_path = get_closest_depth_frame(timestamp, depth_timestamps, depth_folder_path)
            if not depth_frame_path:
                print(f"  Warning: No depth frame found for timestamp {timestamp}")
                continue

            # print(f"  Found depth frame: {depth_frame_path}")
            depth_image = cv2.imread(depth_frame_path, cv2.IMREAD_UNCHANGED)
            if depth_image is None:
                print(f"  Warning: Could not read depth image {depth_frame_path}")
                continue

            # print(f"  Successfully loaded depth image.")
            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = box
                person_image = frame[y1:y2, x1:x2]

                # Check for yellow vest
                if person_image.size > 0:
                    is_vest, vest_mask, yellow_percentage = detect_yellow_vest(person_image)
                    
                    # --- Debugging Visualization ---
                    if args.debug >= 2:
                        # Show the color mask for the current person
                        cv2.imshow(f"Vest Mask ID: {track_id}", vest_mask)
                        print(f"Frame {frame_id}, Track ID {track_id}: Yellow Percentage = {yellow_percentage:.2f}%")

                    if is_vest:
                        # Calculate 3D position
                        x_cam, y_cam, z_cam = get_3d_position(box, depth_image, K_rgb)

                        # Transform to xTrack base frame
                        pos_camera_frame = np.array([x_cam, y_cam, z_cam, 1])
                        pos_base_frame = T_base_rgb_camera @ pos_camera_frame
                        x, y, z = pos_base_frame[:3]

                        # Log data in real-time
                        log_entry = pd.DataFrame([[timestamp, frame_id, track_id, x, y, z]], columns=['timestamp', 'frame_id', 'object_id', 'x_position', 'y_position', 'z_position'])
                        with open(output_csv_path, 'a', newline='') as f:
                            log_entry.to_csv(f, header=False, index=False)

                        # Visualization
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2) # Yellow box for vest
                        cv2.putText(frame, f"ID: {track_id} (Vest)", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                        cv2.putText(frame, f"Pos: ({x:.2f}, {y:.2f}, {z:.2f})m", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                    else:
                        # Optional: visualize non-vest detections differently
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1) # Red box for no vest
                        cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)


        # Display the resulting frame
        cv2.imshow('xTrack Person Tracking', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_id += 1

    # --- Cleanup ---
    cap.release()
    cv2.destroyAllWindows()
    print(f"Tracking data saved to {output_csv_path}")

if __name__ == "__main__":
    main()

import time
import warnings
from collections import deque

warnings.filterwarnings("ignore")

import cv2
import numpy as np
from ultralytics import YOLO

import config
from coordinate_frames import K_rgb, T_base_rgb_camera, T_camera_lidar
from lidar_utils import (
    find_closest_lidar_file,
    get_3d_position_from_lidar,
    load_point_cloud,
)
from localization import get_3d_position, get_closest_depth_frame
from reid import cosine_similarity, get_appearance_embedding
from utils import load_timestamps, log_result, parse_arguments, setup_logging
from vest_classifier import VestClassifier
from vision import detect_yellow_vest


class PersonTracker:
    def __init__(self, args):
        self.args = args
        self.paths = config.get_dataset_paths(args.dataset)
        self.output_csv_path = (
            config.OUTPUT_DIR / f"tracking_log_{self.args.dataset}.csv"
        )
        self._load_timestamps()
        setup_logging(self.output_csv_path)

        self.model = None
        self.vest_classifier = None
        self._load_models()

        # Re-ID Data Structures
        self.track_embeddings = {}
        self.lost_tracks = {}
        self.id_links = {}
        self.next_person_id = 1
        self.previous_track_ids = set()

        # Performance Benchmarking
        if self.args.benchmark:
            self.perf_data = {
                "total_frame_time": deque(maxlen=100),
                "yolo_time": deque(maxlen=100),
                "reid_time": deque(maxlen=100),
                "vest_depth_time": deque(maxlen=100),
            }

    def _load_timestamps(self):
        self.video_timestamps = load_timestamps(self.paths["video_timestamps_path"])
        self.depth_timestamps = load_timestamps(self.paths["depth_timestamps_path"])
        if self.args.localization_method == "lidar":
            self.lidar_timestamps = load_timestamps(
                self.paths["lidar_timestamps_path"], scale_factor=1e-9
            )
            self.lidar_pcd_files = list(self.paths["lidar_folder_path"].glob("*.pcd"))
            if not self.lidar_pcd_files:
                print(
                    f"Warning: No LiDAR .pcd files found in {self.paths['lidar_folder_path']}. LiDAR localization will not work."
                )

    def _load_models(self):
        print("Loading models...")
        self.model = YOLO("yolo11n.pt")
        self.model.to(self.args.device)

        if self.args.vest_detection == "model":
            self.vest_classifier = VestClassifier(
                model_path=self.args.vest_model_path, device=self.args.device
            )
        print("Models loaded.")

    def run(self):
        cap = cv2.VideoCapture(str(self.paths["video_path"]))
        if not cap.isOpened():
            print(f"Error: Could not open video {self.paths['video_path']}")
            return

        frame_id = 0
        while cap.isOpened():
            if frame_id >= len(self.video_timestamps):
                break

            start_time = time.time()

            ret, frame = cap.read()
            if not ret:
                break

            timestamp = self.video_timestamps[frame_id]

            self._process_frame(frame, frame_id, timestamp)

            if self.args.benchmark:
                self.perf_data["total_frame_time"].append(time.time() - start_time)

            cv2.imshow("xTrack Person Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            frame_id += 1

        cap.release()
        cv2.destroyAllWindows()
        print(f"Tracking data saved to {self.output_csv_path}")

        if self.args.benchmark:
            self._print_benchmark_results()

    def _process_frame(self, frame, frame_id, timestamp):
        yolo_start = time.time()
        tracker_config_path = (
            config.PROJECT_DIR / f"src/trackers/{self.args.tracker}.yaml"
        )
        results = self.model.track(
            frame,
            persist=True,
            classes=[0],
            conf=0.5,
            tracker=str(tracker_config_path),
            verbose=False,
        )
        if self.args.benchmark:
            self.perf_data["yolo_time"].append(time.time() - yolo_start)

        if results[0].boxes.id is None:
            return

        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        ultralytics_ids = results[0].boxes.id.cpu().numpy().astype(int)

        reid_start = time.time()
        if self.args.reid_method == "custom":
            processed_ids = self._handle_custom_reid(
                frame, boxes, ultralytics_ids, frame_id
            )
        else:
            processed_ids = list(zip(boxes, ultralytics_ids))
        if self.args.benchmark:
            self.perf_data["reid_time"].append(time.time() - reid_start)

        vest_depth_start = time.time()
        for box, display_id in processed_ids:
            if display_id is None:
                continue
            self._process_detection(frame, box, display_id, frame_id, timestamp)
        if self.args.benchmark:
            self.perf_data["vest_depth_time"].append(time.time() - vest_depth_start)

    def _handle_custom_reid(self, frame, boxes, ultralytics_ids, frame_id):
        current_ultralytics_ids = set(ultralytics_ids)

        # Update embeddings for active tracks
        for box, track_id in zip(boxes, ultralytics_ids):
            person_image = frame[box[1] : box[3], box[0] : box[2]]
            embedding = get_appearance_embedding(person_image)
            if embedding is not None:
                self.track_embeddings[track_id] = embedding

        # Identify and handle lost tracks
        lost_ids = self.previous_track_ids - current_ultralytics_ids
        for track_id in lost_ids:
            original_id = self.id_links.get(track_id, track_id)
            if self.track_embeddings.get(track_id) is not None:
                self.lost_tracks[original_id] = {
                    "embedding": self.track_embeddings[track_id],
                    "last_seen": frame_id,
                }
            if track_id in self.track_embeddings:
                del self.track_embeddings[track_id]

        # Process all current tracks
        processed_ids = []
        for box, ultralytics_id in zip(boxes, ultralytics_ids):
            display_id = self.id_links.get(ultralytics_id)

            if display_id is None:  # New or re-identified track
                embedding = self.track_embeddings.get(ultralytics_id)
                if embedding is not None:
                    best_match_id, best_similarity = -1, -1
                    for lost_id, data in list(self.lost_tracks.items()):
                        if (
                            frame_id - data["last_seen"] > config.MAX_LOST_FRAMES
                        ):  # MAX_LOST_FRAMES
                            del self.lost_tracks[lost_id]
                            continue
                        similarity = cosine_similarity(embedding, data["embedding"])
                        if similarity > best_similarity:
                            best_similarity, best_match_id = similarity, lost_id

                    if (
                        best_similarity > config.REID_SIMILARITY_THRESHOLD
                    ):  # REID_SIMILARITY_THRESHOLD
                        display_id = best_match_id
                        self.id_links[ultralytics_id] = display_id
                        if self.args.debug >= 1:
                            print(
                                f"Re-identified track {ultralytics_id} as {display_id} with similarity {best_similarity:.2f}"
                            )
                        if best_match_id in self.lost_tracks:
                            del self.lost_tracks[best_match_id]
                    else:
                        display_id = self.next_person_id
                        self.id_links[ultralytics_id] = display_id
                        self.next_person_id += 1

            processed_ids.append((box, display_id))
        self.previous_track_ids = current_ultralytics_ids
        return processed_ids

    def _process_detection(self, frame, box, display_id, frame_id, timestamp):
        x1, y1, x2, y2 = box
        person_image = frame[y1:y2, x1:x2]

        if person_image.size == 0:
            return

        if self.args.vest_detection == "model":
            is_vest, confidence = self.vest_classifier.predict(person_image)
            if self.args.debug >= 2:
                print(
                    f"Frame {frame_id}, Track ID {display_id}: Vest Confidence = {confidence:.2f}"
                )
        else:
            is_vest, vest_mask, yellow_percentage = detect_yellow_vest(person_image)
            if self.args.debug >= 2:
                if is_vest:
                    cv2.imshow(f"Vest Mask ID: {display_id}", vest_mask)
                print(
                    f"Frame {frame_id}, Track ID {display_id}: Yellow Percentage = {yellow_percentage * 100:.2f}%"
                )

        if is_vest:
            x, y, z = self._get_3d_position(box, timestamp)
            if x != 0 or y != 0 or z != 0:
                log_result(
                    self.output_csv_path, timestamp, frame_id, display_id, x, y, z
                )
                self._draw_on_frame(frame, box, display_id, (x, y, z), is_vest=True)
        else:
            self._draw_on_frame(frame, box, display_id, is_vest=False)

    def _get_3d_position(self, box, timestamp):
        if self.args.localization_method == "lidar":
            lidar_file = find_closest_lidar_file(
                timestamp, self.lidar_timestamps, self.lidar_pcd_files
            )
            point_cloud = load_point_cloud(lidar_file)
            if point_cloud is not None:
                x_cam, y_cam, z_cam = get_3d_position_from_lidar(
                    box, point_cloud, K_rgb, T_camera_lidar
                )
                pos_camera_frame = np.array([x_cam, y_cam, z_cam, 1])
                pos_base_frame = T_base_rgb_camera @ pos_camera_frame
                return pos_base_frame[:3]
        else:  # Default to depth camera
            depth_frame_path = get_closest_depth_frame(
                timestamp, self.depth_timestamps, self.paths["depth_folder_path"]
            )
            if depth_frame_path:
                depth_image = cv2.imread(str(depth_frame_path), cv2.IMREAD_UNCHANGED)
                if depth_image is not None:
                    x_cam, y_cam, z_cam = get_3d_position(box, depth_image, K_rgb)
                    pos_camera_frame = np.array([x_cam, y_cam, z_cam, 1])
                    pos_base_frame = T_base_rgb_camera @ pos_camera_frame
                    return pos_base_frame[:3]
        return 0, 0, 0

    def _draw_on_frame(self, frame, box, display_id, position=None, is_vest=False):
        x1, y1, x2, y2 = box
        if is_vest:
            color = (0, 255, 255)
            thickness = 2
            label = f"ID: {display_id} (Vest)"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                thickness,
            )
            if position:
                pos_text = (
                    f"Pos: ({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f})m"
                )
                cv2.putText(
                    frame,
                    pos_text,
                    (x1, y2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    thickness,
                )
        else:
            color = (0, 0, 255)
            thickness = 1
            label = f"ID: {display_id}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                thickness,
            )

    def _print_benchmark_results(self):
        print("\n--- Performance Benchmark Results ---")
        total_time = sum(self.perf_data["total_frame_time"])
        if total_time > 0:
            avg_fps = len(self.perf_data["total_frame_time"]) / total_time
            print(f"Average FPS: {avg_fps:.2f}")

        for name, data in self.perf_data.items():
            if len(data) > 0:
                avg_time = sum(data) / len(data)
                print(f"Average {name.replace('_', ' ')}: {avg_time * 1000:.2f} ms")
        print("------------------------------------")


def main():
    args = parse_arguments()
    if args.reid_method == "botsort" and args.tracker != "botsort":
        print(
            "Error: BoTSORT Re-ID can only be used with the BoTSORT tracker. Please set --tracker to 'botsort'."
        )
        return

    tracker = PersonTracker(args)
    tracker.run()


if __name__ == "__main__":
    main()

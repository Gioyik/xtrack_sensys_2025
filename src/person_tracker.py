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
    fuse_depth_and_lidar,
)
from localization import get_3d_position, get_closest_depth_frame
from reid import cosine_similarity, get_appearance_embedding, initialize_reid_model
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
        
        # Vest Detection Persistence Tracking
        self.vest_detection_history = {}  # track_id -> list of recent vest detections

        # Performance Benchmarking
        if self.args.benchmark:
            self.perf_data = {
                "total_frame_time": deque(maxlen=100),
                "yolo_time": deque(maxlen=100),
                "reid_time": deque(maxlen=100),
                "vest_depth_time": deque(maxlen=100),
                "localization_time": deque(maxlen=100),
            }
            self.detection_stats = {
                "total_detections": 0,
                "vest_detections": 0,
                "successful_localizations": 0,
                "failed_localizations": 0,
                "track_ids_seen": set(),
                "yellow_percentages": [],
                "reid_events": 0,  # Count of re-identification events
                "max_track_id": 0,  # Highest track ID seen
                "reid_messages": [],  # Store re-ID events for analysis
            }
            self.start_time = None

    def _load_timestamps(self):
        self.video_timestamps = load_timestamps(self.paths["video_timestamps_path"])
        self.depth_timestamps = load_timestamps(self.paths["depth_timestamps_path"])
        if self.args.localization_method in ["lidar", "fusion"]:
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

        # Initialize ReID model with specified device if using custom ReID
        if self.args.reid_method == "custom":
            initialize_reid_model(self.args.device)

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
        processed_frames = 0
        skipped_frames = 0
        
        if self.args.benchmark:
            self.start_time = time.time()
        
        print(f"Processing every {self.args.jump_frames + 1} frames (skipping {self.args.jump_frames} frames)")
        
        while cap.isOpened():
            if frame_id >= len(self.video_timestamps):
                break

            ret, frame = cap.read()
            if not ret:
                break

            # Implement frame skipping logic
            if self.args.jump_frames > 0 and (frame_id % (self.args.jump_frames + 1)) != 0:
                skipped_frames += 1
                frame_id += 1
                continue

            start_time = time.time()
            timestamp = self.video_timestamps[frame_id]

            self._process_frame(frame, frame_id, timestamp)
            processed_frames += 1

            if self.args.benchmark:
                self.perf_data["total_frame_time"].append(time.time() - start_time)

            cv2.imshow("xTrack Person Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            frame_id += 1

        cap.release()
        cv2.destroyAllWindows()
        print(f"Tracking data saved to {self.output_csv_path}")
        print(f"Processed {processed_frames} frames, skipped {skipped_frames} frames")

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
            # Update detection statistics
            self.detection_stats["total_detections"] += len(processed_ids)
            valid_ids = [display_id for _, display_id in processed_ids if display_id is not None]
            self.detection_stats["track_ids_seen"].update(valid_ids)
            if valid_ids:
                self.detection_stats["max_track_id"] = max(self.detection_stats["max_track_id"], max(valid_ids))

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
                            frame_id - data["last_seen"] > self.args.max_lost_frames
                        ):  # Use configurable max lost frames
                            del self.lost_tracks[lost_id]
                            continue
                        similarity = cosine_similarity(embedding, data["embedding"])
                        if similarity > best_similarity:
                            best_similarity, best_match_id = similarity, lost_id

                    if (
                        best_similarity > self.args.reid_threshold
                    ):  # Use configurable threshold
                        display_id = best_match_id
                        self.id_links[ultralytics_id] = display_id
                        
                        # Track re-identification events for benchmarking
                        if self.args.benchmark:
                            self.detection_stats["reid_events"] += 1
                            self.detection_stats["reid_messages"].append({
                                "frame": frame_id,
                                "old_id": ultralytics_id,
                                "new_id": display_id,
                                "similarity": best_similarity
                            })
                        
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
            is_vest, vest_mask, yellow_percentage = detect_yellow_vest(person_image, self.args.vest_threshold)
            if self.args.debug >= 2:
                if is_vest:
                    cv2.imshow(f"Vest Mask ID: {display_id}", vest_mask)
                print(
                    f"Frame {frame_id}, Track ID {display_id}: Yellow Percentage = {yellow_percentage * 100:.2f}%"
                )
            
            # Collect benchmarking data
            if self.args.benchmark:
                self.detection_stats["yellow_percentages"].append(yellow_percentage * 100)

        # Apply temporal persistence filtering for vest detection
        is_vest_confirmed = self._apply_vest_persistence(display_id, is_vest)

        # Always draw vest detection result regardless of 3D localization success
        if is_vest_confirmed:
            if self.args.benchmark:
                self.detection_stats["vest_detections"] += 1
                
            localization_start = time.time()
            x, y, z = self._get_3d_position(box, timestamp)
            if self.args.benchmark:
                self.perf_data["localization_time"].append(time.time() - localization_start)
                
            if x != 0 or y != 0 or z != 0:
                # Valid 3D position - log to CSV and draw with position
                if self.args.benchmark:
                    self.detection_stats["successful_localizations"] += 1
                log_result(
                    self.output_csv_path, timestamp, frame_id, display_id, x, y, z
                )
                self._draw_on_frame(frame, box, display_id, (x, y, z), is_vest=True)
                if self.args.debug >= 1:
                    print(f"Frame {frame_id}, Track ID {display_id}: Vest detected with 3D position ({x:.2f}, {y:.2f}, {z:.2f})")
            else:
                # Vest detected but no valid 3D position - still draw vest indicator
                if self.args.benchmark:
                    self.detection_stats["failed_localizations"] += 1
                self._draw_on_frame(frame, box, display_id, None, is_vest=True)
                if self.args.debug >= 1:
                    print(f"Frame {frame_id}, Track ID {display_id}: Vest detected but 3D localization failed ({self.args.localization_method})")
        else:
            self._draw_on_frame(frame, box, display_id, is_vest=False)

    def _apply_vest_persistence(self, track_id, is_vest_detected):
        """
        Apply temporal filtering to vest detection to reduce false positives.
        
        Args:
            track_id: The track ID of the person
            is_vest_detected: Boolean indicating if vest was detected in current frame
            
        Returns:
            Boolean indicating if vest detection is confirmed after persistence filtering
        """
        # Initialize history for new tracks
        if track_id not in self.vest_detection_history:
            self.vest_detection_history[track_id] = []
        
        # Add current detection to history
        self.vest_detection_history[track_id].append(is_vest_detected)
        
        # Keep only recent history (sliding window)
        max_history = max(self.args.vest_persistence, 5)  # At least 5 frames of history
        if len(self.vest_detection_history[track_id]) > max_history:
            self.vest_detection_history[track_id] = self.vest_detection_history[track_id][-max_history:]
        
        # Check if vest has been detected for required consecutive frames
        if len(self.vest_detection_history[track_id]) >= self.args.vest_persistence:
            recent_detections = self.vest_detection_history[track_id][-self.args.vest_persistence:]
            return all(recent_detections)  # All recent frames must have vest detection
        
        # Not enough history yet, use current detection if persistence is 1
        return is_vest_detected if self.args.vest_persistence == 1 else False

    def _get_3d_position(self, box, timestamp):
        """
        Get 3D position using the specified localization method.
        
        Args:
            box: 2D bounding box (x1, y1, x2, y2)
            timestamp: Frame timestamp for synchronization
            
        Returns:
            3D position in base frame (x, y, z) or (0, 0, 0) if failed
        """
        if self.args.localization_method == "lidar":
            # LiDAR-only localization
            lidar_file = find_closest_lidar_file(
                timestamp, self.lidar_timestamps, self.lidar_pcd_files
            )
            point_cloud = load_point_cloud(lidar_file)
            if point_cloud is not None:
                x_cam, y_cam, z_cam = get_3d_position_from_lidar(
                    box, point_cloud, K_rgb, T_camera_lidar
                )
                if x_cam != 0 or y_cam != 0 or z_cam != 0:
                    pos_camera_frame = np.array([x_cam, y_cam, z_cam, 1])
                    pos_base_frame = T_base_rgb_camera @ pos_camera_frame
                    return pos_base_frame[:3]
                    
        elif self.args.localization_method == "fusion":
            # Sensor fusion: combine depth camera and LiDAR
            depth_frame_path = get_closest_depth_frame(
                timestamp, self.depth_timestamps, self.paths["depth_folder_path"]
            )
            lidar_file = find_closest_lidar_file(
                timestamp, self.lidar_timestamps, self.lidar_pcd_files
            )
            
            if depth_frame_path and lidar_file:
                depth_image = cv2.imread(str(depth_frame_path), cv2.IMREAD_UNCHANGED)
                point_cloud = load_point_cloud(lidar_file)
                
                if depth_image is not None and point_cloud is not None:
                    x_cam, y_cam, z_cam = fuse_depth_and_lidar(
                        box, depth_image, point_cloud, K_rgb, K_rgb, T_camera_lidar
                    )
                    if x_cam != 0 or y_cam != 0 or z_cam != 0:
                        pos_camera_frame = np.array([x_cam, y_cam, z_cam, 1])
                        pos_base_frame = T_base_rgb_camera @ pos_camera_frame
                        return pos_base_frame[:3]
            
            # Fallback to individual sensors if fusion fails
            if depth_frame_path:
                depth_image = cv2.imread(str(depth_frame_path), cv2.IMREAD_UNCHANGED)
                if depth_image is not None:
                    x_cam, y_cam, z_cam = get_3d_position(box, depth_image, K_rgb)
                    if x_cam != 0 or y_cam != 0 or z_cam != 0:
                        pos_camera_frame = np.array([x_cam, y_cam, z_cam, 1])
                        pos_base_frame = T_base_rgb_camera @ pos_camera_frame
                        return pos_base_frame[:3]
                        
        else:  # Default to depth camera only
            depth_frame_path = get_closest_depth_frame(
                timestamp, self.depth_timestamps, self.paths["depth_folder_path"]
            )
            if depth_frame_path:
                depth_image = cv2.imread(str(depth_frame_path), cv2.IMREAD_UNCHANGED)
                if depth_image is not None:
                    x_cam, y_cam, z_cam = get_3d_position(box, depth_image, K_rgb)
                    if x_cam != 0 or y_cam != 0 or z_cam != 0:
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
        print("\n" + "="*60)
        print("           COMPREHENSIVE BENCHMARK RESULTS")
        print("="*60)
        
        # Calculate total processing time
        total_processing_time = time.time() - self.start_time if self.start_time else 0
        
        # Performance Metrics
        print(f"\nPERFORMANCE METRICS")
        print("-" * 30)
        total_time = sum(self.perf_data["total_frame_time"])
        if total_time > 0:
            avg_fps = len(self.perf_data["total_frame_time"]) / total_time
            print(f"Average FPS: {avg_fps:.2f}")
            print(f"Total Processing Time: {total_processing_time:.1f} seconds")

        print(f"\nCOMPONENT LATENCY")
        print("-" * 30)
        for name, data in self.perf_data.items():
            if len(data) > 0:
                avg_time = sum(data) / len(data)
                max_time = max(data)
                min_time = min(data)
                print(f"{name.replace('_', ' ').title():<20}: {avg_time * 1000:6.2f} ms (avg) | {min_time * 1000:6.2f} ms (min) | {max_time * 1000:6.2f} ms (max)")
        
        # Detection Statistics
        print(f"\nDETECTION STATISTICS")
        print("-" * 30)
        print(f"Total Detections: {self.detection_stats['total_detections']}")
        print(f"Vest Detections: {self.detection_stats['vest_detections']}")
        print(f"Unique Track IDs: {len(self.detection_stats['track_ids_seen'])}")
        
        if self.detection_stats['vest_detections'] > 0:
            vest_rate = (self.detection_stats['vest_detections'] / self.detection_stats['total_detections']) * 100
            print(f"Vest Detection Rate: {vest_rate:.1f}%")
        
        # ReID Performance Analysis
        print(f"\nRE-IDENTIFICATION ANALYSIS")
        print("-" * 30)
        print(f"Total Re-ID Events: {self.detection_stats['reid_events']}")
        print(f"Unique Track IDs: {len(self.detection_stats['track_ids_seen'])}")
        print(f"Highest Track ID: {self.detection_stats['max_track_id']}")
        
        if self.detection_stats['reid_events'] > 0 and self.detection_stats['total_detections'] > 0:
            reid_rate = (self.detection_stats['reid_events'] / self.detection_stats['total_detections']) * 100
            print(f"Re-ID Rate: {reid_rate:.2f}% of detections")
            
            # Analyze re-ID similarity distribution
            similarities = [event['similarity'] for event in self.detection_stats['reid_messages']]
            if similarities:
                print(f"Re-ID Similarity - Mean: {np.mean(similarities):.3f}, Min: {np.min(similarities):.3f}, Max: {np.max(similarities):.3f}")
        
        # Track ID efficiency analysis
        if self.detection_stats['max_track_id'] > 0:
            id_efficiency = len(self.detection_stats['track_ids_seen']) / self.detection_stats['max_track_id'] * 100
            print(f"Track ID Efficiency: {id_efficiency:.1f}% (lower = more fragmentation)")
        
        # ReID Quality Assessment
        if self.detection_stats['reid_events'] > 0:
            print(f"\nReID Quality Indicators:")
            print(f"  - Frequent Re-IDs (>10% of detections): {'  YES' if (self.detection_stats['reid_events'] / max(self.detection_stats['total_detections'], 1)) > 0.1 else ' NO'}")
            print(f"  - Many Unique IDs (efficiency <80%): {'  YES' if len(self.detection_stats['track_ids_seen']) > 0 and (len(self.detection_stats['track_ids_seen']) / self.detection_stats['max_track_id'] * 100 < 80) else ' NO'}")
            print(f"  - Threshold seems: {' Too Low' if (self.detection_stats['reid_events'] / max(self.detection_stats['total_detections'], 1)) > 0.15 else ' Moderate' if (self.detection_stats['reid_events'] / max(self.detection_stats['total_detections'], 1)) > 0.05 else ' Good'}")
        
        # Localization Statistics
        total_localizations = self.detection_stats['successful_localizations'] + self.detection_stats['failed_localizations']
        if total_localizations > 0:
            success_rate = (self.detection_stats['successful_localizations'] / total_localizations) * 100
            print(f"\nLOCALIZATION PERFORMANCE")
            print("-" * 30)
            print(f"Successful Localizations: {self.detection_stats['successful_localizations']}")
            print(f"Failed Localizations: {self.detection_stats['failed_localizations']}")
            print(f"Localization Success Rate: {success_rate:.1f}%")
        
        # Yellow Percentage Statistics
        if self.detection_stats['yellow_percentages']:
            yellow_percentages = self.detection_stats['yellow_percentages']
            print(f"\nVEST DETECTION ANALYSIS")
            print("-" * 30)
            print(f"Mean Yellow Percentage: {np.mean(yellow_percentages):.2f}%")
            print(f"Median Yellow Percentage: {np.median(yellow_percentages):.2f}%")
            print(f"Max Yellow Percentage: {np.max(yellow_percentages):.2f}%")
            print(f"Min Yellow Percentage: {np.min(yellow_percentages):.2f}%")
            print(f"Std Yellow Percentage: {np.std(yellow_percentages):.2f}%")
            
            # Threshold analysis
            current_threshold = self.args.vest_threshold
            above_threshold = sum(1 for p in yellow_percentages if p > current_threshold)
            below_threshold = len(yellow_percentages) - above_threshold
            print(f"Above {current_threshold}% threshold: {above_threshold} ({above_threshold/len(yellow_percentages)*100:.1f}%)")
            print(f"Below {current_threshold}% threshold: {below_threshold} ({below_threshold/len(yellow_percentages)*100:.1f}%)")
        
        # Configuration Summary
        print(f"\nCONFIGURATION USED")
        print("-" * 30)
        print(f"Dataset: {self.args.dataset}")
        print(f"Tracker: {self.args.tracker}")
        print(f"ReID Method: {self.args.reid_method}")
        print(f"Localization: {self.args.localization_method}")
        print(f"Vest Detection: {self.args.vest_detection}")
        print(f"Device: {self.args.device}")
        print(f"Frame Skipping: {self.args.jump_frames}")
        print(f"Vest Threshold: {self.args.vest_threshold}%")
        print(f"Vest Persistence: {self.args.vest_persistence} frames")
        print(f"ReID Threshold: {self.args.reid_threshold}")
        print(f"Max Lost Frames: {self.args.max_lost_frames}")
        print(f"Debug Level: {self.args.debug}")
        
        print("\n" + "="*60)


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

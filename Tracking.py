from ultralytics import YOLO
import cv2
import numpy as np

from video_stream import VideoStream


def calculate_distance(object_position, track_positions):
    distances = np.sqrt(np.sum((track_positions - object_position) ** 2, axis=1))
    minima = np.argmin(distances)
    return distances[minima], minima


def create_tracker(tracker_type):
    if tracker_type == "MIL":
        return cv2.TrackerMIL_create()
    elif tracker_type == "VIT":
        return cv2.TrackerVit_create()
    elif tracker_type == "NANO":
        return cv2.TrackerNano_create()
    elif tracker_type == "GOTURN":
        return cv2.TrackerGOTURN_create()
    elif tracker_type == "RPN":
        return cv2.TrackerDaSiamRPN_create()
    else:
        raise ValueError(f"Unknown tracker type: {tracker_type}")


def detect_vest(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([15, 100, 100])
    upper_yellow = np.array([35, 255, 255])
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    # Dilate the mask to fill in holes
    mask = cv2.dilate(mask, kernel, iterations=3)
    boolean_mask = np.zeros_like(mask, dtype=bool)
    boolean_mask[mask!=0] = True
    return boolean_mask


class Tracker:
    def __init__(self, frame, bounding_box, tracker_id, point_3d, model_type):
        self.model_type = model_type
        self.bounding_box = bounding_box
        self.model = None
        self.init_model(frame)
        self.point_3d = None
        self.active = 1
        self.id = tracker_id
        self.has_vest = False
        self.had_vest = 0
        self.vest_pct = 0

    def init_model(self, frame):
        self.model = create_tracker(self.model_type)
        self.model.init(frame, tuple(self.bounding_box.astype(int)))

    def update(self, frame, depth_frame=None):
        self.had_vest -= 1
        self.active -= 1
        if self.active == 0:
            (success, box) = self.model.update(frame)
            if success:
                self.active = 1
                self.bounding_box = np.array(list(box))
                if self.point_3d is not None and depth_frame is not None:
                    self.point_3d = None

    def refresh_data(self, frame, new_box, new_3d_point):
        self.active = 1
        self.bounding_box = new_box
        self.point_3d = new_3d_point
        self.init_model(frame)


class PersonTracker:
    def __init__(self, detection_model_version, detection_frame_rate, distance_threshold, active_threshold, min_vest_pct, tracker_kwargs):
        self.detection_model = YOLO(detection_model_version)
        self.detection_frame_rate = detection_frame_rate
        self.tracker_kwargs = tracker_kwargs
        self.tracker_objects = []
        self.next_id = 0
        self.distance_threshold = distance_threshold
        self.active_threshold = active_threshold
        self.min_vest_pct = min_vest_pct
        self.had_vest_threshold = -10

    def detect(self, frame):
        objects_in_frame = self.detection_model(frame, verbose=False)
        objects_in_frame = objects_in_frame[0].boxes.data.cpu().numpy()
        # Confidence > 0.5
        objects_in_frame = objects_in_frame[objects_in_frame[:, 4] > 0.5]
        # Object type == 0 (Person)
        objects_in_frame = objects_in_frame[objects_in_frame[:, -1] == 0]
        # Width Height
        objects_in_frame[:, 2] = objects_in_frame[:, 2] - objects_in_frame[:, 0]
        objects_in_frame[:, 3] = objects_in_frame[:, 3] - objects_in_frame[:, 1]
        return objects_in_frame[:, :4]

    def track(self, frame, depth_frame=None):
        for tracker_object in self.tracker_objects:
            tracker_object.update(frame, depth_frame)

    def update(self, frame, n_frame, depth_frame=None):
        self.track(frame, depth_frame)
        for i in range(len(self.tracker_objects)-1, -1, -1):
            if self.tracker_objects[i].active < self.active_threshold:
                self.tracker_objects.pop(i)
        if (n_frame-1) % self.detection_frame_rate == 0:
            for tracker_object in self.tracker_objects:
                tracker_object.active -= 1
            new_objects = self.detect(frame)
            old_positions = np.array([tracker_object.bounding_box for tracker_object in self.tracker_objects])
            if len(old_positions) > 0:
                old_positions = old_positions[:,:2]
            for i, new_object in enumerate(new_objects):
                object_position = new_object[:2]
                if old_positions.size > 0:
                    min_distance, min_index = calculate_distance(object_position, old_positions)
                    if min_distance <= self.distance_threshold:
                        self.tracker_objects[min_index].refresh_data(frame, new_object, None)
                        break
                self.tracker_objects.append(Tracker(frame, new_object, self.next_id, None, **self.tracker_kwargs))
                self.next_id += 1

        for i in range(len(self.tracker_objects)-1, -1, -1):
            if self.tracker_objects[i].active < self.active_threshold:
                self.tracker_objects.pop(i)
            elif self.tracker_objects[i].has_vest and self.tracker_objects[i].had_vest < self.had_vest_threshold:
                self.tracker_objects[i].has_vest = False

        vest_mask = detect_vest(frame)
        for tracker_object in self.tracker_objects:
            x1, y1, w, h = tracker_object.bounding_box.astype(int)
            x2, y2 = x1+w, y1+h
            vest_pct = np.mean(vest_mask[y1:y2, x1:x2])
            if vest_pct > self.min_vest_pct:
                tracker_object.has_vest = True
                tracker_object.vest_pct = vest_pct
                tracker_object.had_vest = 0

    def visualize(self, frame):
        for tracker_object in self.tracker_objects:
            if tracker_object.active < 0:
                continue
            x, y, w, h = tracker_object.bounding_box.astype(int)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (25, 178, 178) if tracker_object.has_vest else (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"id={tracker_object.id},{round(tracker_object.vest_pct * 100, 2)}",
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2,
            )
        cv2.imshow("Tracker", frame)
        cv2.waitKey(1)


if __name__ == "__main__":
    video_ = VideoStream("data/data_outdoor/camera/d435i/color/cam_2025_06_04_09_41_51_color_frames",
                         "data/data_outdoor/camera/d435i/depth/cam_2025_06_04_09_41_51_depth_frames",
                         start_frame=3000)
    tracker = PersonTracker("yolo11n", 10, 30, 0, 0.1, {"model_type":"MIL"})

    while video_.active:
        color_frame, depth_frame = video_.next_frame()
        tracker.update(color_frame, video_.color_n_frame, depth_frame)
        tracker.visualize(color_frame)
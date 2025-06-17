import numpy as np
import cv2


def load_timestamps(path):
    timestamps = []
    with open(path) as f:
        for line in f:
            try:
                timestamps.append(float(line))
            except Exception as e:
                pass
    return np.array(timestamps)


def get_nearest_timestamp(timestamps, timestamp):
    return np.argmax(timestamps[timestamps<=timestamp])


def load_image(path, index):
    try:
        frame_path = path + "/" + ("00000" + str(index))[-6:] + ".png"
        frame = cv2.imread(frame_path, cv2.IMREAD_UNCHANGED)
        return True, frame
    except FileNotFoundError:
        return False, None


class VideoStream:
    def __init__(self, color_path, depth_path=None, start_frame=0):
        self.color_path = color_path
        self.depth_path = depth_path
        self.color_timestamps = load_timestamps(self.color_path.split("_color")[0]+"_timestamps.txt")
        self.depth_timestamps = load_timestamps(self.depth_path.split("_depth")[0]+"_timestamps.txt")
        self.stream = cv2.VideoCapture(self.color_path)
        self.color_n_frame = start_frame
        self.active = True

    def next_frame(self):
        next_depth_frame = None
        ret, next_color_frame = load_image(self.color_path, self.color_n_frame)
        self.color_n_frame += 1
        if not ret:
            self.active = False
            return None, None
        if self.color_n_frame is not None:
            depth_index = get_nearest_timestamp(self.depth_timestamps, self.color_timestamps[self.color_n_frame])
            next_depth_frame = load_image(self.depth_path, depth_index)
        return next_color_frame, next_depth_frame

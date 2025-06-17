import os.path
import cv2
from tqdm import tqdm


def video_to_frames(video_path, frame_directory):
    if not os.path.exists(frame_directory):
        os.mkdir(frame_directory)

    cam = cv2.VideoCapture(video_path)
    index = 0

    pBar = tqdm(range(8800))
    while True:
        (ret, frame) = cam.read()
        if not ret:
            break
        frame_path = frame_directory + "/" + ("00000" + str(index))[-6:] + ".png"
        cv2.imwrite(frame_path, frame)
        index += 1
        pBar.update(1)


if __name__ == "__main__":
    video_to_frames("data/data_outdoor/camera/d435i/color/cam_2025_06_04_09_41_51.avi",
                    "data/data_outdoor/camera/d435i/color/cam_2025_06_04_09_41_51_color_frames")
import cv2
import numpy as np
import os
import mediapipe as mp

# Local imports
from src.constants import *
from utils.extract_keypoints import mp_holistic, mediapipe_detection, draw_landmarks, extract_keypoints


def main():
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        for root, dirs, files in os.walk(DATA_PATH):
            for file in files:
                video_path = os.path.join(root, file)
                action_folder = os.path.basename(os.path.dirname(video_path))
                action = os.path.splitext(os.path.basename(video_path))[0]
                npy_folder = os.path.join(SAVED_DATA_PATH, action_folder, action)
                os.makedirs(npy_folder, exist_ok=True)

                capture = cv2.VideoCapture(video_path)
                frame_num = 0
                while capture.isOpened():
                    ret, frame = capture.read()
                    if not ret:
                        break

                    image, results = mediapipe_detection(frame, holistic)
                    draw_landmarks(image, results)

                    keypoints = extract_keypoints(results)
                    npy_path = os.path.join(npy_folder, f"{frame_num}.npy")
                    np.save(npy_path, keypoints)

                    frame_num += 1

                    cv2.imshow("Sign Language Detection", image)
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break

                capture.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
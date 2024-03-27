import mediapipe as mp
import argparse

from utils.detector import SingleFileSignLanguageDetector


if __name__ == "__main__":
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils

    parser = argparse.ArgumentParser(description='Sign Language Detection')
    parser.add_argument('--video_path', type=str, help='Path to the video file (optional)')
    args = parser.parse_args()
    if args.video_path:
        detector = SingleFileSignLanguageDetector(video_path=args.video_path)
        detector.detect_sign_language()
    else:
        print("Please provide the video path.")
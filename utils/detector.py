import os
import cv2
import torch
import mediapipe as mp
import arabic_reshaper
from bidi.algorithm import get_display
from PIL import Image, ImageDraw, ImageFont
from deep_translator import GoogleTranslator

# Local imports
from src.constants import *
from src.model import CustomLSTM
from utils.extract_keypoints import *
from utils.create_label_mappings import index_to_word

class SingleFileSignLanguageDetector:
    def __init__(self, video_path=None):
        self.video_path = video_path
        self.model = CustomLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES)
        self.model.load_state_dict(torch.load(CHECKPOINT_PATH))
        self.model.eval()

    def detect_sign_language(self):
        cap = cv2.VideoCapture(self.video_path if self.video_path else 0)

        with mp.solutions.holistic.Holistic(
            min_detection_confidence=MP_HOLISTIC_PARAMS, min_tracking_confidence=MP_HOLISTIC_PARAMS
        ) as holistic:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                image, results = mediapipe_detection(frame, holistic)
                draw_landmarks(image, results)

                keypoints = extract_keypoints(results)
                inputs = torch.tensor(keypoints, dtype=torch.float32).unsqueeze(0)

                with torch.no_grad():
                    outputs = self.model(inputs)

                _, predicted_label = torch.max(outputs, 1)
                predicted_label = predicted_label.item()

                predicted_word = str(index_to_word(predicted_label))               

                translated = GoogleTranslator(source="auto", target="en").translate(
                    predicted_word
                )
                print(predicted_word,"====>", translated)
                
                reshaped_text = arabic_reshaper.reshape(predicted_word)
                bidi_text = get_display(reshaped_text)

                font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
                img_pil = Image.fromarray(image)
                draw = ImageDraw.Draw(img_pil)

                text_start_x = 50
                text_start_y = 80
                text_spacing = 40

                draw.text(
                    (text_start_x, text_start_y), bidi_text, font=font, fill=FONT_COLOR
                )
                
                # Uncomment this line to display the translated(english) word in the frame.
                draw.text((text_start_x, text_start_y + text_spacing),translated,font=font,fill=FONT_COLOR,)

                img = np.array(img_pil)
                cv2.imshow("Sign Language Detection", img)

                if cv2.waitKey(10) & 0xFF == ord("q"):
                    break

            cap.release()

        cv2.destroyAllWindows()


import torch
import cv2
import numpy as np
import arabic_reshaper
from bidi.algorithm import get_display
from PIL import Image, ImageDraw, ImageFont

# local imports
from src.model import CustomLSTM
from utils.extract_keypoints import (
    mediapipe_detection,
    draw_landmarks,
    extract_keypoints,
    mp_drawing,
    mp_holistic,
)
from src.constants import *
from utils.create_label_mappings import index_to_word


# Initialize the model and load trained weights
model = CustomLSTM()
model.load_state_dict(torch.load(CHECKPOINT_PATH))
model.eval()

# Set up camera or video stream
cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(
    min_detection_confidence=MP_HOLISTIC_PARAMS,
    min_tracking_confidence=MP_HOLISTIC_PARAMS,
) as holistic:
    capture = cv2.VideoCapture(0)
    frame_num = 0
    while capture.isOpened():
        ret, frame = capture.read()
        if not ret:
            break

        image, results = mediapipe_detection(frame, holistic)
        draw_landmarks(image, results)

        # Extract keypoints from MediaPipe results
        keypoints = extract_keypoints(results)
        inputs = torch.tensor(keypoints, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            outputs = model(inputs)

        _, predicted_label = torch.max(outputs, 1)
        predicted_label = predicted_label.item()

        predicted_word = str(index_to_word(predicted_label))

        from deep_translator import GoogleTranslator

        to_translate = predicted_word
        translated = GoogleTranslator(source="auto", target="en").translate(
            to_translate
        )

        print(predicted_word, "====>", translated)

        reshaped_text = arabic_reshaper.reshape(predicted_word)
        bidi_text = get_display(reshaped_text)

        font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
        img_pil = Image.fromarray(image)
        draw = ImageDraw.Draw(img_pil)

        text_start_x = 50
        text_start_y = 80
        text_spacing = 40

        draw.text((text_start_x, text_start_y), bidi_text, font=font, fill=FONT_COLOR)

        # Uncomment this line to display the translated(english) word in the frame. in realtime, its slow.
        # draw.text((text_start_x, text_start_y + text_spacing),translated,font=font,fill=FONT_COLOR,)

        img = np.array(img_pil)
        cv2.imshow("Sign Language Detection", img)

        # # Break the loop if 'q' is pressed
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

    capture.release()
cv2.destroyAllWindows()

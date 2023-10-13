import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import numpy

model_path = 'gesture_recognizer.task'

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.GestureRecognizerOptions(base_options=base_options)
recognizer = vision.GestureRecognizer.create_from_options(options)

cap = cv2.VideoCapture(0) # 0 for webcam

while True:
    data, image = cap.read()

    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB) # flip the image horizontally
    
    # image_data = image.tobytes()
    image_format = vision.ImageFormat.Format.SRGB

    recognition_result = recognizer.recognize(image, image_format)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # convert the image back to BGR

    #results = landmarks (joints, fingertips, palm - all 21 landmarks) in 2d coordinates, handedness (left or right), handedness_confidence (whether hand is detected or not)

    top_gesture = recognition_result.gestures[0][0]
    hand_landmarks = recognition_result.hand_landmarks
            
    cv2.imshow('Gesture Recognizer', image) # show the image
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
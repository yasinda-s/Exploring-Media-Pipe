import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils # module for drawing utilities (for visualization purposes)
mp_drawing_styles = mp.solutions.drawing_styles # module for editing the drawing styles in drawing_utils
mp_hands = mp.solutions.hands # module for hand tracking

cap = cv2.VideoCapture(0) # 0 for webcam
hands = mp_hands.Hands() # initialize the hand tracking module

while True:
    data, image = cap.read()
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB) # flip the image horizontally
    results = hands.process(image) # process the image for hand tracking
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # convert the image back to BGR

    #results = landmarks (joints, fingertips, palm - all 21 landmarks) in 2d coordinates, handedness (left or right), handedness_confidence (whether hand is detected or not)

    if results.multi_hand_landmarks: 
        for hand_landmarks in results.multi_hand_landmarks: # iterate through all the hands detected
            mp_drawing.draw_landmarks( # draw the landmarks on the image
                image, hand_landmarks, 
                mp_hands.HAND_CONNECTIONS # draw the connections between the landmarks
                )
            
    cv2.imshow('Hand Tracking', image) # show the image
    cv2.waitKey(1) 
    



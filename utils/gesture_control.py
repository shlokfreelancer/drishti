import cv2
import mediapipe as mp
import numpy as np
from hardware import speaker
from modules import text_reader, object_detection, face_recognition

# Initialize MediaPipe Hand Tracking
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Gesture Commands
GESTURE_COMMANDS = {
    "thumbs_up": "read_text",
    "open_palm": "identify_object",
    "victory": "recognize_face",
    "fist": "exit",
}

def detect_gesture(frame):
    """ Detect hand gestures and return a command. """
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract key points
            landmarks = []
            for lm in hand_landmarks.landmark:
                h, w, c = frame.shape
                landmarks.append((int(lm.x * w), int(lm.y * h)))

            # Simple gesture recognition (example)
            if landmarks[4][1] < landmarks[3][1]:  # Thumb up
                return "thumbs_up"
            elif landmarks[8][1] < landmarks[6][1] and landmarks[12][1] < landmarks[10][1]:  # Open Palm
                return "open_palm"
            elif landmarks[12][1] < landmarks[10][1] and landmarks[16][1] < landmarks[14][1]:  # Victory
                return "victory"
            elif all(landmarks[i][1] > landmarks[0][1] for i in range(1, 5)):  # Fist
                return "fist"
    
    return None

def process_gesture_control():
    """ Real-time gesture control processing. """
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        
        gesture = detect_gesture(frame)
        
        if gesture and gesture in GESTURE_COMMANDS:
            command = GESTURE_COMMANDS[gesture]
            if command == "read_text":
                speaker.speak("Reading text...")
                text = text_reader.read_text()
                speaker.speak(text)
            elif command == "identify_object":
                speaker.speak("Identifying objects...")
                objects = object_detection.identify_object()
                if objects:
                    for obj in objects:
                        speaker.speak(f"Detected {obj}")
            elif command == "recognize_face":
                speaker.speak("Recognizing faces...")
                face = face_recognition.recognize_face()
                if face:
                    speaker.speak(f"Recognized {face}")
                else:
                    speaker.speak("No known faces detected.")
            elif command == "exit":
                speaker.speak("Shutting down gesture control.")
                break

        cv2.imshow("Gesture Control", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

import cv2
import threading
import time
from modules import (
    text_reader, face_recognition, object_detection, banknote_id, 
    color_detection, scene_description, obstacle_detection, ai_assistant
)
from hardware import camera, mic, speaker
from utils import gesture_control, voice_control

# Initialize Camera
cap = cv2.VideoCapture(0)  # Using laptop camera

# Function to process video frames
def process_video():
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        
        # Perform real-time processing
        detected_text = text_reader.read_text(frame)
        recognized_face = face_recognition.recognize_face(frame)
        detected_object = object_detection.identify_object(frame)
        detected_currency = banknote_id.detect_currency(frame)
        detected_color = color_detection.detect_color(frame)
        scene_info = scene_description.describe_scene(frame)
        obstacle_warning = obstacle_detection.detect_obstacles(frame)

        # Provide audio feedback
        if detected_text:
            speaker.speak(f"Text detected: {detected_text}")
        if recognized_face:
            speaker.speak(f"Face recognized: {recognized_face}")
        if detected_object:
            speaker.speak(f"Object identified: {detected_object}")
        if detected_currency:
            speaker.speak(f"Currency detected: {detected_currency}")
        if detected_color:
            speaker.speak(f"Color detected: {detected_color}")
        if scene_info:
            speaker.speak(f"Scene: {scene_info}")
        if obstacle_warning:
            speaker.speak(f"Warning: {obstacle_warning}")
            

        # Display frame (for debugging)
        cv2.imshow("Drishti Smart Glasses - Live Feed", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break  # Press 'q' to quit

# Function to listen for voice commands
def listen_for_commands():
    while True:
        command = mic.listen()
        ai_assistant.handle_command(command)

# Function to process gesture controls
def listen_for_gestures():
    while True:
        gesture = gesture_control.detect_gesture()
        ai_assistant.handle_command(gesture)

# Start all functionalities in parallel
video_thread = threading.Thread(target=process_video)
audio_thread = threading.Thread(target=listen_for_commands)
gesture_thread = threading.Thread(target=listen_for_gestures)

video_thread.start()
audio_thread.start()
gesture_thread.start()

video_thread.join()
audio_thread.join()
gesture_thread.join()

cap.release()
cv2.destroyAllWindows()

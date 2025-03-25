import cv2
import torch
import numpy as np
import os
import json
import speech_recognition as sr
import pyttsx3
from ultralytics import YOLO
from hardware import speaker

# Load YOLOv8 model
yolo_model = YOLO("yolov8n.pt")  # Using the smallest YOLOv8 model for efficiency

# Initialize AI Assistant
recognizer = sr.Recognizer()
engine = pyttsx3.init()

# File to store learned objects
LEARNED_OBJECTS_FILE = "learned_objects.json"

# Load or initialize learned objects
def load_learned_objects():
    if os.path.exists(LEARNED_OBJECTS_FILE):
        with open(LEARNED_OBJECTS_FILE, "r") as file:
            return json.load(file)
    return {}

def save_learned_objects(objects):
    with open(LEARNED_OBJECTS_FILE, "w") as file:
        json.dump(objects, file)

learned_objects = load_learned_objects()

def identify_object(frame):
    """ Detects objects in the frame and returns detected object names. """
    results = yolo_model(frame)
    detected_objects = []
    
    for result in results:
        boxes = result.boxes  # Bounding boxes
        names = result.names  # Object names
        
        for box in boxes:
            class_id = int(box.cls.item())  # Class ID
            object_name = names[class_id]  # Object name
            detected_objects.append(object_name)
    
    return detected_objects if detected_objects else None

# AI Assistant Function
def ai_assistant():
    with sr.Microphone() as source:
        print("Listening for commands...")
        recognizer.adjust_for_ambient_noise(source)
        try:
            audio = recognizer.listen(source)
            command = recognizer.recognize_google(audio).lower()
            print(f"Command: {command}")
            
            if "identify object" in command:
                engine.say("Identifying objects.")
                engine.runAndWait()
            elif "learn object" in command:
                engine.say("Look at the object and say its name.")
                engine.runAndWait()
            elif "list objects" in command:
                engine.say("Listing learned objects.")
                for obj, name in learned_objects.items():
                    engine.say(f"{name}")
                engine.runAndWait()
            elif "exit" in command:
                engine.say("Shutting down.")
                engine.runAndWait()
                return False
        except sr.UnknownValueError:
            print("Could not understand the command.")
        except sr.RequestError:
            print("Error with voice recognition service.")
    return True

# Real-Time Object Detection with Learning & Obstacle Detection
cap = cv2.VideoCapture(0)
running = True
while running:
    ret, frame = cap.read()
    if not ret:
        continue
    
    objects = identify_object(frame)
    
    if objects:
        for obj in objects:
            if obj in learned_objects:
                speaker.speak(f"Detected {learned_objects[obj]}")
            else:
                speaker.speak(f"Detected {obj}")
    
    # Obstacle Detection using Edge Detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    
    obstacle_percentage = np.sum(edges) / edges.size
    if obstacle_percentage > 0.1:  # Arbitrary threshold for obstacle detection
        speaker.speak("Obstacle detected ahead. Please be careful.")
    
    cv2.imshow("Object Detection & Obstacle Detection", frame)
    key = cv2.waitKey(1) & 0xFF
    
    # Press 'L' to learn a new object
    if key == ord('l'):
        print("Look at the object and enter its name: ")
        obj_name = input("Enter object name: ")
        for obj in objects:
            learned_objects[obj] = obj_name
        save_learned_objects(learned_objects)
        print("Object learned!")
    
    # Press 'Q' to quit
    if key == ord('q'):
        break
    
    # Listen for voice commands
    running = ai_assistant()

cap.release()
cv2.destroyAllWindows()

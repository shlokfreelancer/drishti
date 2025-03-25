import cv2
import face_recognition
import numpy as np
import os
import pickle

# Directory to store known face encodings
FACE_DB = "known_faces.pkl"

# Load existing face database
if os.path.exists(FACE_DB):
    with open(FACE_DB, "rb") as f:
        known_faces = pickle.load(f)
else:
    known_faces = {}

def save_faces():
    """ Save learned faces to disk """
    with open(FACE_DB, "wb") as f:
        pickle.dump(known_faces, f)

def recognize_face(frame):
    """ Recognize faces in a frame """
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    
    recognized_names = []
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(list(known_faces.values()), face_encoding, tolerance=0.5)
        name = "Unknown"
        
        if True in matches:
            first_match_index = matches.index(True)
            name = list(known_faces.keys())[first_match_index]
        
        recognized_names.append(name)
    
    return face_locations, recognized_names

def learn_new_face(frame, name):
    """ Learn and store a new face """
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    
    if face_encodings:
        known_faces[name] = face_encodings[0]
        save_faces()
        print(f"Saved new face: {name}")

# Real-time Face Recognition
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        continue
    
    face_locations, names = recognize_face(frame)
    
    for (top, right, bottom, left), name in zip(face_locations, names):
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    cv2.imshow("Face Recognition", frame)
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('l'):
        new_name = input("Enter name for new face: ")
        learn_new_face(frame, new_name)
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

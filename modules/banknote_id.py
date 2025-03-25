import cv2
import torch
import numpy as np
import os
import json
import pyttsx3
from ultralytics import YOLO
from hardware import speaker

# Load YOLOv8 currency detection model (Train a custom model with currency images)
currency_model = YOLO("currency_model.pt")  # Replace with actual trained model

# Initialize text-to-speech engine
engine = pyttsx3.init()

# File to store learned currencies
LEARNED_CURRENCY_FILE = "learned_currency.json"

# Load or initialize learned currency data
def load_learned_currency():
    if os.path.exists(LEARNED_CURRENCY_FILE):
        with open(LEARNED_CURRENCY_FILE, "r") as file:
            return json.load(file)
    return {}

def save_learned_currency(currency_data):
    with open(LEARNED_CURRENCY_FILE, "w") as file:
        json.dump(currency_data, file)

learned_currency = load_learned_currency()

def recognize_currency(frame):
    """Detects currency notes in the frame and returns detected denominations."""
    results = currency_model(frame)
    detected_currency = []

    for result in results:
        boxes = result.boxes  # Bounding boxes
        names = result.names  # Currency names

        for box in boxes:
            class_id = int(box.cls.item())  # Class ID
            currency_name = names[class_id]  # Currency denomination
            detected_currency.append(currency_name)

    return detected_currency if detected_currency else None

def process_currency_recognition():
    """ Continuously detects currency notes in real-time and announces them. """
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        currencies = recognize_currency(frame)

        if currencies:
            for currency in currencies:
                if currency in learned_currency:
                    speaker.speak(f"Detected {learned_currency[currency]}")
                else:
                    speaker.speak(f"Detected {currency}")

        cv2.imshow("Currency Recognition", frame)
        key = cv2.waitKey(1) & 0xFF

        # Press 'L' to learn a new currency
        if key == ord('l'):
            print("Hold the currency note in front of the camera and enter its value:")
            currency_value = input("Enter currency value: ")
            for currency in currencies:
                learned_currency[currency] = currency_value
            save_learned_currency(learned_currency)
            print("Currency learned!")

        # Press 'Q' to quit
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

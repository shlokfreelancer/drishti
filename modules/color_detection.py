import cv2
import pandas as pd
import numpy as np
from hardware import speaker  # Import speaker for audio output

# Load color dataset
color_data = pd.read_csv("assets/colors.csv")

def get_color_name(R, G, B):
    """Find the closest matching color name based on RGB values."""
    min_dist = float('inf')
    closest_color = "Unknown"

    for index, row in color_data.iterrows():
        dist = np.sqrt((R - row['R'])**2 + (G - row['G'])**2 + (B - row['B'])**2)
        if dist < min_dist:
            min_dist = dist
            closest_color = row['color']

    return closest_color

def detect_color(frame):
    """Detect color in the center of the frame."""
    height, width, _ = frame.shape
    center_pixel = frame[height // 2, width // 2]
    R, G, B = center_pixel

    detected_color = get_color_name(R, G, B)
    
    speaker.speak(f"Detected color: {detected_color}")
    return detected_color

# Real-time color detection
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    detected_color = detect_color(frame)

    cv2.putText(frame, f"Color: {detected_color}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.imshow("Color Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

import cv2
import pytesseract
import numpy as np

# Configure Tesseract path (only needed for Windows)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def preprocess_image(frame):
    """
    Preprocesses the image to enhance OCR accuracy.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding for better text detection
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 31, 10)
    
    # Use morphological operations to remove noise
    kernel = np.ones((1, 1), np.uint8)
    processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    return processed

def read_text(frame):
    """
    Extracts text from a given frame using Tesseract OCR.
    """
    try:
        processed_frame = preprocess_image(frame)
        
        # List of 40 languages for OCR
        languages = "eng+hin+tam+fra+spa+deu+ita+rus+ara+jpn+kor+chi_sim+chi_tra+por+tur+dut+swe+dan+fin+gre+heb+ind+tha+vie+pol+hun+ces+slk+bul+ron+ukr+hrv+srp+lit+lav+est+mlt+isl+cat+glg"
        
        # Perform OCR with multiple language support
        extracted_text = pytesseract.image_to_string(processed_frame, lang=languages)
        
        return extracted_text.strip() if extracted_text.strip() else None
    
    except Exception as e:
        print(f"Error in OCR: {e}")
        return None

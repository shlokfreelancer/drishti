import cv2
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration, MarianMTModel, MarianTokenizer
from hardware import speaker

# Load AI Model (Optimized)
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")  
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Load Translation Model (English → Other Languages)
translator_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-multi")
translator_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-multi")

def translate_text(text, target_lang="fr"):  # Change "fr" to any language code (e.g., "hi" for Hindi)
    """ Translates the scene description to another language. """
    inputs = translator_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    translated = translator_model.generate(**inputs)
    return translator_tokenizer.decode(translated[0], skip_special_tokens=True)

def describe_scene(frame, target_lang="en"):
    """ Generates a description of the current scene and translates if needed. """
    try:
        # Convert frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process image
        inputs = processor(images=frame_rgb, return_tensors="pt")
        caption_ids = model.generate(**inputs)
        description = processor.decode(caption_ids[0], skip_special_tokens=True)

        # Translate if needed
        if target_lang != "en":
            description = translate_text(description, target_lang)

        return description
    except Exception as e:
        return f"Error in scene description: {str(e)}"

def process_scene_description(target_lang="en"):
    """ Runs real-time scene description with translation. """
    cap = cv2.VideoCapture(0)

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        
        if frame_count % 5 == 0:  # Process every 5th frame for better performance
            description = describe_scene(frame, target_lang)
            speaker.speak(f"Scene: {description}")

        cv2.imshow("Scene Description", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

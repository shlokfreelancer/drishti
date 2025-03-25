import speech_recognition as sr
import pyttsx3
from googletrans import Translator
from hardware import speaker
from modules import text_reader, object_detection, face_recognition

# Initialize voice engine and translator
engine = pyttsx3.init()
translator = Translator()
recognizer = sr.Recognizer()

# Supported languages
SUPPORTED_LANGUAGES = {
    "english": "en",
    "hindi": "hi",
    "spanish": "es",
    "french": "fr",
    "german": "de",
    "tamil": "ta",
    "telugu": "te",
    "chinese": "zh-cn",
    "japanese": "ja",
    "arabic": "ar",
    "russian": "ru",
}

current_language = "en"  # Default language

def speak(text, lang="en"):
    """ Converts text to speech in the given language. """
    translated_text = translator.translate(text, dest=lang).text
    speaker.speak(translated_text)

def listen_command():
    """ Listens for a voice command and returns recognized text. """
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)
        print("Listening...")
        try:
            audio = recognizer.listen(source)
            command = recognizer.recognize_google(audio).lower()
            print(f"Command: {command}")
            return command
        except sr.UnknownValueError:
            return None
        except sr.RequestError:
            return None

def process_voice_command():
    """ Listens for commands and triggers appropriate actions. """
    global current_language

    while True:
        command = listen_command()
        if not command:
            continue
        
        if "read text" in command:
            speak("Reading text...", current_language)
            text = text_reader.read_text()
            speak(text, current_language)
        
        elif "identify object" in command:
            speak("Identifying objects...", current_language)
            objects = object_detection.identify_object()
            if objects:
                for obj in objects:
                    speak(f"Detected {obj}", current_language)
            else:
                speak("No objects detected.", current_language)

        elif "recognize face" in command:
            speak("Recognizing faces...", current_language)
            face = face_recognition.recognize_face()
            if face:
                speak(f"Recognized {face}", current_language)
            else:
                speak("No known faces detected.", current_language)

        elif "change language" in command:
            speak("Which language do you want?", current_language)
            new_lang = listen_command()
            if new_lang in SUPPORTED_LANGUAGES:
                current_language = SUPPORTED_LANGUAGES[new_lang]
                speak(f"Language changed to {new_lang}", current_language)
            else:
                speak("Language not supported.", current_language)

        elif "exit" in command:
            speak("Shutting down...", current_language)
            break

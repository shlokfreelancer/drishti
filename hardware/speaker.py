import pyttsx3

# Initialize Text-to-Speech Engine
engine = pyttsx3.init()

def speak(text):
    """Converts text to speech and speaks it."""
    engine.say(text)
    engine.runAndWait()

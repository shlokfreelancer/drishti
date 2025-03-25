import speech_recognition as sr

# Initialize the recognizer
recognizer = sr.Recognizer()

def listen():
    """Captures voice commands using the microphone and converts them to text."""
    with sr.Microphone() as source:
        print("Listening for a command...")
        recognizer.adjust_for_ambient_noise(source, duration=1)  # Reduce background noise
        try:
            audio = recognizer.listen(source, timeout=5)  # Listen for 5 seconds max
            command = recognizer.recognize_google(audio).lower()
            print(f"Command received: {command}")
            return command
        except sr.UnknownValueError:
            print("Could not understand the command.")
            return None
        except sr.RequestError:
            print("Error connecting to the speech recognition service.")
            return None
        except sr.WaitTimeoutError:
            print("Listening timeout reached. No command received.")
            return None

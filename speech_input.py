# speech_input.py
import speech_recognition as sr

def get_user_query():
    """
    Listens to user's voice and converts it into text.
    Returns:
        str: User's query in text
    """
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Please speak your query:")
        r.adjust_for_ambient_noise(source)
        audio = r.listen(source)
    try:
        query = r.recognize_google(audio)
        print("You said:", query)
        return query
    except sr.UnknownValueError:
        print("Sorry, could not understand your speech.")
        return None
    except sr.RequestError as e:
        print("Could not request results; {0}".format(e))
        return None

# # # # # speech_input.py
# # # # import speech_recognition as sr

# # # # def get_user_query():
# # # #     """
# # # #     Listens to user's voice and converts it into text.
# # # #     Returns:
# # # #         str: User's query in text
# # # #     """
# # # #     r = sr.Recognizer()
# # # #     with sr.Microphone() as source:
# # # #         print("Please speak your query:")
# # # #         r.adjust_for_ambient_noise(source)
# # # #         audio = r.listen(source)
# # # #     try:
# # # #         query = r.recognize_google(audio)
# # # #         print("You said:", query)
# # # #         return query
# # # #     except sr.UnknownValueError:
# # # #         print("Sorry, could not understand your speech.")
# # # #         return None
# # # #     except sr.RequestError as e:
# # # #         print("Could not request results; {0}".format(e))
# # # #         return None


# # from google import genai
# # import os

# # client = genai.Client(api_key=os.getenv("GEMINI_LLM_KEY"))

# # # List available models
# # models = client.models.list()
# # for m in models:
# #     print(m.name)
# models/gemini-2.5-flash
# models/gemini-2.5-pro
# models/gemini-2.0-flash
# models/gemini-2.0-flash-001
# models/gemini-2.0-flash-exp-image-generation
# models/gemini-2.0-flash-lite-001
# models/gemini-2.0-flash-lite
# models/gemini-exp-1206
# models/gemini-2.5-flash-preview-tts
# models/gemini-2.5-pro-preview-tts
# models/gemma-3-1b-it
# models/gemma-3-4b-it
# models/gemma-3-12b-it
# models/gemma-3-27b-it
# models/gemma-3n-e4b-it
# models/gemma-3n-e2b-it
# models/gemini-flash-latest
# models/gemini-flash-lite-latest
# models/gemini-pro-latest
# models/gemini-2.5-flash-lite
# models/gemini-2.5-flash-image
# models/gemini-2.5-flash-preview-09-2025
# models/gemini-2.5-flash-lite-preview-09-2025
# models/gemini-3-pro-preview
# models/gemini-3-flash-preview
# models/gemini-3-pro-image-preview
# models/nano-banana-pro-preview
# models/gemini-robotics-er-1.5-preview
# models/gemini-2.5-computer-use-preview-10-2025
# models/deep-research-pro-preview-12-2025
# models/embedding-001
# models/text-embedding-004
# models/gemini-embedding-001
# models/aqa






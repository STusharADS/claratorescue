import pyttsx3

engine = pyttsx3.init('espeak')
engine.setProperty('rate', 150)

voices = engine.getProperty('voices')

engine.setProperty('voice', voices[26].id) 
# engine.setProperty('pitch', 30)

engine.say("This is a test of the male voice variant. pneumonoultramicroscopicsilicovolcanoconiosis")
engine.runAndWait()

# import pyttsx3

# engine = pyttsx3.init('espeak')
# engine.setProperty('rate', 160)

# # Use a female variant of English
# engine.setProperty('voice', 'en+f1')  # Try f1, f2, f3, f4
# engine.setProperty('pitch', 60)

# engine.say("This is a test of the female voice variant.")
# engine.runAndWait()

# import pyttsx3

# engine = pyttsx3.init('espeak')
# engine.setProperty('rate', 160)

# # Try different female variants:
# female_voices = ['en+f1', 'en+f2', 'en+f3', 'en+f4', 'en-us+f1', 'en-us+f2', 'en-us+f3', 'english+f1', 'english+f2', 'english+f3', 'english_rp+f3', 'english_rp+f4']

# for voice_code in female_voices:
#     try:
#         engine.setProperty('voice', voice_code)
#         print(f"Testing voice: {voice_code}")
#         engine.say(f"This is a test of voice {voice_code}")
#         engine.runAndWait()
#     except:
#         print(f"Voice {voice_code} not available")
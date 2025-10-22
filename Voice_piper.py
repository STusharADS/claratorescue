import sounddevice as sd
import numpy as np
from piper import PiperVoice

# --- Configuration ---
# 1. Download voice model files from: https://huggingface.co/rhasspy/piper-voices/tree/main
#    For this example, download: en_US-lessac-medium.onnx and en_US-lessac-medium.onnx.json
# 2. Place them in a folder (e.g., 'models')
VOICE_MODEL_PATH = './models/en_US-lessac-medium.onnx' # Change this path

print("Loading Piper voice model...")
voice = PiperVoice.from_onnx(VOICE_MODEL_PATH)
print("Model loaded.")

def speak_piper(text_to_speak):
    """
    Uses Piper to generate speech and play it directly.
    """
    print(f"Speaking: {text_to_speak}")
    try:
        # Synthesize audio from text
        # The result is raw audio data (a bytes object)
        audio_bytes = b"".join(voice.synthesize_stream_raw(text_to_speak))
        
        # Convert bytes to a numpy array for playback
        audio_np = np.frombuffer(audio_bytes, dtype=np.int16)
        
        # Get the sample rate from the model's config
        sample_rate = voice.config.sample_rate
        
        # Play the audio
        sd.play(audio_np, samplerate=sample_rate)
        sd.wait()
        
    except Exception as e:
        print(f"An error occurred: {e}")

# --- Example of Real-time Usage ---
speak_piper("Hello, my name is Piper. I am a fast and local text to speech engine.")
speak_piper("My response time is very low, making me great for real-time assistants.")
# pip install opencv-python websockets
import asyncio, websockets, json, threading, cv2
from dotenv import load_dotenv  # <-- Added
import os  # <-- Added

# Load environment variables from .env file
load_dotenv()  # <-- Added

# If you use Gemini API in this file, get the key like:
# GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

ESP_IP = "192.168.1.xxx"

def video_thread():
    cap = cv2.VideoCapture(f"http://{ESP_IP}:81/stream")
    if not cap.isOpened():
        print("Could not open video stream"); return
    while True:
        ok, frame = cap.read()
        if not ok: break
        # TODO: your detection here
        cv2.imshow("ESP32-CAM", frame)
        if cv2.waitKey(1) == 27: break  # ESC
    cap.release(); cv2.destroyAllWindows()

async def telemetry():
    uri = f"ws://{ESP_IP}:82"
    async with websockets.connect(uri, ping_interval=20, ping_timeout=20) as ws:
        async for msg in ws:
            data = json.loads(msg)   # {"d":<mm>, "b":0/1}
            print(data)

if __name__ == "__main__":
    threading.Thread(target=video_thread, daemon=True).start()
    asyncio.run(telemetry())




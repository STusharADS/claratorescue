import cv2
import time
import torch
import pyttsx3
import requests
import numpy as np
import google.generativeai as genai
import os
from threading import Thread, Lock
from ultralytics import YOLO
from PIL import Image

# -------------------------------
# CONFIGURATION
# -------------------------------
ESP32_IP = "172.20.180.190"  # Replace with your ESP32's IP
DISTANCE_ENDPOINT = f"http://{ESP32_IP}/distance"
DISTANCE_UPDATE_INTERVAL = 1.0  # seconds

# --- SECURITY IMPROVEMENT ---
# Load API key from environment variable for better security
# For quick testing, you can hardcode it like this, but be careful.
GEMINI_API_KEY = "AIzaSyAm9XWsI_kDGmtUehlE5BpcDGWkdfeEVXc" 

# Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)
# Use gemini-1.5-flash for faster and cheaper responses in real-time applications
model = genai.GenerativeModel("gemini-1.5-flash") 

# Setup device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Text-to-speech engine
try:
    engine = pyttsx3.init(driverName='espeak') # Force espeak on Linux for better performance
except Exception:
    engine = pyttsx3.init() # Default for other OS

# ===================================================================
# MODIFIED SECTION: Auto-select Indian Accent Voice
# ===================================================================
voices = engine.getProperty('voices')
selected_voice = None

for voice in voices:
    if "en-in" in voice.id.lower() or "hindi" in voice.id.lower() or "indian" in voice.name.lower():
        selected_voice = voice.id
        break

if selected_voice:
    engine.setProperty('voice', selected_voice)
    print(f"Using voice: {selected_voice}")
else:
    print("Indian voice not found, using default voice.")
# ===================================================================

engine.setProperty('rate', 160) # Slightly slower rate for clarity

# Cooldown for announcements
last_spoken = {}
cooldown = 5  # seconds

# Load YOLO model
model_yolo = YOLO("yolov8n.pt")
model_yolo.conf = 0.5
names = model_yolo.names

ignore_classes = ["suitcase", "toothbrush", "handbag", "refrigerator",
                  "hair drier", "hair dryer", "teddy bear", "toilet", "airplane", "aeroplane"]

# Setup video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

frame_lock = Lock()
current_frame = None
pause_yolo = False

# Distance tracking
last_distance_update = 0
current_distance = -1.0

# -------------------------------
# FUNCTIONS
# -------------------------------
def get_distance_from_esp32():
    global current_distance
    try:
        response = requests.get(DISTANCE_ENDPOINT, timeout=1.0)
        if response.status_code == 200:
            data = response.json()
            current_distance = data.get("distance_cm", -1.0)
    except requests.exceptions.RequestException as e:
        if current_distance != -1.0:
           print(f"Could not update distance from ESP32: {e}")
        current_distance = -1.0


def calculate_quadrant(cx, cy, w, h):
    col = min(2, cx // (w // 3))
    row = min(2, cy // (h // 3))
    return f"Quadrant {row * 3 + col + 1}"

def draw_grid(frame, quadrant_counts):
    h, w, _ = frame.shape
    for i in range(1, 3):
        cv2.line(frame, (i * w // 3, 0), (i * w // 3, h), (0, 0, 255), 1)
        cv2.line(frame, (0, i * h // 3), (w, i * h // 3), (0, 0, 255), 1)
    for r in range(3):
        for c in range(3):
            q_num = r * 3 + c + 1
            q_name = f"Quadrant {q_num}"
            count = quadrant_counts.get(q_name, 0)
            cv2.putText(frame, f"Q{q_num}: {count}",
                        (c * w // 3 + 10, r * h // 3 + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

def announce_object(obj_name, quadrant, dist):
    def speak():
        now = time.time()
        key = (obj_name, quadrant)
        if now - last_spoken.get(key, 0) >= cooldown:
            dist_str = f"{dist:.0f} centimeters away" if dist > 0 else ""
            
            # Remove the word "Quadrant " from the string before speaking
            spoken_quadrant = quadrant.replace("Quadrant ", "")
            
            message = f"{obj_name} {spoken_quadrant}"
            engine.say(message)
            engine.runAndWait()
            last_spoken[key] = now
    Thread(target=speak).start()

def describe_with_gemini(frame):
    """
    Captures a frame, sends it to the Gemini API for a description,
    and speaks the result. This version is corrected to work with the Python SDK.
    """
    global pause_yolo
    print("Pausing YOLO and generating Gemini description...")
    pause_yolo = True

    try:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb_frame)
        response = model.generate_content(["Describe this scene in brief from the perspective of a person looking at it in 20 Words", img])
        description = response.text
        print("Gemini Description:", description)
        def speak_description():
            engine.say(description)
            engine.runAndWait()
        
        Thread(target=speak_description).start()

    except Exception as e:
        error_message = f"Error with Gemini API: {e}"
        print(error_message)
        engine.say("Sorry, I could not describe the scene.")
        engine.runAndWait()

    finally:
        pause_yolo = False
        print("Resuming YOLO detection.")

# -------------------------------
# MAIN LOOP
# -------------------------------
fps_time = time.time()
distance_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    with frame_lock:
        current_frame = frame.copy()

    now = time.time()
    if now - distance_time >= DISTANCE_UPDATE_INTERVAL:
        Thread(target=get_distance_from_esp32).start()
        distance_time = now

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("Quitting...")
        break
    elif key == ord('c'):
        with frame_lock:
            snapshot = current_frame.copy()
        Thread(target=describe_with_gemini, args=(snapshot,)).start()
        cv2.imshow("Smart Object Tracker", current_frame)
        continue

    if pause_yolo:
        cv2.putText(current_frame, "Generating Description...", (150, 200), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Smart Object Tracker", current_frame)
        continue

    results = model_yolo.predict(source=current_frame, verbose=False)[0]
    boxes = results.boxes

    quadrant_counts = {}
    h, w, _ = current_frame.shape

    if boxes is not None and len(boxes) > 0:
        detections = []
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            name = names[cls_id]

            if name in ignore_classes:
                continue

            area = (x2 - x1) * (y2 - y1)
            detections.append({"name": name, "coords": (x1, y1, x2, y2), "area": area})

        top_detections = sorted(detections, key=lambda x: x["area"], reverse=True)[:3]

        for det in top_detections:
            x1, y1, x2, y2 = det["coords"]
            name = det["name"]
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            quadrant = calculate_quadrant(cx, cy, w, h)
            
            quadrant_counts[quadrant] = quadrant_counts.get(quadrant, 0) + 1
            announce_object(name, quadrant, current_distance)

            cv2.rectangle(current_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(current_frame, (cx, cy), 4, (0, 0, 255), -1)
            cv2.putText(current_frame, name, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    draw_grid(current_frame, quadrant_counts)

    now = time.time()
    fps = 1 / (now - fps_time)
    fps_time = now
    cv2.putText(current_frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    dist_text = f"Distance: {current_distance:.1f} cm" if current_distance > 0 else "Distance: N/A"
    cv2.putText(current_frame, dist_text, (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("Smart Object Tracker", current_frame)

cap.release()
cv2.destroyAllWindows()

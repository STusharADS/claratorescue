import cv2
import numpy as np
import os
import pickle
import time
import torch
import pyttsx3
import requests
import google.generativeai as genai
from threading import Thread, Lock
from ultralytics import YOLO
from PIL import Image
from dotenv import load_dotenv


VIDEO_ENDPOINT = "http://10.151.122.93:81/stream"
ESP32_IP = "http://10.151.122.128"
DATA_ENDPOINT = f"http://{ESP32_IP}/data"
DISTANCE_UPDATE_INTERVAL = 1.0

VIBRATE_ENDPOINT = f"http://{ESP32_IP}/m"
VIBRATE_THRESHOLD = 30.0
VIBRATE_COOLDOWN = 2.0
last_vibrate_time = 0


load_dotenv()


GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-1.5-flash")


try:
    engine = pyttsx3.init(driverName='espeak')
except Exception:
    engine = pyttsx3.init()

engine.setProperty('rate', 160)


last_spoken = {}
cooldown = 5


# Load the optimized OpenVINO model
yolo_model = YOLO("yolov8n_openvino_model/")
yolo_model.conf = 0.5
names = yolo_model.names
# "person" has been removed from this list to enable general person detection
ignore_classes = ["suitcase", "toothbrush", "handbag", "refrigerator",
                  "hair drier", "hair dryer", "teddy bear", "toilet", "airplane", "aeroplane"]


# Video capture
cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture(VIDEO_ENDPOINT)
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

frame_lock = Lock()
current_frame = None
pause_detection = False


last_distance_update = 0
current_distance = -1.0
current_button = "N/A"
previous_button = "RELEASED"


# Modified button layout since "Register" is removed
BUTTON_HEIGHT = 50
BUTTON_WIDTH = 640 // 2
QUIT_RECT = (0, 360 - BUTTON_HEIGHT, BUTTON_WIDTH, 360)
DESCRIBE_RECT = (BUTTON_WIDTH, 360 - BUTTON_HEIGHT, 2 * BUTTON_WIDTH, 360)


quit_flag = False
describe_flag = False


def get_distance_from_esp32():
    global current_distance, current_button
    try:
        response = requests.get(DATA_ENDPOINT, timeout=1.0)
        if response.status_code == 200:
            data = response.json()
            current_distance = data.get("d", -1.0)
            current_button = data.get("b", "N/A")
    except Exception as e:
        current_distance = -1.0
        current_button = "N/A"

def trigger_vibration():
    try:
        response = requests.get(VIBRATE_ENDPOINT, timeout=1.0)
        if response.status_code == 200:
            print("[INFO] Vibration triggered.")
    except Exception as e:
        print(f"[WARN] Could not trigger vibration: {e}")

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
            count = quadrant_counts.get(f"Quadrant {q_num}", 0)
            cv2.putText(frame, f"Q{q_num}: {count}", (c * w // 3 + 10, r * h // 3 + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

def announce_object(obj_name, quadrant, dist):
    def speak():
        now = time.time()
        key = (obj_name, quadrant)
        if now - last_spoken.get(key, 0) >= cooldown:
            dist_str = f"{dist:.0f}" if dist > 0 else ""
            spoken_quadrant = quadrant.replace("Quadrant ", "")
            message = f"{obj_name} {spoken_quadrant} {dist_str}"
            engine.say(message)
            engine.runAndWait()
            last_spoken[key] = now
    Thread(target=speak).start()

def describe_with_gemini(frame):
    global pause_detection
    print("Pausing detection and generating Gemini description...")
    pause_detection = True
    try:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb_frame)
        response = gemini_model.generate_content(["From a first person perspective describe the scene in this image and include any prominent text.limit the entire response to 25 words and directly start saying do not say I am describing or something like that. ", img])
        description = response.text
        print("Gemini Description:", description)
        def speak_description():
            engine.say(description)
            engine.runAndWait()
        Thread(target=speak_description).start()
    except Exception as e:
        print(f"Error with Gemini API: {e}")
        engine.say("Sorry, I could not describe the scene.")
        engine.runAndWait()
    finally:
        pause_detection = False
        print("Resuming detection.")

def on_mouse(event, x, y, flags, param):
    global quit_flag, describe_flag
    if event == cv2.EVENT_LBUTTONDOWN:
        if QUIT_RECT[0] <= x < QUIT_RECT[2] and QUIT_RECT[1] <= y < QUIT_RECT[3]:
            quit_flag = True
        elif DESCRIBE_RECT[0] <= x < DESCRIBE_RECT[2] and DESCRIBE_RECT[1] <= y < DESCRIBE_RECT[3]:
            describe_flag = True

def draw_buttons(frame):
    # Quit button
    cv2.rectangle(frame, (QUIT_RECT[0], QUIT_RECT[1]), (QUIT_RECT[2], QUIT_RECT[3]), (200, 200, 200), -1)
    cv2.putText(frame, "Quit", (QUIT_RECT[0] + 120, QUIT_RECT[1] + 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    # Describe button
    cv2.rectangle(frame, (DESCRIBE_RECT[0], DESCRIBE_RECT[1]), (DESCRIBE_RECT[2], DESCRIBE_RECT[3]), (200, 200, 200), -1)
    cv2.putText(frame, "Describe", (DESCRIBE_RECT[0] + 90, DESCRIBE_RECT[1] + 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)


print("[INFO] Press 'c' to describe scene, 'q' to quit. Or use on-screen buttons.")

fps_time = time.time()
distance_time = time.time()

cv2.namedWindow("Smart Recognition")
cv2.setMouseCallback("Smart Recognition", on_mouse)

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

    if current_distance > 0 and current_distance < VIBRATE_THRESHOLD and now - last_vibrate_time >= VIBRATE_COOLDOWN:
        Thread(target=trigger_vibration).start()
        last_vibrate_time = now

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("Quitting...")
        break
    elif key == ord('c'):
        describe_flag = True

    if previous_button == 'RELEASED' and current_button == 'PRESSED':
        describe_flag = True
    previous_button = current_button

    if quit_flag:
        print("Quitting...")
        break

    if describe_flag:
        with frame_lock:
            snapshot = current_frame.copy()
        Thread(target=describe_with_gemini, args=(snapshot,)).start()
        describe_flag = False
        cv2.imshow("Smart Recognition", current_frame)
        continue

    if pause_detection:
        cv2.putText(current_frame, "Generating Description...", (150, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Smart Recognition", current_frame)
        continue

    # The model is pre-configured for the GPU, so no device argument is needed here.
    yolo_results = yolo_model.predict(source=current_frame, verbose=False)[0]
    
    boxes = yolo_results.boxes
    object_detections = []
    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            name = names[cls_id]
            if name in ignore_classes:
                continue
            area = (x2 - x1) * (y2 - y1)
            object_detections.append({"name": name, "coords": (x1, y1, x2, y2), "area": area})

    h, w, _ = current_frame.shape
    quadrant_counts = {}

    top_objects = sorted(object_detections, key=lambda x: x["area"], reverse=True)[:3]
    for det in top_objects:
        x1, y1, x2, y2 = det["coords"]
        name = det["name"]
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        quadrant = calculate_quadrant(cx, cy, w, h)
        quadrant_counts[quadrant] = quadrant_counts.get(quadrant, 0) + 1
        announce_object(name, quadrant, current_distance)
        cv2.rectangle(current_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(current_frame, (cx, cy), 4, (0, 0, 255), -1)
        cv2.putText(current_frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    if (len(object_detections) > 0) and current_distance > 0 and current_distance < VIBRATE_THRESHOLD and now - last_vibrate_time >= VIBRATE_COOLDOWN:
        Thread(target=trigger_vibration).start()
        last_vibrate_time = now

    draw_grid(current_frame, quadrant_counts)

    now = time.time()
    fps = 1 / (now - fps_time)
    fps_time = now
    cv2.putText(current_frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    dist_text = f"Distance: {current_distance:.1f} cm" if current_distance > 0 else "Distance: N/A"
    cv2.putText(current_frame, dist_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    button_text = f"Button: {current_button}"
    cv2.putText(current_frame, button_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    draw_buttons(current_frame)

    cv2.imshow("Smart Recognition", current_frame)

cap.release()
cv2.destroyAllWindows()
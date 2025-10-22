import cv2
import time
import torch
import pyttsx3
import requests
import numpy as np
from PIL import Image
from threading import Thread, Lock
from ultralytics import YOLO
from transformers import Blip2Processor, Blip2ForConditionalGeneration

# ESP32 Configuration
ESP32_IP = "172.20.180.190"  # Replace with your ESP32's IP
DISTANCE_ENDPOINT = f"http://{ESP32_IP}/distance"
DISTANCE_UPDATE_INTERVAL = 1.0  # seconds

# Setup device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Text-to-speech engine
engine = pyttsx3.init(driverName='espeak')  # Force espeak backend
engine.setProperty('rate', 200)


# Cooldown for announcements
last_spoken = {}
cooldown = 5  # seconds

# Load YOLO model
# model_yolo = YOLO("yolov8x-seg.pt")
model_yolo = YOLO("yoloe-11l-seg-pf.pt")
model_yolo.conf = 0.5
names = model_yolo.names

ignore_classes = ["Suitcase", "Toothbrush", "Handbag","Refrigerator","Hair Drier","Hair Dryer","Teddy Bear", "airplane", "aeroplane"]

# Load BLIP model for image captioning
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model_blip = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b",
    device_map="auto",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
)

# Setup video capture
cap = cv2.VideoCapture(0)  # or use your stream URL
# cap = cv2.VideoCapture(f"http://{ESP32_IP}:81/stream")  # or use your stream URL
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

frame_lock = Lock()
current_frame = None
pause_yolo = False

# Distance tracking
last_distance_update = 0
current_distance = -1.0

def get_distance_from_esp32():
  global current_distance
  try:
      response = requests.get(DISTANCE_ENDPOINT, timeout=1.0)
      if response.status_code == 200:
          data = response.json()
          current_distance = data.get("distance_cm", -1.0)
          print(f"Distance: {current_distance} cm")
  except Exception as e:
      print(f"Error fetching distance: {e}")

def calculate_quadrant(cx, cy, w, h):
    return min(2, cx // (w // 3)) + 1 + 3 * min(2, cy // (h // 3))

def draw_grid(frame, quadrant_counts):
    h, w = frame.shape[:2]
    for i in range(1, 3):
        cv2.line(frame, (i * w // 3, 0), (i * w // 3, h), (0, 0, 255), 1)
        cv2.line(frame, (0, i * h // 3), (w, i * h // 3), (0, 0, 255), 1)
    for r in range(3):
        for c in range(3):
            q = r * 3 + c + 1
            cv2.putText(frame, f"Q{q}: {quadrant_counts[q]}", (c * w // 3 + 10, r * h // 3 + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

def announce_object(obj_name, quadrant, current_distance):
    def speak():
        now = time.time()
        key = (obj_name, quadrant)
        if now - last_spoken.get(key, 0) >= cooldown:
            engine.say(f"{quadrant} {obj_name} {current_distance}")
            engine.runAndWait()
            last_spoken[key] = now
    Thread(target=speak).start()

def describe_with_blip(frame):
    global pause_yolo
    pause_yolo = True

    rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_img)
    inputs = processor(images=pil_image, return_tensors="pt").to(device, torch.float16 if device == "cuda" else torch.float32)

    with torch.no_grad():
        generated_ids = model_blip.generate(**inputs, max_new_tokens=60)
        description = processor.decode(generated_ids[0], skip_special_tokens=True)

    print("BLIP Description:", description)
    engine.say(description)
    engine.runAndWait()

    pause_yolo = False

# FPS measurement
fps_time = time.time()
distance_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    with frame_lock:
        current_frame = frame.copy()

    # Check if it's time to update distance
    now = time.time()
    if now - distance_time >= DISTANCE_UPDATE_INTERVAL:
        Thread(target=get_distance_from_esp32).start()
        distance_time = now

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('c'):
        with frame_lock:
            snap = current_frame.copy()
        Thread(target=describe_with_blip, args=(snap,)).start()
        continue

    if pause_yolo:
        cv2.imshow("Smart Object Tracker", current_frame)
        continue

    rgb_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
    results = model_yolo.predict(source=rgb_frame, verbose=False)[0]
    boxes = results.boxes

    quadrant_counts = {i: 0 for i in range(1, 10)}
    h, w = current_frame.shape[:2]

    if boxes is not None and len(boxes) > 0:
        detections = []
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            name = names[cls_id]

            # Skip unwanted classes
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
            quadrant_counts[quadrant] += 1
            announce_object(name, quadrant, current_distance)

            cv2.rectangle(current_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(current_frame, (cx, cy), 4, (0, 0, 255), -1)
            cv2.putText(current_frame, name, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    draw_grid(current_frame, quadrant_counts)

    # Show FPS and distance
    now = time.time()
    fps = 1 / (now - fps_time)
    fps_time = now
    cv2.putText(current_frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(current_frame, f"Distance: {current_distance:.1f} cm", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("Smart Object Tracker", current_frame)

cap.release()
cv2.destroyAllWindows()
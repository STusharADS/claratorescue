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
from insightface.app import FaceAnalysis

# -------------------------------
# CONFIGURATION
# -------------------------------
ESP32_IP = "172.20.180.190"  # Replace with your ESP32's IP
DISTANCE_ENDPOINT = f"http://{ESP32_IP}/distance"
DISTANCE_UPDATE_INTERVAL = 1.0  # seconds

# Gemini API Key (use env var for security: os.getenv('GEMINI_API_KEY'))
GEMINI_API_KEY = "AIzaSyAm9XWsI_kDGmtUehlE5BpcDGWkdfeEVXc"
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Text-to-speech engine
try:
    engine = pyttsx3.init(driverName='espeak')  # Force espeak on Linux
except Exception:
    engine = pyttsx3.init()
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
engine.setProperty('rate', 160)

# Cooldown for announcements
last_spoken = {}
cooldown = 5  # seconds

# Models
# InsightFace for faces
face_app = FaceAnalysis(name="buffalo_l")
face_app.prepare(ctx_id=0 if device == "cuda" else -1, det_size=(640, 640))

# YOLO for objects
yolo_model = YOLO("yolov8n.pt")
yolo_model.conf = 0.5
names = yolo_model.names
ignore_classes = ["person", "suitcase", "toothbrush", "handbag", "refrigerator",
                  "hair drier", "hair dryer", "teddy bear", "toilet", "airplane", "aeroplane"]

# Face DB
face_db = {}
def save_db(path="face_db.pkl"):
    with open(path, "wb") as f:
        pickle.dump(face_db, f)

def load_db(path="face_db.pkl"):
    global face_db
    if os.path.exists(path):
        with open(path, "rb") as f:
            face_db = pickle.load(f)

def cosine(a, b):
    return float(np.dot(a, b))

def recognize_embedding(emb, threshold=0.60):
    best_name, best_score = "Person", -1.0
    for name, embs in face_db.items():
        ref = np.mean(np.vstack(embs), axis=0)
        score = cosine(emb, ref)
        if score > best_score:
            best_score, best_name = score, name
    print(f"[DEBUG] best={best_name} score={best_score:.3f}")
    return best_name if best_score >= threshold else "Person"

def add_sample(name, emb):
    emb = emb / np.linalg.norm(emb)
    face_db.setdefault(name, []).append(emb)

def build_face_db(dataset_dir="faces_dataset"):
    if not os.path.isdir(dataset_dir):
        return
    for person in os.listdir(dataset_dir):
        pdir = os.path.join(dataset_dir, person)
        if not os.path.isdir(pdir):
            continue
        added = 0
        for img_name in os.listdir(pdir):
            img = cv2.imread(os.path.join(pdir, img_name))
            if img is None:
                continue
            faces = face_app.get(img)
            if not faces:
                continue
            face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
            emb = getattr(face, "normed_embedding", face.embedding / np.linalg.norm(face.embedding))
            add_sample(person, emb)
            added += 1
        if added:
            print(f"[INFO] {person}: added {added} samples")

# Video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

frame_lock = Lock()
current_frame = None
pause_detection = False

# Distance
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
            current_distance = response.json().get("distance_cm", -1.0)
    except Exception as e:
        # print(f"Could not update distance: {e}")  # Commented out to avoid console spam
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
            count = quadrant_counts.get(f"Quadrant {q_num}", 0)
            cv2.putText(frame, f"Q{q_num}: {count}", (c * w // 3 + 10, r * h // 3 + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

def announce_object(obj_name, quadrant, dist):
    def speak():
        now = time.time()
        key = (obj_name, quadrant)
        if now - last_spoken.get(key, 0) >= cooldown:
            dist_str = f"{dist:.0f} centimeters away" if dist > 0 else ""
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
        response = gemini_model.generate_content(["Describe this scene in brief from the perspective of a person looking at it in 20 Words", img])
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

# -------------------------------
# MAIN LOOP
# -------------------------------
load_db()
build_face_db("faces_dataset")
save_db()

print("[INFO] Press 'r' to register face, 'c' to describe scene, 'q' to quit.")
last_faces = []  # For registration

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
    elif key == ord('r'):
        if not last_faces:
            print("[WARN] No face in frame to register.")
            continue
        face = max(last_faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
        emb = getattr(face, "normed_embedding", face.embedding / np.linalg.norm(face.embedding))
        name = input("Enter name: ").strip()
        if not name:
            print("[WARN] Empty name; skipped.")
            continue
        add_sample(name, emb)
        for _ in range(4):
            time.sleep(0.15)
            ok, fr = cap.read()
            if not ok: break
            f2 = face_app.get(fr)
            if not f2: continue
            face2 = max(f2, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
            emb2 = getattr(face2, "normed_embedding", face2.embedding / np.linalg.norm(face2.embedding))
            add_sample(name, emb2)
        save_db()
        print(f"[INFO] Registered {name} (now {len(face_db[name])} samples).")
        continue
    elif key == ord('c'):
        with frame_lock:
            snapshot = current_frame.copy()
        Thread(target=describe_with_gemini, args=(snapshot,)).start()
        cv2.imshow("Smart Recognition", current_frame)
        continue

    if pause_detection:
        cv2.putText(current_frame, "Generating Description...", (150, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Smart Recognition", current_frame)
        continue

    # Face processing
    faces = face_app.get(current_frame)
    last_faces = faces
    face_detections = []
    for face in faces:
        emb = getattr(face, "normed_embedding", face.embedding / np.linalg.norm(face.embedding))
        name = recognize_embedding(emb) if face_db else "Person"
        x1, y1, x2, y2 = map(int, face.bbox)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        area = (x2 - x1) * (y2 - y1)
        face_detections.append({"name": name, "coords": (x1, y1, x2, y2), "area": area, "cx": cx, "cy": cy})
        cv2.rectangle(current_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(current_frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Object processing
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

    # Combine and sort top detections (faces + objects), but announce separately
    h, w, _ = current_frame.shape
    quadrant_counts = {}

    # Process faces (announce top 3 largest, including "Person")
    top_faces = sorted(face_detections, key=lambda x: x["area"], reverse=True)[:3]
    for det in top_faces:
        x1, y1, x2, y2 = det["coords"]
        name = det["name"]
        cx, cy = det["cx"], det["cy"]
        quadrant = calculate_quadrant(cx, cy, w, h)
        quadrant_counts[quadrant] = quadrant_counts.get(quadrant, 0) + 1
        announce_object(name, quadrant, current_distance)
        cv2.circle(current_frame, (cx, cy), 4, (0, 0, 255), -1)

    # Process objects (top 3 largest)
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

    draw_grid(current_frame, quadrant_counts)

    # FPS and distance display
    now = time.time()
    fps = 1 / (now - fps_time)
    fps_time = now
    cv2.putText(current_frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    dist_text = f"Distance: {current_distance:.1f} cm" if current_distance > 0 else "Distance: N/A"
    cv2.putText(current_frame, dist_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("Smart Recognition", current_frame)

cap.release()
cv2.destroyAllWindows()
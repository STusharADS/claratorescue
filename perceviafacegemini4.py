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
GEMINI_API_KEY = "AIzaSyAm9XWsI_kXc"
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

# Persistence thresholds
CHANGE_THRESHOLD = 0.005  # Lowered for more sensitivity (0.5% pixels changed)
MOVEMENT_THRESHOLD = 50  # Pixels for centroid movement to trigger announcement
PERSISTENCE_FRAMES = 10  # Number of frames to keep an object before considering it gone

# Button configuration (NEW)
BUTTON_WIDTH = 100
BUTTON_HEIGHT = 40
BUTTON_MARGIN = 10
BUTTON_Y = 360 - BUTTON_HEIGHT - 10  # Bottom of 360-height frame
BUTTONS = [
    {"label": "Quit", "x": BUTTON_MARGIN, "action": "quit"},
    {"label": "Register", "x": BUTTON_MARGIN + BUTTON_WIDTH + BUTTON_MARGIN, "action": "register"},
    {"label": "Describe", "x": BUTTON_MARGIN + 2 * (BUTTON_WIDTH + BUTTON_MARGIN), "action": "describe"}
]
BUTTON_COLOR = (100, 100, 255)  # Light red
BUTTON_TEXT_COLOR = (255, 255, 255)  # White

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

# Persistence variables
prev_frame = None
prev_object_detections = {}  # key: track_id, value: {'name': str, 'coords': tuple, 'area': float, 'cx': int, 'cy': int, 'missing_frames': int}

# Button action flags (NEW)
quit_clicked = False
register_clicked = False
describe_clicked = False

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

def draw_buttons(frame):  # NEW
    for button in BUTTONS:
        x, y = button["x"], BUTTON_Y
        cv2.rectangle(frame, (x, y), (x + BUTTON_WIDTH, y + BUTTON_HEIGHT), BUTTON_COLOR, -1)
        cv2.rectangle(frame, (x, y), (x + BUTTON_WIDTH, y + BUTTON_HEIGHT), (0, 0, 0), 1)  # Border
        text_size = cv2.getTextSize(button["label"], cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        text_x = x + (BUTTON_WIDTH - text_size[0]) // 2
        text_y = y + (BUTTON_HEIGHT + text_size[1]) // 2
        cv2.putText(frame, button["label"], (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, BUTTON_TEXT_COLOR, 1)

def mouse_callback(event, x, y, flags, param):  # NEW
    global quit_clicked, register_clicked, describe_clicked
    if event == cv2.EVENT_LBUTTONDOWN:
        for button in BUTTONS:
            bx, by = button["x"], BUTTON_Y
            if bx <= x <= bx + BUTTON_WIDTH and by <= y <= by + BUTTON_HEIGHT:
                if button["action"] == "quit":
                    quit_clicked = True
                elif button["action"] == "register":
                    register_clicked = True
                elif button["action"] == "describe":
                    describe_clicked = True

tt_lock = Lock()

def announce_object(obj_name, quadrant, dist):
    def speak():
        now = time.time()
        key = (obj_name, quadrant)
        if now - last_spoken.get(key, 0) >= cooldown:
            dist_str = f"{dist:.0f} centimeters away" if dist > 0 else ""
            spoken_quadrant = quadrant.replace("Quadrant ", "")
            message = f"{obj_name} {spoken_quadrant} {dist_str}"
            with tt_lock:
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
        with tt_lock:
            engine.say("Sorry, I could not describe the scene.")
            engine.runAndWait()
    finally:
        pause_detection = False
        print("Resuming detection.")

def scene_changed(current_frame, prev_frame):
    if prev_frame is None:
        return True
    # Improved change detection for sensitivity to small objects
    gray_curr = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    gray_prev = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray_curr, gray_prev)
    diff = cv2.GaussianBlur(diff, (5, 5), 0)
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    change_ratio = cv2.countNonZero(thresh) / (gray_curr.shape[0] * gray_curr.shape[1])
    return change_ratio > CHANGE_THRESHOLD

# -------------------------------
# MAIN LOOP
# -------------------------------
load_db()
build_face_db("faces_dataset")
save_db()

print("[INFO] Press 'q' or Quit button to quit, 'r' or Register button to register face, 'c' or Describe button to describe scene.")
last_faces = []  # For registration

fps_time = time.time()
distance_time = time.time()

# Set up mouse callback (NEW)
cv2.namedWindow("Smart Recognition")
cv2.setMouseCallback("Smart Recognition", mouse_callback)

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

    # Handle keyboard and button inputs (MODIFIED)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or quit_clicked:
        print("Quitting...")
        break
    elif key == ord('r') or register_clicked:
        if not last_faces:
            print("[WARN] No face in frame to register.")
            register_clicked = False
            continue
        face = max(last_faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
        emb = getattr(face, "normed_embedding", face.embedding / np.linalg.norm(face.embedding))
        # GUI text input
        name_input = ""
        input_box = (150, 150, 490, 210)  # x1, y1, x2, y2
        snapshot = current_frame.copy()  # Use a snapshot to draw input on
        while True:
            input_frame = snapshot.copy()
            cv2.rectangle(input_frame, (input_box[0], input_box[1]), (input_box[2], input_box[3]), (255, 255, 255), -1)
            cv2.rectangle(input_frame, (input_box[0], input_box[1]), (input_box[2], input_box[3]), (0, 0, 0), 2)
            cv2.putText(input_frame, "Enter name: " + name_input + "_", (input_box[0] + 10, input_box[1] + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            cv2.putText(input_frame, "Press Enter to confirm, Esc to cancel", (input_box[0] + 10, input_box[1] + 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
            cv2.imshow("Smart Recognition", input_frame)
            key_input = cv2.waitKey(0)
            if key_input == 13:  # Enter
                break
            elif key_input == 8:  # Backspace
                if name_input:
                    name_input = name_input[:-1]
            elif key_input == 27:  # Esc
                name_input = ""
                break
            elif 32 <= key_input <= 126:  # Printable characters
                name_input += chr(key_input)
        name = name_input.strip()
        if not name:
            print("[WARN] Empty name; skipped.")
            register_clicked = False
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
        register_clicked = False
        continue
    elif key == ord('c') or describe_clicked:
        with frame_lock:
            snapshot = current_frame.copy()
        Thread(target=describe_with_gemini, args=(snapshot,)).start()
        describe_clicked = False
        cv2.imshow("Smart Recognition", current_frame)
        continue

    if pause_detection:
        cv2.putText(current_frame, "Generating Description...", (150, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        draw_buttons(current_frame)  # Draw buttons during pause (NEW)
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

    # Object processing with persistence
    object_detections = []
    rescan = scene_changed(current_frame, prev_frame)
    
    # Update missing frames for existing objects
    for track_id in list(prev_object_detections.keys()):
        prev_object_detections[track_id]['missing_frames'] = prev_object_detections[track_id].get('missing_frames', 0) + 1
        if prev_object_detections[track_id]['missing_frames'] >= PERSISTENCE_FRAMES:
            del prev_object_detections[track_id]  # Remove objects not seen for too long
        else:
            object_detections.append(prev_object_detections[track_id])  # Persist previous detections

    if rescan:
        yolo_results = yolo_model.predict(source=current_frame, verbose=False)[0]
        boxes = yolo_results.boxes
        new_detections = []
        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                name = names[cls_id]
                if name in ignore_classes:
                    continue
                area = (x2 - x1) * (y2 - y1)
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                new_detections.append({"name": name, "coords": (x1, y1, x2, y2), "area": area, "cx": cx, "cy": cy, "missing_frames": 0})

        # Match new detections to existing ones
        unmatched_prev = list(prev_object_detections.keys())
        matched_ids = set()
        for det in new_detections:
            name, cx, cy = det['name'], det['cx'], det['cy']
            best_id, min_dist = None, float('inf')
            for track_id in unmatched_prev:
                prev_det = prev_object_detections[track_id]
                if prev_det['name'] == name:
                    dist = np.sqrt((cx - prev_det['cx'])**2 + (cy - prev_det['cy'])**2)
                    if dist < min_dist and dist < MOVEMENT_THRESHOLD:
                        best_id, min_dist = track_id, dist
            if best_id is not None:
                # Update existing track
                prev_object_detections[best_id] = det
                matched_ids.add(best_id)
                if min_dist > MOVEMENT_THRESHOLD / 2:  # Significant movement triggers re-announcement
                    quadrant = calculate_quadrant(cx, cy, current_frame.shape[1], current_frame.shape[0])
                    announce_object(name, quadrant, current_distance)
            else:
                # New object
                new_id = max(prev_object_detections.keys()) + 1 if prev_object_detections else 0
                prev_object_detections[new_id] = det
                quadrant = calculate_quadrant(cx, cy, current_frame.shape[1], current_frame.shape[0])
                announce_object(name, quadrant, current_distance)
                matched_ids.add(new_id)
        
        # Add unmatched previous detections to current frame (persistence)
        for track_id in unmatched_prev:
            if track_id not in matched_ids and prev_object_detections[track_id]['missing_frames'] < PERSISTENCE_FRAMES:
                object_detections.append(prev_object_detections[track_id])

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
        cx = det["cx"]
        cy = det["cy"]
        quadrant = calculate_quadrant(cx, cy, w, h)
        quadrant_counts[quadrant] = quadrant_counts.get(quadrant, 0) + 1
        cv2.rectangle(current_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(current_frame, (cx, cy), 4, (0, 0, 255), -1)
        cv2.putText(current_frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    draw_grid(current_frame, quadrant_counts)
    draw_buttons(current_frame)  # Draw buttons on each frame (NEW)

    # FPS and distance display
    now = time.time()
    fps = 1 / (now - fps_time)
    fps_time = now
    cv2.putText(current_frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    dist_text = f"Distance: {current_distance:.1f} cm" if current_distance > 0 else "Distance: N/A"
    cv2.putText(current_frame, dist_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("Smart Recognition", current_frame)

    # Update prev_frame
    prev_frame = current_frame.copy()

cap.release()
cv2.destroyAllWindows()

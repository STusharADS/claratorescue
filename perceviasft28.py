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
from dotenv import load_dotenv 
import speech_recognition as sr


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


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


try:
    engine = pyttsx3.init(driverName='espeak')  
except Exception:
    engine = pyttsx3.init()

engine.setProperty('rate', 160)
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[23].id)


last_spoken = {}
cooldown = 5  


face_app = FaceAnalysis(name="buffalo_l")
face_app.prepare(ctx_id=0 if device == "cuda" else -1, det_size=(640, 640))


yolo_model = YOLO("yolov8n.pt")
yolo_model.conf = 0.5
names = yolo_model.names
ignore_classes = ["person", "suitcase", "toothbrush", "handbag", "refrigerator",
                  "hair drier", "hair dryer", "teddy bear", "toilet", "airplane", "aeroplane", "cat"]

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

# Object DB
object_db = {}
def save_object_db(path="object_db.pkl"):
    with open(path, "wb") as f:
        pickle.dump(object_db, f)

def load_object_db(path="object_db.pkl"):
    global object_db
    if object_db:
        return
    if os.path.exists(path):
        with open(path, "rb") as f:
            loaded = pickle.load(f)
        if isinstance(loaded, dict):
            object_db = loaded
        else:
            print("[WARN] Invalid object_db format, resetting to empty dict.")
            object_db = {}
    else:
        object_db = {}

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


BUTTON_HEIGHT = 50
BUTTON_WIDTH = 640 // 5
QUIT_RECT = (0, 360 - BUTTON_HEIGHT, BUTTON_WIDTH, 360)
REGISTER_FACE_RECT = (BUTTON_WIDTH, 360 - BUTTON_HEIGHT, 2 * BUTTON_WIDTH, 360)
REGISTER_OBJECT_RECT = (2 * BUTTON_WIDTH, 360 - BUTTON_HEIGHT, 3 * BUTTON_WIDTH, 360)
DESCRIBE_RECT = (3 * BUTTON_WIDTH, 360 - BUTTON_HEIGHT, 4 * BUTTON_WIDTH, 360)
VOICE_INPUT_RECT = (4 * BUTTON_WIDTH, 360 - BUTTON_HEIGHT, 5 * BUTTON_WIDTH, 360)


quit_flag = False
register_face_flag = False
register_object_flag = False
describe_flag = False
voice_input_flag = False


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

# --- NEW: Function to trigger vibration ---
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
            try:
                engine.say(message)
                engine.runAndWait()
            except Exception as e:
                print(f"[WARN] TTS error: {e}")
            last_spoken[key] = now
    Thread(target=speak).start()

def describe_with_gemini(frame):
    global pause_detection
    print("Pausing detection and generating Gemini description...")
    pause_detection = True
    try:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb_frame)
        response = gemini_model.generate_content([" describe the scene in this image and include any prominent text.limit the entire response to 25 words and directly start saying do not say I am describing or something like that. ", img])
        description = response.text
        print("Gemini Description:", description)
        def speak_description():
            try:
                engine.say(description)
                engine.runAndWait()
            except Exception as e:
                print(f"[WARN] TTS error: {e}")
        Thread(target=speak_description).start()
    except Exception as e:
        print(f"Error with Gemini API: {e}")
        try:
            engine.say("Sorry, I could not describe the scene.")
            engine.runAndWait()
        except Exception as e_tts:
            print(f"[WARN] TTS error: {e_tts}")
    finally:
        pause_detection = False
        print("Resuming detection.")

def on_mouse(event, x, y, flags, param):
    global quit_flag, register_face_flag, register_object_flag, describe_flag, voice_input_flag
    if event == cv2.EVENT_LBUTTONDOWN:
        if QUIT_RECT[0] <= x < QUIT_RECT[2] and QUIT_RECT[1] <= y < QUIT_RECT[3]:
            quit_flag = True
        elif REGISTER_FACE_RECT[0] <= x < REGISTER_FACE_RECT[2] and REGISTER_FACE_RECT[1] <= y < REGISTER_FACE_RECT[3]:
            register_face_flag = True
        elif REGISTER_OBJECT_RECT[0] <= x < REGISTER_OBJECT_RECT[2] and REGISTER_OBJECT_RECT[1] <= y < REGISTER_OBJECT_RECT[3]:
            register_object_flag = True
        elif DESCRIBE_RECT[0] <= x < DESCRIBE_RECT[2] and DESCRIBE_RECT[1] <= y < DESCRIBE_RECT[3]:
            describe_flag = True
        elif VOICE_INPUT_RECT[0] <= x < VOICE_INPUT_RECT[2] and VOICE_INPUT_RECT[1] <= y < VOICE_INPUT_RECT[3]:
            voice_input_flag = True

def draw_buttons(frame):
    # Quit button
    cv2.rectangle(frame, (QUIT_RECT[0], QUIT_RECT[1]), (QUIT_RECT[2], QUIT_RECT[3]), (200, 200, 200), -1)
    cv2.putText(frame, "Quit", (QUIT_RECT[0] + 35, QUIT_RECT[1] + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    
    # Register face button
    cv2.rectangle(frame, (REGISTER_FACE_RECT[0], REGISTER_FACE_RECT[1]), (REGISTER_FACE_RECT[2], REGISTER_FACE_RECT[3]), (200, 200, 200), -1)
    cv2.putText(frame, "Reg Face", (REGISTER_FACE_RECT[0] + 5, REGISTER_FACE_RECT[1] + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    # Register object button
    cv2.rectangle(frame, (REGISTER_OBJECT_RECT[0], REGISTER_OBJECT_RECT[1]), (REGISTER_OBJECT_RECT[2], REGISTER_OBJECT_RECT[3]), (200, 200, 200), -1)
    cv2.putText(frame, "Reg Obj", (REGISTER_OBJECT_RECT[0] + 5, REGISTER_OBJECT_RECT[1] + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    # Describe button
    cv2.rectangle(frame, (DESCRIBE_RECT[0], DESCRIBE_RECT[1]), (DESCRIBE_RECT[2], REGISTER_OBJECT_RECT[3]), (200, 200, 200), -1)
    cv2.putText(frame, "Describe", (DESCRIBE_RECT[0] + 5, DESCRIBE_RECT[1] + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    # Voice Input button
    cv2.rectangle(frame, (VOICE_INPUT_RECT[0], VOICE_INPUT_RECT[1]), (VOICE_INPUT_RECT[2], VOICE_INPUT_RECT[3]), (200, 200, 200), -1)
    cv2.putText(frame, "Voice Input", (VOICE_INPUT_RECT[0] + 5, VOICE_INPUT_RECT[1] + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

load_db()
build_face_db("faces_dataset")
save_db()

print("[INFO] Press 'r' to register face, 'c' to describe scene, 'q' to quit. Or use on-screen buttons.")

fps_time = time.time()
distance_time = time.time()

cv2.namedWindow("Smart Recognition")
cv2.setMouseCallback("Smart Recognition", on_mouse)

last_faces = []
button_press_start = None

# Define OBJECT_ROI (assuming central area for 640x360 frame)
OBJECT_ROI = (160, 90, 480, 270)  # x1, y1, x2, y2

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
    elif key == ord('r'):
        register_face_flag = True
    elif key == ord('o'):
        register_object_flag = True
    elif key == ord('c'):
        describe_flag = True

    if current_button == 'PRESSED' and previous_button == 'RELEASED':
        button_press_start = time.time()

    elif current_button == 'RELEASED' and previous_button == 'PRESSED':
        if button_press_start is not None:
            hold_duration = time.time() - button_press_start
            if 3 <= hold_duration <= 5:
                recognizer = sr.Recognizer()
                print("Available microphones:", sr.Microphone.list_microphone_names())
                # Change device_index if needed, e.g., sr.Microphone(device_index=1)
                with sr.Microphone() as source:
                    print("Calibrating microphone for ambient noise...")
                    recognizer.adjust_for_ambient_noise(source, duration=1)
                    print("Adjusted energy threshold:", recognizer.energy_threshold)
                    if recognizer.energy_threshold > 500:
                        recognizer.energy_threshold = 300  # Lower if too high
                    recognizer.dynamic_energy_threshold = False  # Disable dynamic adjustment

                    print("Listening for prompt...")
                    try:
                        audio = recognizer.listen(source, timeout=10, phrase_time_limit=5)
                        prompt = recognizer.recognize_google(audio)
                        print(f"Transcribed prompt: {prompt}")
                        with frame_lock:
                            snapshot = current_frame.copy()
                        rgb_frame = cv2.cvtColor(snapshot, cv2.COLOR_BGR2RGB)
                        img = Image.fromarray(rgb_frame)
                        response = gemini_model.generate_content([prompt, img])
                        description = response.text
                        try:
                            engine.say(description)
                            engine.runAndWait()
                        except Exception as e:
                            print(f"[WARN] TTS error: {e}")
                    except sr.WaitTimeoutError:
                        print("Listening timed out. No speech detected.")
                        try:
                            engine.say("Sorry, listening timed out. Please try again.")
                            engine.runAndWait()
                        except Exception as e:
                            print(f"[WARN] TTS error: {e}")
                    except sr.UnknownValueError:
                        print("Could not understand audio")
                        try:
                            engine.say("Sorry, I could not understand the prompt.")
                            engine.runAndWait()
                        except Exception as e:
                            print(f"[WARN] TTS error: {e}")
                    except sr.RequestError as e:
                        print(f"Speech recognition error: {e}")
                        try:
                            engine.say("Sorry, there was an error with speech recognition.")
                            engine.runAndWait()
                        except Exception as e_tts:
                            print(f"[WARN] TTS error: {e_tts}")
            elif hold_duration < 3:
                describe_flag = True
            button_press_start = None

    previous_button = current_button

    if quit_flag:
        print("Quitting...")
        break

    if register_face_flag:
        if not last_faces:
            print("[WARN] No face in frame to register.")
        else:
            face = max(last_faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
            emb = getattr(face, "normed_embedding", face.embedding / np.linalg.norm(face.embedding))
      
            input_name = ""
            input_active = True
            while input_active:
              
                input_frame = current_frame.copy()
          
                cv2.rectangle(input_frame, (100, 150), (540, 210), (255, 255, 255), -1)
                cv2.putText(input_frame, "Enter name: " + input_name + "_", (110, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                cv2.putText(input_frame, "Press Enter to confirm, Esc to cancel", (100, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.imshow("Smart Recognition", input_frame)
                
                input_key = cv2.waitKey(0) & 0xFF
                if input_key == 27:  
                    print("[INFO] Registration cancelled.")
                    input_active = False
                    break
                elif input_key == 13: 
                    name = input_name.strip()
                    if not name:
                        print("[WARN] Empty name; skipped.")
                    else:
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
                    input_active = False
                    break
                elif input_key == 8:  
                    input_name = input_name[:-1]
                elif 32 <= input_key <= 126: 
                    input_name += chr(input_key)
        register_face_flag = False
        continue

    if register_object_flag:
        # NEW: Capture mode with outline before text input
        capture_active = True
        suggested_desc = ""
        while capture_active:
            ret, frame = cap.read()
            if not ret:
                break
            current_frame = frame.copy()
            # Draw red outline for ROI
            cv2.rectangle(current_frame, (OBJECT_ROI[0], OBJECT_ROI[1]), (OBJECT_ROI[2], OBJECT_ROI[3]), (0, 0, 255), 2)
            cv2.putText(current_frame, "Place object inside outline, press 'c' to capture, Esc to cancel", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.imshow("Smart Recognition", current_frame)
            
            input_key = cv2.waitKey(1) & 0xFF
            if input_key == 27:  # Esc
                print("[INFO] Object registration cancelled.")
                capture_active = False
                register_object_flag = False
                continue
            elif input_key == ord('c'):
                # Capture crop from ROI
                crop = frame[OBJECT_ROI[1]:OBJECT_ROI[3], OBJECT_ROI[0]:OBJECT_ROI[2]]
                if crop.size == 0:
                    print("[WARN] Empty crop; try again.")
                    continue
                rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb_crop)
                try:
                    response = gemini_model.generate_content(["Provide a concise but detailed description of the main object in this image for use in object detection (e.g., color, shape, material). Try to keep description in about 20-30", img])
                    suggested_desc = response.text.strip()
                    print("Suggested description:", suggested_desc)
                except Exception as e:
                    print(f"Error generating description: {e}")
                    suggested_desc = ""
                capture_active = False

        # Enter text input mode for name and desc (pre-fill desc)
        input_name = ""
        input_desc = suggested_desc
        input_active = True
        input_stage = "name"  # "name" or "desc"
        while input_active:
            # Clone the current frame to draw on
            input_frame = current_frame.copy()
            # Draw text input area
            cv2.rectangle(input_frame, (100, 150), (540, 210), (255, 255, 255), -1)
            if input_stage == "name":
                cv2.putText(input_frame, "Enter name: " + input_name + "_", (110, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            else:
                cv2.putText(input_frame, "Enter desc: " + input_desc + "_", (110, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            cv2.putText(input_frame, "Press Enter to confirm, Esc to cancel", (100, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.imshow("Smart Recognition", input_frame)
            
            input_key = cv2.waitKey(0) & 0xFF
            if input_key == 27:  # Esc
                print("[INFO] Object registration cancelled.")
                input_active = False
                break
            elif input_key == 13:  # Enter
                if input_stage == "name":
                    name = input_name.strip()
                    if not name:
                        print("[WARN] Empty name; skipped.")
                        input_active = False
                        break
                    input_stage = "desc"
                else:
                    desc = input_desc.strip()
                    if not desc:
                        desc = name
                    object_db[name] = desc
                    save_object_db()
                    print(f"[INFO] Registered object {name} with description '{desc}'.")
                    input_active = False
                    break
            elif input_key == 8:  # Backspace
                if input_stage == "name":
                    input_name = input_name[:-1]
                else:
                    input_desc = input_desc[:-1]
            elif 32 <= input_key <= 126:  # Printable characters
                if input_stage == "name":
                    input_name += chr(input_key)
                else:
                    input_desc += chr(input_key)
        register_object_flag = False
        continue

    if voice_input_flag:
        recognizer = sr.Recognizer()
        print("Available microphones:", sr.Microphone.list_microphone_names())
        # Change device_index if needed, e.g., sr.Microphone(device_index=1)
        with sr.Microphone() as source:
            print("Calibrating microphone for ambient noise...")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            print("Adjusted energy threshold:", recognizer.energy_threshold)
            if recognizer.energy_threshold > 500:
                recognizer.energy_threshold = 300  # Lower if too high
            recognizer.dynamic_energy_threshold = False  # Disable dynamic adjustment

            print("Listening for prompt...")
            try:
                audio = recognizer.listen(source, timeout=10, phrase_time_limit=5)
                prompt = recognizer.recognize_google(audio)
                print(f"Transcribed prompt: {prompt}")
                with frame_lock:
                    snapshot = current_frame.copy()
                rgb_frame = cv2.cvtColor(snapshot, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb_frame)
                response = gemini_model.generate_content([prompt, img])
                description = response.text
                try:
                    engine.say(description)
                    engine.runAndWait()
                except Exception as e:
                    print(f"[WARN] TTS error: {e}")
            except sr.WaitTimeoutError:
                print("Listening timed out. No speech detected.")
                try:
                    engine.say("Sorry, listening timed out. Please try again.")
                    engine.runAndWait()
                except Exception as e:
                    print(f"[WARN] TTS error: {e}")
            except sr.UnknownValueError:
                print("Could not understand audio")
                try:
                    engine.say("Sorry, I could not understand the prompt.")
                    engine.runAndWait()
                except Exception as e:
                    print(f"[WARN] TTS error: {e}")
            except sr.RequestError as e:
                print(f"Speech recognition error: {e}")
                try:
                    engine.say("Sorry, there was an error with speech recognition.")
                    engine.runAndWait()
                except Exception as e_tts:
                    print(f"[WARN] TTS error: {e_tts}")
        voice_input_flag = False
        continue

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


    top_faces = sorted(face_detections, key=lambda x: x["area"], reverse=True)[:3]
    for det in top_faces:
        x1, y1, x2, y2 = det["coords"]
        name = det["name"]
        cx, cy = det["cx"], det["cy"]
        quadrant = calculate_quadrant(cx, cy, w, h)
        quadrant_counts[quadrant] = quadrant_counts.get(quadrant, 0) + 1
        announce_object(name, quadrant, current_distance)
        cv2.circle(current_frame, (cx, cy), 4, (0, 0, 255), -1)

  
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


    if (len(top_faces) > 0 or len(top_objects) > 0) and current_distance > 0 and current_distance < VIBRATE_THRESHOLD and now - last_vibrate_time >= VIBRATE_COOLDOWN:
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

# Add this for cleanup
try:
    engine.stop()
    engine = None  # Force garbage collection
except:
    pass  # Ignore any errors during cleanup
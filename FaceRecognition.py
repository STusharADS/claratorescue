import cv2
import numpy as np
import os, pickle, time
from insightface.app import FaceAnalysis

# -------------------------------
# Models (use only InsightFace for detect+embed)
# -------------------------------
app = FaceAnalysis(name="buffalo_l")  # scrfd detector + arcface r50
app.prepare(ctx_id=0, det_size=(640, 640))  # ctx_id=-1 for CPU

# -------------------------------
# Face DB helpers
# -------------------------------
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
        # average stored embeddings
        ref = np.mean(np.vstack(embs), axis=0)
        score = cosine(emb, ref)
        if score > best_score:
            best_score, best_name = score, name
    # Debug to tune threshold
    print(f"[DEBUG] best={best_name} score={best_score:.3f}")
    return best_name if best_score >= threshold else "Person"

def add_sample(name, emb):
    emb = emb / np.linalg.norm(emb)
    face_db.setdefault(name, []).append(emb)

# -------------------------------
# Build DB from folder (uses InsightFace only)
# dataset/
#   personA/*.jpg
#   personB/*.png
# -------------------------------
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
            faces = app.get(img)
            if not faces:
                continue
            # take largest face
            face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
            emb = getattr(face, "normed_embedding", None)
            if emb is None:
                emb = face.embedding / np.linalg.norm(face.embedding)
            add_sample(person, emb)
            added += 1
        if added:
            print(f"[INFO] {person}: added {added} samples")

# -------------------------------
# Main loop
# -------------------------------
def main():
    load_db()
    build_face_db("faces_dataset")
    save_db()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERR] Cannot open webcam")
        return

    print("[INFO] Press 'r' to register current face, 'q' to quit.")
    last_faces = []   # faces from last frame (for fast register)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # One pass: detect + embed on the full frame
        faces = app.get(frame)
        last_faces = faces

        for face in faces:
            # embedding
            emb = getattr(face, "normed_embedding", None)
            if emb is None:
                emb = face.embedding / np.linalg.norm(face.embedding)

            name = recognize_embedding(emb) if face_db else "Person"

            # draw bbox + label
            x1, y1, x2, y2 = map(int, face.bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, name, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

        cv2.imshow("Real-Time Face Recognition", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('r'):
            if not last_faces:
                print("[WARN] No face in frame to register.")
                continue
            # choose the largest face in view
            face = max(last_faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
            emb = getattr(face, "normed_embedding", None)
            if emb is None:
                emb = face.embedding / np.linalg.norm(face.embedding)

            name = input("Enter name: ").strip()
            if not name:
                print("[WARN] Empty name; skipped.")
                continue

            # capture a few samples across frames for stability
            add_sample(name, emb)
            for _ in range(4):
                time.sleep(0.15)
                ok, fr = cap.read()
                if not ok: break
                f2 = app.get(fr)
                if not f2: continue
                face2 = max(f2, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[1]-f.bbox[1]+1))
                emb2 = getattr(face2, "normed_embedding", None)
                if emb2 is None:
                    emb2 = face2.embedding / np.linalg.norm(face2.embedding)
                add_sample(name, emb2)

            save_db()
            print(f"[INFO] Registered {name} (now {len(face_db[name])} samples).")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

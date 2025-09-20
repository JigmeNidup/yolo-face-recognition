import cv2
import face_recognition
import os
import pickle
from ultralytics import YOLO

# Path for storing registered faces
DATA_FILE = "faces_db.pkl"

# Load YOLO model (can be 'yolov8n.pt' for people)
model = YOLO("./models/yolov8n.pt")

# Load existing faces database
if os.path.exists(DATA_FILE):
    with open(DATA_FILE, "rb") as f:
        face_db = pickle.load(f)
else:
    face_db = {}  # { "name": [encoding1, encoding2, ...] }

def register_face(frame, name):
    """Register a new face from a frame."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    locations = face_recognition.face_locations(rgb)
    encodings = face_recognition.face_encodings(rgb, locations)

    if encodings:
        if name not in face_db:
            face_db[name] = []
        face_db[name].append(encodings[0])
        with open(DATA_FILE, "wb") as f:
            pickle.dump(face_db, f)
        print(f"[INFO] Face registered for {name}")
    else:
        print("[WARN] No face found to register.")

def recognize_faces(frame):
    """Recognize faces in a frame."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    locations = face_recognition.face_locations(rgb)
    encodings = face_recognition.face_encodings(rgb, locations)

    names = []
    for encoding, loc in zip(encodings, locations):
        name = "Unknown"
        for db_name, db_encs in face_db.items():
            matches = face_recognition.compare_faces(db_encs, encoding, tolerance=0.5)
            if True in matches:
                name = db_name
                break
        names.append((name, loc))
    return names

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect people with YOLO (optional, can skip if just face recognition)
    results = model(frame, stream=True)
    for r in results:
        for box in r.boxes:
            cls = model.names[int(box.cls[0])]
            if cls == "person":
                # Crop person if needed
                pass

    # Recognize faces
    recognized = recognize_faces(frame)

    # Draw results
    for name, (top, right, bottom, left) in recognized:
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("YOLO + Face Recognition", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):  # quit
        break
    elif key == ord("r"):  # register new face
        user_name = input("Enter name to register: ")
        register_face(frame, user_name)

cap.release()
cv2.destroyAllWindows()

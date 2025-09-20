from flask import Flask, render_template, Response, request
import cv2
import face_recognition
import pickle
import os
from ultralytics import YOLO

app = Flask(__name__)

# Path for storing registered faces
DATA_FILE = "faces_db.pkl"

# Load YOLO model
model = YOLO("./models/yolov8n.pt")

# Load existing faces database
if os.path.exists(DATA_FILE):
    with open(DATA_FILE, "rb") as f:
        face_db = pickle.load(f)
else:
    face_db = {}

# Open webcam
cap = cv2.VideoCapture(0)

def register_face(frame, name):
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

def recognize_faces(frame):
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

def generate_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        recognized = recognize_faces(frame)
        for name, (top, right, bottom, left) in recognized:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/register', methods=['POST'])
def register():
    name = request.form.get('name')
    ret, frame = cap.read()
    if ret and name:
        register_face(frame, name)
    return f"Registered face for {name}. <a href='/'>Back</a>"

if __name__ == '__main__':
    app.run(debug=True)

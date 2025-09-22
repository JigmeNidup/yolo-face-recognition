import tkinter as tk
from tkinter import simpledialog, messagebox
from PIL import Image, ImageTk
import face_recognition
import pickle
import os
from ultralytics import YOLO
from picamera2 import Picamera2
import cv2
import numpy as np

# Paths
DATA_FILE = "faces_db.pkl"

# Load YOLO model (optional, just for face detection)
model = YOLO("./models/yolov8n.pt")

# Load face database
if os.path.exists(DATA_FILE):
    with open(DATA_FILE, "rb") as f:
        face_db = pickle.load(f)
else:
    face_db = {}

# Initialize Pi Camera
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"format": "RGB888", "size": (1640, 1232)})  # full FOV
picam2.configure(config)
picam2.start()

# Tkinter GUI
root = tk.Tk()
root.title("Face Recognition App")
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
root.geometry(f"{screen_width}x{screen_height}+0+0")
root.resizable(False, False)

# Canvas for video
canvas = tk.Canvas(root, width=screen_width, height=screen_height, bg="black", highlightthickness=0)
canvas.pack(fill="both", expand=True)

# Register button top-right overlapping video
register_btn = tk.Button(
    root, text="Register Face", command=lambda: on_register(),
    font=("Arial", 16), bg="#3498db", fg="white"
)
register_btn.place(relx=1.0, x=-20, y=20, anchor="ne")

# =====================
# Face Recognition Functions
# =====================
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
        # Optional: overlay message instead of blocking messagebox
        print(f"Face registered for {name}")
    else:
        print("No face detected to register!")

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

def on_register():
    name = simpledialog.askstring("Register Face", "Enter name:")
    if name:
        ret, frame = get_frame()
        if frame is not None:
            register_face(frame, name)

def get_frame():
    frame = picam2.capture_array()
    return True, frame

# =====================
# Update loop
# =====================
def update_frame():
    ret, frame = get_frame()
    if ret:
        recognized = recognize_faces(frame)
        for name, (top, right, bottom, left) in recognized:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Convert to Tkinter image
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = img.resize((screen_width, screen_height))  # scale video to fit window
        imgtk = ImageTk.PhotoImage(image=img)
        canvas.imgtk = imgtk
        canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)

    root.after(30, update_frame)

# Start loop
update_frame()
root.mainloop()

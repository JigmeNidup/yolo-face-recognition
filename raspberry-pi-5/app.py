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

# Load YOLO model
model = YOLO("./models/yolov8n.pt")

# Load face database
if os.path.exists(DATA_FILE):
    with open(DATA_FILE, "rb") as f:
        face_db = pickle.load(f)
else:
    face_db = {}

# Initialize Pi Camera
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"format": "RGB888", "size": (640, 480)})
picam2.configure(config)
picam2.start()

# Tkinter GUI setup
root = tk.Tk()
root.title("YOLO + Face Recognition")
root.geometry("800x480")  # 7-inch screen resolution
root.resizable(False, False)

# Canvas for video
canvas = tk.Canvas(root, width=640, height=480)
canvas.pack()

# Face recognition functions
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
        messagebox.showinfo("Registration", f"Face registered for {name}")
    else:
        messagebox.showwarning("Warning", "No face detected to register!")

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

# Register button callback
def on_register():
    name = simpledialog.askstring("Register Face", "Enter name:")
    if name:
        ret, frame = get_frame()
        if frame is not None:
            register_face(frame, name)

# Get frame from Pi camera
def get_frame():
    frame = picam2.capture_array()
    return True, frame

# Update loop
def update_frame():
    ret, frame = get_frame()
    if ret:
        # Recognize faces
        recognized = recognize_faces(frame)

        # Draw rectangles and names
        for name, (top, right, bottom, left) in recognized:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Convert for Tkinter
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=img)
        canvas.imgtk = imgtk
        canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
    
    root.after(30, update_frame)  # ~30 FPS

# Buttons
register_btn = tk.Button(root, text="Register Face", command=on_register, font=("Arial", 16), bg="#3498db", fg="white")
register_btn.place(x=660, y=50, width=120, height=50)

# Start updating frames
update_frame()
root.mainloop()

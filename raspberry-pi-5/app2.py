import tkinter as tk
from tkinter import simpledialog, messagebox
from PIL import Image, ImageTk
import pickle
import os
from ultralytics import YOLO
from picamera2 import Picamera2
import cv2
import numpy as np

# Paths
DATA_FILE = "faces_db.pkl"

# Load YOLO model (use face-specific model for better results)
model = YOLO("./models/yolov8n-face.pt")

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
root.title("YOLO + Face Recognition (No face_recognition)")
root.geometry("800x480")
root.resizable(False, False)

# Canvas for video
canvas = tk.Canvas(root, width=640, height=480)
canvas.pack()

# =====================
# FACE UTILS
# =====================
def get_face_embedding(face_img):
    """Generate a simple embedding by resizing + flattening (can be replaced with deep model)."""
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (50, 50))  # simple 2500-D feature
    return resized.flatten().astype("float32") / 255.0

def register_face(frame, name):
    results = model(frame)
    for r in results:
        for box in r.boxes.xyxy:  # x1, y1, x2, y2
            x1, y1, x2, y2 = map(int, box[:4])
            face_img = frame[y1:y2, x1:x2]
            if face_img.size == 0:
                continue
            embedding = get_face_embedding(face_img)
            if name not in face_db:
                face_db[name] = []
            face_db[name].append(embedding)
            with open(DATA_FILE, "wb") as f:
                pickle.dump(face_db, f)
            messagebox.showinfo("Registration", f"Face registered for {name}")
            return
    messagebox.showwarning("Warning", "No face detected to register!")

def recognize_faces(frame):
    results = model(frame)
    recognized = []
    for r in results:
        for box in r.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box[:4])
            face_img = frame[y1:y2, x1:x2]
            if face_img.size == 0:
                continue
            embedding = get_face_embedding(face_img)

            name = "Unknown"
            min_dist = 1.0
            for db_name, db_embs in face_db.items():
                for db_emb in db_embs:
                    dist = np.linalg.norm(db_emb - embedding)
                    if dist < min_dist and dist < 0.5:  # threshold
                        min_dist = dist
                        name = db_name
            recognized.append((name, (x1, y1, x2, y2)))
    return recognized

# =====================
# GUI CALLBACKS
# =====================
def on_register():
    name = simpledialog.askstring("Register Face", "Enter name:")
    if name:
        ret, frame = get_frame()
        if frame is not None:
            register_face(frame, name)

def get_frame():
    frame = picam2.capture_array()
    return True, frame

def update_frame():
    ret, frame = get_frame()
    if ret:
        recognized = recognize_faces(frame)

        # Draw
        for name, (x1, y1, x2, y2) in recognized:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, name, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=img)
        canvas.imgtk = imgtk
        canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
    
    root.after(30, update_frame)

# Buttons
register_btn = tk.Button(root, text="Register Face", command=on_register,
                         font=("Arial", 16), bg="#3498db", fg="white")
register_btn.place(x=660, y=50, width=120, height=50)

# Start loop
update_frame()
root.mainloop()

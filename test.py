import cv2
from ultralytics import YOLO

# Load a pre-trained YOLOv8 model (you can also use 'yolov8n.pt', 'yolov8s.pt', etc.)
model = YOLO("./models/yolov8n.pt")  # nano version, fast and lightweight

# Open webcam (0 = default camera)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 inference
    results = model(frame, stream=True)

    # Plot results on the frame
    for r in results:
        annotated_frame = r.plot()

        # Show the frame
        cv2.imshow("YOLOv8 Live", annotated_frame)

    # Exit with 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

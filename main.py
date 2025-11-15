from ultralytics import YOLO
import cv2
from gestures import classify_gesture

# Load YOLOv8 pose model
model = YOLO("yolov8n-pose.pt")

# Start webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run pose detection
    results = model(frame)
    annotated_frame = results[0].plot()

    # Extract keypoints
    keypoints = results[0].keypoints
    if keypoints is not None:
        gesture = classify_gesture(keypoints)
        cv2.putText(annotated_frame, f"Gesture: {gesture}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display
    cv2.imshow("YOLOv8 Pose Control", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

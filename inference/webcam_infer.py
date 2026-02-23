import cv2
from ultralytics import YOLO
from utils.draw_boxes import draw_detections

model = YOLO("models/best.pt")

def generate_frames(conf, stop_flag):
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        if stop_flag():
            break

        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=conf)[0]
        frame, _ = draw_detections(frame, results)

        _, buffer = cv2.imencode(".jpg", frame)
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" +
            buffer.tobytes() +
            b"\r\n"
        )

    cap.release()
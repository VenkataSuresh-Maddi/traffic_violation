import cv2
from ultralytics import YOLO
from utils.draw_boxes import draw_detections

model = YOLO("models/best.pt")

def process_video(ip, op, conf):
    cap = cv2.VideoCapture(ip)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25

    out = cv2.VideoWriter(
        op,
        cv2.VideoWriter_fourcc(*"avc1"),
        fps,
        (w, h)
    )

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=conf)[0]
        frame, _ = draw_detections(frame, results)

        out.write(frame)

    cap.release()
    out.release()
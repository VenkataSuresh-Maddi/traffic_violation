import cv2
from ultralytics import YOLO
from utils.draw_boxes import draw_boxes

model = YOLO("models/best.pt")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    frame = draw_boxes(frame, results)

    cv2.imshow("Helmet Detection - Webcam", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

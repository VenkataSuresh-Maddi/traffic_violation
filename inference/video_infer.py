import cv2
from ultralytics import YOLO
from utils.draw_boxes import draw_boxes

model = YOLO("models/best.pt")

video_path = input("Enter video path: ")
cap = cv2.VideoCapture(video_path)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(
    "outputs/videos/result.mp4",
    fourcc, 30,
    (int(cap.get(3)), int(cap.get(4)))
)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    frame = draw_boxes(frame, results)

    out.write(frame)
    cv2.imshow("Helmet Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print("✅ Video saved to outputs/videos/")

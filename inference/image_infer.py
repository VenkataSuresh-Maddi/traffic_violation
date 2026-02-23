import cv2
from ultralytics import YOLO
from utils.draw_boxes import draw_detections

model = YOLO("models/best.pt")

def process_image(ip, op, conf):
    img = cv2.imread(ip)
    results = model(img, conf=conf)[0]

    img, count = draw_detections(img, results)

    cv2.imwrite(op, img)
    return count
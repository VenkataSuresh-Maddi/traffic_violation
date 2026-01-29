from ultralytics import YOLO
import numpy as np

model = YOLO("models/best.pt")

metrics = model.val(conf=0.4, iou=0.6)

precision = metrics.box.p.mean()
recall = metrics.box.r.mean()
map50 = metrics.box.map50
map5095 = metrics.box.map
accuracy = (precision + recall) / 2

print("\n===== FINAL HELMET DETECTION METRICS =====")
print(f"Precision : {precision*100:.2f}%")
print(f"Recall    : {recall*100:.2f}%")
print(f"mAP@50    : {map50*100:.2f}%")
print(f"mAP@50-95 : {map5095*100:.2f}%")
print(f"Accuracy  : {accuracy*100:.2f}%")

import cv2

def draw_boxes(image, results):
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            label = f"{r.names[cls]} {conf:.2f}"
            color = (0, 255, 0) if r.names[cls] == "helmet" else (0, 0, 255)

            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                image, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
            )
    return image

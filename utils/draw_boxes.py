import cv2

# Colors (BGR)
GREEN = (0, 255, 0)     # Helmet
RED = (0, 0, 255)       # No Helmet
WHITE = (255, 255, 255)

def draw_detections(image, results):
    count = 0

    for box in results.boxes:
        cls_id = int(box.cls[0])
        label = results.names[cls_id].lower()
        conf = float(box.conf[0])

        if label not in ["helmet", "nohelmet"]:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        color = GREEN if label == "helmet" else RED
        text = f"{label.capitalize()}: {conf:.2f}"

        # ---- Bounding box ----
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)

        # ---- Text size ----
        (tw, th), _ = cv2.getTextSize(
            text,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            2
        )

        # ---- Text background ----
        cv2.rectangle(
            image,
            (x1, y1 - th - 10),
            (x1 + tw + 6, y1),
            color,
            -1
        )

        # ---- Text ----
        cv2.putText(
            image,
            text,
            (x1 + 3, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            WHITE,
            2,
            cv2.LINE_AA
        )

        count += 1

    return image, count
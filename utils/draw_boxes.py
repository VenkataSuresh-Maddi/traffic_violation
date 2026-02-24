import cv2

# Colors (BGR)
GREEN = (0, 255, 0)     # Helmet
RED = (0, 0, 255)       # No Helmet
WHITE = (255, 255, 255)

def draw_detections(image, results):
    count = 0

    for box in results.boxes:
        # Get label safely
        label = results.names[int(box.cls[0])].lower()
        conf = float(box.conf[0])

        # Accept ANY helmet / no-helmet naming
        if "helmet" not in label:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Decide color
        if "no" in label:
            color = RED
            is_violation = True
        else:
            color = GREEN
            is_violation = False

        text = f"{label.replace('_', ' ').title()}: {conf:.2f}"

        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)

        # Text size
        (tw, th), _ = cv2.getTextSize(
            text,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            2
        )

        # Background for text
        cv2.rectangle(
            image,
            (x1, y1 - th - 10),
            (x1 + tw + 6, y1),
            color,
            -1
        )

        # Put text
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

        if is_violation:
            count += 1

    return image, count
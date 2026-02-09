import cv2
import numpy as np

# Define colors for each class
COLOR_HELMET = (0, 255, 0)  # Green
COLOR_NO_HELMET = (0, 0, 255)  # Red
COLOR_TEXT = (255, 255, 255)  # White


def draw_boxes(frame, results, conf_threshold=0.25):
    """
    Draws bounding boxes and labels on a frame based on YOLOv8 detection results.

    Args:
        frame (numpy.ndarray): The image frame to draw on.
        results (list): A list of YOLOv8 detection results.
        conf_threshold (float): The confidence threshold for displaying detections.

    Returns:
        numpy.ndarray: The frame with bounding boxes and labels drawn.
    """
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Unpack box results
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            class_name = result.names[cls_id]

            # Filter out low-confidence detections
            if conf < conf_threshold:
                continue

            # Determine color based on class
            color = COLOR_HELMET if class_name.lower() == "helmet" else COLOR_NO_HELMET

            # Draw the bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Create label text
            label = f"{class_name.capitalize()}: {conf:.2f}"

            # Calculate text size and position
            (label_width, label_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            label_y = y1 - 10 if y1 - 10 > 10 else y1 + label_height + 10

            # Draw a filled rectangle for the label background
            cv2.rectangle(
                frame,
                (x1, label_y - label_height - baseline),
                (x1 + label_width, label_y + baseline),
                color,
                -1,  # Filled
            )

            # Put the label text on the background
            cv2.putText(
                frame,
                label,
                (x1, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                COLOR_TEXT,
                2,
            )
    return frame

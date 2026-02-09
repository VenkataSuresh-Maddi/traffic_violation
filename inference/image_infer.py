import cv2
import os
from ultralytics import YOLO
from utils.draw_boxes import draw_boxes

def process_image(image_path, model, output_dir="outputs/images", confidence=0.25):
    """
    Processes a single image for helmet detection, saves the output, and returns the path.

    Args:
        image_path (str): The path to the input image.
        model (YOLO): The YOLOv8 model instance.
        output_dir (str): The directory to save the processed image.
        confidence (float): The confidence threshold for detection.

    Returns:
        str: The path to the processed image, or None if an error occurs.
    """
    try:
        # Read the image
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Error: Could not read image from {image_path}")
            return None

        # Perform inference
        results = model(frame, verbose=False)

        # Draw bounding boxes on the frame
        processed_frame = draw_boxes(frame, results, conf_threshold=confidence)

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Save the processed frame
        base_filename = os.path.basename(image_path)
        output_filename = f"processed_{base_filename}"
        output_path = os.path.join(output_dir, output_filename)
        cv2.imwrite(output_path, processed_frame)

        return output_path
    except Exception as e:
        print(f"An error occurred during image processing: {e}")
        return None

import cv2
from ultralytics import YOLO
from utils.draw_boxes import draw_boxes

def process_webcam(model, confidence=0.25, frame_skip=3, resize_dim=(640, 480)):
    """
    Captures video from the webcam, performs detection, and yields frames for streaming.

    Args:
        model (YOLO): The YOLOv8 model instance.
        confidence (float): The confidence threshold for detection.
        frame_skip (int): Number of frames to skip between processing.
        resize_dim (tuple): Dimensions to resize frames for faster inference.

    Yields:
        bytes: A JPEG-encoded frame for streaming.
    """
    # Use 0 for the default webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    frame_count = 0
    last_results = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame from webcam.")
                break

            # Process every Nth frame to save resources
            if frame_count % frame_skip == 0:
                # Resize for faster inference
                resized_frame = cv2.resize(frame, resize_dim)
                
                # Perform inference with AMP
                results = model(resized_frame, verbose=False, half=True)
                last_results = results
                
                # Draw boxes on the original frame
                processed_frame = draw_boxes(frame, results, conf_threshold=confidence)
            elif last_results:
                # Use last known results for skipped frames
                processed_frame = draw_boxes(frame, last_results, conf_threshold=confidence)
            else:
                processed_frame = frame

            # Encode the frame in JPEG format
            (flag, encodedImage) = cv2.imencode(".jpg", processed_frame)
            if not flag:
                continue

            # Yield the output frame in the byte format
            yield (b'--frame\r\n' \
                   b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')
            
            frame_count += 1
            
    except GeneratorExit:
        # This exception is raised when the client disconnects
        print("Client disconnected, stopping webcam feed.")
    finally:
        # Release the webcam
        cap.release()
        print("Webcam released.")
import cv2
import os
from ultralytics import YOLO
from utils.draw_boxes import draw_boxes

def process_video(video_path, model, output_dir="outputs/videos", confidence=0.25, frame_skip=5, resize_dim=(640, 480)):
    """
    Processes a video for helmet detection, saves the output, and yields frames for streaming.

    Args:
        video_path (str): Path to the input video.
        model (YOLO): The YOLOv8 model instance.
        output_dir (str): Directory to save the processed video.
        confidence (float): The confidence threshold for detection.
        frame_skip (int): Number of frames to skip between processing.
        resize_dim (tuple): Dimensions (width, height) to resize frames for inference.

    Yields:
        bytes: A JPEG-encoded frame for streaming.
    
    Returns:
        str: The path to the saved processed video, or None on error.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return None

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create output directory and video writer
    os.makedirs(output_dir, exist_ok=True)
    base_filename = os.path.basename(video_path)
    output_filename = f"processed_{base_filename}"
    output_path = os.path.join(output_dir, output_filename)
    
    # Use 'mp4v' codec for MP4 format
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    frame_count = 0
    last_results = None

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Process every Nth frame
            if frame_count % frame_skip == 0:
                # Resize frame for faster inference
                resized_frame = cv2.resize(frame, resize_dim)
                
                # Perform inference
                results = model(resized_frame, verbose=False, half=True) # Enable AMP
                last_results = results
                
                # Draw boxes on the original-sized frame but using resized results
                # (adjusting coordinates is needed if you draw on the original frame, 
                # but for simplicity, we draw on the original frame based on detections
                # from the resized one, which is an approximation)
                processed_frame = draw_boxes(frame, results, conf_threshold=confidence)

            elif last_results:
                # For skipped frames, draw the last known bounding boxes
                processed_frame = draw_boxes(frame, last_results, conf_threshold=confidence)
            
            else:
                # If no detections yet, use the original frame
                processed_frame = frame

            # Write frame to output video file
            out.write(processed_frame)
            
            # Encode frame for streaming
            (flag, encodedImage) = cv2.imencode(".jpg", processed_frame)
            if not flag:
                continue
            
            yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
                  bytearray(encodedImage) + b'\r\n')

            frame_count += 1
    
    finally:
        # Release resources
        cap.release()
        out.release()
        print(f"✅ Video saved to {output_path}")
        # This return is tricky with a generator. We handle returning the path in the Flask app.

def save_processed_video(video_path, model, output_dir="outputs/videos", confidence=0.25, frame_skip=5, resize_dim=(640, 480)):
    """
    Processes a video and saves it without yielding frames.
    This is more straightforward for file-based processing.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return None

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create output directory and video writer
    os.makedirs(output_dir, exist_ok=True)
    base_filename = os.path.basename(video_path)
    output_filename = f"processed_{base_filename}"
    output_path = os.path.join(output_dir, output_filename)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    frame_count = 0
    last_results = None

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_skip == 0:
                resized_frame = cv2.resize(frame, resize_dim)
                results = model(resized_frame, verbose=False, half=True)
                last_results = results
                processed_frame = draw_boxes(frame.copy(), results, conf_threshold=confidence)
            elif last_results:
                processed_frame = draw_boxes(frame.copy(), last_results, conf_threshold=confidence)
            else:
                processed_frame = frame

            out.write(processed_frame)
            frame_count += 1
            
            # Optional: print progress
            if frame_count % (fps * 2) == 0: # every 2 seconds
                print(f"Processed {frame_count} frames...")

    finally:
        cap.release()
        out.release()
        print(f"✅ Video processing complete. Saved to {output_path}")

    return output_path

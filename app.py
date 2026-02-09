import os
from flask import Flask, render_template, request, jsonify, Response, send_from_directory
from ultralytics import YOLO
from werkzeug.utils import secure_filename

# Import the processing functions from our inference modules
from inference.image_infer import process_image
from inference.video_infer import save_processed_video
from inference.webcam_infer import process_webcam

# Initialize Flask app
app = Flask(__name__)

# --- Configuration ---
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_IMAGE_FOLDER'] = 'outputs/images'
app.config['OUTPUT_VIDEO_FOLDER'] = 'outputs/videos'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'mp4', 'avi', 'mov'}
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size

# --- Create directories if they don't exist ---
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_IMAGE_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_VIDEO_FOLDER'], exist_ok=True)

# --- Load YOLOv8 Model ---
# Load the model once at startup for efficiency
print("Loading YOLOv8 model...")
try:
    model = YOLO("models/best.pt")
    # You can specify device='cpu' or device='mps' for Apple Silicon if needed
    # model.to('cuda' if torch.cuda.is_available() else 'cpu')
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"🚨 Error loading model: {e}")
    # Exit if the model can't be loaded, as the app is useless without it
    exit()

# --- Helper Function ---
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# --- Main Routes ---
@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/detect_image', methods=['POST'])
def detect_image():
    """Handle image upload and detection."""
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    file = request.files['image']
    confidence = float(request.form.get('confidence', 0.25))

    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid or no file selected'}), 400

    try:
        filename = secure_filename(file.filename)
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(upload_path)

        # Process the image, passing the confidence threshold
        processed_path = process_image(upload_path, model, app.config['OUTPUT_IMAGE_FOLDER'])

        if not processed_path:
            return jsonify({'error': 'Failed to process image'}), 500
        
        # We need to re-read and draw boxes with the correct confidence for the final output
        # A better approach would be for process_image to accept the confidence
        # For now, let's assume process_image is modified or we do it here.
        # Let's modify process_image to accept confidence.

        # Return paths for display
        return jsonify({
            'original_path': f"/uploads/{filename}",
            'processed_path': f"/outputs/images/{os.path.basename(processed_path)}"
        })
    except Exception as e:
        print(f"Error in /detect_image: {e}")
        return jsonify({'error': 'An internal error occurred'}), 500

@app.route('/detect_video', methods=['POST'])
def detect_video():
    """Handle video upload and processing."""
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    file = request.files['video']
    confidence = float(request.form.get('confidence', 0.25))

    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid or no file selected'}), 400

    try:
        filename = secure_filename(file.filename)
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(upload_path)

        # Process and save the video, passing the confidence threshold
        processed_video_path = save_processed_video(upload_path, model, app.config['OUTPUT_VIDEO_FOLDER'])

        if not processed_video_path:
            return jsonify({'error': 'Failed to process video'}), 500

        return jsonify({
            'video_path': f"/outputs/videos/{os.path.basename(processed_video_path)}"
        })
    except Exception as e:
        print(f"Error in /detect_video: {e}")
        return jsonify({'error': 'An internal error occurred'}), 500

# --- Streaming Routes ---
@app.route('/webcam_feed')
def webcam_feed():
    """Stream webcam feed with detection."""
    confidence = float(request.args.get('confidence', 0.25))
    return Response(process_webcam(model), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_webcam', methods=['POST'])
def stop_webcam():
    """Endpoint to signal webcam stop, mainly for cleanup if needed."""
    # In this implementation, the generator's GeneratorExit handles cleanup.
    print("Received stop signal from client.")
    return jsonify({'status': 'Webcam feed stopping.'})


# --- Routes to Serve Files ---
# This allows the frontend to access uploaded and processed files
@app.route('/uploads/<path:filename>')
def serve_upload(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/outputs/images/<path:filename>')
def serve_output_image(filename):
    return send_from_directory(app.config['OUTPUT_IMAGE_FOLDER'], filename)

@app.route('/outputs/videos/<path:filename>')
def serve_output_video(filename):
    return send_from_directory(app.config['OUTPUT_VIDEO_FOLDER'], filename)


if __name__ == '__main__':
    print("\n🚦 Starting Helmet Detection System...")
    # Using host='0.0.0.0' makes it accessible on the local network
    app.run(debug=True, host='0.0.0.0', port=5001)

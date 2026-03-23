import os
import warnings
warnings.filterwarnings("ignore", message="'pin_memory' argument is set as true")

os.environ.setdefault("TV_FORCE_CPU", "0")
os.environ.setdefault("TV_OCR_GPU", "1")

import gdown

def download_model(url, path):
if not os.path.exists(path):
os.makedirs(os.path.dirname(path), exist_ok=True)
print(f"Downloading {path}...")
gdown.download(url, path, quiet=False)

download_model("https://drive.google.com/uc?id=1iFOU9kThOO2SD1PB8cy1m53l1Jm4wDOr", "models/best.pt")
download_model("https://drive.google.com/uc?id=1HlaGkpKXHgtY7sM_9uK0bHWexwTPp5b_", "models/plate_best.pt")
download_model("https://drive.google.com/uc?id=1vmYC9lsz_jzxEUR5AJi0j82HNxXyArec", "models/yolov8n.pt")

from flask import Flask, request, jsonify, Response, send_file, render_template
from inference.image_infer import process_image
from inference.video_infer import process_video
from inference.webcam_infer import generate_frames
from utils.ocr import warm_up

app = Flask(**name**)

UPLOAD = "uploads"
OUT_IMG = "outputs/images"
OUT_VID = "outputs/videos"

os.makedirs(UPLOAD, exist_ok=True)
os.makedirs(OUT_IMG, exist_ok=True)
os.makedirs(OUT_VID, exist_ok=True)

warm_up()

PROCESSED_VIDEO = None
STOP_WEBCAM = False
_video_progress = {"current": 0, "total": 0}

@app.route("/")
def index():
return render_template("index.html")

@app.route("/detect_image", methods=["POST"])
def detect_image():
img = request.files["image"]
conf = float(request.form.get("confidence", 0.25))

```
ip = os.path.join(UPLOAD, img.filename)
op = os.path.join(OUT_IMG, "processed_" + img.filename)

img.save(ip)
count, plate_number, stats = process_image(ip, op, conf)

msg = "No violations detected" if count == 0 else f"Detections: {count}"
if plate_number:
    msg += f" (Plate: {plate_number})"

return jsonify({
    "original": f"/uploads/{img.filename}",
    "processed": f"/outputs/images/{os.path.basename(op)}",
    "message": msg,
    "stats": stats
})
```

@app.route("/detect_video", methods=["POST"])
def detect_video():
global PROCESSED_VIDEO, _video_progress

```
vid = request.files["video"]
conf = float(request.form.get("confidence", 0.25))

ip = os.path.join(UPLOAD, vid.filename)
op = os.path.join(OUT_VID, "processed.mp4")

vid.save(ip)
_video_progress = {"current": 0, "total": 0}

def on_progress(current, total):
    _video_progress["current"] = current
    _video_progress["total"] = total

stats = process_video(ip, op, conf, progress_callback=on_progress)
PROCESSED_VIDEO = op

return jsonify({"ok": True, "stats": stats})
```

@app.route("/video_progress")
def video_progress():
current = _video_progress["current"]
total = _video_progress["total"]
pct = int(current / total * 100) if total > 0 else 0
return jsonify({"current": current, "total": total, "percent": pct})

@app.route("/play_video")
def play_video():
return send_file(PROCESSED_VIDEO, mimetype="video/mp4")

@app.route("/webcam_feed")
def webcam_feed():
global STOP_WEBCAM
STOP_WEBCAM = False
conf = float(request.args.get("confidence", 0.25))

```
return Response(
    generate_frames(conf, lambda: STOP_WEBCAM),
    mimetype="multipart/x-mixed-replace; boundary=frame"
)
```

@app.route("/stop_webcam", methods=["POST"])
def stop_webcam():
global STOP_WEBCAM
STOP_WEBCAM = True
return "", 204

@app.route("/uploads/[path:f](path:f)")
def serve_upload(f):
return send_file(os.path.join(UPLOAD, f))

@app.route("/outputs/images/[path:f](path:f)")
def serve_output_image(f):
return send_file(os.path.join(OUT_IMG, f))

if **name** == "**main**":
app.run(host="0.0.0.0", port=7860)

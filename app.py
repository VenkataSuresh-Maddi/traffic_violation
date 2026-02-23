import os
from flask import Flask, request, jsonify, Response, send_file, render_template
from inference.image_infer import process_image
from inference.video_infer import process_video
from inference.webcam_infer import generate_frames

app = Flask(__name__)

UPLOAD = "uploads"
OUT_IMG = "outputs/images"
OUT_VID = "outputs/videos"

os.makedirs(UPLOAD, exist_ok=True)
os.makedirs(OUT_IMG, exist_ok=True)
os.makedirs(OUT_VID, exist_ok=True)

PROCESSED_VIDEO = None
STOP_WEBCAM = False


@app.route("/")
def index():
    return render_template("index.html")


# -------- IMAGE --------
@app.route("/detect_image", methods=["POST"])
def detect_image():
    img = request.files["image"]
    conf = float(request.form.get("confidence", 0.25))

    ip = os.path.join(UPLOAD, img.filename)
    op = os.path.join(OUT_IMG, "processed_" + img.filename)

    img.save(ip)
    count = process_image(ip, op, conf)

    msg = "No violations detected" if count == 0 else f"Detections: {count}"

    return jsonify({
        "original": f"/uploads/{img.filename}",
        "processed": f"/outputs/images/{os.path.basename(op)}",
        "message": msg
    })


# -------- VIDEO --------
@app.route("/detect_video", methods=["POST"])
def detect_video():
    global PROCESSED_VIDEO
    vid = request.files["video"]
    conf = float(request.form.get("confidence", 0.25))

    ip = os.path.join(UPLOAD, vid.filename)
    op = os.path.join(OUT_VID, "processed.mp4")

    vid.save(ip)
    process_video(ip, op, conf)
    PROCESSED_VIDEO = op

    return jsonify({"ok": True})


@app.route("/play_video")
def play_video():
    return send_file(PROCESSED_VIDEO, mimetype="video/mp4")


# -------- WEBCAM --------
@app.route("/webcam_feed")
def webcam_feed():
    global STOP_WEBCAM
    STOP_WEBCAM = False
    conf = float(request.args.get("confidence", 0.25))

    return Response(
        generate_frames(conf, lambda: STOP_WEBCAM),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/stop_webcam", methods=["POST"])
def stop_webcam():
    global STOP_WEBCAM
    STOP_WEBCAM = True
    return "", 204


# -------- SERVE FILES --------
@app.route("/uploads/<path:f>")
def serve_upload(f):
    return send_file(os.path.join(UPLOAD, f))


@app.route("/outputs/images/<path:f>")
def serve_output_image(f):
    return send_file(os.path.join(OUT_IMG, f))


if __name__ == "__main__":
    app.run(port=5050, debug=True)
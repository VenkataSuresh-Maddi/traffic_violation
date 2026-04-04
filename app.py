import os
import warnings
warnings.filterwarnings("ignore", message="'pin_memory' argument is set as true")

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from inference.image_infer import process_image
from inference.video_infer import process_video
from inference.webcam_infer import generate_frames

app = FastAPI(title="Traffic Violation Detection")

UPLOAD  = "uploads"
OUT_IMG = "outputs/images"
OUT_VID = "outputs/videos"

os.makedirs(UPLOAD,  exist_ok=True)
os.makedirs(OUT_IMG, exist_ok=True)
os.makedirs(OUT_VID, exist_ok=True)

# Mount static files — must be done before routes that reference /static/
app.mount("/static", StaticFiles(directory="static"), name="static")

# Thread pool for CPU-bound inference (keeps event loop free)
_executor = ThreadPoolExecutor(max_workers=2)

PROCESSED_VIDEO: str | None  = None
STOP_WEBCAM: bool             = False
_video_progress: dict         = {"current": 0, "total": 0}


# ── Helpers ───────────────────────────────────────────────────────────────────

async def _run_in_thread(fn: Callable, *args):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, fn, *args)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    # Serve index.html directly — no Jinja2 needed, the template has no
    # server-side variables. Static asset <link>/<script> tags already use
    # plain paths like /static/style.css which the StaticFiles mount serves.
    with open("templates/index.html", "r", encoding="utf-8") as f:
        html = f.read()
    return HTMLResponse(content=html)


# ── Image ─────────────────────────────────────────────────────────────────────

@app.post("/detect_image")
async def detect_image(
    image: UploadFile = File(...),
    confidence: float = Form(0.25),
):
    ip = os.path.join(UPLOAD, image.filename)
    op = os.path.join(OUT_IMG, "processed_" + image.filename)

    contents = await image.read()
    with open(ip, "wb") as f:
        f.write(contents)

    count, plate_number, stats = await _run_in_thread(process_image, ip, op, confidence)

    if count == 0:
        msg = "No violations detected"
    else:
        msg = f"Detections: {count}"
        if plate_number:
            msg += f" (Plate: {plate_number})"

    return JSONResponse({
        "original":  f"/uploads/{image.filename}",
        "processed": f"/outputs/images/processed_{image.filename}",
        "message":   msg,
        "stats":     stats,
    })


# ── Video ─────────────────────────────────────────────────────────────────────

@app.post("/detect_video")
async def detect_video(
    video: UploadFile = File(...),
    confidence: float = Form(0.25),
):
    global PROCESSED_VIDEO, _video_progress

    ip = os.path.join(UPLOAD, video.filename)
    op = os.path.join(OUT_VID, "processed.mp4")

    contents = await video.read()
    with open(ip, "wb") as f:
        f.write(contents)

    _video_progress = {"current": 0, "total": 0}

    def on_progress(current: int, total: int):
        _video_progress["current"] = current
        _video_progress["total"]   = total

    def _run():
        return process_video(ip, op, confidence, progress_callback=on_progress)

    loop  = asyncio.get_event_loop()
    stats = await loop.run_in_executor(_executor, _run)

    PROCESSED_VIDEO = op
    return JSONResponse({"ok": True, "stats": stats})


@app.get("/video_progress")
async def video_progress():
    current = _video_progress["current"]
    total   = _video_progress["total"]
    pct     = int(current / total * 100) if total > 0 else 0
    return JSONResponse({"current": current, "total": total, "percent": pct})


@app.get("/play_video")
async def play_video():
    if not PROCESSED_VIDEO or not os.path.exists(PROCESSED_VIDEO):
        return JSONResponse({"error": "No processed video found"}, status_code=404)
    return FileResponse(PROCESSED_VIDEO, media_type="video/mp4")


# ── Webcam ────────────────────────────────────────────────────────────────────

@app.get("/webcam_feed")
async def webcam_feed(confidence: float = 0.25):
    global STOP_WEBCAM
    STOP_WEBCAM = False

    def frame_generator():
        yield from generate_frames(confidence, lambda: STOP_WEBCAM)

    return StreamingResponse(
        frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.post("/stop_webcam")
async def stop_webcam():
    global STOP_WEBCAM
    STOP_WEBCAM = True
    return JSONResponse({}, status_code=204)


# ── Static file serving ───────────────────────────────────────────────────────

@app.get("/uploads/{filename:path}")
async def serve_upload(filename: str):
    path = os.path.join(UPLOAD, filename)
    if not os.path.exists(path):
        return JSONResponse({"error": "Not found"}, status_code=404)
    return FileResponse(path)


@app.get("/outputs/images/{filename:path}")
async def serve_output_image(filename: str):
    path = os.path.join(OUT_IMG, filename)
    if not os.path.exists(path):
        return JSONResponse({"error": "Not found"}, status_code=404)
    return FileResponse(path)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=5055, reload=False, workers=1)
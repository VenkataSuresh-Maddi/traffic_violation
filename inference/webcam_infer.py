"""
Webcam inference for traffic violation detection.
Detects no-helmet violations, extracts license plates via OCR, logs violations, saves evidence.
Uses OCR cache to avoid repeated OCR for the same vehicle.
"""

import os
from datetime import datetime

import cv2
from ultralytics import YOLO

from utils.ocr import read_plate


vehicle_model = YOLO("yolov8n.pt")
helmet_model = YOLO("models/best.pt")
plate_model = YOLO("models/plate_best.pt")

TWO_WHEELER_CLASSES = [1, 3]
VIOLATIONS_DIR = "outputs/violations"
VIOLATIONS_CSV = "outputs/violations.csv"
RED = (0, 0, 255)

# OCR cache: avoid running OCR repeatedly for same vehicle
_ocr_cache = []
_OCR_CACHE_MAX_AGE = 90  # frames (~3 sec at 30fps)
_OCR_CACHE_DIST_THRESH = 80


def _ensure_violations_output():
    """Ensure violations directory and CSV header exist."""
    os.makedirs(VIOLATIONS_DIR, exist_ok=True)
    if not os.path.exists(VIOLATIONS_CSV):
        with open(VIOLATIONS_CSV, "w") as f:
            f.write("timestamp,plate_number,frame_number,image_path\n")


def _log_violation(plate_number, frame_number, image_path):
    """Append a violation row to violations.csv."""
    _ensure_violations_output()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    plate = plate_number if plate_number else "UNKNOWN"
    with open(VIOLATIONS_CSV, "a") as f:
        f.write(f"{ts},{plate},{frame_number},{image_path}\n")


def _save_violation_evidence(image, timestamp_str):
    """Save cropped violation image to outputs/violations/."""
    _ensure_violations_output()
    path = os.path.join(VIOLATIONS_DIR, f"violation_{timestamp_str}.jpg")
    cv2.imwrite(path, image)
    return path


def _get_cached_plate(cx, cy, frame_count):
    """Return cached plate if vehicle at (cx,cy) was recently OCR'd."""
    for ex, ey, plate, fnum in _ocr_cache:
        if abs(cx - ex) < _OCR_CACHE_DIST_THRESH and abs(cy - ey) < _OCR_CACHE_DIST_THRESH:
            if frame_count - fnum <= _OCR_CACHE_MAX_AGE:
                return plate
    return None


def _add_to_ocr_cache(cx, cy, plate, frame_count):
    """Add plate result to cache. Evict old entries."""
    global _ocr_cache
    _ocr_cache.append((cx, cy, plate, frame_count))
    _ocr_cache = [
        (x, y, p, f) for x, y, p, f in _ocr_cache
        if frame_count - f <= _OCR_CACHE_MAX_AGE
    ]


def _get_distance(h_box, v_box):
    """Calculate spatial distance between helmet box and vehicle box."""
    h_center_x = (h_box[0] + h_box[2]) / 2.0
    v_center_x = (v_box[0] + v_box[2]) / 2.0
    h_bottom_y = h_box[3]
    v_top_y = v_box[1]
    x_dist = abs(h_center_x - v_center_x)
    y_dist = max(0, v_top_y - h_bottom_y)
    return x_dist, y_dist


def _has_rider_nearby(helmet_results, vehicle_box, names):
    """Check if a rider is detected near the vehicle."""
    vx1, vy1, vx2, vy2 = vehicle_box
    v_width = vx2 - vx1
    v_height = vy2 - vy1
    for hbox in helmet_results.boxes:
        label = names[int(hbox.cls[0])].lower()
        if "rider" not in label:
            continue
        hx1, hy1, hx2, hy2 = map(int, hbox.xyxy[0])
        h_box = (hx1, hy1, hx2, hy2)
        x_dist, y_dist = _get_distance(h_box, vehicle_box)
        if x_dist < v_width * 0.6 and y_dist < v_height * 0.6:
            return True
    return False


def _detect_plate_in_motorcycle(motorcycle_crop, plate_model):
    """Run plate model on motorcycle crop and return (plate_crop, global_plate_box) or None."""
    if motorcycle_crop.size == 0:
        return None
    plate_results = plate_model(motorcycle_crop, conf=0.01)[0]
    if len(plate_results.boxes) == 0:
        return None
    best_pbox = None
    best_pconf = -1
    for pbox in plate_results.boxes:
        c = float(pbox.conf[0])
        if c > best_pconf:
            best_pconf = c
            best_pbox = pbox
    if best_pbox is None:
        return None
    px1, py1, px2, py2 = map(int, best_pbox.xyxy[0])
    plate_crop = motorcycle_crop[py1:py2, px1:px2]
    if plate_crop.size == 0:
        return None
    return plate_crop, (px1, py1, px2, py2)


def generate_frames(conf, stop_flag):
    """
    Generate frames from webcam with violation detection.
    Draws red box and "NO HELMET | Plate: X" for violations, logs and saves evidence.
    """
    cap = cv2.VideoCapture(0)
    frame_count = 0
    _ensure_violations_output()

    while cap.isOpened():
        if stop_flag():
            break

        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        vehicle_results = vehicle_model(frame, conf=conf, classes=TWO_WHEELER_CLASSES)[0]
        helmet_results = helmet_model(frame, conf=conf)[0]
        names = helmet_results.names

        vehicles = []
        for vbox in vehicle_results.boxes:
            vx1, vy1, vx2, vy2 = map(int, vbox.xyxy[0])
            vehicles.append({
                "box": (vx1, vy1, vx2, vy2),
                "has_violation": False,
            })

        def get_distance(h_box, v_box):
            h_center_x = (h_box[0] + h_box[2]) / 2.0
            v_center_x = (v_box[0] + v_box[2]) / 2.0
            h_bottom_y = h_box[3]
            v_top_y = v_box[1]
            x_dist = abs(h_center_x - v_center_x)
            y_dist = max(0, v_top_y - h_bottom_y)
            return x_dist, y_dist

        for hbox in helmet_results.boxes:
            hx1, hy1, hx2, hy2 = map(int, hbox.xyxy[0])
            h_box = (hx1, hy1, hx2, hy2)

            matched_vehicle = None
            min_distance = float("inf")
            for v in vehicles:
                vx1, vy1, vx2, vy2 = v["box"]
                v_width = vx2 - vx1
                v_height = vy2 - vy1
                x_dist, y_dist = get_distance(h_box, v["box"])
                if x_dist < v_width * 0.6 and y_dist < v_height * 0.6:
                    total_dist = x_dist + y_dist
                    if total_dist < min_distance:
                        min_distance = total_dist
                        matched_vehicle = v

            if not matched_vehicle:
                continue

            label = names[int(hbox.cls[0])].lower()
            h_conf = float(hbox.conf[0])

            if "no" in label and "helmet" in label:
                has_rider = _has_rider_nearby(helmet_results, matched_vehicle["box"], names)
                rider_required = any("rider" in names[i].lower() for i in range(len(names)))
                if not rider_required or has_rider:
                    matched_vehicle["has_violation"] = True

            color = RED if matched_vehicle["has_violation"] else (0, 255, 0)
            text = f"{label.replace('_', ' ').title()}: {h_conf:.2f}"
            cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), color, 3)
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(
                frame, (hx1, hy1 - th - 10), (hx1 + tw + 6, hy1), color, -1
            )
            cv2.putText(
                frame, text, (hx1 + 3, hy1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA
            )

        for v in vehicles:
            vx1, vy1, vx2, vy2 = v["box"]
            cv2.rectangle(frame, (vx1, vy1), (vx2, vy2), (0, 255, 255), 2)
            cv2.putText(
                frame, "Two-Wheeler", (vx1, vy1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
            )

            if v["has_violation"]:
                motorcycle_crop = frame[vy1:vy2, vx1:vx2]
                cx = (vx1 + vx2) // 2
                cy = (vy1 + vy2) // 2

                result = _detect_plate_in_motorcycle(motorcycle_crop, plate_model)
                plate_num = _get_cached_plate(cx, cy, frame_count)
                ran_ocr = False

                if plate_num is None:
                    if result is not None:
                        plate_crop, (px1, py1, px2, py2) = result
                        plate_num = read_plate(plate_crop)
                        _add_to_ocr_cache(cx, cy, plate_num or "", frame_count)
                        ran_ocr = True
                    else:
                        plate_num = ""

                # Draw red box and label for violation
                cv2.rectangle(frame, (vx1, vy1), (vx2, vy2), RED, 3)
                label_text = f"NO HELMET | Plate: {plate_num if plate_num else 'N/A'}"
                cv2.putText(
                    frame, label_text, (vx1, vy1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED, 2, cv2.LINE_AA
                )

                if result is not None:
                    _, (px1, py1, px2, py2) = result
                    global_px1, global_py1 = vx1 + px1, vy1 + py1
                    global_px2, global_py2 = vx1 + px2, vy1 + py2
                    cv2.rectangle(
                        frame, (global_px1, global_py1), (global_px2, global_py2),
                        (255, 0, 0), 2
                    )

                if ran_ocr:
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{frame_count}"
                    evidence_path = _save_violation_evidence(motorcycle_crop, ts)
                    _log_violation(plate_num or "UNKNOWN", frame_count, evidence_path)

        _, buffer = cv2.imencode(".jpg", frame)
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
        )

    cap.release()

"""
Webcam inference — detects motorcycle + rider together.
Persons (class 0) are paired with their nearest motorcycle (class 3)
and the two boxes are merged so the bounding box covers the full rider+bike.
"""

import os
from datetime import datetime
from typing import Callable, Generator

import cv2

from inference.models import vehicle_model, helmet_model, plate_model
from inference.tracker import VehicleTracker
from utils.ocr import read_plate

MOTORCYCLE_CLASS = [3]
PERSON_CLASS     = [0]

VIOLATIONS_DIR = "outputs/violations"
VIOLATIONS_CSV = "outputs/violations.csv"
RED    = (0, 0, 255)
GREEN  = (0, 200, 80)
CYAN   = (0, 220, 220)
BLUE   = (255, 100, 0)
YELLOW = (0, 215, 255)
WHITE  = (255, 255, 255)

MIN_AREA_RATIO   = 0.008
MIN_VEHICLE_CONF = 0.20
MIN_PERSON_CONF  = 0.25
NMS_IOU_THRESH   = 0.45
PERSON_BIKE_X_RATIO = 1.2
PERSON_BIKE_Y_RATIO = 1.5

_ocr_cache: list = []
_OCR_CACHE_MAX_AGE     = 90
_OCR_CACHE_DIST_THRESH = 90


def _ensure_output():
    os.makedirs(VIOLATIONS_DIR, exist_ok=True)
    if not os.path.exists(VIOLATIONS_CSV):
        with open(VIOLATIONS_CSV, "w") as f:
            f.write("timestamp,plate_number,frame_number,image_path\n")


def _log_violation(plate, frame, path):
    _ensure_output()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(VIOLATIONS_CSV, "a") as f:
        f.write(f"{ts},{plate or 'UNKNOWN'},{frame},{path}\n")


def _save_evidence(img, ts_str):
    _ensure_output()
    path = os.path.join(VIOLATIONS_DIR, f"violation_{ts_str}.jpg")
    cv2.imwrite(path, img)
    return path


def _get_cached_plate(cx, cy, frame_count):
    for ex, ey, plate, fnum in _ocr_cache:
        if abs(cx-ex) < _OCR_CACHE_DIST_THRESH and abs(cy-ey) < _OCR_CACHE_DIST_THRESH:
            if frame_count - fnum <= _OCR_CACHE_MAX_AGE:
                return plate
    return None


def _cache_plate(cx, cy, plate, frame_count):
    global _ocr_cache
    _ocr_cache.append((cx, cy, plate, frame_count))
    _ocr_cache = [(x, y, p, f) for x, y, p, f in _ocr_cache
                  if frame_count - f <= _OCR_CACHE_MAX_AGE]


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    if inter == 0:
        return 0.0
    union = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter / union if union > 0 else 0.0


def _nms(boxes):
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda v: v["conf"], reverse=True)
    kept, suppressed = [], set()
    for i, vi in enumerate(boxes):
        if i in suppressed:
            continue
        kept.append(vi)
        for j, vj in enumerate(boxes):
            if j <= i or j in suppressed:
                continue
            if _iou(vi["box"], vj["box"]) > NMS_IOU_THRESH:
                suppressed.add(j)
    return kept


def _merge_box(a, b):
    return (min(a[0], b[0]), min(a[1], b[1]),
            max(a[2], b[2]), max(a[3], b[3]))


def _pair_persons_to_bikes(bike_boxes, person_boxes):
    used_persons  = set()
    paired_persons = []
    for bi, bike in enumerate(bike_boxes):
        bx1, by1, bx2, by2 = bike["box"]
        bw = bx2 - bx1; bh = by2 - by1
        b_cx = (bx1 + bx2) / 2.0
        best_pi, best_score = None, float("inf")
        for pi, pb in enumerate(person_boxes):
            if pi in used_persons:
                continue
            px1, py1, px2, py2 = pb["box"]
            p_cx = (px1 + px2) / 2.0
            x_dist = abs(p_cx - b_cx)
            if x_dist > bw * PERSON_BIKE_X_RATIO:
                continue
            y_gap = py1 - by2
            if y_gap > bh * PERSON_BIKE_Y_RATIO:
                continue
            score = x_dist + abs(py2 - by1)
            if score < best_score:
                best_score = score; best_pi = pi
        if best_pi is not None:
            pb = person_boxes[best_pi]
            bike["box"]        = _merge_box(bike["box"], pb["box"])
            bike["person_box"] = pb["box"]
            used_persons.add(best_pi)
            paired_persons.append(pb["box"])
    return bike_boxes, paired_persons


def _helmet_distance(h_box, v_box):
    h_cx = (h_box[0] + h_box[2]) / 2.0
    v_cx = (v_box[0] + v_box[2]) / 2.0
    x_dist = abs(h_cx - v_cx)
    v_top_zone = v_box[1] + (v_box[3] - v_box[1]) * 0.4
    y_dist = max(0.0, v_box[1] - h_box[3])
    if h_box[3] >= v_box[1] and h_box[1] <= v_top_zone:
        y_dist = 0.0
    return x_dist, y_dist


def _best_plate_in_crop(crop):
    if crop is None or crop.size == 0:
        return None
    results = plate_model(crop, conf=0.01, device="mps")[0]
    if not results.boxes:
        return None
    best_conf, best_box = -1, None
    for pb in results.boxes:
        c = float(pb.conf[0])
        if c > best_conf:
            best_conf = c; best_box = pb
    if best_box is None:
        return None
    px1, py1, px2, py2 = map(int, best_box.xyxy[0])
    plate_crop = crop[py1:py2, px1:px2]
    return (plate_crop, (px1, py1, px2, py2)) if plate_crop.size else None


def _draw_label(img, text, x1, y1, color):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.60, 2)
    cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
    cv2.putText(img, text, (x1 + 3, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.60, WHITE, 2, cv2.LINE_AA)


def generate_frames(conf: float, stop_flag: Callable[[], bool]) -> Generator[bytes, None, None]:
    global _ocr_cache
    _ocr_cache = []

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return

    frame_count      = 0
    tracker          = VehicleTracker()
    logged_track_ids: set = set()
    _ensure_output()
    rider_class_exists = None

    try:
        while cap.isOpened():
            if stop_flag():
                break

            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            fh, fw = frame.shape[:2]
            min_area = fw * fh * MIN_AREA_RATIO

            # ── Detect motorcycles ────────────────────────────────────────
            bike_res   = vehicle_model(frame, conf=MIN_VEHICLE_CONF, classes=MOTORCYCLE_CLASS, device="mps")[0]
            person_res = vehicle_model(frame, conf=MIN_PERSON_CONF,  classes=PERSON_CLASS,     device="mps")[0]
            helmet_res = helmet_model(frame, conf=conf, device="mps")[0]
            names      = helmet_res.names

            if rider_class_exists is None:
                rider_class_exists = any("rider" in names[i].lower() for i in range(len(names)))

            raw_bikes = []
            for vb in bike_res.boxes:
                x1, y1, x2, y2 = map(int, vb.xyxy[0])
                if (x2-x1)*(y2-y1) < min_area:
                    continue
                raw_bikes.append({"box": (x1, y1, x2, y2), "conf": float(vb.conf[0]),
                                  "violation": False, "has_any_helmet": False,
                                  "plate": "", "person_box": None})

            raw_persons = []
            for pb in person_res.boxes:
                x1, y1, x2, y2 = map(int, pb.xyxy[0])
                raw_persons.append({"box": (x1, y1, x2, y2), "conf": float(pb.conf[0])})

            bikes   = _nms(raw_bikes)
            persons = _nms(raw_persons)

            # ── Pair persons → bikes ──────────────────────────────────────
            vehicles, paired_person_boxes = _pair_persons_to_bikes(bikes, persons)

            # ── Match helmets → vehicles ──────────────────────────────────
            for hb in helmet_res.boxes:
                hx1, hy1, hx2, hy2 = map(int, hb.xyxy[0])
                h_box  = (hx1, hy1, hx2, hy2)
                label  = names[int(hb.cls[0])].lower()
                h_conf = float(hb.conf[0])

                best_v, best_d = None, float("inf")
                for v in vehicles:
                    vx1, vy1, vx2, vy2 = v["box"]
                    vw, vh = vx2-vx1, vy2-vy1
                    xd, yd = _helmet_distance(h_box, v["box"])
                    if xd < vw * 0.70 and yd < vh * 0.80:
                        d = xd + yd
                        if d < best_d:
                            best_d = d; best_v = v

                if best_v is None:
                    continue
                best_v["has_any_helmet"] = True

                if "no" in label and "helmet" in label:
                    if rider_class_exists:
                        for hb2 in helmet_res.boxes:
                            lbl2 = names[int(hb2.cls[0])].lower()
                            if "rider" not in lbl2:
                                continue
                            rx1, ry1, rx2, ry2 = map(int, hb2.xyxy[0])
                            vx1, vy1, vx2, vy2 = best_v["box"]
                            vw, vh = vx2-vx1, vy2-vy1
                            rxd, ryd = _helmet_distance((rx1, ry1, rx2, ry2), best_v["box"])
                            if rxd < vw * 0.70 and ryd < vh * 0.80:
                                best_v["violation"] = True
                                break
                    else:
                        best_v["violation"] = True

                color = RED if best_v["violation"] else GREEN
                cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), color, 2)
                _draw_label(frame, f"{label.replace('_',' ').title()}: {h_conf:.2f}", hx1, hy1, color)

            # ── Draw person boxes ─────────────────────────────────────────
            for pb in paired_person_boxes:
                px1, py1, px2, py2 = pb
                cv2.rectangle(frame, (px1, py1), (px2, py2), YELLOW, 2)

            # ── Tracker + draw vehicles ───────────────────────────────────
            dets = [{"box": v["box"], "violation": v["violation"], "plate": v["plate"]}
                    for v in vehicles]
            active_tracks = tracker.update(dets)

            for v in vehicles:
                vx1, vy1, vx2, vy2 = v["box"]
                track    = next((t for t in active_tracks if t["box"] == v["box"]), None)
                track_id = track["track_id"] if track else -1

                if v["violation"]:
                    crop = frame[vy1:vy2, vx1:vx2]
                    cx   = (vx1+vx2)//2; cy = (vy1+vy2)//2

                    res       = _best_plate_in_crop(crop)
                    plate_num = _get_cached_plate(cx, cy, frame_count)
                    ran_ocr   = False

                    if plate_num is None:
                        if res:
                            plate_crop, _ = res
                            plate_num = read_plate(plate_crop)
                            _cache_plate(cx, cy, plate_num or "", frame_count)
                            ran_ocr = True
                        else:
                            plate_num = ""

                    cv2.rectangle(frame, (vx1, vy1), (vx2, vy2), RED, 3)
                    _draw_label(frame, f"#{track_id} VIOLATION | {plate_num or 'N/A'}", vx1, vy1, RED)

                    if res:
                        _, (px1, py1, px2, py2) = res
                        cv2.rectangle(frame, (vx1+px1, vy1+py1), (vx1+px2, vy1+py2), BLUE, 2)

                    if ran_ocr and track_id not in logged_track_ids:
                        logged_track_ids.add(track_id)
                        ts_str  = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{frame_count}"
                        ev_path = _save_evidence(crop, ts_str)
                        _log_violation(plate_num or "UNKNOWN", frame_count, ev_path)
                else:
                    cv2.rectangle(frame, (vx1, vy1), (vx2, vy2), CYAN, 2)
                    _draw_label(frame, f"#{track_id} Rider+Bike", vx1, vy1, (0, 140, 140))

            # HUD
            live_stats = tracker.get_stats()
            cv2.rectangle(frame, (0, 0), (250, 88), (0, 0, 0), -1)
            cv2.putText(frame, f"Total Counted  : {live_stats['total']}",
                        (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, WHITE, 2)
            cv2.putText(frame, f"Followed Rule  : {live_stats['safe']}",
                        (6, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.58, GREEN, 2)
            cv2.putText(frame, f"Violated Rule  : {live_stats['violations']}",
                        (6, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.58, RED, 2)

            ok, buffer = cv2.imencode(".jpg", frame)
            if not ok:
                continue
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")

    finally:
        cap.release()
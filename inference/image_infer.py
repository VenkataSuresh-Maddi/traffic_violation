"""
Image inference for traffic violation detection.
Detects no-helmet violations, extracts license plates via OCR,
logs violations, and saves evidence.

Logic:
- Vehicle detected + "nohelmet" label nearby  → VIOLATION
- Vehicle detected + "helmet" label nearby    → SAFE
- Vehicle detected + no helmet detection      → draw as safe (model uncertain)
- Vehicle too small (< 0.8% image area)       → skip background noise
- Overlapping vehicle boxes (IoU > 0.35)      → NMS keeps best one
"""

import os
from datetime import datetime

import cv2

from inference.models import vehicle_model, helmet_model, plate_model
from utils.ocr import read_plate

TWO_WHEELER_CLASSES = [3]

VIOLATIONS_DIR = "outputs/violations"
VIOLATIONS_CSV = "outputs/violations.csv"
RED   = (0, 0, 255)
GREEN = (0, 200, 80)
CYAN  = (0, 220, 220)
BLUE  = (255, 100, 0)
WHITE = (255, 255, 255)

# ── Tuning ────────────────────────────────────────────────────────────────────
MIN_AREA_RATIO   = 0.008  # 0.8% — catches distant/side bikes in wide scenes
MIN_VEHICLE_CONF = 0.20   # low threshold; NMS removes duplicates
NMS_IOU_THRESH   = 0.35   # tighter NMS for dense scenes
INFER_SIZE       = 1280   # high-res single pass — detects small bikes
# ─────────────────────────────────────────────────────────────────────────────


def _ensure_output():
    os.makedirs(VIOLATIONS_DIR, exist_ok=True)
    if not os.path.exists(VIOLATIONS_CSV):
        with open(VIOLATIONS_CSV, "w") as f:
            f.write("timestamp,plate_number,frame_number,image_path\n")


def _log_violation(plate_number: str, frame_number: int, image_path: str):
    _ensure_output()
    ts    = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    plate = plate_number if plate_number else "UNKNOWN"
    with open(VIOLATIONS_CSV, "a") as f:
        f.write(f"{ts},{plate},{frame_number},{image_path}\n")


def _save_evidence(image, ts_str: str) -> str:
    _ensure_output()
    path = os.path.join(VIOLATIONS_DIR, f"violation_{ts_str}.jpg")
    cv2.imwrite(path, image)
    return path


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    union = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter / union if union > 0 else 0.0


def _nms_vehicles(vehicles):
    if not vehicles:
        return []
    vehicles = sorted(vehicles, key=lambda v: v["conf"], reverse=True)
    kept, suppressed = [], set()
    for i, vi in enumerate(vehicles):
        if i in suppressed:
            continue
        kept.append(vi)
        for j, vj in enumerate(vehicles):
            if j <= i or j in suppressed:
                continue
            if _iou(vi["box"], vj["box"]) > NMS_IOU_THRESH:
                suppressed.add(j)
    return kept


def _helmet_distance(h_box, v_box):
    """
    Returns (x_dist, y_dist) — strict geometry to avoid cross-bike matches.
    Helmet must be horizontally within the vehicle width and
    vertically within the TOP 55% of the vehicle box.
    """
    h_cx = (h_box[0] + h_box[2]) / 2.0
    v_cx = (v_box[0] + v_box[2]) / 2.0
    x_dist = abs(h_cx - v_cx)

    # Helmet must sit in the upper portion of the vehicle bounding box
    v_upper_limit = v_box[1] + (v_box[3] - v_box[1]) * 0.55
    y_dist = max(0.0, v_box[1] - h_box[3])

    # If helmet centre is below the upper 55% of the vehicle → penalise heavily
    h_cy = (h_box[1] + h_box[3]) / 2.0
    if h_cy > v_upper_limit:
        y_dist += (h_cy - v_upper_limit)

    # If helmet overlaps the top zone of the vehicle, y_dist = 0
    v_top_zone = v_box[1] + (v_box[3] - v_box[1]) * 0.35
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
            best_conf = c
            best_box  = pb
    if best_box is None:
        return None
    px1, py1, px2, py2 = map(int, best_box.xyxy[0])
    plate_crop = crop[py1:py2, px1:px2]
    return (plate_crop, (px1, py1, px2, py2)) if plate_crop.size else None


def _draw_label(img, text: str, x1: int, y1: int, color, box_h: int = 80):
    scale = max(0.35, min(0.65, box_h / 130.0))
    thick = 2 if scale > 0.5 else 1
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 6, y1), color, -1)
    cv2.putText(img, text, (x1 + 3, y1 - 3),
                cv2.FONT_HERSHEY_SIMPLEX, scale, WHITE, thick, cv2.LINE_AA)


def process_image(input_path: str, output_path: str, conf: float):
    """
    Process a single image for helmet-violation detection.
    Returns (total_violations, plate_number_found, stats_dict)
    """
    image = cv2.imread(input_path)
    if image is None:
        return 0, None, {"total": 0, "violations": 0, "safe": 0}

    out    = image.copy()
    ih, iw = image.shape[:2]
    min_area = iw * ih * MIN_AREA_RATIO
    _ensure_output()

    # ── 1. Detect vehicles at 1280px ────────────────────────────────────────
    vehicle_res = vehicle_model(
        image, conf=MIN_VEHICLE_CONF,
        classes=TWO_WHEELER_CLASSES,
        imgsz=INFER_SIZE, device="mps"
    )[0]

    raw_vehicles = []
    for vb in vehicle_res.boxes:
        x1, y1, x2, y2 = map(int, vb.xyxy[0])
        vconf = float(vb.conf[0])
        if (x2 - x1) * (y2 - y1) < min_area:
            continue
        raw_vehicles.append({
            "box":       (x1, y1, x2, y2),
            "conf":      vconf,
            "violation": False,
            "has_any_helmet": False,
            "plate":     "",
            "plate_box": None,
        })

    vehicles = _nms_vehicles(raw_vehicles)

    # ── 2. Detect helmets at 1280px ─────────────────────────────────────────
    helmet_res = helmet_model(
        image, conf=0.10,  # low threshold to catch weak nohelmet detections
        imgsz=INFER_SIZE, device="mps"
    )[0]
    names = helmet_res.names
    rider_class_exists = any("rider" in names[i].lower() for i in range(len(names)))

    # ── 3. Match helmet detections → nearest vehicle (strict, exclusive) ────────
    # Each helmet is assigned to exactly ONE vehicle — the closest qualifying one.
    # This prevents a helmet from a neighbouring bike being assigned to the wrong vehicle.
    for hb in helmet_res.boxes:
        hx1, hy1, hx2, hy2 = map(int, hb.xyxy[0])
        h_box  = (hx1, hy1, hx2, hy2)
        label  = names[int(hb.cls[0])].lower()
        h_conf = float(hb.conf[0])

        # Score = normalised distance — must be within strict bounds
        best_v, best_score = None, float("inf")
        for v in vehicles:
            vx1, vy1, vx2, vy2 = v["box"]
            vw = vx2 - vx1
            vh = vy2 - vy1
            if vw == 0 or vh == 0:
                continue
            xd, yd = _helmet_distance(h_box, v["box"])
            # Normalise by vehicle size so distant small bikes aren't penalised
            norm_xd = xd / vw
            norm_yd = yd / vh
            if norm_xd < 0.50 and norm_yd < 0.60:
                score = norm_xd + norm_yd
                if score < best_score:
                    best_score = score
                    best_v     = v

        if best_v is None:
            continue

        # "helmet" (safe) label: require conf >= 0.25 to trust it
        # "nohelmet" (violation) label: accept from conf=0.10
        # This prevents low-confidence false "helmet" detections from hiding violations
        if "no" not in label and h_conf < 0.25:
            continue  # ignore weak helmet-safe detections

        best_v["has_any_helmet"] = True

        # Only flag violation when model explicitly detects "nohelmet"
        if "no" in label and "helmet" in label:
            if rider_class_exists:
                for hb2 in helmet_res.boxes:
                    lbl2 = names[int(hb2.cls[0])].lower()
                    if "rider" not in lbl2:
                        continue
                    rx1, ry1, rx2, ry2 = map(int, hb2.xyxy[0])
                    vx1, vy1, vx2, vy2 = best_v["box"]
                    vw = vx2 - vx1; vh = vy2 - vy1
                    rxd, ryd = _helmet_distance((rx1, ry1, rx2, ry2), best_v["box"])
                    if rxd < vw * 0.70 and ryd < vh * 0.80:
                        best_v["violation"] = True
                        break
            else:
                best_v["violation"] = True

        # Draw helmet box
        color = RED if best_v["violation"] else GREEN
        txt   = f"{label.replace('_', ' ').title()}: {h_conf:.2f}"
        vx1, vy1, vx2, vy2 = best_v["box"]
        cv2.rectangle(out, (hx1, hy1), (hx2, hy2), color, 2)
        _draw_label(out, txt, hx1, hy1, color, vy2 - vy1)

    # ── 4. Large close vehicles with NO helmet detection = violation ───────────
    # If the helmet model scanned a large/close bike and found nothing, the rider
    # is not wearing a helmet. Small distant bikes get benefit of the doubt.
    LARGE_VEHICLE_AREA = iw * ih * 0.025  # >= 2.5% of frame area
    for v in vehicles:
        if not v.get("has_any_helmet", False):
            vx1, vy1, vx2, vy2 = v["box"]
            area = (vx2 - vx1) * (vy2 - vy1)
            if area >= LARGE_VEHICLE_AREA:
                v["violation"] = True

    # ── 5. Plate detection, OCR, evidence saving ─────────────────────────────
    plate_found  = None
    viol_counter = 0

    for v in vehicles:
        vx1, vy1, vx2, vy2 = v["box"]
        box_h = vy2 - vy1
        crop  = image[vy1:vy2, vx1:vx2]

        if v["violation"]:
            res = _best_plate_in_crop(crop)
            plate_num = ""
            if res:
                plate_crop, (px1, py1, px2, py2) = res
                plate_num      = read_plate(plate_crop)
                v["plate"]     = plate_num
                v["plate_box"] = (vx1 + px1, vy1 + py1, vx1 + px2, vy1 + py2)
                if plate_num and not plate_found:
                    plate_found = plate_num

            viol_counter += 1
            ts_str  = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{viol_counter}"
            ev_path = _save_evidence(crop, ts_str)
            _log_violation(plate_num or "UNKNOWN", 0, ev_path)

            cv2.rectangle(out, (vx1, vy1), (vx2, vy2), RED, 2)
            lbl = f"VIOLATION | Plate: {plate_num if plate_num else 'N/A'}"
            _draw_label(out, lbl, vx1, vy1, RED, box_h)

            if v["plate_box"]:
                px1, py1, px2, py2 = v["plate_box"]
                cv2.rectangle(out, (px1, py1), (px2, py2), BLUE, 2)
        else:
            cv2.rectangle(out, (vx1, vy1), (vx2, vy2), CYAN, 2)
            _draw_label(out, "Two-Wheeler", vx1, vy1, (0, 140, 140), box_h)

    # ── 6. Stats + HUD overlay ───────────────────────────────────────────────
    stats = {
        "total":      len(vehicles),
        "violations": sum(1 for v in vehicles if v["violation"]),
        "safe":       sum(1 for v in vehicles if not v["violation"]),
    }

    hud_y = 30
    cv2.rectangle(out, (0, 0), (320, 95), (0, 0, 0), -1)
    cv2.putText(out, f"Total Vehicles : {stats['total']}",
                (10, hud_y),    cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2)
    cv2.putText(out, f"Followed Rule  : {stats['safe']}",
                (10, hud_y+28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, GREEN, 2)
    cv2.putText(out, f"Violated Rule  : {stats['violations']}",
                (10, hud_y+56), cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED,   2)

    cv2.imwrite(output_path, out)
    return stats["violations"], plate_found, stats
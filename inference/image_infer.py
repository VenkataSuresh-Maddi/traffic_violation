"""
Image inference — detects motorcycle + rider together.
Persons (class 0) are paired with their nearest motorcycle (class 3)
and the two boxes are merged so the violation crop always contains the full rider.
"""

import os
from datetime import datetime

import cv2

from inference.models import vehicle_model, helmet_model, plate_model
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

MIN_AREA_RATIO      = 0.03
MIN_VEHICLE_CONF    = 0.35
MIN_PERSON_CONF     = 0.25
NMS_IOU_THRESH      = 0.45
PERSON_BIKE_X_RATIO = 1.2
PERSON_BIKE_Y_RATIO = 1.5


def _ensure_output():
    os.makedirs(VIOLATIONS_DIR, exist_ok=True)
    if not os.path.exists(VIOLATIONS_CSV):
        with open(VIOLATIONS_CSV, "w") as f:
            f.write("timestamp,plate_number,frame_number,image_path\n")


def _log_violation(plate_number, frame_number, image_path):
    _ensure_output()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(VIOLATIONS_CSV, "a") as f:
        f.write(f"{ts},{plate_number or 'UNKNOWN'},{frame_number},{image_path}\n")


def _save_evidence(image, ts_str):
    _ensure_output()
    path = os.path.join(VIOLATIONS_DIR, f"violation_{ts_str}.jpg")
    cv2.imwrite(path, image)
    return path


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


def _nms(vehicles):
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


def _merge_box(a, b):
    return (min(a[0], b[0]), min(a[1], b[1]),
            max(a[2], b[2]), max(a[3], b[3]))


def _pair_persons_to_bikes(bike_boxes, person_boxes):
    used_persons   = set()
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
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    cv2.rectangle(img, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)
    cv2.putText(img, text, (x1 + 3, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, WHITE, 2, cv2.LINE_AA)


def process_image(input_path, output_path, conf):
    image = cv2.imread(input_path)
    if image is None:
        return 0, None, {"total": 0, "violations": 0, "safe": 0}

    out    = image.copy()
    ih, iw = image.shape[:2]
    min_area = iw * ih * MIN_AREA_RATIO
    _ensure_output()

    # ── Detect motorcycles ────────────────────────────────────────────────────
    bike_res   = vehicle_model(image, conf=max(conf, MIN_VEHICLE_CONF), classes=MOTORCYCLE_CLASS, device="mps")[0]
    person_res = vehicle_model(image, conf=MIN_PERSON_CONF,             classes=PERSON_CLASS,     device="mps")[0]

    raw_bikes = []
    for vb in bike_res.boxes:
        x1, y1, x2, y2 = map(int, vb.xyxy[0])
        vconf = float(vb.conf[0])
        if (x2-x1)*(y2-y1) < min_area or vconf < MIN_VEHICLE_CONF:
            continue
        raw_bikes.append({"box": (x1, y1, x2, y2), "conf": vconf,
                           "violation": False, "has_any_helmet": False,
                           "plate": "", "plate_box": None, "person_box": None})

    raw_persons = []
    for pb in person_res.boxes:
        x1, y1, x2, y2 = map(int, pb.xyxy[0])
        raw_persons.append({"box": (x1, y1, x2, y2), "conf": float(pb.conf[0])})

    bikes   = _nms(raw_bikes)
    persons = _nms(raw_persons)

    # ── Pair persons → bikes ──────────────────────────────────────────────────
    vehicles, paired_person_boxes = _pair_persons_to_bikes(bikes, persons)

    # ── Detect helmets ────────────────────────────────────────────────────────
    helmet_res = helmet_model(image, conf=conf, device="mps")[0]
    names      = helmet_res.names
    rider_class_exists = any("rider" in names[i].lower() for i in range(len(names)))

    for hb in helmet_res.boxes:
        hx1, hy1, hx2, hy2 = map(int, hb.xyxy[0])
        h_box  = (hx1, hy1, hx2, hy2)
        label  = names[int(hb.cls[0])].lower()
        h_conf = float(hb.conf[0])

        best_v, best_dist = None, float("inf")
        for v in vehicles:
            vx1, vy1, vx2, vy2 = v["box"]
            vw = vx2-vx1; vh = vy2-vy1
            xd, yd = _helmet_distance(h_box, v["box"])
            if xd < vw * 0.70 and yd < vh * 0.80:
                d = xd + yd
                if d < best_dist:
                    best_dist = d; best_v = v

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
                    vw = vx2-vx1; vh = vy2-vy1
                    rxd, ryd = _helmet_distance((rx1, ry1, rx2, ry2), best_v["box"])
                    if rxd < vw * 0.70 and ryd < vh * 0.80:
                        best_v["violation"] = True
                        break
            else:
                best_v["violation"] = True

        color = RED if best_v["violation"] else GREEN
        cv2.rectangle(out, (hx1, hy1), (hx2, hy2), color, 2)
        _draw_label(out, f"{label.replace('_',' ').title()}: {h_conf:.2f}", hx1, hy1, color)

    # ── Draw person boxes (yellow) ────────────────────────────────────────────
    for pb in paired_person_boxes:
        px1, py1, px2, py2 = pb
        cv2.rectangle(out, (px1, py1), (px2, py2), YELLOW, 2)
        _draw_label(out, "Rider", px1, py1, YELLOW)

    # ── Plate + evidence ──────────────────────────────────────────────────────
    plate_found  = None
    viol_counter = 0

    for v in vehicles:
        vx1, vy1, vx2, vy2 = v["box"]
        crop = image[vy1:vy2, vx1:vx2]

        if v["violation"]:
            res = _best_plate_in_crop(crop)
            plate_num = ""
            if res:
                plate_crop, (px1, py1, px2, py2) = res
                plate_num      = read_plate(plate_crop)
                v["plate"]     = plate_num
                v["plate_box"] = (vx1+px1, vy1+py1, vx1+px2, vy1+py2)
                if plate_num and not plate_found:
                    plate_found = plate_num

            viol_counter += 1
            ts_str  = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{viol_counter}"
            ev_path = _save_evidence(crop, ts_str)
            _log_violation(plate_num or "UNKNOWN", 0, ev_path)

            cv2.rectangle(out, (vx1, vy1), (vx2, vy2), RED, 3)
            _draw_label(out, f"VIOLATION | Plate: {plate_num or 'N/A'}", vx1, vy1, RED)

            if v["plate_box"]:
                px1, py1, px2, py2 = v["plate_box"]
                cv2.rectangle(out, (px1, py1), (px2, py2), BLUE, 2)
        else:
            cv2.rectangle(out, (vx1, vy1), (vx2, vy2), CYAN, 2)
            _draw_label(out, "Rider+Bike ✓", vx1, vy1, (0, 140, 140))

    stats = {
        "total":      len(vehicles),
        "violations": sum(1 for v in vehicles if v["violation"]),
        "safe":       sum(1 for v in vehicles if not v["violation"]),
    }

    cv2.rectangle(out, (0, 0), (320, 95), (0, 0, 0), -1)
    cv2.putText(out, f"Total Vehicles : {stats['total']}",     (10, 30),   cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2)
    cv2.putText(out, f"Followed Rule  : {stats['safe']}",      (10, 58),   cv2.FONT_HERSHEY_SIMPLEX, 0.7, GREEN, 2)
    cv2.putText(out, f"Violated Rule  : {stats['violations']}", (10, 86),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED,   2)

    cv2.imwrite(output_path, out)
    return stats["violations"], plate_found, stats
"""
Video inference for traffic violation detection.
Uses tiled detection + VehicleTracker to count every unique vehicle
exactly once across all frames — no double counting.

Tiled detection: splits each frame into overlapping tiles and merges
results with global NMS. This catches small/distant bikes that a single
640px pass misses entirely.
"""

import os
from datetime import datetime

import cv2

from inference.models import vehicle_model, helmet_model, plate_model
from inference.tracker import VehicleTracker
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
MIN_AREA_RATIO   = 0.0008  # 0.08% — catches small distant bikes in wide shots
MIN_VEHICLE_CONF = 0.20    # low threshold; NMS cleans false positives
NMS_IOU_THRESH   = 0.35    # global NMS across tiles — tighter for dense traffic
PROCESS_INTERVAL = 3       # run detection every N frames
TILE_OVERLAP     = 0.25    # 25% overlap between tiles
# ─────────────────────────────────────────────────────────────────────────────

_ocr_cache: list = []
_OCR_CACHE_MAX_AGE     = 45
_OCR_CACHE_DIST_THRESH = 90


def _ensure_output():
    os.makedirs(VIOLATIONS_DIR, exist_ok=True)
    if not os.path.exists(VIOLATIONS_CSV):
        with open(VIOLATIONS_CSV, "w") as f:
            f.write("timestamp,plate_number,frame_number,image_path\n")


def _log_violation(plate: str, frame: int, path: str):
    _ensure_output()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(VIOLATIONS_CSV, "a") as f:
        f.write(f"{ts},{plate or 'UNKNOWN'},{frame},{path}\n")


def _save_evidence(img, ts_str: str) -> str:
    _ensure_output()
    path = os.path.join(VIOLATIONS_DIR, f"violation_{ts_str}.jpg")
    cv2.imwrite(path, img)
    return path


def _get_cached_plate(cx, cy, frame_count):
    for ex, ey, plate, fnum in _ocr_cache:
        if abs(cx - ex) < _OCR_CACHE_DIST_THRESH and abs(cy - ey) < _OCR_CACHE_DIST_THRESH:
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
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    union = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter / union if union > 0 else 0.0


def _global_nms(boxes_confs, iou_thresh=NMS_IOU_THRESH):
    """
    boxes_confs: list of (x1,y1,x2,y2,conf)
    Returns filtered list after NMS.
    """
    if not boxes_confs:
        return []
    boxes_confs = sorted(boxes_confs, key=lambda x: x[4], reverse=True)
    kept, suppressed = [], set()
    for i, bi in enumerate(boxes_confs):
        if i in suppressed:
            continue
        kept.append(bi)
        for j, bj in enumerate(boxes_confs):
            if j <= i or j in suppressed:
                continue
            if _iou(bi[:4], bj[:4]) > iou_thresh:
                suppressed.add(j)
    return kept


def _detect_vehicles_tiled(frame, conf_thresh):
    """
    Run vehicle detection on the full frame + a 2x2 tile grid.
    Merge all boxes with global NMS.
    Returns list of dicts: {box, conf}
    """
    fh, fw = frame.shape[:2]
    all_boxes = []

    # Pass 1: full frame at 1280px for global context
    res_full = vehicle_model(frame, conf=conf_thresh, classes=TWO_WHEELER_CLASSES, imgsz=1280, device="mps")[0]
    for vb in res_full.boxes:
        x1, y1, x2, y2 = map(int, vb.xyxy[0])
        all_boxes.append((x1, y1, x2, y2, float(vb.conf[0])))

    # Pass 2: 2x2 tiles with overlap for small/distant vehicles
    step_x = int(fw * (1 - TILE_OVERLAP) / 2)
    step_y = int(fh * (1 - TILE_OVERLAP) / 2)
    tile_w  = fw - step_x
    tile_h  = fh - step_y

    for row in range(2):
        for col in range(2):
            tx1 = col * step_x
            ty1 = row * step_y
            tx2 = min(tx1 + tile_w, fw)
            ty2 = min(ty1 + tile_h, fh)
            tile = frame[ty1:ty2, tx1:tx2]
            if tile.size == 0:
                continue
            res_tile = vehicle_model(tile, conf=conf_thresh, classes=TWO_WHEELER_CLASSES, imgsz=640, device="mps")[0]
            for vb in res_tile.boxes:
                bx1, by1, bx2, by2 = map(int, vb.xyxy[0])
                # Translate back to full-frame coords
                all_boxes.append((
                    bx1 + tx1, by1 + ty1,
                    bx2 + tx1, by2 + ty1,
                    float(vb.conf[0])
                ))

    # Global NMS to remove duplicates across tiles
    kept = _global_nms(all_boxes)
    return [{"box": (b[0], b[1], b[2], b[3]), "conf": b[4]} for b in kept]


def _detect_helmets_tiled(frame, conf_thresh):
    """
    Run helmet detection on full frame + tiles.
    Returns list of (hx1, hy1, hx2, hy2, label, conf).
    """
    fh, fw = frame.shape[:2]
    all_helmets = []
    names = None

    # Full frame
    res_full = helmet_model(frame, conf=conf_thresh, imgsz=1280, device="mps")[0]
    names = res_full.names
    for hb in res_full.boxes:
        x1, y1, x2, y2 = map(int, hb.xyxy[0])
        all_helmets.append((x1, y1, x2, y2, float(hb.conf[0]),
                            names[int(hb.cls[0])].lower()))

    # 2x2 tiles
    step_x = int(fw * (1 - TILE_OVERLAP) / 2)
    step_y = int(fh * (1 - TILE_OVERLAP) / 2)
    tile_w  = fw - step_x
    tile_h  = fh - step_y

    for row in range(2):
        for col in range(2):
            tx1 = col * step_x
            ty1 = row * step_y
            tx2 = min(tx1 + tile_w, fw)
            ty2 = min(ty1 + tile_h, fh)
            tile = frame[ty1:ty2, tx1:tx2]
            if tile.size == 0:
                continue
            res_tile = helmet_model(tile, conf=conf_thresh, imgsz=640, device="mps")[0]
            tn = res_tile.names
            for hb in res_tile.boxes:
                bx1, by1, bx2, by2 = map(int, hb.xyxy[0])
                all_helmets.append((
                    bx1 + tx1, by1 + ty1,
                    bx2 + tx1, by2 + ty1,
                    float(hb.conf[0]),
                    tn[int(hb.cls[0])].lower()
                ))

    # NMS on helmets too
    helm_bc = [(h[0], h[1], h[2], h[3], h[4]) for h in all_helmets]
    kept_bc  = _global_nms(helm_bc)
    kept_set = {(b[0], b[1], b[2], b[3]) for b in kept_bc}

    deduped = []
    seen = set()
    for h in all_helmets:
        key = (h[0], h[1], h[2], h[3])
        if key in kept_set and key not in seen:
            seen.add(key)
            deduped.append(h)

    return deduped, names


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
            best_conf = c
            best_box  = pb
    if best_box is None:
        return None
    px1, py1, px2, py2 = map(int, best_box.xyxy[0])
    plate_crop = crop[py1:py2, px1:px2]
    return (plate_crop, (px1, py1, px2, py2)) if plate_crop.size else None


def _draw_label(img, text: str, x1: int, y1: int, color, box_h: int = 80):
    # Scale font with box height — small distant bikes get smaller labels
    scale = max(0.30, min(0.55, box_h / 150.0))
    thick = 1
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw + 4, y1), color, -1)
    cv2.putText(img, text, (x1 + 2, y1 - 2),
                cv2.FONT_HERSHEY_SIMPLEX, scale, WHITE, thick, cv2.LINE_AA)


def process_video(ip: str, op: str, conf: float, progress_callback=None):
    global _ocr_cache
    _ocr_cache = []

    cap = cv2.VideoCapture(ip)
    if not cap.isOpened():
        return {"total": 0, "violations": 0, "safe": 0}

    w            = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h            = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    min_area     = w * h * MIN_AREA_RATIO

    out = cv2.VideoWriter(op, cv2.VideoWriter_fourcc(*"avc1"), fps, (w, h))

    tracker          = VehicleTracker()
    frame_count      = 0
    cached_helmets   = []
    cached_vehicles  = []
    cached_plates    = []
    logged_track_ids: set = set()

    _ensure_output()
    rider_class_exists = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        if frame_count % PROCESS_INTERVAL == 1:
            if progress_callback:
                progress_callback(frame_count, total_frames)

            cached_helmets  = []
            cached_vehicles = []
            cached_plates   = []

            # ── Tiled detection ───────────────────────────────────────────
            raw_dets   = _detect_vehicles_tiled(frame, MIN_VEHICLE_CONF)
            helm_dets, names = _detect_helmets_tiled(frame, conf)

            if rider_class_exists is None and names:
                rider_class_exists = any(
                    "rider" in names[i].lower() for i in range(len(names))
                )

            # Build vehicle list
            vehicles = []
            for det in raw_dets:
                x1, y1, x2, y2 = det["box"]
                area = (x2 - x1) * (y2 - y1)
                if area < min_area:
                    continue
                vehicles.append({
                    "box":            (x1, y1, x2, y2),
                    "conf":           det["conf"],
                    "violation":      False,
                    "has_any_helmet": False,
                    "plate":          "",
                })

            # Match helmets → vehicles
            for (hx1, hy1, hx2, hy2, h_conf, label) in helm_dets:
                h_box = (hx1, hy1, hx2, hy2)

                best_v, best_d = None, float("inf")
                for v in vehicles:
                    vx1, vy1, vx2, vy2 = v["box"]
                    vw, vh = vx2 - vx1, vy2 - vy1
                    xd, yd = _helmet_distance(h_box, v["box"])
                    if xd < vw * 0.70 and yd < vh * 0.80:
                        d = xd + yd
                        if d < best_d:
                            best_d = d
                            best_v = v

                if best_v is None:
                    continue

                best_v["has_any_helmet"] = True

                if "no" in label and "helmet" in label:
                    if rider_class_exists:
                        for (rx1, ry1, rx2, ry2, r_conf, rlabel) in helm_dets:
                            if "rider" not in rlabel:
                                continue
                            vx1, vy1, vx2, vy2 = best_v["box"]
                            vw, vh = vx2 - vx1, vy2 - vy1
                            rxd, ryd = _helmet_distance(
                                (rx1, ry1, rx2, ry2), best_v["box"]
                            )
                            if rxd < vw * 0.70 and ryd < vh * 0.80:
                                best_v["violation"] = True
                                break
                    else:
                        best_v["violation"] = True

                color = RED if best_v["violation"] else GREEN
                txt   = f"{label.replace('_', ' ').title()}: {h_conf:.2f}"
                cached_helmets.append((hx1, hy1, hx2, hy2, color, txt))

            # Note: only flag violation when model explicitly detects "nohelmet"
            # Do NOT flag vehicles with no helmet detection as violations —
            # the helmet model misses many helmeted riders on overhead footage.

            # Feed into tracker
            dets = [{"box": v["box"], "violation": v["violation"], "plate": v["plate"]}
                    for v in vehicles]
            active_tracks = tracker.update(dets)

            for v in vehicles:
                vx1, vy1, vx2, vy2 = v["box"]
                track    = next((t for t in active_tracks if t["box"] == v["box"]), None)
                track_id = track["track_id"] if track else -1

                cached_vehicles.append((vx1, vy1, vx2, vy2, v["violation"], track_id))

                if v["violation"]:
                    crop = frame[vy1:vy2, vx1:vx2]
                    cx   = (vx1 + vx2) // 2
                    cy   = (vy1 + vy2) // 2

                    res       = _best_plate_in_crop(crop)
                    plate_num = _get_cached_plate(cx, cy, frame_count)
                    ran_ocr   = False

                    if plate_num is None:
                        if res:
                            plate_crop, (px1, py1, px2, py2) = res
                            plate_num = read_plate(plate_crop)
                            _cache_plate(cx, cy, plate_num or "", frame_count)
                            ran_ocr = True
                        else:
                            plate_num = ""

                    gx1 = gy1 = gx2 = gy2 = 0
                    if res:
                        _, (px1, py1, px2, py2) = res
                        gx1, gy1 = vx1 + px1, vy1 + py1
                        gx2, gy2 = vx1 + px2, vy1 + py2

                    cached_plates.append((
                        vx1, vy1, vx2, vy2,
                        gx1, gy1, gx2, gy2,
                        plate_num, track_id
                    ))

                    if ran_ocr and track_id not in logged_track_ids:
                        logged_track_ids.add(track_id)
                        ts_str  = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{frame_count}"
                        ev_path = _save_evidence(crop, ts_str)
                        _log_violation(plate_num or "UNKNOWN", frame_count, ev_path)

        # ── DRAWING ───────────────────────────────────────────────────────────
        for hx1, hy1, hx2, hy2, color, txt in cached_helmets:
            cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), color, 1)
            _draw_label(frame, txt, hx1, hy1, color, hy2 - hy1)

        for (vx1, vy1, vx2, vy2, is_viol, tid) in cached_vehicles:
            if not is_viol:
                cv2.rectangle(frame, (vx1, vy1), (vx2, vy2), CYAN, 1)
                _draw_label(frame, f"#{tid} Safe", vx1, vy1, (0, 140, 140), vy2 - vy1)

        for item in cached_plates:
            vx1, vy1, vx2, vy2, px1, py1, px2, py2, plate_num, tid = item
            cv2.rectangle(frame, (vx1, vy1), (vx2, vy2), RED, 2)
            lbl = f"#{tid} VIOL | {plate_num if plate_num else 'N/A'}"
            _draw_label(frame, lbl, vx1, vy1, RED, vy2 - vy1)
            if px1 != px2 or py1 != py2:
                cv2.rectangle(frame, (px1, py1), (px2, py2), BLUE, 1)

        # HUD
        live_stats = tracker.get_stats()
        cv2.rectangle(frame, (0, 0), (250, 88), (0, 0, 0), -1)
        cv2.putText(frame, f"Total Counted  : {live_stats['total']}",
                    (6, 24),  cv2.FONT_HERSHEY_SIMPLEX, 0.58, WHITE, 2)
        cv2.putText(frame, f"Followed Rule  : {live_stats['safe']}",
                    (6, 52),  cv2.FONT_HERSHEY_SIMPLEX, 0.58, GREEN, 2)
        cv2.putText(frame, f"Violated Rule  : {live_stats['violations']}",
                    (6, 80),  cv2.FONT_HERSHEY_SIMPLEX, 0.58, RED,   2)

        out.write(frame)

    cap.release()
    out.release()

    if progress_callback:
        progress_callback(total_frames, total_frames)

    return tracker.get_stats()
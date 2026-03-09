"""
Image inference for traffic violation detection.
Detects no-helmet violations, extracts license plates via OCR, logs violations, and saves evidence.
"""

import os
from datetime import datetime

import cv2
from ultralytics import YOLO

from utils.ocr import read_plate


# ---------------- LOAD MODELS ----------------
vehicle_model = YOLO("yolov8n.pt")  # Standard model for vehicle (motorcycle) detection
helmet_model = YOLO("models/best.pt")
plate_model = YOLO("models/plate_best.pt")

# COCO dataset class IDs: 1 (bicycle/scooter), 3 (motorcycle)
TWO_WHEELER_CLASSES = [1, 3]

# Output paths for violations
VIOLATIONS_DIR = "outputs/violations"
VIOLATIONS_CSV = "outputs/violations.csv"
RED = (0, 0, 255)


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
    """Check if a rider is detected near the vehicle (for violation condition)."""
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
    """
    Run plate model on motorcycle crop and return (plate_crop, global_plate_box) or None.
    """
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


def process_image(input_path, output_path, conf):
    """
    Process a single image for traffic violations.

    Violation condition: rider + motorcycle + no_helmet (all associated).
    For each violation: detect plate, run OCR, draw red box, log to CSV, save evidence.
    """
    image = cv2.imread(input_path)
    if image is None:
        return 0, None

    output_image = image.copy()
    total_violations = 0
    plate_number_found = None
    violation_counter = 0

    _ensure_violations_output()

    # -------- STEP 1 & 2: VEHICLE AND HELMET DETECTION --------
    vehicle_results = vehicle_model(image, conf=conf, classes=TWO_WHEELER_CLASSES)[0]
    helmet_results = helmet_model(image, conf=conf)[0]
    names = helmet_results.names

    vehicles = []
    for vbox in vehicle_results.boxes:
        vx1, vy1, vx2, vy2 = map(int, vbox.xyxy[0])
        vehicles.append({
            "box": (vx1, vy1, vx2, vy2),
            "has_violation": False,
            "plate_text": "",
            "plate_box": None,
        })

    # -------- STEP 3: MATCH HELMETS TO VEHICLES, CHECK VIOLATION --------
    for hbox in helmet_results.boxes:
        hx1, hy1, hx2, hy2 = map(int, hbox.xyxy[0])
        h_box = (hx1, hy1, hx2, hy2)

        matched_vehicle = None
        min_distance = float("inf")
        for v in vehicles:
            vx1, vy1, vx2, vy2 = v["box"]
            v_width = vx2 - vx1
            v_height = vy2 - vy1
            x_dist, y_dist = _get_distance(h_box, v["box"])
            if x_dist < v_width * 0.6 and y_dist < v_height * 0.6:
                total_dist = x_dist + y_dist
                if total_dist < min_distance:
                    min_distance = total_dist
                    matched_vehicle = v

        if not matched_vehicle:
            continue

        label = names[int(hbox.cls[0])].lower()
        h_conf = float(hbox.conf[0])

        # Violation: no_helmet + rider + motorcycle (rider check if model has it)
        if "no" in label and "helmet" in label:
            has_rider = _has_rider_nearby(helmet_results, matched_vehicle["box"], names)
            # If model has rider class, require it; else allow motorcycle + no_helmet only
            rider_required = any("rider" in names[i].lower() for i in range(len(names)))
            if not rider_required or has_rider:
                matched_vehicle["has_violation"] = True
                total_violations += 1

        color = RED if matched_vehicle["has_violation"] else (0, 255, 0)
        text = f"{label.replace('_', ' ').title()}: {h_conf:.2f}"
        cv2.rectangle(output_image, (hx1, hy1), (hx2, hy2), color, 3)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(
            output_image,
            (hx1, hy1 - th - 10),
            (hx1 + tw + 6, hy1),
            color,
            -1,
        )
        cv2.putText(
            output_image, text, (hx1 + 3, hy1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA
        )

    # -------- STEP 4: PLATE DETECTION, OCR, LOGGING, DRAWING FOR VIOLATORS --------
    for v in vehicles:
        vx1, vy1, vx2, vy2 = v["box"]

        if v["has_violation"]:
            motorcycle_crop = image[vy1:vy2, vx1:vx2]
            result = _detect_plate_in_motorcycle(motorcycle_crop, plate_model)

            plate_num = ""
            if result is not None:
                plate_crop, (px1, py1, px2, py2) = result
                plate_num = read_plate(plate_crop)
                if plate_num and not plate_number_found:
                    plate_number_found = plate_num
                v["plate_text"] = plate_num
                v["plate_box"] = (vx1 + px1, vy1 + py1, vx1 + px2, vy1 + py2)

            # Draw RED bounding box for violation (motorcycle region)
            cv2.rectangle(output_image, (vx1, vy1), (vx2, vy2), RED, 3)
            label_text = f"NO HELMET | Plate: {plate_num if plate_num else 'N/A'}"
            cv2.putText(
                output_image,
                label_text,
                (vx1, vy1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                RED,
                2,
                cv2.LINE_AA,
            )

            # Log violation and save evidence (unique timestamp per violation)
            violation_counter += 1
            ts = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{violation_counter}"
            evidence_path = _save_violation_evidence(motorcycle_crop, ts)
            _log_violation(plate_num or "UNKNOWN", 0, evidence_path)
        else:
            cv2.rectangle(output_image, (vx1, vy1), (vx2, vy2), (0, 255, 255), 2)
            cv2.putText(
                output_image, "Two-Wheeler", (vx1, vy1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
            )

        # Draw plate box if detected
        if v.get("plate_box"):
            px1, py1, px2, py2 = v["plate_box"]
            cv2.rectangle(output_image, (px1, py1), (px2, py2), (255, 0, 0), 2)

    cv2.imwrite(output_path, output_image)
    return total_violations, plate_number_found

import cv2
from ultralytics import YOLO

from utils.draw_boxes import draw_detections
from utils.ocr import read_plate


# ---------------- LOAD MODELS ----------------
vehicle_model = YOLO("yolov8n.pt") # Standard model for vehicle (motorcycle) detection
helmet_model  = YOLO("models/best.pt")
plate_model   = YOLO("models/plate_best.pt")

# COCO dataset class IDs: 1 (bicycle/scooter), 3 (motorcycle)
TWO_WHEELER_CLASSES = [1, 3]


def process_image(input_path, output_path, conf):
    image = cv2.imread(input_path)
    if image is None:
        return 0, None

    output_image = image.copy()
    total_violations = 0
    plate_number_found = None

    # -------- STEP 1 & 2: VEHICLE AND HELMET DETECTION (FULL IMAGE) --------
    vehicle_results = vehicle_model(image, conf=conf, classes=TWO_WHEELER_CLASSES)[0]
    helmet_results = helmet_model(image, conf=conf)[0]
    
    # Pre-calculate vehicle boxes for intersection logic
    vehicles = []
    for vbox in vehicle_results.boxes:
        vx1, vy1, vx2, vy2 = map(int, vbox.xyxy[0])
        vehicles.append({
            "box": (vx1, vy1, vx2, vy2),
            "has_violation": False,
            "drawn": False
        })
        
    def get_distance(h_box, v_box):
        # Calculate horizontal center distance to ensure helmet is over the bike
        h_center_x = (h_box[0] + h_box[2]) / 2.0
        v_center_x = (v_box[0] + v_box[2]) / 2.0
        
        # Calculate vertical distance from bottom of helmet to top of vehicle
        h_bottom_y = h_box[3]
        v_top_y = v_box[1]
        
        # Simple distance heuristic
        x_dist = abs(h_center_x - v_center_x)
        y_dist = max(0, v_top_y - h_bottom_y) # Distance from helmet bottom to vehicle top
        
        return x_dist, y_dist

    # -------- STEP 3: MATCH HELMETS TO VEHICLES --------
    for hbox in helmet_results.boxes:
        hx1, hy1, hx2, hy2 = map(int, hbox.xyxy[0])
        h_box_tuple = (hx1, hy1, hx2, hy2)
        
        # Check if this helmet belongs to any detected two-wheeler
        matched_vehicle = None
        min_distance = float('inf')
        
        for v in vehicles:
            vx1, vy1, vx2, vy2 = v["box"]
            v_width = vx2 - vx1
            v_height = vy2 - vy1
            
            x_dist, y_dist = get_distance(h_box_tuple, v["box"])
            
            # Match if the helmet is horizontally strictly aligned with the bike's center (v_width * 0.5) 
            # and vertically within a tight logical distance physically touching or right above it (v_height * 0.5)
            # This strict constraint prevents pedestrians on the sidewalk from linking to motorcycles.
            if x_dist < v_width * 0.6 and y_dist < v_height * 0.6:
                total_dist = x_dist + y_dist
                if total_dist < min_distance:
                    min_distance = total_dist
                    matched_vehicle = v
                
        if not matched_vehicle:
            # Ignore helmets not logically near two-wheelers
            continue

        label = helmet_results.names[int(hbox.cls[0])].lower()
        h_conf = float(hbox.conf[0])

        if "no" in label:
            color = (0, 0, 255) # RED
            matched_vehicle["has_violation"] = True
            total_violations += 1
        else:
            color = (0, 255, 0) # GREEN

        # Draw Helmet Box
        text = f"{label.replace('_', ' ').title()}: {h_conf:.2f}"
        cv2.rectangle(output_image, (hx1, hy1), (hx2, hy2), color, 3)
        
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(output_image, (hx1, hy1 - th - 10), (hx1 + tw + 6, hy1), color, -1)
        cv2.putText(output_image, text, (hx1 + 3, hy1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    # -------- STEP 4: DRAW VEHICLES & RUN OCR FOR VIOLATORS --------
    for v in vehicles:
        vx1, vy1, vx2, vy2 = v["box"]
        
        # Draw vehicle box (YELLOW for motorcycle)
        cv2.rectangle(output_image, (vx1, vy1), (vx2, vy2), (0, 255, 255), 2)
        cv2.putText(output_image, "Two-Wheeler", (vx1, vy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        if v["has_violation"]:
            motorcycle_crop = image[vy1:vy2, vx1:vx2]
            if motorcycle_crop.size == 0:
                continue

            plate_results = plate_model(motorcycle_crop, conf=0.3)[0]

            for pbox in plate_results.boxes:
                px1, py1, px2, py2 = map(int, pbox.xyxy[0])
                plate_crop = motorcycle_crop[py1:py2, px1:px2]

                if plate_crop.size == 0:
                    continue

                plate_num = read_plate(plate_crop)
                if plate_num and not plate_number_found:
                    plate_number_found = plate_num # Keep first plate found for summary

                # Translate plate coordinates to original image space
                global_px1, global_py1 = vx1 + px1, vy1 + py1
                global_px2, global_py2 = vx1 + px2, vy1 + py2

                # Draw plate box (BLUE)
                cv2.rectangle(output_image, (global_px1, global_py1), (global_px2, global_py2), (255, 0, 0), 2)

                if plate_num:
                    cv2.putText(
                        output_image,
                        plate_num,
                        (global_px1, global_py1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (255, 0, 0),
                        2,
                        cv2.LINE_AA
                    )
                break

    cv2.imwrite(output_path, output_image)
    return total_violations, plate_number_found
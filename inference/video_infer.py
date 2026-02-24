import cv2
from ultralytics import YOLO
from utils.ocr import read_plate

vehicle_model = YOLO("yolov8n.pt")
helmet_model  = YOLO("models/best.pt")
plate_model   = YOLO("models/plate_best.pt")

TWO_WHEELER_CLASSES = [1, 3]

def process_video(ip, op, conf):
    cap = cv2.VideoCapture(ip)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25

    out = cv2.VideoWriter(
        op,
        cv2.VideoWriter_fourcc(*"avc1"),
        fps,
        (w, h)
    )

    frame_count = 0
    process_interval = 5  # Extract heavy DL features only once every 5 frames (gives 5x speed boost)

    cached_helmets = []
    cached_vehicles = []
    cached_plates = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1

        if frame_count % process_interval == 1:
            cached_helmets = []
            cached_vehicles = []
            cached_plates = []
            
            vehicle_results = vehicle_model(frame, conf=conf, classes=TWO_WHEELER_CLASSES)[0]
            helmet_results = helmet_model(frame, conf=conf)[0]

            vehicles = []
            for vbox in vehicle_results.boxes:
                vx1, vy1, vx2, vy2 = map(int, vbox.xyxy[0])
                vehicles.append({
                    "box": (vx1, vy1, vx2, vy2),
                    "has_violation": False
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
                min_distance = float('inf')
                
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

                label = helmet_results.names[int(hbox.cls[0])].lower()
                h_conf = float(hbox.conf[0])

                if "no" in label:
                    color = (0, 0, 255) # RED
                    matched_vehicle["has_violation"] = True
                else:
                    color = (0, 255, 0) # GREEN

                text = f"{label.replace('_', ' ').title()}: {h_conf:.2f}"
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                
                cached_helmets.append((hx1, hy1, hx2, hy2, color, text, tw, th))

            for v in vehicles:
                vx1, vy1, vx2, vy2 = v["box"]
                cached_vehicles.append((vx1, vy1, vx2, vy2))

                if v["has_violation"]:
                    motorcycle_crop = frame[vy1:vy2, vx1:vx2]
                    if motorcycle_crop.size == 0:
                        continue

                    plate_results = plate_model(motorcycle_crop, conf=0.01)[0]
                    if len(plate_results.boxes) == 0:
                        continue
                        
                    best_pbox = None
                    best_pconf = -1
                    
                    for pbox in plate_results.boxes:
                        p_conf = float(pbox.conf[0])
                        if p_conf > best_pconf:
                            best_pconf = p_conf
                            best_pbox = pbox

                    if best_pbox is not None:
                        px1, py1, px2, py2 = map(int, best_pbox.xyxy[0])
                        plate_crop = motorcycle_crop[py1:py2, px1:px2]

                        if plate_crop.size > 0:
                            # EasyOCR is intensely heavy, only run it on evaluated violation crops
                            plate_num = read_plate(plate_crop)

                            global_px1, global_py1 = vx1 + px1, vy1 + py1
                            global_px2, global_py2 = vx1 + px2, vy1 + py2
                            
                            cached_plates.append((global_px1, global_py1, global_px2, global_py2, plate_num))

        # --- DRAWING ---
        for hx1, hy1, hx2, hy2, color, text, tw, th in cached_helmets:
            cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), color, 3)
            cv2.rectangle(frame, (hx1, hy1 - th - 10), (hx1 + tw + 6, hy1), color, -1)
            cv2.putText(frame, text, (hx1 + 3, hy1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            
        for vx1, vy1, vx2, vy2 in cached_vehicles:
            cv2.rectangle(frame, (vx1, vy1), (vx2, vy2), (0, 255, 255), 2)
            cv2.putText(frame, "Two-Wheeler", (vx1, vy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
        for px1, py1, px2, py2, plate_num in cached_plates:
            cv2.rectangle(frame, (px1, py1), (px2, py2), (255, 0, 0), 2)
            if plate_num:
                cv2.putText(frame, plate_num, (px1, py1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2, cv2.LINE_AA)

        out.write(frame)

    cap.release()
    out.release()
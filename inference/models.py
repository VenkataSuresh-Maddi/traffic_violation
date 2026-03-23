"""
Shared model loading and MPS warmup.
Loaded once at import time; all inference modules reuse these instances.
"""

import numpy as np
from ultralytics import YOLO

vehicle_model = YOLO("yolov8n.pt").to("mps")
helmet_model = YOLO("models/best.pt").to("mps")
plate_model = YOLO("models/plate_best.pt").to("mps")

# Warmup: pre-compile MPS compute graph so first real inference isn't slow
_dummy = np.zeros((416, 416, 3), dtype=np.uint8)
vehicle_model(_dummy, verbose=False)
helmet_model(_dummy, verbose=False)
plate_model(_dummy, verbose=False)
del _dummy
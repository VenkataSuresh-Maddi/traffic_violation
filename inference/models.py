"""Shared model loading with safe device selection and warmup."""

import os

import numpy as np
import torch
from ultralytics import YOLO


def _pick_device() -> str:
    # Allow forcing CPU for stability troubleshooting.
    if os.getenv("TV_FORCE_CPU", "0") == "1":
        return "cpu"
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    return "cpu"


MODEL_DEVICE = _pick_device()

vehicle_model = YOLO("yolov8n.pt").to(MODEL_DEVICE)
helmet_model = YOLO("models/best.pt").to(MODEL_DEVICE)
plate_model = YOLO("models/plate_best.pt").to(MODEL_DEVICE)

# Warm up model graphs once. If warmup fails, continue without crashing startup.
try:
    _dummy = np.zeros((416, 416, 3), dtype=np.uint8)
    vehicle_model(_dummy, verbose=False, device=MODEL_DEVICE)
    helmet_model(_dummy, verbose=False, device=MODEL_DEVICE)
    plate_model(_dummy, verbose=False, device=MODEL_DEVICE)
    del _dummy
except Exception:
    pass
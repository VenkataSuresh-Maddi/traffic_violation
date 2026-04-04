from ultralytics import YOLO

model = YOLO("yolov8m.pt")

model.train(
    data="data/helmet_dataset/data.yaml",
    epochs=40,
    imgsz=608,
    batch=32,
    optimizer="AdamW",
    lr0=0.001,
    lrf=0.01,
    patience=15,
    project=".",
    name="helmet_training",
    device="mps"
)

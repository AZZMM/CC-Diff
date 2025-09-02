from ultralytics import YOLO

# Load a model
model = YOLO('runs/detect/yolov8n_epochs50_batch16/weights/best.pt')

# Customize validation settings
validation_results = model.val(data="config.yaml", imgsz=800, name='yolov8n_val', batch=16)
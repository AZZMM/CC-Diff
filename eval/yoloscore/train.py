from ultralytics import YOLO
model = YOLO('yolov8n.yaml')

results = model.train(data='config.yaml', imgsz=800, epochs=50, batch=16, name='yolov8n_epochs50_batch16')
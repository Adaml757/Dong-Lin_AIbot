from ultralytics import YOLO

# load YOLOv8 PyTorch model
model = YOLO("yolov8n.pt")

# export model to NCNN format
model.export(format="ncnn")
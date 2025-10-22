from ultralytics import YOLO

# 1. Load your original PyTorch model
model = YOLO('yoloe-11l-seg-pf.pt') 

# 2. Export the model to ONNX format for a specific image size
model.export(format='onnx', imgsz=640)
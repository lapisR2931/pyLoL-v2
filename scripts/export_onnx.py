"""
YOLOv8モデルをONNX形式にエクスポート
"""
from ultralytics import YOLO
from pathlib import Path

model_path = Path(r"C:\dataset_yolo\runs\ward_detection_v2\weights\best.pt")

print(f"Loading model: {model_path}")
model = YOLO(str(model_path))

print("Exporting to ONNX...")
model.export(format='onnx', imgsz=512, simplify=True)

print("Export complete!")
print(f"ONNX model: {model_path.with_suffix('.onnx')}")

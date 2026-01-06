"""
YOLOv8でward検出モデルを学習
少量データ（27枚）での学習のため、データ拡張を強化
"""
from ultralytics import YOLO
from pathlib import Path

# データセットパス
data_yaml = Path(r"C:\dataset_yolo\data.yaml")

# モデル設定
model_name = "yolov8n.pt"  # nano版（軽量・高速）

# 出力先
project_dir = Path(r"C:\dataset_yolo\runs")
run_name = "ward_detection_v3"

def main():
    # モデルをロード（事前学習済み）
    model = YOLO(model_name)

    # 学習実行
    results = model.train(
        data=str(data_yaml),
        epochs=100,
        imgsz=512,
        batch=8,
        patience=20,  # 早期停止
        project=str(project_dir),
        name=run_name,
        exist_ok=True,
        # データ拡張（少量データのため強化）
        augment=True,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        # その他
        verbose=True,
        seed=42,
    )

    print("\n学習完了")
    print(f"Best model: {project_dir / run_name / 'weights' / 'best.pt'}")
    print(f"Last model: {project_dir / run_name / 'weights' / 'last.pt'}")

if __name__ == "__main__":
    main()

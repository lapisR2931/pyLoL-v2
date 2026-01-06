"""
アノテーションデータをYOLO形式に変換し、train/val分割を行う
"""
import json
import shutil
import random
from pathlib import Path

# 設定
annotation_dir = Path(r"C:\dataset_annotation")
output_dir = Path(r"C:\dataset_yolo")
val_ratio = 0.2  # 検証データの割合

# クラスマッピング（ラベル名の揺れに対応）
class_mapping = {
    "stealth_ward": 0,
    "stealth ward": 0,
    "stealth_ward_enemy": 1,
    "stealth ward enemy": 1,
    "control_ward": 2,
    "control ward": 2,
    "control_ward_enemy": 3,
    "control ward enemy": 3,
}

def convert_json_to_yolo(json_path: Path, output_txt_path: Path, img_width: int, img_height: int):
    """X-AnyLabeling JSONをYOLO形式に変換"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    lines = []
    for shape in data.get("shapes", []):
        label = shape["label"]
        if label not in class_mapping:
            print(f"Warning: Unknown label '{label}' in {json_path}")
            continue

        class_id = class_mapping[label]
        points = shape["points"]

        # rectangle形式: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]] または [[x1,y1], [x2,y2]]
        if len(points) >= 2:
            x_coords = [p[0] for p in points]
            y_coords = [p[1] for p in points]
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)

            # YOLO形式に変換（正規化）
            x_center = ((x_min + x_max) / 2) / img_width
            y_center = ((y_min + y_max) / 2) / img_height
            width = (x_max - x_min) / img_width
            height = (y_max - y_min) / img_height

            # 範囲チェック
            x_center = max(0, min(1, x_center))
            y_center = max(0, min(1, y_center))
            width = max(0, min(1, width))
            height = max(0, min(1, height))

            lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    with open(output_txt_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    return len(lines)

def main():
    # 出力ディレクトリ構造を作成
    (output_dir / "train" / "images").mkdir(parents=True, exist_ok=True)
    (output_dir / "train" / "labels").mkdir(parents=True, exist_ok=True)
    (output_dir / "val" / "images").mkdir(parents=True, exist_ok=True)
    (output_dir / "val" / "labels").mkdir(parents=True, exist_ok=True)

    # アノテーション済みファイルを収集
    json_files = list(annotation_dir.glob("*.json"))
    print(f"アノテーション済みファイル数: {len(json_files)}")

    # クラス別カウント
    class_counts = {v: 0 for v in set(class_mapping.values())}

    # ファイルリストをシャッフル
    random.seed(42)
    random.shuffle(json_files)

    # train/val分割
    val_count = max(1, int(len(json_files) * val_ratio))
    val_files = json_files[:val_count]
    train_files = json_files[val_count:]

    print(f"Train: {len(train_files)}枚, Val: {val_count}枚")

    total_annotations = 0

    for split, files in [("train", train_files), ("val", val_files)]:
        for json_path in files:
            # 対応する画像ファイル
            img_path = json_path.with_suffix('.png')
            if not img_path.exists():
                print(f"Warning: Image not found for {json_path}")
                continue

            # JSONを読み込んで画像サイズを取得
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            img_width = data.get("imageWidth", 512)
            img_height = data.get("imageHeight", 512)

            # 出力先
            output_img = output_dir / split / "images" / img_path.name
            output_txt = output_dir / split / "labels" / (json_path.stem + ".txt")

            # 画像をコピー
            shutil.copy(img_path, output_img)

            # YOLO形式に変換
            num_annotations = convert_json_to_yolo(json_path, output_txt, img_width, img_height)
            total_annotations += num_annotations

            # クラスカウント
            for shape in data.get("shapes", []):
                label = shape["label"]
                if label in class_mapping:
                    class_counts[class_mapping[label]] += 1

            print(f"[{split}] {json_path.name} -> {num_annotations}個のアノテーション")

    # data.yaml作成
    class_names = ["stealth_ward", "stealth_ward_enemy", "control_ward", "control_ward_enemy"]
    yaml_content = f"""# Ward Detection Dataset
path: {output_dir}
train: train/images
val: val/images

nc: {len(class_names)}
names: {class_names}
"""

    with open(output_dir / "data.yaml", 'w', encoding='utf-8') as f:
        f.write(yaml_content)

    print(f"\n完了:")
    print(f"  総アノテーション数: {total_annotations}")
    print(f"  クラス別カウント:")
    for class_id, name in enumerate(class_names):
        print(f"    {name}: {class_counts.get(class_id, 0)}")
    print(f"  出力先: {output_dir}")
    print(f"  data.yaml: {output_dir / 'data.yaml'}")

if __name__ == "__main__":
    main()

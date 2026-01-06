"""
ward検出結果の可視化ツール

機能:
1. 特定のwardについて、開始/終了フレームを表示
2. 検出座標にバウンディングボックスを描画
3. HTMLレポート生成（全wardのサムネイル一覧）

使用方法:
    # 特定wardの確認
    python scripts/visualize_wards.py --match JP1-551272474 --ward 10

    # HTMLレポート生成
    python scripts/visualize_wards.py --match JP1-551272474 --report

    # 全wardの画像生成
    python scripts/visualize_wards.py --match JP1-551272474 --all
"""

import argparse
import csv
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass

import cv2
import numpy as np

# 設定
DATASET_DIR = Path(r"C:\dataset")
OUTPUT_DIR = Path(r"C:\dataset\visualizations")

# クラス別の色（BGR）
CLASS_COLORS = {
    0: (0, 255, 0),      # stealth_ward: 緑
    1: (0, 0, 255),      # stealth_ward_enemy: 赤
    2: (255, 255, 0),    # control_ward: シアン
    3: (0, 165, 255),    # control_ward_enemy: オレンジ
}

CLASS_NAMES = {
    0: "stealth_ward",
    1: "stealth_ward_enemy",
    2: "control_ward",
    3: "control_ward_enemy",
}


@dataclass
class Ward:
    ward_id: int
    class_id: int
    class_name: str
    x: float
    y: float
    frame_start: int
    frame_end: int
    detection_count: int
    confidence_avg: float


def load_wards(match_dir: Path) -> List[Ward]:
    """wards.csvを読み込む"""
    wards_path = match_dir / "wards.csv"
    if not wards_path.exists():
        print(f"wards.csvが見つかりません: {wards_path}")
        return []

    wards = []
    with open(wards_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            wards.append(Ward(
                ward_id=int(row['ward_id']),
                class_id=int(row['class_id']),
                class_name=row['class_name'],
                x=float(row['x']),
                y=float(row['y']),
                frame_start=int(row['frame_start']),
                frame_end=int(row['frame_end']),
                detection_count=int(row['detection_count']),
                confidence_avg=float(row['confidence_avg']),
            ))
    return wards


def draw_ward_box(img: np.ndarray, ward: Ward, img_size: int = 512) -> np.ndarray:
    """画像上にwardのバウンディングボックスを描画"""
    img_copy = img.copy()

    # 正規化座標をピクセル座標に変換
    cx = int(ward.x * img_size)
    cy = int(ward.y * img_size)

    # wardのサイズ（固定: 約20x20ピクセル）
    box_size = 20
    x1 = max(0, cx - box_size // 2)
    y1 = max(0, cy - box_size // 2)
    x2 = min(img_size, cx + box_size // 2)
    y2 = min(img_size, cy + box_size // 2)

    color = CLASS_COLORS.get(ward.class_id, (255, 255, 255))

    # バウンディングボックス
    cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)

    # ラベル
    label = f"#{ward.ward_id} {ward.class_name}"
    font_scale = 0.4
    thickness = 1
    (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

    # ラベル背景
    cv2.rectangle(img_copy, (x1, y1 - text_h - 4), (x1 + text_w + 4, y1), color, -1)
    cv2.putText(img_copy, label, (x1 + 2, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (0, 0, 0), thickness)

    return img_copy


def visualize_ward(match_dir: Path, ward: Ward, output_dir: Path) -> Path:
    """
    特定のwardの開始/中間/終了フレームを並べた画像を生成
    """
    frame_dir = match_dir / "0"

    # 3つのフレームを選択（開始、中間、終了）
    frames_to_show = [
        ward.frame_start,
        (ward.frame_start + ward.frame_end) // 2,
        ward.frame_end
    ]

    images = []
    for frame_num in frames_to_show:
        frame_path = frame_dir / f"{frame_num}.png"
        if frame_path.exists():
            img = cv2.imread(str(frame_path))
            img = draw_ward_box(img, ward)
            # フレーム番号を追加
            cv2.putText(img, f"Frame: {frame_num}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            images.append(img)

    if not images:
        return None

    # 横に連結
    combined = np.hstack(images)

    # 保存
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"ward_{ward.ward_id:03d}.png"
    cv2.imwrite(str(output_path), combined)

    return output_path


def generate_html_report(match_dir: Path, wards: List[Ward], output_dir: Path):
    """全wardのサムネイル付きHTMLレポートを生成"""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 各wardの画像を生成
    print(f"各wardの画像を生成中...")
    for ward in wards:
        visualize_ward(match_dir, ward, output_dir)

    # HTML生成
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Ward Detection Report - {match_dir.name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #1a1a1a; color: #fff; }}
        h1 {{ color: #4CAF50; }}
        .summary {{ background: #2a2a2a; padding: 15px; border-radius: 8px; margin-bottom: 20px; }}
        .ward-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(800px, 1fr)); gap: 20px; }}
        .ward-card {{ background: #2a2a2a; border-radius: 8px; overflow: hidden; }}
        .ward-card img {{ width: 100%; }}
        .ward-info {{ padding: 10px; }}
        .stealth_ward {{ border-left: 4px solid #00ff00; }}
        .stealth_ward_enemy {{ border-left: 4px solid #ff0000; }}
        .control_ward {{ border-left: 4px solid #00ffff; }}
        .control_ward_enemy {{ border-left: 4px solid #ffa500; }}
        .stats {{ display: flex; gap: 20px; }}
        .stat {{ background: #333; padding: 10px 20px; border-radius: 4px; }}
    </style>
</head>
<body>
    <h1>Ward Detection Report</h1>
    <div class="summary">
        <h2>Match: {match_dir.name}</h2>
        <div class="stats">
            <div class="stat">Total Wards: <strong>{len(wards)}</strong></div>
            <div class="stat">stealth_ward: <strong>{sum(1 for w in wards if w.class_id == 0)}</strong></div>
            <div class="stat">stealth_ward_enemy: <strong>{sum(1 for w in wards if w.class_id == 1)}</strong></div>
            <div class="stat">control_ward: <strong>{sum(1 for w in wards if w.class_id == 2)}</strong></div>
            <div class="stat">control_ward_enemy: <strong>{sum(1 for w in wards if w.class_id == 3)}</strong></div>
        </div>
    </div>
    <div class="ward-grid">
"""

    for ward in wards:
        img_path = f"ward_{ward.ward_id:03d}.png"
        duration_sec = (ward.frame_end - ward.frame_start)
        html_content += f"""
        <div class="ward-card {ward.class_name}">
            <img src="{img_path}" alt="Ward #{ward.ward_id}">
            <div class="ward-info">
                <strong>Ward #{ward.ward_id}</strong> - {ward.class_name}<br>
                Frames: {ward.frame_start} - {ward.frame_end} ({duration_sec}s)<br>
                Detections: {ward.detection_count} | Confidence: {ward.confidence_avg:.3f}<br>
                Position: ({ward.x:.3f}, {ward.y:.3f})
            </div>
        </div>
"""

    html_content += """
    </div>
</body>
</html>
"""

    report_path = output_dir / "report.html"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"HTMLレポート生成完了: {report_path}")
    return report_path


def main():
    parser = argparse.ArgumentParser(description="ward検出結果の可視化")
    parser.add_argument("--match", type=str, required=True, help="試合ID")
    parser.add_argument("--ward", type=int, help="確認するward_id")
    parser.add_argument("--report", action="store_true", help="HTMLレポート生成")
    parser.add_argument("--all", action="store_true", help="全wardの画像生成")
    args = parser.parse_args()

    match_dir = DATASET_DIR / args.match
    if not match_dir.exists():
        print(f"試合ディレクトリが見つかりません: {match_dir}")
        return

    output_dir = OUTPUT_DIR / args.match

    # wardsを読み込み
    wards = load_wards(match_dir)
    if not wards:
        return

    print(f"読み込んだward数: {len(wards)}")

    if args.ward:
        # 特定のwardを確認
        ward = next((w for w in wards if w.ward_id == args.ward), None)
        if ward:
            path = visualize_ward(match_dir, ward, output_dir)
            if path:
                print(f"画像生成完了: {path}")
        else:
            print(f"ward_id={args.ward}が見つかりません")

    elif args.report:
        # HTMLレポート生成
        generate_html_report(match_dir, wards, output_dir)

    elif args.all:
        # 全wardの画像生成
        print(f"全{len(wards)}wardの画像を生成中...")
        for ward in wards:
            path = visualize_ward(match_dir, ward, output_dir)
            if path:
                print(f"  Ward #{ward.ward_id}: {path}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()

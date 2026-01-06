"""
timeline_onlyのwardを可視化するスクリプト

問題のwardがなぜマッチしなかったかを画像で確認する
"""

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional
import cv2
import numpy as np


def load_wards_matched(path: Path) -> List[dict]:
    """wards_matched.csvを読み込み"""
    wards = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            wards.append(row)
    return wards


def load_wards_csv(path: Path) -> List[dict]:
    """wards.csv (YOLO検出結果) を読み込み"""
    wards = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            wards.append(row)
    return wards


def load_frame_timestamps(path: Path) -> Dict[int, int]:
    """frame_timestamps.csvを読み込み"""
    frame_map = {}
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame = int(row['frame_number'])
            time_ms = int(row['game_time_ms'])
            if time_ms >= 0:
                frame_map[frame] = time_ms
    return frame_map


def timestamp_to_frame(timestamp_ms: int, frame_timestamps: Dict[int, int]) -> int:
    """タイムスタンプから最も近いフレームを取得"""
    best_frame = 0
    best_diff = float('inf')
    for frame, time_ms in frame_timestamps.items():
        diff = abs(time_ms - timestamp_ms)
        if diff < best_diff:
            best_diff = diff
            best_frame = frame
    return best_frame


def get_class_color(class_name: str) -> tuple:
    """クラス名に応じた色を返す (BGR)"""
    if 'control_ward' in class_name:
        if 'enemy' in class_name:
            return (0, 0, 255)  # 赤
        else:
            return (255, 0, 0)  # 青
    else:  # stealth_ward
        if 'enemy' in class_name:
            return (0, 128, 255)  # オレンジ
        else:
            return (255, 255, 0)  # シアン
    return (255, 255, 255)


def visualize_ward_issue(
    match_dir: Path,
    ward: dict,
    all_wards: List[dict],
    frame_timestamps: Dict[int, int],
    output_dir: Path
):
    """
    問題のwardを可視化

    - タイムラインのframe_expected時点のフレーム画像を表示
    - そのフレーム付近で検出されたwardをオーバーレイ
    - マッチすべきだったwardを強調表示
    """
    timeline_ward_id = ward.get('timeline_ward_id', 'unknown')
    ward_type = ward.get('ward_type', '')
    team = ward.get('team', '')
    frame_expected = int(ward.get('frame_start', 0))
    timestamp = int(ward.get('timestamp_placed', 0)) if ward.get('timestamp_placed') else 0

    # フレーム画像のパス
    frame_dir = match_dir / "0"
    frame_path = frame_dir / f"{frame_expected}.png"

    if not frame_path.exists():
        print(f"フレーム画像が見つかりません: {frame_path}")
        # 前後のフレームを探す
        for offset in range(-5, 6):
            alt_path = frame_dir / f"{frame_expected + offset}.png"
            if alt_path.exists():
                frame_path = alt_path
                frame_expected = frame_expected + offset
                break
        else:
            return None

    # 画像読み込み
    img = cv2.imread(str(frame_path))
    if img is None:
        print(f"画像読み込み失敗: {frame_path}")
        return None

    h, w = img.shape[:2]

    # 画像を拡大 (見やすくするため)
    scale = 2
    img = cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)
    h, w = img.shape[:2]

    # 周辺のwardを描画
    nearby_wards = []
    for yw in all_wards:
        yolo_frame_start = int(yw['frame_start'])
        yolo_frame_end = int(yw['frame_end'])
        frame_diff = yolo_frame_start - frame_expected

        # 検出期間にframe_expectedが含まれているか、周辺かをチェック
        is_visible = yolo_frame_start <= frame_expected <= yolo_frame_end
        is_nearby = -30 <= frame_diff <= 50

        if is_visible or is_nearby:
            nearby_wards.append({
                **yw,
                'frame_diff': frame_diff,
                'is_visible': is_visible
            })

    # 周辺wardを描画
    for nw in nearby_wards:
        x_norm = float(nw['x'])
        y_norm = float(nw['y'])
        x = int(x_norm * w)
        y = int(y_norm * h)

        class_name = nw['class_name']
        color = get_class_color(class_name)
        conf = float(nw['confidence_avg'])
        is_visible = nw['is_visible']

        # 可視のwardは実線、周辺のwardは点線風に
        thickness = 2 if is_visible else 1

        # バウンディングボックス (推定)
        box_size = int(20 * scale)
        cv2.rectangle(img,
                      (x - box_size // 2, y - box_size // 2),
                      (x + box_size // 2, y + box_size // 2),
                      color, thickness)

        # ラベル
        label = f"{nw['ward_id']}: {class_name.replace('stealth_ward', 'SW').replace('control_ward', 'CW')}"
        label += f" f={nw['frame_start']}"
        if not is_visible:
            label += f" (diff={nw['frame_diff']:+d})"

        # ラベル背景
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.35 * scale, 1)
        cv2.rectangle(img,
                      (x - box_size // 2, y - box_size // 2 - text_h - 4),
                      (x - box_size // 2 + text_w + 2, y - box_size // 2),
                      color, -1)
        cv2.putText(img, label,
                    (x - box_size // 2 + 1, y - box_size // 2 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35 * scale, (0, 0, 0), 1)

    # タイトル情報
    title_lines = [
        f"timeline_ward_id={timeline_ward_id}: {ward_type} ({team})",
        f"timestamp={timestamp}ms ({timestamp//1000}s), frame_expected={frame_expected}",
        f"Status: timeline_only (YOLO detection missing)"
    ]

    y_offset = 20 * scale
    for line in title_lines:
        cv2.putText(img, line, (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale, (255, 255, 255), 2)
        cv2.putText(img, line, (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale, (0, 0, 0), 1)
        y_offset += 25 * scale

    # 凡例
    legend_y = h - 100 * scale
    legends = [
        ("SW (blue)", (255, 255, 0)),
        ("SW_enemy (red)", (0, 128, 255)),
        ("CW (blue)", (255, 0, 0)),
        ("CW_enemy (red)", (0, 0, 255)),
    ]
    for i, (leg_text, leg_color) in enumerate(legends):
        cv2.rectangle(img, (10, legend_y + i * 20 * scale), (30, legend_y + i * 20 * scale + 15 * scale), leg_color, -1)
        cv2.putText(img, leg_text, (35, legend_y + i * 20 * scale + 12 * scale),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4 * scale, (255, 255, 255), 1)

    # 出力
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"timeline_only_{timeline_ward_id}_frame{frame_expected}.png"
    cv2.imwrite(str(output_path), img)
    print(f"保存: {output_path}")

    return output_path


def main():
    # パス設定
    dataset_dir = Path(r"C:\dataset_20260105")
    timeline_dir = Path("data/timeline")
    output_dir = Path("debug/visualizations")

    match_id = "JP1-555621265"
    match_dir = dataset_dir / match_id

    # データ読み込み
    wards_matched = load_wards_matched(match_dir / "wards_matched.csv")
    wards_csv = load_wards_csv(match_dir / "wards.csv")
    frame_timestamps = load_frame_timestamps(match_dir / "frame_timestamps.csv")

    # timeline_onlyのwardを抽出
    timeline_only = [w for w in wards_matched if w['match_status'] == 'timeline_only']

    print(f"timeline_onlyのward: {len(timeline_only)}件")
    print(f"出力先: {output_dir}")

    # 各timeline_onlyを可視化 (最初の5件)
    for i, ward in enumerate(timeline_only[:5]):
        print(f"\n--- {i+1}/{len(timeline_only[:5])} ---")
        visualize_ward_issue(match_dir, ward, wards_csv, frame_timestamps, output_dir)


if __name__ == "__main__":
    main()

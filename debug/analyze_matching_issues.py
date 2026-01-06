"""
バッチ推論精度問題の診断スクリプト

問題点を可視化し、根本原因を特定する
"""

import csv
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple


def load_wards_matched(path: Path) -> List[dict]:
    """wards_matched.csvを読み込み"""
    wards = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            wards.append(row)
    return wards


def load_wards_csv(path: Path) -> List[dict]:
    """wards.csv（YOLO検出結果）を読み込み"""
    wards = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            wards.append(row)
    return wards


def load_timeline(path: Path) -> dict:
    """タイムラインJSONを読み込み"""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


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


def extract_ward_events(timeline: dict) -> List[dict]:
    """タイムラインからWARD_PLACEDイベントを抽出"""
    events = []
    for frame in timeline.get("info", {}).get("frames", []):
        for event in frame.get("events", []):
            if event.get("type") == "WARD_PLACED":
                events.append(event)
    return events


def timestamp_to_frame(timestamp_ms: int, frame_timestamps: Dict[int, int]) -> int:
    """タイムスタンプからフレームを推定"""
    best_frame = 0
    best_diff = float('inf')
    for frame, time_ms in frame_timestamps.items():
        diff = abs(time_ms - timestamp_ms)
        if diff < best_diff:
            best_diff = diff
            best_frame = frame
    return best_frame


def analyze_match(match_dir: Path, timeline_dir: Path):
    """1試合を分析"""
    match_id = match_dir.name
    match_id_num = match_id.replace("JP1-", "")

    # ファイルパス
    wards_matched_path = match_dir / "wards_matched.csv"
    wards_csv_path = match_dir / "wards.csv"
    timeline_path = timeline_dir / f"JP1_{match_id_num}.json"
    frame_timestamps_path = match_dir / "frame_timestamps.csv"

    if not wards_matched_path.exists():
        print(f"wards_matched.csv が見つかりません: {match_dir}")
        return None

    # データ読み込み
    wards_matched = load_wards_matched(wards_matched_path)
    wards_csv = load_wards_csv(wards_csv_path)
    timeline = load_timeline(timeline_path)
    frame_timestamps = load_frame_timestamps(frame_timestamps_path)
    ward_events = extract_ward_events(timeline)

    print(f"\n{'='*70}")
    print(f"試合: {match_id}")
    print(f"{'='*70}")

    # 基本統計
    status_counts = defaultdict(int)
    for w in wards_matched:
        status_counts[w['match_status']] += 1

    print(f"\n【基本統計】")
    print(f"  タイムラインWARD_PLACED: {len(ward_events)}件")
    print(f"  YOLO検出ward数: {len(wards_csv)}件")
    print(f"  出力ward数: {len(wards_matched)}件")
    print(f"    - matched: {status_counts['matched']}件")
    print(f"    - timeline_only: {status_counts['timeline_only']}件")
    print(f"    - detection_only: {status_counts['detection_only']}件")

    matched_count = status_counts['matched']
    valid_events = matched_count + status_counts['timeline_only']
    if valid_events > 0:
        print(f"  マッチング率: {matched_count/valid_events*100:.1f}%")

    # ============================================
    # 1. timeline_only の分析
    # ============================================
    timeline_only = [w for w in wards_matched if w['match_status'] == 'timeline_only']

    if timeline_only:
        print(f"\n{'='*70}")
        print(f"【問題1: timeline_only ({len(timeline_only)}件)】")
        print(f"  タイムラインにあるがYOLO検出されなかったward")
        print(f"{'='*70}")

        # 各timeline_onlyの詳細を分析
        for w in timeline_only:
            timeline_ward_id = w.get('timeline_ward_id', '')
            ward_type = w.get('ward_type', '')
            team = w.get('team', '')
            frame_expected = int(w.get('frame_start', 0))
            timestamp = int(w.get('timestamp_placed', 0)) if w.get('timestamp_placed') else 0

            # 実際のゲーム内時間（frame_timestampsから）
            actual_game_time = frame_timestamps.get(frame_expected, 0)

            print(f"\n  ward_id={timeline_ward_id}: {ward_type} ({team})")
            print(f"    timestamp_placed: {timestamp}ms ({timestamp//1000}s)")
            print(f"    frame_expected: {frame_expected}")
            print(f"    実際のゲーム時間@frame: {actual_game_time}ms ({actual_game_time//1000}s)")

            # frame_expected周辺の検出wardを検索
            nearby_wards = []
            for yw in wards_csv:
                frame_start = int(yw['frame_start'])
                frame_diff = frame_start - frame_expected
                if -30 <= frame_diff <= 50:  # 広めの範囲で検索
                    nearby_wards.append({
                        'ward_id': yw['ward_id'],
                        'class_name': yw['class_name'],
                        'frame_start': frame_start,
                        'frame_diff': frame_diff,
                        'confidence': float(yw['confidence_avg'])
                    })

            if nearby_wards:
                print(f"    周辺の検出ward（frame差-30〜+50）:")
                for nw in sorted(nearby_wards, key=lambda x: abs(x['frame_diff'])):
                    marker = "  <-- チーム/タイプ不一致?" if nw['class_name'] != w.get('class_name') else ""
                    print(f"      ward_id={nw['ward_id']}: {nw['class_name']}, frame={nw['frame_start']} (diff={nw['frame_diff']:+d}), conf={nw['confidence']:.3f}{marker}")
            else:
                print(f"    周辺に検出wardなし（YOLOが検出していない）")

    # ============================================
    # 2. detection_only の分析
    # ============================================
    detection_only = [w for w in wards_matched if w['match_status'] == 'detection_only']

    if detection_only:
        print(f"\n{'='*70}")
        print(f"【問題2: detection_only ({len(detection_only)}件)】")
        print(f"  YOLO検出されたがタイムラインにマッチしなかったward")
        print(f"{'='*70}")

        # タイプ別に集計
        type_counts = defaultdict(int)
        team_counts = defaultdict(int)
        confidence_stats = []

        for w in detection_only:
            class_name = w.get('class_name', '')
            team = w.get('team', '')
            conf = float(w.get('confidence_avg', 0)) if w.get('confidence_avg') else 0

            type_counts[class_name] += 1
            team_counts[team] += 1
            confidence_stats.append(conf)

        print(f"\n  クラス別内訳:")
        for cls, cnt in sorted(type_counts.items()):
            print(f"    {cls}: {cnt}件")

        print(f"\n  チーム別内訳:")
        for team, cnt in sorted(team_counts.items()):
            print(f"    {team}: {cnt}件")

        if confidence_stats:
            avg_conf = sum(confidence_stats) / len(confidence_stats)
            min_conf = min(confidence_stats)
            max_conf = max(confidence_stats)
            print(f"\n  信頼度統計:")
            print(f"    平均: {avg_conf:.3f}")
            print(f"    最小: {min_conf:.3f}")
            print(f"    最大: {max_conf:.3f}")

        # 低信頼度のwardを抽出
        low_conf_wards = [w for w in detection_only
                         if float(w.get('confidence_avg', 1)) < 0.5]
        if low_conf_wards:
            print(f"\n  低信頼度（<0.5）のward: {len(low_conf_wards)}件（誤検出の可能性）")
            for w in low_conf_wards[:5]:  # 最大5件表示
                print(f"    ward_id={w.get('ward_id')}: {w.get('class_name')}, conf={w.get('confidence_avg')}, frame={w.get('frame_start')}")

        # 検出期間が短いwardを抽出
        short_duration = []
        for w in detection_only:
            frame_start = int(w.get('frame_start', 0))
            frame_end = int(w.get('frame_end', 0)) if w.get('frame_end') else frame_start
            duration = frame_end - frame_start
            if duration < 10:  # 10フレーム未満
                short_duration.append({**w, 'duration': duration})

        if short_duration:
            print(f"\n  検出期間が短い（<10フレーム）ward: {len(short_duration)}件（ノイズの可能性）")

    # ============================================
    # 3. フレーム↔タイムスタンプ変換の精度分析
    # ============================================
    print(f"\n{'='*70}")
    print(f"【診断3: フレーム↔タイムスタンプ変換の精度】")
    print(f"{'='*70}")

    # matchedのwardでframe_startとtimestamp_placedの関係を分析
    matched_wards = [w for w in wards_matched if w['match_status'] == 'matched']

    frame_diffs = []
    for w in matched_wards:
        frame_start = int(w.get('frame_start', 0))
        timestamp = int(w.get('timestamp_placed', 0)) if w.get('timestamp_placed') else 0

        # timestamp_placedをフレームに変換
        expected_frame = timestamp_to_frame(timestamp, frame_timestamps)
        diff = frame_start - expected_frame
        frame_diffs.append(diff)

    if frame_diffs:
        avg_diff = sum(frame_diffs) / len(frame_diffs)
        min_diff = min(frame_diffs)
        max_diff = max(frame_diffs)

        print(f"\n  matchedのward ({len(matched_wards)}件) のフレーム差分析:")
        print(f"    frame_start - expected_frame の統計:")
        print(f"    平均: {avg_diff:.1f}")
        print(f"    最小: {min_diff}")
        print(f"    最大: {max_diff}")

        # 分布
        bins = defaultdict(int)
        for d in frame_diffs:
            if d < -10:
                bins["< -10"] += 1
            elif d < 0:
                bins["-10〜-1"] += 1
            elif d < 10:
                bins["0〜9"] += 1
            elif d < 20:
                bins["10〜19"] += 1
            elif d < 30:
                bins["20〜29"] += 1
            else:
                bins[">= 30"] += 1

        print(f"\n    フレーム差の分布:")
        for label in ["< -10", "-10〜-1", "0〜9", "10〜19", "20〜29", ">= 30"]:
            cnt = bins.get(label, 0)
            bar = "#" * (cnt // 2)
            print(f"      {label:>10}: {cnt:3d} {bar}")

    # ============================================
    # 4. チーム判定の精度分析
    # ============================================
    print(f"\n{'='*70}")
    print(f"【診断4: チーム判定の整合性】")
    print(f"{'='*70}")

    # matchedのwardでチーム判定を確認
    team_match = 0
    team_mismatch = []

    for w in matched_wards:
        timeline_team = w.get('team', '')
        class_name = w.get('class_name', '')

        # class_nameからYOLO検出チームを推定
        if 'enemy' in class_name:
            yolo_team = 'red'
        else:
            yolo_team = 'blue'

        if timeline_team == yolo_team:
            team_match += 1
        else:
            team_mismatch.append(w)

    print(f"\n  チーム判定の一致: {team_match}/{len(matched_wards)}")

    if team_mismatch:
        print(f"  チーム不一致: {len(team_mismatch)}件")
        print(f"    ※現在のマッチングロジックではチーム不一致はマッチしないはず")
        print(f"    ※不一致があれば、ハンガリアン法 + ignore_team使用の可能性")

    return {
        'match_id': match_id,
        'matched': status_counts['matched'],
        'timeline_only': status_counts['timeline_only'],
        'detection_only': status_counts['detection_only']
    }


def main():
    # パス設定
    dataset_dir = Path(r"C:\dataset_20260105")
    timeline_dir = Path("data/timeline")

    # JP1-555621265を分析（wards_matched.csvがある試合）
    match_dir = dataset_dir / "JP1-555621265"

    if match_dir.exists():
        analyze_match(match_dir, timeline_dir)
    else:
        print(f"試合ディレクトリが見つかりません: {match_dir}")


if __name__ == "__main__":
    main()

"""
バッチ推論＋クラスタリングによるward座標抽出

処理の流れ:
1. 指定した試合フォルダ内の全フレームをYOLOv8で推論
2. 生の検出結果をdetections_raw.csvに保存
3. 空間クラスタリングで同一wardをグループ化
4. ward_idを付与してwards.csvに保存
5. (オプション) タイムラインデータとマッチングしてwards_matched.csvに保存

使用方法:
    python scripts/batch_inference_wards.py --match JP1-551272474
    python scripts/batch_inference_wards.py --all  # 全試合処理
    python scripts/batch_inference_wards.py --all --with-timeline  # タイムライン統合
"""

import argparse
import csv
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from collections import defaultdict

import numpy as np
from tqdm import tqdm
from ultralytics import YOLO


# =============================================================================
# 設定
# =============================================================================

# モデルパス（v4を使用）
MODEL_PATH = Path(__file__).parent.parent / "models" / "best.pt"

# データセットパス（デフォルト）
DATASET_DIR = Path(r"C:\dataset_20260105")

# 推論設定
CONFIDENCE_THRESHOLD = 0.6  # 検出信頼度の閾値
IMAGE_SIZE = 512  # 推論時の画像サイズ

# クラスタリング設定
DISTANCE_THRESHOLD = 0.01  # 同一wardと判定する座標距離（正規化座標、約5px）
MIN_FRAMES = 3  # ノイズ除去：最小連続フレーム数
GAP_TOLERANCE = 10  # 検出が途切れても同一wardとみなすフレーム数

# タイムラインデータディレクトリ
TIMELINE_DIR = Path("data/timeline")


# =============================================================================
# データ構造
# =============================================================================

@dataclass
class Detection:
    """1フレームでの検出結果"""
    frame: int
    class_id: int
    class_name: str
    x: float  # 正規化座標 (0-1)
    y: float
    w: float
    h: float
    confidence: float


@dataclass
class Ward:
    """クラスタリング後のward"""
    ward_id: int
    class_id: int
    class_name: str
    x: float  # 平均座標
    y: float
    frame_start: int
    frame_end: int
    detections: List[Detection] = field(default_factory=list)

    @property
    def confidence_avg(self) -> float:
        if not self.detections:
            return 0.0
        return sum(d.confidence for d in self.detections) / len(self.detections)

    @property
    def detection_count(self) -> int:
        return len(self.detections)


# =============================================================================
# 推論処理
# =============================================================================

def run_inference(model: YOLO, match_dir: Path, conf: float = CONFIDENCE_THRESHOLD) -> List[Detection]:
    """
    1試合分の全フレームを推論し、検出結果を返す
    """
    frame_dir = match_dir / "0"
    if not frame_dir.exists():
        print(f"フレームディレクトリが見つかりません: {frame_dir}")
        return []

    # フレーム一覧（ソート済み）
    frame_files = sorted(frame_dir.glob("*.png"), key=lambda p: int(p.stem))
    if not frame_files:
        print(f"フレームが見つかりません: {frame_dir}")
        return []

    detections: List[Detection] = []

    # バッチ推論（メモリ効率のため1枚ずつ）
    for frame_path in tqdm(frame_files, desc=f"推論中 ({match_dir.name})"):
        frame_num = int(frame_path.stem)

        # 推論実行
        results = model(str(frame_path), imgsz=IMAGE_SIZE, conf=conf, verbose=False)

        # 検出結果を抽出
        for box in results[0].boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            x, y, w, h = box.xywhn[0].tolist()  # 正規化座標
            confidence = float(box.conf[0])

            detections.append(Detection(
                frame=frame_num,
                class_id=class_id,
                class_name=class_name,
                x=x,
                y=y,
                w=w,
                h=h,
                confidence=confidence
            ))

    return detections


def save_raw_detections(detections: List[Detection], output_path: Path):
    """生の検出結果をCSVに保存"""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['frame', 'class_id', 'class_name', 'x', 'y', 'w', 'h', 'confidence'])

        for d in detections:
            writer.writerow([
                d.frame, d.class_id, d.class_name,
                f"{d.x:.6f}", f"{d.y:.6f}", f"{d.w:.6f}", f"{d.h:.6f}",
                f"{d.confidence:.4f}"
            ])

    print(f"生の検出結果を保存: {output_path} ({len(detections)}件)")


# =============================================================================
# クラスタリング処理
# =============================================================================

def distance(d1: Detection, d2: Detection) -> float:
    """2つの検出間のユークリッド距離（正規化座標）"""
    return np.sqrt((d1.x - d2.x) ** 2 + (d1.y - d2.y) ** 2)


def cluster_detections(detections: List[Detection],
                       distance_threshold: float = DISTANCE_THRESHOLD,
                       min_frames: int = MIN_FRAMES,
                       gap_tolerance: int = GAP_TOLERANCE) -> List[Ward]:
    """
    検出結果をクラスタリングしてward単位にまとめる

    アルゴリズム:
    1. フレーム順にソート
    2. 各検出を既存のwardクラスタに割り当てるか、新規wardを作成
    3. 同一クラス＆座標が近い場合は同一wardとみなす
    4. min_frames未満の検出はノイズとして除去
    """
    if not detections:
        return []

    # フレーム順にソート
    sorted_detections = sorted(detections, key=lambda d: d.frame)

    # アクティブなwardクラスタ（クラスごとに管理）
    active_wards: Dict[int, List[Ward]] = defaultdict(list)

    # 完了したward
    completed_wards: List[Ward] = []

    # 次のward_id
    next_ward_id = 1

    for det in sorted_detections:
        matched = False

        # 同じクラスのアクティブなwardを検索
        for ward in active_wards[det.class_id]:
            # 座標が近く、フレームが連続（gap_tolerance以内）の場合
            if (distance(det, ward.detections[-1]) < distance_threshold and
                det.frame - ward.frame_end <= gap_tolerance):
                # 既存のwardに追加
                ward.detections.append(det)
                ward.frame_end = det.frame
                # 座標を更新（移動平均）
                ward.x = sum(d.x for d in ward.detections) / len(ward.detections)
                ward.y = sum(d.y for d in ward.detections) / len(ward.detections)
                matched = True
                break

        if not matched:
            # 新規wardを作成
            new_ward = Ward(
                ward_id=next_ward_id,
                class_id=det.class_id,
                class_name=det.class_name,
                x=det.x,
                y=det.y,
                frame_start=det.frame,
                frame_end=det.frame,
                detections=[det]
            )
            active_wards[det.class_id].append(new_ward)
            next_ward_id += 1

        # 古いwardを完了リストに移動
        current_frame = det.frame
        for class_id in list(active_wards.keys()):
            still_active = []
            for ward in active_wards[class_id]:
                if current_frame - ward.frame_end > gap_tolerance:
                    # gap_toleranceを超えて検出がない → 完了
                    completed_wards.append(ward)
                else:
                    still_active.append(ward)
            active_wards[class_id] = still_active

    # 残りのアクティブなwardを完了リストに追加
    for class_id in active_wards:
        completed_wards.extend(active_wards[class_id])

    # ノイズ除去（min_frames未満は除外）
    filtered_wards = [w for w in completed_wards if w.detection_count >= min_frames]

    # ward_idを振り直し
    for i, ward in enumerate(sorted(filtered_wards, key=lambda w: w.frame_start), start=1):
        ward.ward_id = i

    return sorted(filtered_wards, key=lambda w: w.frame_start)


def save_wards(wards: List[Ward], output_path: Path):
    """クラスタリング後のward情報をCSVに保存"""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'ward_id', 'class_id', 'class_name',
            'x', 'y',
            'frame_start', 'frame_end',
            'detection_count', 'confidence_avg'
        ])

        for w in wards:
            writer.writerow([
                w.ward_id, w.class_id, w.class_name,
                f"{w.x:.6f}", f"{w.y:.6f}",
                w.frame_start, w.frame_end,
                w.detection_count, f"{w.confidence_avg:.4f}"
            ])

    print(f"ward情報を保存: {output_path} ({len(wards)}個のward)")


# =============================================================================
# 統計表示
# =============================================================================

def print_statistics(wards: List[Ward], detections: List[Detection]):
    """処理結果の統計を表示"""
    print("\n=== 処理結果 ===")
    print(f"生の検出数: {len(detections)}")
    print(f"クラスタリング後のward数: {len(wards)}")

    if wards:
        # クラス別集計
        class_counts = defaultdict(int)
        for w in wards:
            class_counts[w.class_name] += 1

        print("\nクラス別ward数:")
        for class_name, count in sorted(class_counts.items()):
            print(f"  {class_name}: {count}")

        # 平均検出フレーム数
        avg_frames = sum(w.detection_count for w in wards) / len(wards)
        print(f"\n平均検出フレーム数/ward: {avg_frames:.1f}")

        # 平均信頼度
        avg_conf = sum(w.confidence_avg for w in wards) / len(wards)
        print(f"平均信頼度: {avg_conf:.3f}")


# =============================================================================
# メイン処理
# =============================================================================

def process_match(model: YOLO, match_dir: Path, with_timeline: bool = False, timeline_dir: Path = TIMELINE_DIR,
                  use_hungarian: bool = False, ignore_team: bool = False):
    """1試合を処理"""
    print(f"\n{'='*60}")
    print(f"処理開始: {match_dir.name}")
    print(f"{'='*60}")

    # 1. 推論
    detections = run_inference(model, match_dir)

    if not detections:
        print("検出なし")
        return

    # 2. 生の検出結果を保存
    raw_output = match_dir / "detections_raw.csv"
    save_raw_detections(detections, raw_output)

    # 3. クラスタリング
    wards = cluster_detections(detections)

    # 4. ward情報を保存
    wards_output = match_dir / "wards.csv"
    save_wards(wards, wards_output)

    # 5. 統計表示
    print_statistics(wards, detections)

    # 6. タイムライン統合（オプション）
    if with_timeline:
        print("\n--- タイムライン統合 ---")
        from autoLeague.dataset.ward_tracker import WardTracker
        tracker = WardTracker(
            timeline_dir=timeline_dir,
            dataset_dir=match_dir.parent,
            use_hungarian=use_hungarian,
            ignore_team=ignore_team
        )
        tracker.process_match(match_dir.name)


def main():
    parser = argparse.ArgumentParser(description="バッチ推論＋クラスタリングによるward座標抽出")
    parser.add_argument("--match", type=str, help="処理する試合ID（例: JP1-551272474）")
    parser.add_argument("--all", action="store_true", help="全試合を処理")
    parser.add_argument("--dataset", type=str, default=str(DATASET_DIR), help="データセットディレクトリ")
    parser.add_argument("--model", type=str, default=str(MODEL_PATH), help="モデルパス")
    parser.add_argument("--conf", type=float, default=CONFIDENCE_THRESHOLD, help="信頼度閾値")
    parser.add_argument("--with-timeline", action="store_true", help="タイムラインデータと統合")
    parser.add_argument("--timeline-dir", type=str, default=str(TIMELINE_DIR), help="タイムラインデータディレクトリ")
    parser.add_argument("--hungarian", action="store_true", help="ハンガリアン法（全体最適）を使用")
    parser.add_argument("--ignore-team", action="store_true", help="チーム不一致でもマッチング候補とする")
    args = parser.parse_args()

    # モデル読み込み
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"エラー: モデルが見つかりません: {model_path}")
        return

    print(f"モデル読み込み: {model_path}")
    model = YOLO(str(model_path))
    print(f"クラス: {model.names}")

    # データセットディレクトリ
    dataset_dir = Path(args.dataset)
    if not dataset_dir.exists():
        print(f"エラー: データセットディレクトリが見つかりません: {dataset_dir}")
        return

    # 処理対象を決定
    if args.match:
        match_dirs = [dataset_dir / args.match]
        if not match_dirs[0].exists():
            print(f"エラー: 試合ディレクトリが見つかりません: {match_dirs[0]}")
            return
    elif args.all:
        match_dirs = sorted(dataset_dir.glob("JP1-*"))
        print(f"全{len(match_dirs)}試合を処理します")
    else:
        parser.print_help()
        return

    # タイムラインディレクトリ
    timeline_dir = Path(args.timeline_dir)

    # 各試合を処理
    for match_dir in match_dirs:
        process_match(model, match_dir, with_timeline=args.with_timeline, timeline_dir=timeline_dir,
                      use_hungarian=args.hungarian, ignore_team=args.ignore_team)

    print("\n処理完了")


if __name__ == "__main__":
    main()

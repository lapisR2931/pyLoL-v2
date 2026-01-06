"""
貪欲法 vs ハンガリアン法のマッチング結果を画像で可視化

使用方法:
    python scripts/visualize_matching_comparison.py --match JP1-555621265
"""

import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

from autoLeague.dataset.ward_tracker import (
    load_ward_events_from_timeline,
    load_detected_wards,
    load_frame_timestamps,
    filter_river_sights,
    match_wards,
    match_wards_hungarian,
    MatchedWard,
    FRAME_TOLERANCE,
    IMAGE_SIZE
)


def plot_wards_on_minimap(
    ax,
    results: List[MatchedWard],
    title: str,
    bg_image: Image.Image = None
):
    """ミニマップ上にwardをプロット"""

    # 背景画像
    if bg_image:
        ax.imshow(bg_image, extent=[0, IMAGE_SIZE, IMAGE_SIZE, 0], alpha=0.5)

    ax.set_xlim(0, IMAGE_SIZE)
    ax.set_ylim(IMAGE_SIZE, 0)  # Y軸反転（画像座標系）
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=14, fontweight='bold')

    # ステータス別に色分け
    colors = {
        "matched": "#2ecc71",        # 緑
        "timeline_only": "#e74c3c",  # 赤
        "detection_only": "#3498db"  # 青
    }

    markers = {
        "matched": "o",
        "timeline_only": "x",
        "detection_only": "s"
    }

    # wardタイプ別にサイズ変更
    def get_size(ward):
        if "control" in ward.ward_type.lower():
            return 120
        return 80

    # プロット
    for status in ["matched", "timeline_only", "detection_only"]:
        wards = [w for w in results if w.match_status == status]

        if not wards:
            continue

        xs = []
        ys = []
        sizes = []

        for w in wards:
            if w.x_pixel >= 0 and w.y_pixel >= 0:
                xs.append(w.x_pixel)
                ys.append(w.y_pixel)
                sizes.append(get_size(w))
            elif status == "timeline_only":
                # timeline_onlyは座標がないので表示しない（またはランダム配置）
                pass

        if xs:
            ax.scatter(
                xs, ys,
                c=colors[status],
                s=sizes,
                marker=markers[status],
                alpha=0.8,
                edgecolors='white',
                linewidths=1,
                label=f"{status} ({len(wards)})"
            )

    ax.legend(loc='upper left', fontsize=9)
    ax.set_xlabel('X (pixel)')
    ax.set_ylabel('Y (pixel)')


def plot_comparison(
    greedy_results: List[MatchedWard],
    hungarian_results: List[MatchedWard],
    bg_image: Image.Image,
    match_id: str,
    placed_count: int,
    output_path: Path = None
):
    """2つの手法を並べて比較表示"""

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # 統計計算
    greedy_matched = len([w for w in greedy_results if w.match_status == "matched"])
    hungarian_matched = len([w for w in hungarian_results if w.match_status == "matched"])

    greedy_rate = greedy_matched / placed_count * 100 if placed_count > 0 else 0
    hungarian_rate = hungarian_matched / placed_count * 100 if placed_count > 0 else 0

    # 貪欲法
    plot_wards_on_minimap(
        axes[0],
        greedy_results,
        f"貪欲法 (Greedy)\nマッチング率: {greedy_rate:.1f}% ({greedy_matched}/{placed_count})",
        bg_image
    )

    # ハンガリアン法
    plot_wards_on_minimap(
        axes[1],
        hungarian_results,
        f"ハンガリアン法 (Hungarian)\nマッチング率: {hungarian_rate:.1f}% ({hungarian_matched}/{placed_count})",
        bg_image
    )

    # 全体タイトル
    fig.suptitle(
        f"Ward Matching Comparison - {match_id}\n"
        f"緑=matched, 赤=timeline_only(検出失敗), 青=detection_only(タイムラインなし)",
        fontsize=12
    )

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"画像を保存: {output_path}")

    plt.show()


def plot_difference_only(
    greedy_results: List[MatchedWard],
    hungarian_results: List[MatchedWard],
    bg_image: Image.Image,
    match_id: str,
    output_path: Path = None
):
    """差分のあるwardのみを表示"""

    # timeline_ward_idでインデックス化
    greedy_by_tid = {w.timeline_ward_id: w for w in greedy_results if w.timeline_ward_id}
    hungarian_by_tid = {w.timeline_ward_id: w for w in hungarian_results if w.timeline_ward_id}

    common_tids = set(greedy_by_tid.keys()) & set(hungarian_by_tid.keys())

    # 差分を抽出
    diff_wards = []
    for tid in common_tids:
        g = greedy_by_tid[tid]
        h = hungarian_by_tid[tid]
        if g.match_status != h.match_status:
            diff_wards.append({
                "tid": tid,
                "greedy": g,
                "hungarian": h
            })

    if not diff_wards:
        print("両手法で差分はありませんでした。")
        return

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    if bg_image:
        ax.imshow(bg_image, extent=[0, IMAGE_SIZE, IMAGE_SIZE, 0], alpha=0.5)

    ax.set_xlim(0, IMAGE_SIZE)
    ax.set_ylim(IMAGE_SIZE, 0)
    ax.set_aspect('equal')

    # 差分をプロット
    for d in diff_wards:
        h = d["hungarian"]
        if h.x_pixel >= 0 and h.y_pixel >= 0:
            # ハンガリアン法でマッチしたものを表示
            color = "#2ecc71" if h.match_status == "matched" else "#e74c3c"
            ax.scatter(
                h.x_pixel, h.y_pixel,
                c=color,
                s=150,
                marker='o',
                edgecolors='black',
                linewidths=2,
                alpha=0.9
            )
            ax.annotate(
                f"TL:{d['tid']}",
                (h.x_pixel, h.y_pixel),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=8,
                color='white',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7)
            )

    ax.set_title(
        f"差分のあるward ({len(diff_wards)}件)\n"
        f"緑=ハンガリアン法でマッチ成功, 赤=失敗",
        fontsize=12,
        fontweight='bold'
    )

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"差分画像を保存: {output_path}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(description="マッチング結果を画像で可視化")
    parser.add_argument("--match", type=str, required=True, help="試合ID")
    parser.add_argument("--timeline-dir", type=str, default="data/timeline")
    parser.add_argument("--dataset", type=str, default=r"C:\dataset_20260105")
    parser.add_argument("--tolerance", type=int, default=FRAME_TOLERANCE)
    parser.add_argument("--output", type=str, help="出力画像パス")
    parser.add_argument("--diff-only", action="store_true", help="差分のみ表示")
    parser.add_argument("--frame", type=int, default=100, help="背景に使用するフレーム番号")
    args = parser.parse_args()

    # パス設定
    match_id = args.match
    if match_id.startswith("JP1-"):
        match_id_num = match_id.replace("JP1-", "")
    else:
        match_id_num = match_id
        match_id = f"JP1-{match_id}"

    timeline_dir = Path(args.timeline_dir)
    dataset_dir = Path(args.dataset)

    timeline_path = timeline_dir / f"JP1_{match_id_num}.json"
    match_dir = dataset_dir / match_id
    wards_csv_path = match_dir / "wards.csv"
    frame_timestamps_path = match_dir / "frame_timestamps.csv"
    frame_dir = match_dir / "0"

    # ファイル存在チェック
    if not timeline_path.exists():
        print(f"エラー: タイムラインが見つかりません: {timeline_path}")
        return

    if not wards_csv_path.exists():
        print(f"エラー: 検出結果が見つかりません: {wards_csv_path}")
        return

    # 背景画像読み込み
    bg_image = None
    frame_path = frame_dir / f"{args.frame}.png"
    if frame_path.exists():
        bg_image = Image.open(frame_path)
        print(f"背景画像: {frame_path}")
    else:
        # 最初のフレームを使用
        frames = sorted(frame_dir.glob("*.png"))
        if frames:
            bg_image = Image.open(frames[0])
            print(f"背景画像: {frames[0]}")

    # データ読み込み
    frame_timestamps = load_frame_timestamps(frame_timestamps_path)
    ms_per_frame = None

    placed_events, killed_events, filter_stats = load_ward_events_from_timeline(
        timeline_path, ms_per_frame, frame_timestamps
    )

    detected_wards_greedy = load_detected_wards(wards_csv_path)
    detected_wards_greedy, river_count = filter_river_sights(detected_wards_greedy, frame_timestamps, ms_per_frame)
    detected_wards_hungarian = load_detected_wards(wards_csv_path)
    detected_wards_hungarian, _ = filter_river_sights(detected_wards_hungarian, frame_timestamps, ms_per_frame)

    print(f"タイムラインward: {filter_stats['valid']}件")
    print(f"YOLO検出ward: {len(detected_wards_greedy)}件 (リバーサイト除外: {river_count}件)")

    # マッチング実行
    print("貪欲法でマッチング...")
    greedy_results = match_wards(
        placed_events, killed_events, detected_wards_greedy,
        args.tolerance, ms_per_frame, frame_timestamps
    )

    print("ハンガリアン法でマッチング...")
    hungarian_results = match_wards_hungarian(
        placed_events, killed_events, detected_wards_hungarian,
        args.tolerance, ms_per_frame, frame_timestamps
    )

    # 可視化
    output_path = Path(args.output) if args.output else None

    if args.diff_only:
        plot_difference_only(
            greedy_results, hungarian_results,
            bg_image, match_id, output_path
        )
    else:
        plot_comparison(
            greedy_results, hungarian_results,
            bg_image, match_id,
            filter_stats['valid'],
            output_path
        )


if __name__ == "__main__":
    main()

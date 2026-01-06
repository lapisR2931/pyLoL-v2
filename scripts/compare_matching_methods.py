"""
貪欲法 vs ハンガリアン法のマッチング結果比較スクリプト

使用方法:
    python scripts/compare_matching_methods.py --match JP1-555621265
"""

import argparse
from pathlib import Path
from typing import List, Dict

from autoLeague.dataset.ward_tracker import (
    WardTracker,
    load_ward_events_from_timeline,
    load_detected_wards,
    load_frame_timestamps,
    filter_river_sights,
    match_wards,
    match_wards_hungarian,
    MatchedWard,
    FRAME_TOLERANCE
)


def compare_results(
    greedy_results: List[MatchedWard],
    hungarian_results: List[MatchedWard]
) -> Dict:
    """両手法の結果を比較"""

    # ステータス別集計
    def count_by_status(results):
        counts = {"matched": 0, "timeline_only": 0, "detection_only": 0}
        for w in results:
            counts[w.match_status] += 1
        return counts

    greedy_stats = count_by_status(greedy_results)
    hungarian_stats = count_by_status(hungarian_results)

    return {
        "greedy": greedy_stats,
        "hungarian": hungarian_stats
    }


def print_comparison_table(
    greedy_results: List[MatchedWard],
    hungarian_results: List[MatchedWard],
    placed_count: int
):
    """比較結果をテーブル形式で表示"""

    stats = compare_results(greedy_results, hungarian_results)

    print("\n" + "=" * 70)
    print("マッチング手法比較")
    print("=" * 70)

    print(f"\nタイムラインward設置イベント数: {placed_count}")

    print("\n{:<20} {:>15} {:>15}".format("", "貪欲法", "ハンガリアン法"))
    print("-" * 50)
    print("{:<20} {:>15} {:>15}".format(
        "matched",
        stats["greedy"]["matched"],
        stats["hungarian"]["matched"]
    ))
    print("{:<20} {:>15} {:>15}".format(
        "timeline_only",
        stats["greedy"]["timeline_only"],
        stats["hungarian"]["timeline_only"]
    ))
    print("{:<20} {:>15} {:>15}".format(
        "detection_only",
        stats["greedy"]["detection_only"],
        stats["hungarian"]["detection_only"]
    ))
    print("-" * 50)

    greedy_rate = stats["greedy"]["matched"] / placed_count * 100 if placed_count > 0 else 0
    hungarian_rate = stats["hungarian"]["matched"] / placed_count * 100 if placed_count > 0 else 0

    print("{:<20} {:>14.1f}% {:>14.1f}%".format(
        "マッチング率",
        greedy_rate,
        hungarian_rate
    ))

    diff = hungarian_rate - greedy_rate
    print(f"\n差分: {diff:+.1f}%")


def print_detailed_comparison(
    greedy_results: List[MatchedWard],
    hungarian_results: List[MatchedWard]
):
    """詳細な比較（どのwardで差が出たか）"""

    # timeline_ward_idでインデックス化
    greedy_by_tid = {w.timeline_ward_id: w for w in greedy_results if w.timeline_ward_id}
    hungarian_by_tid = {w.timeline_ward_id: w for w in hungarian_results if w.timeline_ward_id}

    # 両方に存在するtimeline_ward_id
    common_tids = set(greedy_by_tid.keys()) & set(hungarian_by_tid.keys())

    # 差分を検出
    differences = []
    for tid in sorted(common_tids):
        g = greedy_by_tid[tid]
        h = hungarian_by_tid[tid]

        if g.match_status != h.match_status:
            differences.append({
                "timeline_ward_id": tid,
                "ward_type": g.ward_type,
                "team": g.team,
                "greedy_status": g.match_status,
                "hungarian_status": h.match_status,
                "greedy_frame": g.frame_start if g.match_status == "matched" else "-",
                "hungarian_frame": h.frame_start if h.match_status == "matched" else "-",
            })

    if differences:
        print("\n" + "=" * 70)
        print("マッチング結果が異なるward")
        print("=" * 70)

        print("\n{:<8} {:<15} {:<6} {:<15} {:<15}".format(
            "TL_ID", "ward_type", "team", "貪欲法", "ハンガリアン"
        ))
        print("-" * 60)

        for d in differences:
            print("{:<8} {:<15} {:<6} {:<15} {:<15}".format(
                d["timeline_ward_id"],
                d["ward_type"],
                d["team"],
                d["greedy_status"],
                d["hungarian_status"]
            ))

        print(f"\n差分件数: {len(differences)}")
    else:
        print("\n両手法でマッチング結果に差はありませんでした。")


def print_matched_wards_sample(results: List[MatchedWard], method_name: str, limit: int = 20):
    """マッチング結果のサンプル表示"""

    print(f"\n{'=' * 70}")
    print(f"{method_name} - マッチング結果サンプル（先頭{limit}件）")
    print("=" * 70)

    # matchedのものを表示
    matched = [w for w in results if w.match_status == "matched"][:limit]

    print("\n{:<6} {:<8} {:<18} {:<6} {:>8} {:>8} {:>10}".format(
        "ID", "TL_ID", "ward_type", "team", "frame_s", "frame_e", "conf"
    ))
    print("-" * 70)

    for w in matched:
        print("{:<6} {:<8} {:<18} {:<6} {:>8} {:>8} {:>10.3f}".format(
            w.ward_id,
            w.timeline_ward_id or "-",
            w.ward_type,
            w.team,
            w.frame_start,
            w.frame_end if w.frame_end > 0 else "-",
            w.confidence_avg
        ))


def main():
    parser = argparse.ArgumentParser(description="貪欲法 vs ハンガリアン法の比較")
    parser.add_argument("--match", type=str, required=True, help="試合ID（例: JP1-555621265）")
    parser.add_argument("--timeline-dir", type=str, default="data/timeline")
    parser.add_argument("--dataset", type=str, default=r"C:\dataset_20260105")
    parser.add_argument("--tolerance", type=int, default=FRAME_TOLERANCE)
    parser.add_argument("--show-sample", action="store_true", help="マッチング結果サンプルを表示")
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

    # ファイル存在チェック
    if not timeline_path.exists():
        print(f"エラー: タイムラインファイルが見つかりません: {timeline_path}")
        return

    if not wards_csv_path.exists():
        print(f"エラー: 検出結果ファイルが見つかりません: {wards_csv_path}")
        return

    print(f"試合: {match_id}")
    print(f"タイムライン: {timeline_path}")
    print(f"検出結果: {wards_csv_path}")

    # データ読み込み
    frame_timestamps = load_frame_timestamps(frame_timestamps_path)
    ms_per_frame = None

    placed_events, killed_events, filter_stats = load_ward_events_from_timeline(
        timeline_path, ms_per_frame, frame_timestamps
    )

    print(f"\nタイムラインwardイベント: {filter_stats['valid']}件（除外: {filter_stats['total'] - filter_stats['valid']}件）")

    # 検出結果読み込み
    detected_wards_raw = load_detected_wards(wards_csv_path)
    print(f"YOLO検出ward: {len(detected_wards_raw)}件")

    # リバーサイト（スカトルの視界）をフィルタリング
    detected_wards_filtered, river_sight_count = filter_river_sights(
        detected_wards_raw, frame_timestamps, ms_per_frame
    )
    if river_sight_count > 0:
        print(f"リバーサイト除外: {river_sight_count}件")
        print(f"フィルタリング後: {len(detected_wards_filtered)}件")

    # 2回使うのでリストをコピー
    detected_wards_greedy = load_detected_wards(wards_csv_path)
    detected_wards_greedy, _ = filter_river_sights(detected_wards_greedy, frame_timestamps, ms_per_frame)
    detected_wards_hungarian = load_detected_wards(wards_csv_path)
    detected_wards_hungarian, _ = filter_river_sights(detected_wards_hungarian, frame_timestamps, ms_per_frame)

    # 貪欲法でマッチング
    print("\n貪欲法でマッチング中...")
    greedy_results = match_wards(
        placed_events, killed_events, detected_wards_greedy,
        args.tolerance, ms_per_frame, frame_timestamps
    )

    # ハンガリアン法でマッチング
    print("ハンガリアン法でマッチング中...")
    hungarian_results = match_wards_hungarian(
        placed_events, killed_events, detected_wards_hungarian,
        args.tolerance, ms_per_frame, frame_timestamps
    )

    # 比較結果を表示
    print_comparison_table(greedy_results, hungarian_results, filter_stats['valid'])

    # 詳細比較
    print_detailed_comparison(greedy_results, hungarian_results)

    # サンプル表示（オプション）
    if args.show_sample:
        print_matched_wards_sample(greedy_results, "貪欲法")
        print_matched_wards_sample(hungarian_results, "ハンガリアン法")


if __name__ == "__main__":
    main()

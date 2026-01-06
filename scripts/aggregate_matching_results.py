"""
全試合のマッチング結果を集計

使用方法:
    python scripts/aggregate_matching_results.py
"""

import csv
from pathlib import Path
from collections import defaultdict

DATASET_DIR = Path(r"C:\dataset_20260105")


def aggregate_results():
    """全試合のwards_matched.csvを集計"""

    total_stats = {
        "matches": 0,
        "timeline_total": 0,
        "matched": 0,
        "timeline_only": 0,
        "detection_only": 0,
        "river_sights": 0,  # 除外されたリバーサイト（推定）
    }

    per_match_stats = []

    for match_dir in sorted(DATASET_DIR.glob("JP1-*")):
        matched_csv = match_dir / "wards_matched.csv"
        if not matched_csv.exists():
            continue

        match_stats = {
            "match_id": match_dir.name,
            "matched": 0,
            "timeline_only": 0,
            "detection_only": 0,
        }

        with open(matched_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                status = row['match_status']
                if status == "matched":
                    match_stats["matched"] += 1
                elif status == "timeline_only":
                    match_stats["timeline_only"] += 1
                elif status == "detection_only":
                    match_stats["detection_only"] += 1

        # タイムライン総数 = matched + timeline_only
        timeline_total = match_stats["matched"] + match_stats["timeline_only"]
        match_stats["timeline_total"] = timeline_total

        # マッチング率
        if timeline_total > 0:
            match_stats["match_rate"] = match_stats["matched"] / timeline_total * 100
        else:
            match_stats["match_rate"] = 0

        per_match_stats.append(match_stats)

        # 全体集計
        total_stats["matches"] += 1
        total_stats["timeline_total"] += timeline_total
        total_stats["matched"] += match_stats["matched"]
        total_stats["timeline_only"] += match_stats["timeline_only"]
        total_stats["detection_only"] += match_stats["detection_only"]

    return total_stats, per_match_stats


def print_results(total_stats, per_match_stats):
    """結果を表示"""

    print("=" * 70)
    print("全試合マッチング結果集計")
    print("=" * 70)

    # 試合別結果
    print("\n{:<20} {:>8} {:>8} {:>10} {:>12}".format(
        "試合ID", "matched", "TL_only", "det_only", "マッチング率"
    ))
    print("-" * 70)

    for stats in per_match_stats:
        print("{:<20} {:>8} {:>8} {:>10} {:>11.1f}%".format(
            stats["match_id"],
            stats["matched"],
            stats["timeline_only"],
            stats["detection_only"],
            stats["match_rate"]
        ))

    # 全体サマリー
    print("\n" + "=" * 70)
    print("全体サマリー")
    print("=" * 70)

    print(f"\n処理試合数: {total_stats['matches']}")
    print(f"タイムラインward総数: {total_stats['timeline_total']}")
    print(f"  - matched: {total_stats['matched']}")
    print(f"  - timeline_only: {total_stats['timeline_only']}")
    print(f"detection_only総数: {total_stats['detection_only']}")

    if total_stats['timeline_total'] > 0:
        overall_rate = total_stats['matched'] / total_stats['timeline_total'] * 100
        print(f"\n全体マッチング率: {overall_rate:.1f}% ({total_stats['matched']}/{total_stats['timeline_total']})")

    # 平均マッチング率
    if per_match_stats:
        avg_rate = sum(s["match_rate"] for s in per_match_stats) / len(per_match_stats)
        print(f"試合別平均マッチング率: {avg_rate:.1f}%")


def main():
    total_stats, per_match_stats = aggregate_results()

    if not per_match_stats:
        print("処理済みの試合が見つかりませんでした。")
        print("先に以下のコマンドを実行してください:")
        print("  python autoLeague/dataset/ward_tracker.py --all --hungarian")
        return

    print_results(total_stats, per_match_stats)


if __name__ == "__main__":
    main()

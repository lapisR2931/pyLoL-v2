"""
リバーサイト（スカトルの視界）の座標分析

detection_onlyのstealth_wardの座標を分析し、
リバーサイトの固定位置を特定する
"""

import csv
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

DATASET_DIR = Path(r"C:\dataset_20260105")
IMAGE_SIZE = 512


def load_all_detection_only_wards():
    """全試合からdetection_onlyのstealth_wardを収集"""
    wards = []

    for match_dir in sorted(DATASET_DIR.glob("JP1-*")):
        matched_csv = match_dir / "wards_matched.csv"
        if not matched_csv.exists():
            continue

        with open(matched_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['match_status'] == 'detection_only' and 'stealth' in row['class_name']:
                    if row['x_pixel'] and row['y_pixel']:
                        wards.append({
                            'match_id': match_dir.name,
                            'x': int(row['x_pixel']),
                            'y': int(row['y_pixel']),
                            'class_name': row['class_name'],
                            'frame_start': int(row['frame_start']),
                            'frame_end': int(row['frame_end']) if row['frame_end'] else 0,
                            'confidence': float(row['confidence_avg']) if row['confidence_avg'] else 0
                        })

    return wards


def analyze_clusters(wards, cluster_radius=20):
    """座標クラスタリングで頻出位置を特定"""
    if not wards:
        return []

    coords = np.array([[w['x'], w['y']] for w in wards])

    # 簡易クラスタリング（グリッドベース）
    grid_size = cluster_radius
    grid_counts = defaultdict(list)

    for i, (x, y) in enumerate(coords):
        grid_key = (int(x / grid_size), int(y / grid_size))
        grid_counts[grid_key].append(i)

    # 頻出グリッドを抽出
    clusters = []
    for grid_key, indices in grid_counts.items():
        if len(indices) >= 3:  # 3回以上出現
            cluster_coords = coords[indices]
            center_x = np.mean(cluster_coords[:, 0])
            center_y = np.mean(cluster_coords[:, 1])
            clusters.append({
                'center_x': center_x,
                'center_y': center_y,
                'count': len(indices),
                'indices': indices
            })

    # 出現回数でソート
    clusters.sort(key=lambda c: c['count'], reverse=True)
    return clusters


def plot_detection_only_distribution(wards, clusters):
    """detection_onlyの分布を可視化"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # 背景画像（最初の試合のフレームを使用）
    bg_image = None
    for match_dir in DATASET_DIR.glob("JP1-*"):
        frame_path = match_dir / "0" / "100.png"
        if frame_path.exists():
            bg_image = Image.open(frame_path)
            break

    # 左: 全detection_onlyの分布
    ax = axes[0]
    if bg_image:
        ax.imshow(bg_image, extent=[0, IMAGE_SIZE, IMAGE_SIZE, 0], alpha=0.5)

    xs = [w['x'] for w in wards]
    ys = [w['y'] for w in wards]

    ax.scatter(xs, ys, c='blue', s=30, alpha=0.5, label=f'detection_only ({len(wards)})')
    ax.set_xlim(0, IMAGE_SIZE)
    ax.set_ylim(IMAGE_SIZE, 0)
    ax.set_aspect('equal')
    ax.set_title('detection_only stealth_ward 分布', fontsize=12)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend()

    # 右: クラスタ中心をハイライト
    ax = axes[1]
    if bg_image:
        ax.imshow(bg_image, extent=[0, IMAGE_SIZE, IMAGE_SIZE, 0], alpha=0.5)

    ax.scatter(xs, ys, c='lightblue', s=20, alpha=0.3)

    # 上位クラスタをハイライト
    colors = ['red', 'orange', 'yellow', 'green', 'purple']
    for i, cluster in enumerate(clusters[:5]):
        ax.scatter(
            cluster['center_x'], cluster['center_y'],
            c=colors[i % len(colors)],
            s=200,
            marker='*',
            edgecolors='black',
            linewidths=2,
            label=f"クラスタ{i+1}: ({cluster['center_x']:.0f}, {cluster['center_y']:.0f}) n={cluster['count']}"
        )
        # 範囲を円で表示
        circle = plt.Circle(
            (cluster['center_x'], cluster['center_y']),
            25,
            fill=False,
            color=colors[i % len(colors)],
            linewidth=2,
            linestyle='--'
        )
        ax.add_patch(circle)

    ax.set_xlim(0, IMAGE_SIZE)
    ax.set_ylim(IMAGE_SIZE, 0)
    ax.set_aspect('equal')
    ax.set_title('頻出位置（リバーサイト候補）', fontsize=12)
    ax.legend(loc='upper left', fontsize=9)

    plt.tight_layout()
    plt.savefig('river_sight_analysis.png', dpi=150)
    print("画像を保存: river_sight_analysis.png")
    plt.show()


def print_cluster_stats(clusters):
    """クラスタ統計を表示"""
    print("\n" + "=" * 60)
    print("detection_only stealth_ward 頻出位置")
    print("=" * 60)

    print("\n{:<10} {:>10} {:>10} {:>10}".format("クラスタ", "X", "Y", "出現回数"))
    print("-" * 45)

    for i, cluster in enumerate(clusters[:10]):
        print("{:<10} {:>10.0f} {:>10.0f} {:>10}".format(
            f"#{i+1}",
            cluster['center_x'],
            cluster['center_y'],
            cluster['count']
        ))

    # リバーサイト候補の提案
    print("\n" + "=" * 60)
    print("リバーサイト候補座標（フィルタリング用）")
    print("=" * 60)

    # ミニマップ上のリバー位置（概算）
    # バロン側: 左上寄り、ドラゴン側: 右下寄り
    river_candidates = []
    for cluster in clusters[:10]:
        x, y = cluster['center_x'], cluster['center_y']
        # 対角線付近（リバー）にあるクラスタを候補とする
        if 150 < x < 350 and 150 < y < 350:
            river_candidates.append(cluster)

    if river_candidates:
        print("\n推定リバーサイト位置:")
        for i, rc in enumerate(river_candidates[:2]):
            print(f"  位置{i+1}: ({rc['center_x']:.0f}, {rc['center_y']:.0f}) - {rc['count']}回検出")


def main():
    print("detection_only stealth_wardを収集中...")
    wards = load_all_detection_only_wards()
    print(f"収集完了: {len(wards)}件")

    if not wards:
        print("detection_onlyのstealth_wardが見つかりませんでした")
        return

    print("\nクラスタリング中...")
    clusters = analyze_clusters(wards)

    print_cluster_stats(clusters)
    plot_detection_only_distribution(wards, clusters)


if __name__ == "__main__":
    main()

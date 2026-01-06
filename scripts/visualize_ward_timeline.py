"""
タイムラインJSONからward関連イベントを抽出・可視化するスクリプト
"""

import json
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # GUI無効化
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd


def extract_ward_events(timeline_path: Path) -> tuple[list[dict], list[dict]]:
    """タイムラインJSONからWARD_PLACEDとWARD_KILLイベントを抽出"""
    with open(timeline_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    ward_placed = []
    ward_killed = []

    for frame in data['info']['frames']:
        for event in frame['events']:
            if event['type'] == 'WARD_PLACED':
                ward_placed.append({
                    'timestamp': event['timestamp'],
                    'timestamp_min': event['timestamp'] / 60000,  # ミリ秒→分
                    'creator_id': event.get('creatorId', 0),
                    'ward_type': event.get('wardType', 'UNKNOWN'),
                    'team': 'Blue' if event.get('creatorId', 0) <= 5 else 'Red'
                })
            elif event['type'] == 'WARD_KILL':
                ward_killed.append({
                    'timestamp': event['timestamp'],
                    'timestamp_min': event['timestamp'] / 60000,
                    'killer_id': event.get('killerId', 0),
                    'ward_type': event.get('wardType', 'UNKNOWN'),
                    'killer_team': 'Blue' if event.get('killerId', 0) <= 5 else 'Red'
                })

    return ward_placed, ward_killed


def print_ward_summary(ward_placed: list[dict], ward_killed: list[dict]):
    """wardイベントのサマリーを出力"""
    print("=" * 60)
    print("Ward Events Summary")
    print("=" * 60)

    # 設置数
    print(f"\n【WARD_PLACED】 Total: {len(ward_placed)}")
    df_placed = pd.DataFrame(ward_placed)
    if not df_placed.empty:
        print("\nBy Team:")
        print(df_placed.groupby('team').size().to_string())
        print("\nBy Ward Type:")
        print(df_placed.groupby('ward_type').size().to_string())
        print("\nBy Team & Ward Type:")
        print(df_placed.groupby(['team', 'ward_type']).size().to_string())

    # 破壊数
    print(f"\n【WARD_KILL】 Total: {len(ward_killed)}")
    df_killed = pd.DataFrame(ward_killed)
    if not df_killed.empty:
        print("\nBy Killer Team:")
        print(df_killed.groupby('killer_team').size().to_string())
        print("\nBy Ward Type:")
        print(df_killed.groupby('ward_type').size().to_string())


def visualize_ward_timeline(ward_placed: list[dict], ward_killed: list[dict],
                            output_path: Path = None, match_id: str = ""):
    """wardイベントを時間軸で可視化"""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # 色の定義
    team_colors = {'Blue': '#3498db', 'Red': '#e74c3c'}
    ward_markers = {
        'YELLOW_TRINKET': 'o',
        'CONTROL_WARD': 's',
        'BLUE_TRINKET': '^',
        'SIGHT_WARD': 'D',
        'UNDEFINED': 'x'
    }

    # 上段: WARD_PLACED
    ax1 = axes[0]
    ax1.set_title(f'Ward Placed Timeline - {match_id}', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Team')
    ax1.set_yticks([0, 1])
    ax1.set_yticklabels(['Blue (1-5)', 'Red (6-10)'])
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)

    for ward in ward_placed:
        y = 0 if ward['team'] == 'Blue' else 1
        marker = ward_markers.get(ward['ward_type'], 'o')
        ax1.scatter(ward['timestamp_min'], y,
                   c=team_colors[ward['team']],
                   marker=marker, s=80, alpha=0.7,
                   edgecolors='black', linewidths=0.5)

    # 下段: WARD_KILL
    ax2 = axes[1]
    ax2.set_title('Ward Kill Timeline', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Game Time (minutes)')
    ax2.set_ylabel('Killer Team')
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['Blue (1-5)', 'Red (6-10)'])
    ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)

    for ward in ward_killed:
        y = 0 if ward['killer_team'] == 'Blue' else 1
        marker = ward_markers.get(ward['ward_type'], 'o')
        ax2.scatter(ward['timestamp_min'], y,
                   c=team_colors[ward['killer_team']],
                   marker=marker, s=80, alpha=0.7,
                   edgecolors='black', linewidths=0.5)

    # 凡例作成
    legend_elements = [
        mpatches.Patch(color='#3498db', label='Blue Team'),
        mpatches.Patch(color='#e74c3c', label='Red Team'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
                   markersize=10, label='Yellow Trinket'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='gray',
                   markersize=10, label='Control Ward'),
        plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='gray',
                   markersize=10, label='Blue Trinket'),
    ]
    fig.legend(handles=legend_elements, loc='center right', bbox_to_anchor=(0.99, 0.5))

    plt.tight_layout(rect=[0, 0, 0.88, 1])

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved: {output_path}")

    plt.close()


def visualize_ward_counts_over_time(ward_placed: list[dict], ward_killed: list[dict],
                                    output_path: Path = None, match_id: str = ""):
    """時間経過に伴うward設置数の累積グラフ"""
    df_placed = pd.DataFrame(ward_placed)

    if df_placed.empty:
        print("No ward placed events found.")
        return

    # 1分ごとのビンで集計
    df_placed['time_bin'] = (df_placed['timestamp_min'] // 1).astype(int)

    fig, ax = plt.subplots(figsize=(12, 6))

    # チーム別累積カウント
    for team, color in [('Blue', '#3498db'), ('Red', '#e74c3c')]:
        team_data = df_placed[df_placed['team'] == team].copy()
        counts_per_min = team_data.groupby('time_bin').size()
        cumulative = counts_per_min.cumsum()

        # 全時間帯に対応
        max_time = int(df_placed['timestamp_min'].max()) + 1
        full_index = range(max_time)
        cumulative = cumulative.reindex(full_index, method='ffill').fillna(0)

        ax.plot(cumulative.index, cumulative.values,
                color=color, linewidth=2, label=f'{team} Team', marker='o', markersize=3)
        ax.fill_between(cumulative.index, cumulative.values, alpha=0.2, color=color)

    ax.set_title(f'Cumulative Ward Placement Over Time - {match_id}',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Game Time (minutes)')
    ax.set_ylabel('Cumulative Ward Count')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved: {output_path}")

    plt.close()


def visualize_ward_type_distribution(ward_placed: list[dict],
                                     output_path: Path = None, match_id: str = ""):
    """ward種類別の分布（チーム別）"""
    df = pd.DataFrame(ward_placed)

    if df.empty:
        print("No ward placed events found.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    team_colors = {
        'YELLOW_TRINKET': '#f1c40f',
        'CONTROL_WARD': '#e91e63',
        'BLUE_TRINKET': '#2196f3',
        'SIGHT_WARD': '#4caf50',
        'UNDEFINED': '#9e9e9e'
    }

    for idx, team in enumerate(['Blue', 'Red']):
        ax = axes[idx]
        team_data = df[df['team'] == team]
        ward_counts = team_data['ward_type'].value_counts()

        colors = [team_colors.get(wt, '#9e9e9e') for wt in ward_counts.index]
        ax.pie(ward_counts.values, labels=ward_counts.index, autopct='%1.1f%%',
               colors=colors, startangle=90)
        ax.set_title(f'{team} Team Ward Distribution\n(Total: {len(team_data)})')

    plt.suptitle(f'Ward Type Distribution by Team - {match_id}',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved: {output_path}")

    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize ward events from timeline JSON')
    parser.add_argument('timeline_path', type=str, help='Path to timeline JSON file')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for visualization images')
    args = parser.parse_args()

    timeline_path = Path(args.timeline_path)
    if not timeline_path.exists():
        print(f"Error: File not found: {timeline_path}")
        return

    match_id = timeline_path.stem

    # 出力ディレクトリ設定
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = timeline_path.parent / 'ward_visualizations'
    output_dir.mkdir(parents=True, exist_ok=True)

    # wardイベント抽出
    print(f"Loading: {timeline_path}")
    ward_placed, ward_killed = extract_ward_events(timeline_path)

    # サマリー出力
    print_ward_summary(ward_placed, ward_killed)

    # 可視化
    print("\n" + "=" * 60)
    print("Generating Visualizations...")
    print("=" * 60)

    visualize_ward_timeline(
        ward_placed, ward_killed,
        output_path=output_dir / f'{match_id}_ward_timeline.png',
        match_id=match_id
    )

    visualize_ward_counts_over_time(
        ward_placed, ward_killed,
        output_path=output_dir / f'{match_id}_ward_cumulative.png',
        match_id=match_id
    )

    visualize_ward_type_distribution(
        ward_placed,
        output_path=output_dir / f'{match_id}_ward_distribution.png',
        match_id=match_id
    )

    print(f"\nAll visualizations saved to: {output_dir}")


if __name__ == '__main__':
    main()

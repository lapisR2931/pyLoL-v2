"""
タイムラインJSONからward関連イベントをcreatorId別に抽出・可視化
"""

import json
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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
                creator_id = event.get('creatorId', 0)
                ward_placed.append({
                    'timestamp': event['timestamp'],
                    'timestamp_min': event['timestamp'] / 60000,
                    'creator_id': creator_id,
                    'ward_type': event.get('wardType', 'UNKNOWN'),
                    'team': 'Blue' if creator_id <= 5 else 'Red',
                    'role': get_role_name(creator_id)
                })
            elif event['type'] == 'WARD_KILL':
                killer_id = event.get('killerId', 0)
                ward_killed.append({
                    'timestamp': event['timestamp'],
                    'timestamp_min': event['timestamp'] / 60000,
                    'killer_id': killer_id,
                    'ward_type': event.get('wardType', 'UNKNOWN'),
                    'killer_team': 'Blue' if killer_id <= 5 else 'Red',
                    'role': get_role_name(killer_id)
                })

    return ward_placed, ward_killed


def get_role_name(participant_id: int) -> str:
    """participantIdからロール名を推定（一般的な配置）"""
    roles = {
        1: 'Top', 2: 'Jungle', 3: 'Mid', 4: 'ADC', 5: 'Support',
        6: 'Top', 7: 'Jungle', 8: 'Mid', 9: 'ADC', 10: 'Support',
        0: 'Unknown'
    }
    return roles.get(participant_id, 'Unknown')


def print_player_summary(ward_placed: list[dict], ward_killed: list[dict]):
    """プレイヤー別wardイベントのサマリーを出力"""
    df_placed = pd.DataFrame(ward_placed)
    df_killed = pd.DataFrame(ward_killed)

    print("=" * 70)
    print("Ward Events by Player (creatorId)")
    print("=" * 70)

    # 設置数
    print("\n【WARD_PLACED by creatorId】")
    if not df_placed.empty:
        summary = df_placed.groupby(['creator_id', 'team', 'role', 'ward_type']).size().unstack(fill_value=0)
        summary['Total'] = summary.sum(axis=1)
        print(summary.to_string())

        print("\n--- Player Totals ---")
        totals = df_placed.groupby(['creator_id', 'team', 'role']).size().reset_index(name='count')
        totals = totals.sort_values('creator_id')
        for _, row in totals.iterrows():
            print(f"  Player {row['creator_id']:2d} ({row['team']:4s} {row['role']:7s}): {row['count']:3d} wards")

    # 破壊数
    print("\n【WARD_KILL by killerId】")
    if not df_killed.empty:
        summary = df_killed.groupby(['killer_id', 'killer_team', 'role', 'ward_type']).size().unstack(fill_value=0)
        summary['Total'] = summary.sum(axis=1)
        print(summary.to_string())

        print("\n--- Player Totals ---")
        totals = df_killed.groupby(['killer_id', 'killer_team', 'role']).size().reset_index(name='count')
        totals = totals.sort_values('killer_id')
        for _, row in totals.iterrows():
            print(f"  Player {row['killer_id']:2d} ({row['killer_team']:4s} {row['role']:7s}): {row['count']:3d} kills")


def visualize_ward_by_player(ward_placed: list[dict], ward_killed: list[dict],
                              output_path: Path = None, match_id: str = ""):
    """プレイヤー別ward設置数の棒グラフ"""
    df_placed = pd.DataFrame(ward_placed)
    df_killed = pd.DataFrame(ward_killed)

    if df_placed.empty:
        print("No ward placed events found.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 左: Ward設置数
    ax1 = axes[0]
    player_counts = df_placed.groupby(['creator_id', 'team']).size().reset_index(name='count')
    player_counts = player_counts.sort_values('creator_id')

    colors = ['#3498db' if t == 'Blue' else '#e74c3c' for t in player_counts['team']]
    bars = ax1.bar(player_counts['creator_id'].astype(str), player_counts['count'], color=colors)

    ax1.set_xlabel('Player ID (creatorId)')
    ax1.set_ylabel('Ward Placed Count')
    ax1.set_title(f'Ward Placed by Player - {match_id}')

    # 値をバー上に表示
    for bar, count in zip(bars, player_counts['count']):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(count), ha='center', va='bottom', fontsize=10)

    # x軸ラベルにロール追加
    labels = [f"{row['creator_id']}\n({get_role_name(row['creator_id'])})"
              for _, row in player_counts.iterrows()]
    ax1.set_xticks(range(len(labels)))
    ax1.set_xticklabels(labels)

    # 右: Ward破壊数
    ax2 = axes[1]
    if not df_killed.empty:
        killer_counts = df_killed.groupby(['killer_id', 'killer_team']).size().reset_index(name='count')
        killer_counts = killer_counts.sort_values('killer_id')

        colors = ['#3498db' if t == 'Blue' else '#e74c3c' for t in killer_counts['killer_team']]
        bars = ax2.bar(killer_counts['killer_id'].astype(str), killer_counts['count'], color=colors)

        ax2.set_xlabel('Player ID (killerId)')
        ax2.set_ylabel('Ward Kill Count')
        ax2.set_title(f'Ward Kills by Player - {match_id}')

        for bar, count in zip(bars, killer_counts['count']):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    str(count), ha='center', va='bottom', fontsize=10)

        labels = [f"{row['killer_id']}\n({get_role_name(row['killer_id'])})"
                  for _, row in killer_counts.iterrows()]
        ax2.set_xticks(range(len(labels)))
        ax2.set_xticklabels(labels)

    # 凡例
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#3498db', label='Blue Team (1-5)'),
                       Patch(facecolor='#e74c3c', label='Red Team (6-10)')]
    fig.legend(handles=legend_elements, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.02))

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved: {output_path}")

    plt.close()


def visualize_ward_type_by_player(ward_placed: list[dict],
                                   output_path: Path = None, match_id: str = ""):
    """プレイヤー別ward種類の積み上げ棒グラフ"""
    df = pd.DataFrame(ward_placed)

    if df.empty:
        print("No ward placed events found.")
        return

    # ward種類別にピボット
    pivot = df.pivot_table(index='creator_id', columns='ward_type',
                           aggfunc='size', fill_value=0)
    pivot = pivot.reindex(range(1, 11), fill_value=0)

    # 色の定義
    ward_colors = {
        'YELLOW_TRINKET': '#f1c40f',
        'CONTROL_WARD': '#e91e63',
        'BLUE_TRINKET': '#2196f3',
        'SIGHT_WARD': '#4caf50',
        'UNDEFINED': '#9e9e9e',
        'UNKNOWN': '#757575'
    }

    fig, ax = plt.subplots(figsize=(12, 6))

    bottom = [0] * 10
    for ward_type in pivot.columns:
        color = ward_colors.get(ward_type, '#757575')
        ax.bar(pivot.index, pivot[ward_type], bottom=bottom,
               label=ward_type, color=color)
        bottom = [b + v for b, v in zip(bottom, pivot[ward_type])]

    ax.set_xlabel('Player ID')
    ax.set_ylabel('Ward Count')
    ax.set_title(f'Ward Types by Player - {match_id}')
    ax.set_xticks(range(1, 11))

    # チーム境界線
    ax.axvline(x=5.5, color='black', linestyle='--', linewidth=2, alpha=0.5)
    ax.text(3, ax.get_ylim()[1] * 0.95, 'Blue Team', ha='center', fontsize=12, fontweight='bold', color='#3498db')
    ax.text(8, ax.get_ylim()[1] * 0.95, 'Red Team', ha='center', fontsize=12, fontweight='bold', color='#e74c3c')

    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved: {output_path}")

    plt.close()


def visualize_ward_timeline_by_player(ward_placed: list[dict],
                                       output_path: Path = None, match_id: str = ""):
    """プレイヤー別ward設置タイムライン（横軸:時間、縦軸:プレイヤー）"""
    df = pd.DataFrame(ward_placed)

    if df.empty:
        print("No ward placed events found.")
        return

    fig, ax = plt.subplots(figsize=(14, 8))

    # ward種類別のマーカー
    ward_markers = {
        'YELLOW_TRINKET': ('o', '#f1c40f'),
        'CONTROL_WARD': ('s', '#e91e63'),
        'BLUE_TRINKET': ('^', '#2196f3'),
        'SIGHT_WARD': ('D', '#4caf50'),
        'UNDEFINED': ('x', '#9e9e9e'),
    }

    for ward_type, (marker, color) in ward_markers.items():
        subset = df[df['ward_type'] == ward_type]
        if not subset.empty:
            ax.scatter(subset['timestamp_min'], subset['creator_id'],
                      marker=marker, c=color, s=80, alpha=0.7,
                      label=ward_type, edgecolors='black', linewidths=0.5)

    ax.set_xlabel('Game Time (minutes)')
    ax.set_ylabel('Player ID')
    ax.set_title(f'Ward Placement Timeline by Player - {match_id}')
    ax.set_yticks(range(1, 11))

    # ロール名をy軸ラベルに追加
    ylabels = [f"{i} ({get_role_name(i)})" for i in range(1, 11)]
    ax.set_yticklabels(ylabels)

    # チーム境界線
    ax.axhline(y=5.5, color='black', linestyle='--', linewidth=2, alpha=0.5)

    # 背景色でチーム分け
    ax.axhspan(0.5, 5.5, alpha=0.1, color='#3498db')
    ax.axhspan(5.5, 10.5, alpha=0.1, color='#e74c3c')

    ax.legend(loc='upper right')
    ax.grid(axis='x', alpha=0.3)
    ax.set_ylim(0.5, 10.5)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved: {output_path}")

    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize ward events by player')
    parser.add_argument('timeline_path', type=str, help='Path to timeline JSON file')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for visualization images')
    args = parser.parse_args()

    timeline_path = Path(args.timeline_path)
    if not timeline_path.exists():
        print(f"Error: File not found: {timeline_path}")
        return

    match_id = timeline_path.stem

    # 出力ディレクトリ
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = timeline_path.parent / 'ward_visualizations'
    output_dir.mkdir(parents=True, exist_ok=True)

    # wardイベント抽出
    print(f"Loading: {timeline_path}")
    ward_placed, ward_killed = extract_ward_events(timeline_path)

    # サマリー出力
    print_player_summary(ward_placed, ward_killed)

    # 可視化
    print("\n" + "=" * 70)
    print("Generating Visualizations...")
    print("=" * 70)

    visualize_ward_by_player(
        ward_placed, ward_killed,
        output_path=output_dir / f'{match_id}_ward_by_player.png',
        match_id=match_id
    )

    visualize_ward_type_by_player(
        ward_placed,
        output_path=output_dir / f'{match_id}_ward_type_by_player.png',
        match_id=match_id
    )

    visualize_ward_timeline_by_player(
        ward_placed,
        output_path=output_dir / f'{match_id}_ward_timeline_by_player.png',
        match_id=match_id
    )

    print(f"\nAll visualizations saved to: {output_dir}")


if __name__ == '__main__':
    main()

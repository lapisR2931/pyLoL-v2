"""
グリッド特徴量生成 - Phase 5 Task A

wards_matched.csvからward座標を読み込み、時間帯別のグリッド特徴量を生成する。

出力形式:
    ward_grid.npz
    - blue: (3, 32, 32) - 各セルにBlueチームのwardが存在した秒数
    - red: (3, 32, 32) - 各セルにRedチームのwardが存在した秒数
    - match_id: 試合ID
"""

import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# =============================================================================
# 設定
# =============================================================================

GRID_SIZE = 32           # グリッドサイズ（32x32）
MINIMAP_SIZE = 512       # ミニマップサイズ（ピクセル）
CELL_SIZE = MINIMAP_SIZE // GRID_SIZE  # 1セルあたりのピクセル数（16）

# 時間帯定義（ミリ秒）
TIME_PHASES: List[Tuple[int, Optional[int]]] = [
    (0, 10 * 60 * 1000),              # Phase 0: 0-10分
    (10 * 60 * 1000, 20 * 60 * 1000),  # Phase 1: 10-20分
    (20 * 60 * 1000, None),           # Phase 2: 20分以降
]
NUM_PHASES = len(TIME_PHASES)

# ward種別ごとのデフォルト持続時間（ミリ秒）
DEFAULT_DURATION_MS: Dict[str, int] = {
    "YELLOW_TRINKET": 90 * 1000,   # 90秒
    "SIGHT_WARD": 90 * 1000,       # 90秒
    "CONTROL_WARD": 180 * 1000,    # 180秒（打ち切り）
    "BLUE_TRINKET": 60 * 1000,     # 60秒
}


# =============================================================================
# ward持続時間計算
# =============================================================================

def calculate_duration_ms(
    ward_type: str,
    timestamp_placed: int,
    timestamp_killed: Optional[int]
) -> int:
    """
    ward持続時間をミリ秒で計算

    Args:
        ward_type: wardの種別
        timestamp_placed: 設置時刻（ミリ秒）
        timestamp_killed: 破壊時刻（ミリ秒、Noneの場合はデフォルト値を使用）

    Returns:
        持続時間（ミリ秒）
    """
    if timestamp_killed is not None:
        return timestamp_killed - timestamp_placed

    # デフォルト持続時間を使用
    return DEFAULT_DURATION_MS.get(ward_type, 90 * 1000)


def distribute_to_phases(
    timestamp_placed: int,
    duration_ms: int
) -> List[Tuple[int, float]]:
    """
    wardの存在時間を時間帯ごとに分配

    Args:
        timestamp_placed: 設置時刻（ミリ秒）
        duration_ms: 持続時間（ミリ秒）

    Returns:
        [(phase_index, duration_seconds), ...] のリスト
    """
    results: List[Tuple[int, float]] = []
    ward_start = timestamp_placed
    ward_end = timestamp_placed + duration_ms

    for phase_idx, (phase_start, phase_end) in enumerate(TIME_PHASES):
        # 時間帯の終了時刻がNoneの場合は無限大として扱う
        if phase_end is None:
            phase_end = float('inf')

        # wardと時間帯の重なりを計算
        overlap_start = max(ward_start, phase_start)
        overlap_end = min(ward_end, phase_end)

        if overlap_start < overlap_end:
            duration_in_phase_ms = overlap_end - overlap_start
            duration_in_phase_sec = duration_in_phase_ms / 1000.0
            results.append((phase_idx, duration_in_phase_sec))

    return results


# =============================================================================
# グリッド生成
# =============================================================================

def pixel_to_grid(x_pixel: int, y_pixel: int) -> Tuple[int, int]:
    """
    ピクセル座標をグリッドセルに変換

    Args:
        x_pixel: X座標（0-512）
        y_pixel: Y座標（0-512）

    Returns:
        (grid_x, grid_y) - グリッドセル座標（0-31）
    """
    grid_x = min(x_pixel // CELL_SIZE, GRID_SIZE - 1)
    grid_y = min(y_pixel // CELL_SIZE, GRID_SIZE - 1)
    return grid_x, grid_y


def generate_ward_grid(wards_csv_path: Path) -> dict:
    """
    wards_matched.csvからグリッド特徴量を生成

    Args:
        wards_csv_path: wards_matched.csvのパス

    Returns:
        {
            "blue": np.ndarray (3, 32, 32) - 各セルの存在秒数,
            "red": np.ndarray (3, 32, 32),
            "match_id": str
        }
    """
    # グリッドを初期化
    blue_grid = np.zeros((NUM_PHASES, GRID_SIZE, GRID_SIZE), dtype=np.float32)
    red_grid = np.zeros((NUM_PHASES, GRID_SIZE, GRID_SIZE), dtype=np.float32)

    # match_idを抽出（パスから取得）
    match_id = wards_csv_path.parent.name

    # CSVを読み込み
    with open(wards_csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        for row in reader:
            # timeline_onlyは座標がないため除外
            match_status = row.get('match_status', '')
            if match_status == 'timeline_only':
                continue

            # 座標を取得
            x_pixel_str = row.get('x_pixel', '')
            y_pixel_str = row.get('y_pixel', '')
            if not x_pixel_str or not y_pixel_str:
                continue

            x_pixel = int(x_pixel_str)
            y_pixel = int(y_pixel_str)

            # 座標が有効範囲内かチェック
            if x_pixel < 0 or x_pixel >= MINIMAP_SIZE:
                continue
            if y_pixel < 0 or y_pixel >= MINIMAP_SIZE:
                continue

            # チームを取得
            team = row.get('team', '')
            if team not in ('blue', 'red'):
                continue

            # タイムスタンプを取得
            timestamp_placed_str = row.get('timestamp_placed', '')
            if not timestamp_placed_str:
                continue
            timestamp_placed = int(timestamp_placed_str)

            # 破壊時刻を取得（存在しない場合はNone）
            timestamp_killed_str = row.get('timestamp_killed', '')
            timestamp_killed = int(timestamp_killed_str) if timestamp_killed_str else None

            # ward種別を取得
            ward_type = row.get('ward_type', 'SIGHT_WARD')

            # 持続時間を計算
            duration_ms = calculate_duration_ms(ward_type, timestamp_placed, timestamp_killed)

            # グリッド座標に変換
            grid_x, grid_y = pixel_to_grid(x_pixel, y_pixel)

            # 時間帯ごとに分配してグリッドに累積
            phase_durations = distribute_to_phases(timestamp_placed, duration_ms)

            for phase_idx, duration_sec in phase_durations:
                if team == 'blue':
                    blue_grid[phase_idx, grid_y, grid_x] += duration_sec
                else:
                    red_grid[phase_idx, grid_y, grid_x] += duration_sec

    return {
        "blue": blue_grid,
        "red": red_grid,
        "match_id": match_id
    }


def save_ward_grid(grid_data: dict, output_path: Path) -> None:
    """
    グリッドデータをnpz形式で保存

    Args:
        grid_data: generate_ward_gridの出力
        output_path: 保存先パス
    """
    np.savez(
        output_path,
        blue=grid_data["blue"],
        red=grid_data["red"],
        match_id=grid_data["match_id"]
    )
    print(f"保存完了: {output_path}")


# =============================================================================
# 統計表示
# =============================================================================

def print_grid_statistics(grid_data: dict) -> None:
    """グリッドデータの統計を表示"""
    blue = grid_data["blue"]
    red = grid_data["red"]

    print(f"\n=== グリッド統計: {grid_data['match_id']} ===")
    print(f"グリッドサイズ: {GRID_SIZE}x{GRID_SIZE} ({CELL_SIZE}px単位)")

    for phase_idx, (phase_start, phase_end) in enumerate(TIME_PHASES):
        phase_start_min = phase_start // 60000
        phase_end_min = phase_end // 60000 if phase_end else "END"
        print(f"\nPhase {phase_idx} ({phase_start_min}-{phase_end_min}分):")

        blue_phase = blue[phase_idx]
        red_phase = red[phase_idx]

        blue_total = blue_phase.sum()
        red_total = red_phase.sum()
        blue_cells = np.count_nonzero(blue_phase)
        red_cells = np.count_nonzero(red_phase)

        print(f"  Blue: 累積秒数={blue_total:.1f}秒, 使用セル数={blue_cells}")
        print(f"  Red:  累積秒数={red_total:.1f}秒, 使用セル数={red_cells}")


# =============================================================================
# メイン処理
# =============================================================================

def process_match(match_dir: Path) -> Optional[dict]:
    """
    1試合を処理してward_grid.npzを生成

    Args:
        match_dir: 試合ディレクトリ（例: C:/dataset_20260105/JP1-555621265）

    Returns:
        生成したグリッドデータ、または処理失敗時はNone
    """
    wards_csv_path = match_dir / "wards_matched.csv"
    output_path = match_dir / "ward_grid.npz"

    if not wards_csv_path.exists():
        print(f"スキップ: {match_dir.name} (wards_matched.csvが存在しません)")
        return None

    print(f"\n処理中: {match_dir.name}")

    # グリッド生成
    grid_data = generate_ward_grid(wards_csv_path)

    # 統計表示
    print_grid_statistics(grid_data)

    # 保存
    save_ward_grid(grid_data, output_path)

    return grid_data


def process_all(dataset_dir: Path) -> Dict[str, dict]:
    """
    全試合を処理

    Args:
        dataset_dir: データセットディレクトリ（例: C:/dataset_20260105）

    Returns:
        {match_id: grid_data} の辞書
    """
    results = {}

    # 全試合ディレクトリを検索
    match_dirs = sorted(dataset_dir.glob("JP1-*"))
    print(f"全{len(match_dirs)}試合を処理します")

    for match_dir in match_dirs:
        try:
            grid_data = process_match(match_dir)
            if grid_data:
                results[grid_data["match_id"]] = grid_data
        except Exception as e:
            print(f"エラー [{match_dir.name}]: {e}")

    print(f"\n処理完了: {len(results)}試合")
    return results


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="グリッド特徴量生成（Phase 5 Task A）"
    )
    parser.add_argument(
        "--match", type=str,
        help="処理する試合ID（例: JP1-555621265）"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="全試合を処理"
    )
    parser.add_argument(
        "--dataset", type=str, default=r"C:\dataset_20260105",
        help="データセットディレクトリ"
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset)

    if args.match:
        # 1試合処理
        match_id = args.match
        if not match_id.startswith("JP1-"):
            match_id = f"JP1-{match_id}"
        match_dir = dataset_dir / match_id
        process_match(match_dir)

    elif args.all:
        # 全試合処理
        process_all(dataset_dir)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()

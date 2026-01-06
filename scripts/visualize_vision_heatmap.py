"""
視界スコアヒートマップ可視化スクリプト - Phase 5 Task D

使用方法:
    # 基本実行
    python scripts/visualize_vision_heatmap.py

    # 重要度ファイル指定
    python scripts/visualize_vision_heatmap.py --importance models/vision_importance.npy

    # 出力先指定
    python scripts/visualize_vision_heatmap.py --output heatmaps/

    # 背景画像指定
    python scripts/visualize_vision_heatmap.py --bg path/to/minimap.png

    # グリッド表示（全時間帯を1枚に）
    python scripts/visualize_vision_heatmap.py --grid
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from autoLeague.scoring.visualizer import (
    visualize_importance_heatmap,
    visualize_importance_grid,
    visualize_top_cells,
    print_importance_summary,
)


# =============================================================================
# 設定
# =============================================================================

DEFAULT_IMPORTANCE = PROJECT_ROOT / "models" / "vision_importance.npy"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "heatmaps"


# =============================================================================
# メイン処理
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="視界スコアヒートマップ可視化",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--importance",
        type=str,
        default=str(DEFAULT_IMPORTANCE),
        help="重要度ファイルパス (default: models/vision_importance.npy)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="出力ディレクトリ (default: heatmaps/)"
    )
    parser.add_argument(
        "--bg",
        type=str,
        default=None,
        help="ミニマップ背景画像パス"
    )
    parser.add_argument(
        "--grid",
        action="store_true",
        help="全時間帯を1枚の画像に並べて表示"
    )
    parser.add_argument(
        "--top-cells",
        type=int,
        default=10,
        help="上位セルハイライト表示数 (default: 10)"
    )
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="統計情報の表示を抑制"
    )
    args = parser.parse_args()

    # パス設定
    importance_path = Path(args.importance)
    output_dir = Path(args.output)

    print("=" * 60)
    print("視界スコアヒートマップ可視化")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # 1. 重要度データ読み込み
    # -------------------------------------------------------------------------
    print(f"\n[1] 重要度データ読み込み: {importance_path}")

    if not importance_path.exists():
        print(f"エラー: 重要度ファイルが見つかりません: {importance_path}")
        print("Task Cを先に実行してvision_importance.npyを生成してください。")
        sys.exit(1)

    importance = np.load(importance_path)
    print(f"    形状: {importance.shape}")
    print(f"    値域: [{importance.min():.4f}, {importance.max():.4f}]")

    # -------------------------------------------------------------------------
    # 2. 統計情報表示
    # -------------------------------------------------------------------------
    if not args.no_summary:
        print_importance_summary(importance)

    # -------------------------------------------------------------------------
    # 3. ヒートマップ生成
    # -------------------------------------------------------------------------
    print("\n[2] ヒートマップ生成")

    output_files = []

    # 時間帯別 + 統合ヒートマップ
    files = visualize_importance_heatmap(
        importance=importance,
        output_dir=output_dir,
        minimap_bg=args.bg,
    )
    output_files.extend(files)
    for f in files:
        print(f"    生成: {f}")

    # グリッド表示
    if args.grid:
        grid_path = visualize_importance_grid(
            importance=importance,
            output_path=output_dir / "importance_grid.png",
            minimap_bg=args.bg,
        )
        output_files.append(grid_path)
        print(f"    生成: {grid_path}")

    # 上位セルハイライト
    top_cells_path = visualize_top_cells(
        importance=importance,
        output_path=output_dir / "importance_top_cells.png",
        top_n=args.top_cells,
    )
    output_files.append(top_cells_path)
    print(f"    生成: {top_cells_path}")

    # -------------------------------------------------------------------------
    # 完了
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"可視化完了: {len(output_files)}ファイル生成")
    print(f"出力先: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()

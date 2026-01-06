"""
視界スコアヒートマップ可視化 - Phase 5 Task D

特徴量重要度をヒートマップとして可視化し、
「どの座標・時間帯のward配置が勝敗に影響するか」を分析する。

出力:
- 時間帯別ヒートマップ (3枚)
- 全時間帯統合ヒートマップ (1枚)
"""

from pathlib import Path
from typing import Optional, Union, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle


# =============================================================================
# 設定
# =============================================================================

# グリッド設定
GRID_SIZE = 32
N_TIME_PHASES = 3

# 時間帯ラベル
TIME_PHASE_LABELS = [
    "Phase 1: 0-10min (Early)",
    "Phase 2: 10-20min (Mid)",
    "Phase 3: 20min+ (Late)",
]

# ミニマップサイズ（元画像）
MINIMAP_SIZE = 512

# カラーマップ
# 正の値（Blue有利）: 青系
# 負の値（Red有利）: 赤系
CMAP_DIVERGING = "RdBu_r"  # Red-Blue diverging (reversed: blue=positive)

# フォント設定
plt.rcParams['font.family'] = 'DejaVu Sans'


# =============================================================================
# メイン可視化関数
# =============================================================================

def visualize_importance_heatmap(
    importance: np.ndarray,
    output_dir: Union[str, Path],
    minimap_bg: Optional[Union[str, Path]] = None,
    show_colorbar: bool = True,
    figsize: Tuple[int, int] = (8, 8),
    dpi: int = 150,
) -> List[Path]:
    """
    特徴量重要度をヒートマップとして可視化

    Args:
        importance: 重要度配列 (3, 32, 32)
            - 正の値: Blue有利に寄与
            - 負の値: Red有利に寄与
        output_dir: 出力ディレクトリ
        minimap_bg: ミニマップ背景画像パス（オプション）
        show_colorbar: カラーバー表示
        figsize: 図サイズ
        dpi: 解像度

    Returns:
        生成されたファイルパスのリスト
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 背景画像の読み込み
    bg_image = None
    if minimap_bg is not None:
        minimap_bg = Path(minimap_bg)
        if minimap_bg.exists():
            bg_image = plt.imread(minimap_bg)

    # 値の範囲を対称にする（ゼロを中心に）
    vmax = max(abs(importance.min()), abs(importance.max()))
    vmin = -vmax

    output_files = []

    # 時間帯別ヒートマップ
    for phase_idx in range(N_TIME_PHASES):
        phase_importance = importance[phase_idx]  # (32, 32)

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        # 背景画像
        if bg_image is not None:
            ax.imshow(bg_image, extent=[0, GRID_SIZE, GRID_SIZE, 0], alpha=0.3)

        # ヒートマップ
        im = ax.imshow(
            phase_importance,
            cmap=CMAP_DIVERGING,
            vmin=vmin,
            vmax=vmax,
            extent=[0, GRID_SIZE, GRID_SIZE, 0],
            alpha=0.8 if bg_image is not None else 1.0,
        )

        # グリッド線
        ax.set_xticks(np.arange(0, GRID_SIZE + 1, 8))
        ax.set_yticks(np.arange(0, GRID_SIZE + 1, 8))
        ax.grid(True, alpha=0.3, linestyle='--')

        # タイトル
        ax.set_title(TIME_PHASE_LABELS[phase_idx], fontsize=14, fontweight='bold')
        ax.set_xlabel("X (grid)")
        ax.set_ylabel("Y (grid)")

        # カラーバー
        if show_colorbar:
            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label("Importance\n(+: Blue advantage, -: Red advantage)")

        plt.tight_layout()

        # 保存
        output_path = output_dir / f"importance_phase{phase_idx}.png"
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        output_files.append(output_path)

    # 統合ヒートマップ（全時間帯の平均）
    combined_importance = importance.mean(axis=0)  # (32, 32)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    if bg_image is not None:
        ax.imshow(bg_image, extent=[0, GRID_SIZE, GRID_SIZE, 0], alpha=0.3)

    im = ax.imshow(
        combined_importance,
        cmap=CMAP_DIVERGING,
        vmin=vmin,
        vmax=vmax,
        extent=[0, GRID_SIZE, GRID_SIZE, 0],
        alpha=0.8 if bg_image is not None else 1.0,
    )

    ax.set_xticks(np.arange(0, GRID_SIZE + 1, 8))
    ax.set_yticks(np.arange(0, GRID_SIZE + 1, 8))
    ax.grid(True, alpha=0.3, linestyle='--')

    ax.set_title("Combined Importance (All Phases)", fontsize=14, fontweight='bold')
    ax.set_xlabel("X (grid)")
    ax.set_ylabel("Y (grid)")

    if show_colorbar:
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label("Importance\n(+: Blue advantage, -: Red advantage)")

    plt.tight_layout()

    output_path = output_dir / "importance_combined.png"
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    output_files.append(output_path)

    return output_files


def visualize_importance_grid(
    importance: np.ndarray,
    output_path: Union[str, Path],
    minimap_bg: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (16, 5),
    dpi: int = 150,
) -> Path:
    """
    全時間帯を1枚の画像に並べて表示

    Args:
        importance: 重要度配列 (3, 32, 32)
        output_path: 出力ファイルパス
        minimap_bg: ミニマップ背景画像パス
        figsize: 図サイズ
        dpi: 解像度

    Returns:
        出力ファイルパス
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    bg_image = None
    if minimap_bg is not None:
        minimap_bg = Path(minimap_bg)
        if minimap_bg.exists():
            bg_image = plt.imread(minimap_bg)

    vmax = max(abs(importance.min()), abs(importance.max()))
    vmin = -vmax

    fig, axes = plt.subplots(1, 3, figsize=figsize, dpi=dpi)

    for phase_idx, ax in enumerate(axes):
        phase_importance = importance[phase_idx]

        if bg_image is not None:
            ax.imshow(bg_image, extent=[0, GRID_SIZE, GRID_SIZE, 0], alpha=0.3)

        im = ax.imshow(
            phase_importance,
            cmap=CMAP_DIVERGING,
            vmin=vmin,
            vmax=vmax,
            extent=[0, GRID_SIZE, GRID_SIZE, 0],
            alpha=0.8 if bg_image is not None else 1.0,
        )

        ax.set_xticks(np.arange(0, GRID_SIZE + 1, 8))
        ax.set_yticks(np.arange(0, GRID_SIZE + 1, 8))
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_title(TIME_PHASE_LABELS[phase_idx], fontsize=11)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

    # 共通カラーバー
    fig.colorbar(im, ax=axes, shrink=0.6, label="Importance (+: Blue, -: Red)")

    plt.suptitle("Ward Vision Importance by Time Phase", fontsize=14, fontweight='bold')
    plt.tight_layout()

    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

    return output_path


def visualize_top_cells(
    importance: np.ndarray,
    output_path: Union[str, Path],
    top_n: int = 10,
    figsize: Tuple[int, int] = (10, 10),
    dpi: int = 150,
) -> Path:
    """
    重要度が高いセルをハイライト表示

    Args:
        importance: 重要度配列 (3, 32, 32)
        output_path: 出力ファイルパス
        top_n: 表示する上位セル数
        figsize: 図サイズ
        dpi: 解像度

    Returns:
        出力ファイルパス
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    combined = importance.mean(axis=0)  # (32, 32)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    vmax = max(abs(combined.min()), abs(combined.max()))
    im = ax.imshow(combined, cmap=CMAP_DIVERGING, vmin=-vmax, vmax=vmax)

    # 上位セルを特定
    flat_importance = combined.flatten()
    top_positive_idx = np.argsort(flat_importance)[-top_n:]  # Blue有利
    top_negative_idx = np.argsort(flat_importance)[:top_n]   # Red有利

    # ハイライト表示
    for idx in top_positive_idx:
        y, x = divmod(idx, GRID_SIZE)
        rect = Rectangle((x - 0.5, y - 0.5), 1, 1,
                         fill=False, edgecolor='blue', linewidth=2)
        ax.add_patch(rect)

    for idx in top_negative_idx:
        y, x = divmod(idx, GRID_SIZE)
        rect = Rectangle((x - 0.5, y - 0.5), 1, 1,
                         fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)

    ax.set_title(f"Top {top_n} Important Cells\n(Blue box: Blue advantage, Red box: Red advantage)",
                 fontsize=12)
    ax.set_xlabel("X (grid)")
    ax.set_ylabel("Y (grid)")

    plt.colorbar(im, ax=ax, shrink=0.8, label="Importance")
    plt.tight_layout()

    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

    return output_path


def print_importance_summary(importance: np.ndarray) -> None:
    """
    重要度の統計情報を表示

    Args:
        importance: 重要度配列 (3, 32, 32)
    """
    print("\n=== Feature Importance Summary ===")
    print(f"Shape: {importance.shape}")
    print(f"Overall range: [{importance.min():.4f}, {importance.max():.4f}]")
    print()

    for phase_idx in range(N_TIME_PHASES):
        phase_imp = importance[phase_idx]
        print(f"{TIME_PHASE_LABELS[phase_idx]}:")
        print(f"  Range: [{phase_imp.min():.4f}, {phase_imp.max():.4f}]")
        print(f"  Mean:  {phase_imp.mean():.4f}")
        print(f"  Std:   {phase_imp.std():.4f}")

        # 最も重要なセル
        max_idx = np.unravel_index(np.argmax(phase_imp), phase_imp.shape)
        min_idx = np.unravel_index(np.argmin(phase_imp), phase_imp.shape)
        print(f"  Top Blue advantage: cell {max_idx} = {phase_imp[max_idx]:.4f}")
        print(f"  Top Red advantage:  cell {min_idx} = {phase_imp[min_idx]:.4f}")
        print()

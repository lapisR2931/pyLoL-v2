"""
Phase 6: 設定・定数

勝敗予測モデルで使用する設定値を定義。
"""

from typing import List

# =============================================================================
# 評価時点（ミリ秒）
# =============================================================================

PREDICTION_TIMES_MS: List[int] = [
    10 * 60 * 1000,  # 10分
    20 * 60 * 1000,  # 20分
]

PREDICTION_TIMES_NAMES: List[str] = ["10min", "20min"]

# =============================================================================
# 特徴量定義
# =============================================================================

# ベースライン特徴量（20個）
BASELINE_FEATURES: List[str] = [
    # Gold系 (3)
    "blue_total_gold",
    "red_total_gold",
    "gold_diff",
    # XP/Level系 (4)
    "blue_total_xp",
    "red_total_xp",
    "blue_avg_level",
    "red_avg_level",
    # CS系 (2)
    "blue_total_cs",
    "red_total_cs",
    # Kill系 (3)
    "blue_kills",
    "red_kills",
    "kill_diff",
    # Objective系 (6)
    "blue_dragons",
    "red_dragons",
    "blue_heralds",
    "red_heralds",
    "blue_towers",
    "red_towers",
    # Ward基本 (2)
    "blue_wards_placed",
    "red_wards_placed",
]

# Riot visionScore関連特徴量（3個）
RIOT_VISION_FEATURES: List[str] = [
    "blue_vision_score_est",
    "red_vision_score_est",
    "vision_score_diff",
]

# =============================================================================
# Timeline構造
# =============================================================================

# 参加者ID -> チーム
# participantId 1-5: Blue (teamId=100)
# participantId 6-10: Red (teamId=200)
BLUE_PARTICIPANT_IDS = [1, 2, 3, 4, 5]
RED_PARTICIPANT_IDS = [6, 7, 8, 9, 10]

# チームID
BLUE_TEAM_ID = 100
RED_TEAM_ID = 200

# =============================================================================
# モデル設定
# =============================================================================

# ロジスティック回帰
LOGISTIC_C = 0.1  # 正則化強度
LOGISTIC_MAX_ITER = 1000

# =============================================================================
# グリッド設定（Phase 5との整合性）
# =============================================================================

GRID_SIZE = 32
MINIMAP_SIZE = 512
CELL_SIZE = MINIMAP_SIZE // GRID_SIZE  # 16

# ward種別ごとのデフォルト持続時間（ミリ秒）
DEFAULT_WARD_DURATION_MS = {
    "YELLOW_TRINKET": 90 * 1000,
    "SIGHT_WARD": 90 * 1000,
    "CONTROL_WARD": 180 * 1000,
    "BLUE_TRINKET": 60 * 1000,
}

# =============================================================================
# パス設定
# =============================================================================

# デフォルトのデータセットディレクトリ
DEFAULT_DATASET_DIR = r"C:\dataset_20260105"

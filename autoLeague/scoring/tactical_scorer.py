"""
Ward戦術スコア計算 - Phase 5 拡張

ヒートマップの重要度マップを使用して、各wardに戦術的スコアを付与する。

スコアリングルール:
- 設置スコア: 自チームに有利な位置に置けば高スコア
  - Blue: 正の重要度セル → 高スコア
  - Red: 負の重要度セル → 高スコア
- 破壊スコア: 敵wardを破壊した側に付与
  - Blue: 負の重要度セルの敵ward破壊 → 高スコア
  - Red: 正の重要度セルの敵ward破壊 → 高スコア

出力:
- ward_tactical_scores.csv（各試合フォルダ）
"""

import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# =============================================================================
# 設定
# =============================================================================

GRID_SIZE = 32
MINIMAP_SIZE = 512
CELL_SIZE = MINIMAP_SIZE // GRID_SIZE  # 16

# 時間帯定義（ミリ秒）- 5分刻み
TIME_PHASES: List[Tuple[int, Optional[int]]] = [
    (0, 5 * 60 * 1000),               # Phase 0: 0-5分
    (5 * 60 * 1000, 10 * 60 * 1000),  # Phase 1: 5-10分
    (10 * 60 * 1000, 15 * 60 * 1000), # Phase 2: 10-15分
    (15 * 60 * 1000, 20 * 60 * 1000), # Phase 3: 15-20分
    (20 * 60 * 1000, 25 * 60 * 1000), # Phase 4: 20-25分
    (25 * 60 * 1000, 30 * 60 * 1000), # Phase 5: 25-30分
    (30 * 60 * 1000, None),           # Phase 6: 30分以降
]
NUM_PHASES = len(TIME_PHASES)

# スコア閾値
THRESHOLD_HIGH = 0.03  # |importance| >= 0.03 → 2点
THRESHOLD_MID = 0.01   # 0.01 <= |importance| < 0.03 → 1点


# =============================================================================
# TacticalScorer クラス
# =============================================================================

class TacticalScorer:
    """
    wardに戦術スコアを付与するクラス

    使用例:
        scorer = TacticalScorer(Path("models/vision_importance.npy"))
        results = scorer.score_match(Path("C:/dataset/JP1-123/wards_matched.csv"))
        scorer.save_scores(results, Path("C:/dataset/JP1-123/ward_tactical_scores.csv"))
    """

    def __init__(self, importance_map_path: Path):
        """
        Args:
            importance_map_path: 重要度マップファイルパス (7, 32, 32)
        """
        self.importance_map = np.load(importance_map_path)

        if self.importance_map.shape != (NUM_PHASES, GRID_SIZE, GRID_SIZE):
            raise ValueError(
                f"重要度マップの形状が不正: {self.importance_map.shape}, "
                f"期待値: ({NUM_PHASES}, {GRID_SIZE}, {GRID_SIZE})"
            )

    def get_phase(self, timestamp_ms: int) -> int:
        """
        タイムスタンプからPhase番号を取得

        Args:
            timestamp_ms: タイムスタンプ（ミリ秒）

        Returns:
            Phase番号（0-6）
        """
        for phase_idx, (start, end) in enumerate(TIME_PHASES):
            if end is None:
                return phase_idx
            if start <= timestamp_ms < end:
                return phase_idx
        return NUM_PHASES - 1  # 最後のPhase

    def get_grid_coords(self, x_pixel: float, y_pixel: float) -> Tuple[int, int]:
        """
        ピクセル座標をグリッド座標に変換

        Args:
            x_pixel: Xピクセル座標（0-512）
            y_pixel: Yピクセル座標（0-512）

        Returns:
            (grid_x, grid_y) グリッド座標（0-31）
        """
        grid_x = min(int(x_pixel // CELL_SIZE), GRID_SIZE - 1)
        grid_y = min(int(y_pixel // CELL_SIZE), GRID_SIZE - 1)
        return grid_x, grid_y

    def get_importance(self, phase: int, grid_x: int, grid_y: int) -> float:
        """
        指定位置の重要度を取得

        Args:
            phase: Phase番号
            grid_x: グリッドX座標
            grid_y: グリッドY座標

        Returns:
            重要度値（正=Blue有利、負=Red有利）
        """
        return float(self.importance_map[phase, grid_y, grid_x])

    def importance_to_score(self, importance: float) -> int:
        """
        重要度をスコアに変換

        Args:
            importance: 重要度の絶対値

        Returns:
            スコア（0, 1, 2）
        """
        abs_imp = abs(importance)
        if abs_imp >= THRESHOLD_HIGH:
            return 2
        elif abs_imp >= THRESHOLD_MID:
            return 1
        else:
            return 0

    def calc_placement_score(
        self,
        team: str,
        x_pixel: float,
        y_pixel: float,
        timestamp_ms: int
    ) -> int:
        """
        設置スコアを計算

        自チームに有利な位置に置けば高スコア:
        - Blue: 正の重要度 → 高スコア
        - Red: 負の重要度 → 高スコア

        Args:
            team: "blue" or "red"
            x_pixel: Xピクセル座標
            y_pixel: Yピクセル座標
            timestamp_ms: 設置時刻（ミリ秒）

        Returns:
            設置スコア（0, 1, 2）
        """
        phase = self.get_phase(timestamp_ms)
        grid_x, grid_y = self.get_grid_coords(x_pixel, y_pixel)
        importance = self.get_importance(phase, grid_x, grid_y)

        # チームに応じて符号を調整
        # Blue: 正の重要度が有利 → そのまま
        # Red: 負の重要度が有利 → 符号反転
        if team == "red":
            importance = -importance

        # 正の重要度のみスコア化（自チームに有利な位置）
        if importance > 0:
            return self.importance_to_score(importance)
        else:
            return 0

    def calc_deny_score(
        self,
        destroyer_team: str,
        x_pixel: float,
        y_pixel: float,
        timestamp_ms: int
    ) -> int:
        """
        破壊スコアを計算

        敵wardを破壊した側のチームにスコアを付与:
        - Blue(破壊者): 負の重要度セル（Red有利）の敵ward破壊 → 高スコア
        - Red(破壊者): 正の重要度セル（Blue有利）の敵ward破壊 → 高スコア

        Args:
            destroyer_team: 破壊した側のチーム "blue" or "red"
            x_pixel: 破壊されたwardのXピクセル座標
            y_pixel: 破壊されたwardのYピクセル座標
            timestamp_ms: 破壊時刻（ミリ秒）

        Returns:
            破壊スコア（0, 1, 2）
        """
        phase = self.get_phase(timestamp_ms)
        grid_x, grid_y = self.get_grid_coords(x_pixel, y_pixel)
        importance = self.get_importance(phase, grid_x, grid_y)

        # Blueが破壊: 負の重要度（Red有利）のward破壊が価値あり → 符号反転
        # Redが破壊: 正の重要度（Blue有利）のward破壊が価値あり → そのまま
        if destroyer_team == "blue":
            importance = -importance

        # 正の重要度のみスコア化
        if importance > 0:
            return self.importance_to_score(importance)
        else:
            return 0

    def score_match(self, wards_csv_path: Path) -> List[Dict]:
        """
        1試合分のwardにスコアを付与

        Args:
            wards_csv_path: wards_matched.csvのパス

        Returns:
            各wardのスコア情報リスト
        """
        results = []

        with open(wards_csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)

            for row in reader:
                ward_id = row.get('ward_id', '')
                team = row.get('team', '')
                x_pixel_str = row.get('x_pixel', '')
                y_pixel_str = row.get('y_pixel', '')
                timestamp_placed_str = row.get('timestamp_placed', '')
                timestamp_killed_str = row.get('timestamp_killed', '')

                # 必須項目のチェック
                if not all([ward_id, team, x_pixel_str, y_pixel_str, timestamp_placed_str]):
                    continue

                # 座標がない場合（timeline_only等）はスキップ
                try:
                    x_pixel = float(x_pixel_str)
                    y_pixel = float(y_pixel_str)
                    timestamp_placed = int(float(timestamp_placed_str))
                except (ValueError, TypeError):
                    continue

                # 設置スコア計算
                placement_score = self.calc_placement_score(
                    team, x_pixel, y_pixel, timestamp_placed
                )

                # 破壊スコア計算（破壊された場合のみ）
                deny_score = 0
                if timestamp_killed_str:
                    try:
                        timestamp_killed = int(float(timestamp_killed_str))
                        # 破壊した側は敵チーム
                        destroyer_team = "red" if team == "blue" else "blue"
                        deny_score = self.calc_deny_score(
                            destroyer_team, x_pixel, y_pixel, timestamp_killed
                        )
                    except (ValueError, TypeError):
                        pass

                results.append({
                    'ward_id': ward_id,
                    'team': team,
                    'x_pixel': x_pixel,
                    'y_pixel': y_pixel,
                    'timestamp_placed': timestamp_placed,
                    'timestamp_killed': timestamp_killed_str if timestamp_killed_str else '',
                    'placement_score': placement_score,
                    'deny_score': deny_score,
                    'deny_team': ("red" if team == "blue" else "blue") if timestamp_killed_str else '',
                })

        return results

    def save_scores(self, results: List[Dict], output_path: Path) -> None:
        """
        スコア結果をCSVに保存

        Args:
            results: score_matchの出力
            output_path: 出力CSVパス
        """
        if not results:
            print(f"  警告: 結果が空のため保存スキップ: {output_path}")
            return

        fieldnames = [
            'ward_id', 'team', 'x_pixel', 'y_pixel',
            'timestamp_placed', 'timestamp_killed',
            'placement_score', 'deny_score', 'deny_team'
        ]

        with open(output_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

    def process_match(self, match_dir: Path) -> Optional[Dict]:
        """
        1試合を処理

        Args:
            match_dir: 試合ディレクトリ

        Returns:
            処理結果サマリー、失敗時はNone
        """
        wards_csv = match_dir / "wards_matched.csv"
        output_csv = match_dir / "ward_tactical_scores.csv"

        if not wards_csv.exists():
            return None

        results = self.score_match(wards_csv)
        self.save_scores(results, output_csv)

        # サマリー計算
        blue_placement = sum(r['placement_score'] for r in results if r['team'] == 'blue')
        red_placement = sum(r['placement_score'] for r in results if r['team'] == 'red')
        blue_deny = sum(r['deny_score'] for r in results if r.get('deny_team') == 'blue')
        red_deny = sum(r['deny_score'] for r in results if r.get('deny_team') == 'red')

        return {
            'match_id': match_dir.name,
            'ward_count': len(results),
            'blue_placement': blue_placement,
            'red_placement': red_placement,
            'blue_deny': blue_deny,
            'red_deny': red_deny,
            'blue_total': blue_placement + blue_deny,
            'red_total': red_placement + red_deny,
        }

    def process_all(self, dataset_dir: Path) -> List[Dict]:
        """
        全試合を処理

        Args:
            dataset_dir: データセットディレクトリ

        Returns:
            各試合のサマリーリスト
        """
        results = []
        match_dirs = sorted(dataset_dir.glob("JP1-*"))

        print(f"全{len(match_dirs)}試合を処理します")

        for match_dir in match_dirs:
            summary = self.process_match(match_dir)
            if summary:
                results.append(summary)
                print(f"  {summary['match_id']}: "
                      f"Blue={summary['blue_total']} (P:{summary['blue_placement']}/D:{summary['blue_deny']}), "
                      f"Red={summary['red_total']} (P:{summary['red_placement']}/D:{summary['red_deny']})")

        print(f"\n処理完了: {len(results)}試合")
        return results


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Ward戦術スコア計算"
    )
    parser.add_argument(
        "--importance", type=str,
        default="models/vision_importance.npy",
        help="重要度マップファイルパス"
    )
    parser.add_argument(
        "--dataset", type=str,
        default=r"C:\dataset_20260105",
        help="データセットディレクトリ"
    )
    parser.add_argument(
        "--match", type=str,
        help="処理する試合ID（省略時は全試合）"
    )
    args = parser.parse_args()

    importance_path = Path(args.importance)
    dataset_dir = Path(args.dataset)

    if not importance_path.exists():
        print(f"エラー: 重要度マップが見つかりません: {importance_path}")
        return

    scorer = TacticalScorer(importance_path)

    if args.match:
        match_id = args.match
        if not match_id.startswith("JP1-"):
            match_id = f"JP1-{match_id}"
        match_dir = dataset_dir / match_id
        summary = scorer.process_match(match_dir)
        if summary:
            print(f"処理完了: {summary}")
        else:
            print(f"処理失敗: {match_dir}")
    else:
        scorer.process_all(dataset_dir)


if __name__ == "__main__":
    main()

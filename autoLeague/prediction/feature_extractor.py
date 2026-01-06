"""
Phase 6: 特徴量抽出モジュール

Timeline/MatchデータからN分時点の特徴量を抽出する。
"""

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import (
    PREDICTION_TIMES_MS,
    BASELINE_FEATURES,
    RIOT_VISION_FEATURES,
    BLUE_PARTICIPANT_IDS,
    RED_PARTICIPANT_IDS,
    BLUE_TEAM_ID,
    RED_TEAM_ID,
    GRID_SIZE,
    MINIMAP_SIZE,
    CELL_SIZE,
    DEFAULT_WARD_DURATION_MS,
    DEFAULT_DATASET_DIR,
)


# =============================================================================
# Timeline特徴量抽出
# =============================================================================

class TimelineFeatureExtractor:
    """
    タイムラインから指定時刻の特徴量を抽出

    使用例:
        extractor = TimelineFeatureExtractor(timeline_path, match_path)
        features_10min = extractor.extract_at_time(10 * 60 * 1000)
        vision_est = extractor.extract_vision_score_estimate(10 * 60 * 1000)
    """

    def __init__(self, timeline_path: Path, match_path: Path):
        """
        Args:
            timeline_path: Timeline JSONファイルのパス
            match_path: Match JSONファイルのパス
        """
        self.timeline_path = Path(timeline_path)
        self.match_path = Path(match_path)

        self.timeline = self._load_json(self.timeline_path)
        self.match = self._load_json(self.match_path)

        # フレーム間隔（通常60000ms = 1分）
        self.frame_interval = self.timeline["info"]["frameInterval"]
        self.frames = self.timeline["info"]["frames"]

        # 試合情報
        self.game_duration_sec = self.match["info"]["gameDuration"]
        self.game_duration_ms = self.game_duration_sec * 1000
        self.participants = self.match["info"]["participants"]

    def _load_json(self, path: Path) -> dict:
        """JSONファイルを読み込み"""
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _get_frame_index(self, time_ms: int) -> int:
        """
        指定時刻に対応するフレームインデックスを取得

        Args:
            time_ms: 時刻（ミリ秒）

        Returns:
            フレームインデックス
        """
        # フレームは1分ごと、frame[0]は0分時点
        frame_idx = time_ms // self.frame_interval
        return min(frame_idx, len(self.frames) - 1)

    def extract_at_time(self, time_ms: int) -> Dict[str, float]:
        """
        指定時刻のベースライン特徴量を抽出

        Args:
            time_ms: 時刻（ミリ秒）

        Returns:
            特徴量辞書
        """
        frame_idx = self._get_frame_index(time_ms)
        features = {}

        # participantFramesから抽出
        features.update(self._extract_participant_features(frame_idx))

        # eventsから抽出（累積）
        features.update(self._extract_event_features(time_ms))

        return features

    def _extract_participant_features(self, frame_idx: int) -> Dict[str, float]:
        """
        participantFramesから特徴量を抽出

        Args:
            frame_idx: フレームインデックス

        Returns:
            特徴量辞書
        """
        frame = self.frames[frame_idx]
        participant_frames = frame["participantFrames"]

        # チーム別に集計
        blue_gold = 0
        red_gold = 0
        blue_xp = 0
        red_xp = 0
        blue_levels = []
        red_levels = []
        blue_cs = 0
        red_cs = 0

        for pid_str, pf in participant_frames.items():
            pid = int(pid_str)

            gold = pf.get("totalGold", 0)
            xp = pf.get("xp", 0)
            level = pf.get("level", 1)
            minions = pf.get("minionsKilled", 0)
            jungle_minions = pf.get("jungleMinionsKilled", 0)
            cs = minions + jungle_minions

            if pid in BLUE_PARTICIPANT_IDS:
                blue_gold += gold
                blue_xp += xp
                blue_levels.append(level)
                blue_cs += cs
            elif pid in RED_PARTICIPANT_IDS:
                red_gold += gold
                red_xp += xp
                red_levels.append(level)
                red_cs += cs

        return {
            "blue_total_gold": blue_gold,
            "red_total_gold": red_gold,
            "gold_diff": blue_gold - red_gold,
            "blue_total_xp": blue_xp,
            "red_total_xp": red_xp,
            "blue_avg_level": np.mean(blue_levels) if blue_levels else 1.0,
            "red_avg_level": np.mean(red_levels) if red_levels else 1.0,
            "blue_total_cs": blue_cs,
            "red_total_cs": red_cs,
        }

    def _extract_event_features(self, time_ms: int) -> Dict[str, float]:
        """
        eventsから累積特徴量を抽出

        Args:
            time_ms: 時刻（ミリ秒）

        Returns:
            特徴量辞書
        """
        blue_kills = 0
        red_kills = 0
        blue_dragons = 0
        red_dragons = 0
        blue_heralds = 0
        red_heralds = 0
        blue_towers = 0
        red_towers = 0
        blue_wards_placed = 0
        red_wards_placed = 0

        for frame in self.frames:
            for event in frame.get("events", []):
                event_time = event.get("timestamp", 0)
                if event_time > time_ms:
                    continue

                event_type = event.get("type", "")

                if event_type == "CHAMPION_KILL":
                    killer_id = event.get("killerId", 0)
                    if killer_id in BLUE_PARTICIPANT_IDS:
                        blue_kills += 1
                    elif killer_id in RED_PARTICIPANT_IDS:
                        red_kills += 1

                elif event_type == "ELITE_MONSTER_KILL":
                    killer_team_id = event.get("killerTeamId", 0)
                    monster_type = event.get("monsterType", "")

                    if monster_type == "DRAGON":
                        if killer_team_id == BLUE_TEAM_ID:
                            blue_dragons += 1
                        elif killer_team_id == RED_TEAM_ID:
                            red_dragons += 1
                    elif monster_type == "RIFTHERALD":
                        if killer_team_id == BLUE_TEAM_ID:
                            blue_heralds += 1
                        elif killer_team_id == RED_TEAM_ID:
                            red_heralds += 1

                elif event_type == "BUILDING_KILL":
                    building_type = event.get("buildingType", "")
                    team_id = event.get("teamId", 0)  # 破壊されたタワーのチーム

                    if building_type == "TOWER_BUILDING":
                        # teamIdは破壊されたタワーの所属チーム
                        # 敵チームのタワーを破壊 = 自チームのスコア
                        if team_id == RED_TEAM_ID:  # Redのタワーが壊された = Blueが破壊
                            blue_towers += 1
                        elif team_id == BLUE_TEAM_ID:  # Blueのタワーが壊された = Redが破壊
                            red_towers += 1

                elif event_type == "WARD_PLACED":
                    creator_id = event.get("creatorId", 0)
                    if creator_id in BLUE_PARTICIPANT_IDS:
                        blue_wards_placed += 1
                    elif creator_id in RED_PARTICIPANT_IDS:
                        red_wards_placed += 1

        return {
            "blue_kills": blue_kills,
            "red_kills": red_kills,
            "kill_diff": blue_kills - red_kills,
            "blue_dragons": blue_dragons,
            "red_dragons": red_dragons,
            "blue_heralds": blue_heralds,
            "red_heralds": red_heralds,
            "blue_towers": blue_towers,
            "red_towers": red_towers,
            "blue_wards_placed": blue_wards_placed,
            "red_wards_placed": red_wards_placed,
        }

    def extract_vision_score_estimate(self, time_ms: int) -> Dict[str, float]:
        """
        Riot visionScoreの途中時点推定値を抽出

        visionScorePerMinuteを使用して推定:
        estimated_score = visionScorePerMinute * (time_ms / 60000)

        Args:
            time_ms: 時刻（ミリ秒）

        Returns:
            {"blue_vision_score_est": float, "red_vision_score_est": float, ...}
        """
        time_min = time_ms / 60000.0

        blue_vision = 0.0
        red_vision = 0.0

        for p in self.participants:
            team_id = p.get("teamId", 0)

            # visionScorePerMinuteがある場合はそれを使用
            vision_per_min = p.get("challenges", {}).get("visionScorePerMinute", 0)

            # なければ試合終了時のvisionScoreから計算
            if vision_per_min == 0 and self.game_duration_sec > 0:
                vision_score = p.get("visionScore", 0)
                game_min = self.game_duration_sec / 60.0
                vision_per_min = vision_score / game_min if game_min > 0 else 0

            estimated_score = vision_per_min * time_min

            if team_id == BLUE_TEAM_ID:
                blue_vision += estimated_score
            elif team_id == RED_TEAM_ID:
                red_vision += estimated_score

        return {
            "blue_vision_score_est": blue_vision,
            "red_vision_score_est": red_vision,
            "vision_score_diff": blue_vision - red_vision,
        }

    def get_winner(self) -> str:
        """
        勝者チームを取得

        Returns:
            "blue" or "red"
        """
        for team in self.match["info"]["teams"]:
            if team.get("win", False):
                return "blue" if team["teamId"] == BLUE_TEAM_ID else "red"
        return "unknown"


# =============================================================================
# Ward Grid Extractor
# =============================================================================

class WardGridExtractor:
    """
    指定時刻までのward配置をグリッド特徴量に変換

    Phase 5のgrid_generator.pyを時刻指定で拡張したもの。
    """

    def __init__(self, wards_csv_path: Path):
        """
        Args:
            wards_csv_path: wards_matched.csvのパス
        """
        self.wards_csv_path = Path(wards_csv_path)
        self.wards = self._load_wards()

    def _load_wards(self) -> List[dict]:
        """CSVからward情報を読み込み"""
        wards = []

        if not self.wards_csv_path.exists():
            return wards

        with open(self.wards_csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # timeline_onlyは座標がないため除外
                if row.get('match_status', '') == 'timeline_only':
                    continue

                # 座標を取得
                x_pixel_str = row.get('x_pixel', '')
                y_pixel_str = row.get('y_pixel', '')
                if not x_pixel_str or not y_pixel_str:
                    continue

                try:
                    ward = {
                        'x_pixel': int(x_pixel_str),
                        'y_pixel': int(y_pixel_str),
                        'team': row.get('team', ''),
                        'ward_type': row.get('ward_type', 'SIGHT_WARD'),
                        'timestamp_placed': int(row.get('timestamp_placed', 0)),
                        'timestamp_killed': int(row['timestamp_killed']) if row.get('timestamp_killed') else None,
                    }

                    # 座標範囲チェック
                    if 0 <= ward['x_pixel'] < MINIMAP_SIZE and 0 <= ward['y_pixel'] < MINIMAP_SIZE:
                        if ward['team'] in ('blue', 'red'):
                            wards.append(ward)
                except (ValueError, KeyError):
                    continue

        return wards

    def extract_at_time(self, time_ms: int) -> np.ndarray:
        """
        指定時刻までのward配置をグリッド化

        Args:
            time_ms: 時刻（ミリ秒）

        Returns:
            np.ndarray (2, GRID_SIZE, GRID_SIZE) - [blue_grid, red_grid]
            各セルにはwardが存在した秒数が格納される
        """
        blue_grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
        red_grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)

        for ward in self.wards:
            timestamp_placed = ward['timestamp_placed']

            # 指定時刻以降に設置されたwardは除外
            if timestamp_placed > time_ms:
                continue

            # 持続時間を計算
            timestamp_killed = ward['timestamp_killed']
            ward_type = ward['ward_type']

            if timestamp_killed is not None:
                duration_ms = timestamp_killed - timestamp_placed
            else:
                duration_ms = DEFAULT_WARD_DURATION_MS.get(ward_type, 90 * 1000)

            # 指定時刻までの実効持続時間
            ward_end = timestamp_placed + duration_ms
            effective_end = min(ward_end, time_ms)
            effective_duration_ms = max(0, effective_end - timestamp_placed)
            effective_duration_sec = effective_duration_ms / 1000.0

            # グリッド座標に変換
            grid_x = min(ward['x_pixel'] // CELL_SIZE, GRID_SIZE - 1)
            grid_y = min(ward['y_pixel'] // CELL_SIZE, GRID_SIZE - 1)

            # グリッドに累積
            if ward['team'] == 'blue':
                blue_grid[grid_y, grid_x] += effective_duration_sec
            else:
                red_grid[grid_y, grid_x] += effective_duration_sec

        return np.stack([blue_grid, red_grid], axis=0)


# =============================================================================
# データセット構築
# =============================================================================

def build_prediction_dataset(
    timeline_dir: Path,
    match_dir: Path,
    dataset_dir: Path,
    output_path: Path,
    times_ms: Optional[List[int]] = None,
    verbose: bool = True,
) -> Dict:
    """
    全試合を処理してデータセットを構築

    Args:
        timeline_dir: Timeline JSONディレクトリ
        match_dir: Match JSONディレクトリ
        dataset_dir: wards_matched.csv格納ディレクトリ（例: C:/dataset_20260105）
        output_path: 出力先パス
        times_ms: 評価時点リスト（デフォルト: [10分, 20分]）
        verbose: 進捗表示

    Returns:
        {"n_samples": int, "n_features": int, ...}
    """
    if times_ms is None:
        times_ms = PREDICTION_TIMES_MS

    timeline_dir = Path(timeline_dir)
    match_dir = Path(match_dir)
    dataset_dir = Path(dataset_dir)
    output_path = Path(output_path)

    # Match JSONファイルを列挙（Timeline JSONとの共通部分）
    match_files = sorted(match_dir.glob("JP1_*.json"))

    # 結果格納用
    all_baseline_features = []
    all_riot_vision_features = []
    all_ward_grids = []
    all_labels = []
    all_match_ids = []

    n_skipped = 0

    for match_file in match_files:
        match_id = match_file.stem  # 例: JP1_555621265
        timeline_file = timeline_dir / f"{match_id}.json"

        # wards_matched.csvのパス（例: C:/dataset_20260105/JP1-555621265/wards_matched.csv）
        match_id_dash = match_id.replace("_", "-")
        wards_csv_path = dataset_dir / match_id_dash / "wards_matched.csv"

        # 必要なファイルが揃っているか確認
        if not timeline_file.exists():
            if verbose:
                print(f"スキップ: {match_id} (timeline なし)")
            n_skipped += 1
            continue

        if not wards_csv_path.exists():
            if verbose:
                print(f"スキップ: {match_id} (wards_matched.csv なし)")
            n_skipped += 1
            continue

        try:
            # 特徴量抽出器を初期化
            extractor = TimelineFeatureExtractor(timeline_file, match_file)
            ward_extractor = WardGridExtractor(wards_csv_path)

            # 試合が指定時刻より短い場合はスキップ
            if extractor.game_duration_ms < max(times_ms):
                if verbose:
                    print(f"スキップ: {match_id} (試合時間 {extractor.game_duration_sec}秒 < 最大評価時点)")
                n_skipped += 1
                continue

            # 各時点の特徴量を抽出
            baseline_per_time = []
            riot_vision_per_time = []
            ward_grid_per_time = []

            for time_ms in times_ms:
                # ベースライン特徴量
                baseline = extractor.extract_at_time(time_ms)
                baseline_vec = [baseline[f] for f in BASELINE_FEATURES]
                baseline_per_time.append(baseline_vec)

                # Riot visionScore推定値
                riot_vision = extractor.extract_vision_score_estimate(time_ms)
                riot_vision_vec = [riot_vision[f] for f in RIOT_VISION_FEATURES]
                riot_vision_per_time.append(riot_vision_vec)

                # Ward grid
                ward_grid = ward_extractor.extract_at_time(time_ms)
                ward_grid_per_time.append(ward_grid)

            # 勝敗ラベル
            winner = extractor.get_winner()
            label = 1 if winner == "blue" else 0

            # 結果に追加
            all_baseline_features.append(baseline_per_time)
            all_riot_vision_features.append(riot_vision_per_time)
            all_ward_grids.append(ward_grid_per_time)
            all_labels.append(label)
            all_match_ids.append(match_id)

            if verbose:
                print(f"処理完了: {match_id} (勝者: {winner})")

        except Exception as e:
            if verbose:
                print(f"エラー: {match_id} - {e}")
            n_skipped += 1
            continue

    # NumPy配列に変換
    n_samples = len(all_match_ids)
    n_times = len(times_ms)

    if n_samples == 0:
        raise ValueError("処理可能な試合がありません")

    X_baseline = np.array(all_baseline_features, dtype=np.float32)  # (N, T, F_base)
    X_riot_vision = np.array(all_riot_vision_features, dtype=np.float32)  # (N, T, 3)
    X_ward_grid = np.array(all_ward_grids, dtype=np.float32)  # (N, T, 2, 32, 32)
    y = np.array(all_labels, dtype=np.int32)  # (N,)

    # 保存
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        X_baseline=X_baseline,
        X_riot_vision=X_riot_vision,
        X_ward_grid=X_ward_grid,
        y=y,
        match_ids=all_match_ids,
        times_ms=times_ms,
        baseline_features=BASELINE_FEATURES,
        riot_vision_features=RIOT_VISION_FEATURES,
    )

    stats = {
        "n_samples": n_samples,
        "n_skipped": n_skipped,
        "n_times": n_times,
        "n_baseline_features": len(BASELINE_FEATURES),
        "n_riot_vision_features": len(RIOT_VISION_FEATURES),
        "blue_wins": int(y.sum()),
        "red_wins": int(n_samples - y.sum()),
        "output_path": str(output_path),
    }

    if verbose:
        print(f"\n=== データセット構築完了 ===")
        print(f"試合数: {n_samples}")
        print(f"スキップ: {n_skipped}")
        print(f"Blue勝利: {stats['blue_wins']}, Red勝利: {stats['red_wins']}")
        print(f"出力先: {output_path}")

    return stats


def load_prediction_dataset(path: Path) -> Dict:
    """
    データセットを読み込み

    Args:
        path: データセットファイルパス

    Returns:
        データセット辞書
    """
    path = Path(path)
    data = np.load(path, allow_pickle=True)

    return {
        "X_baseline": data["X_baseline"],
        "X_riot_vision": data["X_riot_vision"],
        "X_ward_grid": data["X_ward_grid"],
        "y": data["y"],
        "match_ids": data["match_ids"].tolist(),
        "times_ms": data["times_ms"].tolist(),
        "baseline_features": data["baseline_features"].tolist(),
        "riot_vision_features": data["riot_vision_features"].tolist(),
    }


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Phase 6: 特徴量抽出"
    )
    parser.add_argument(
        "--timeline-dir", type=str,
        default=r"c:\Users\lapis\Desktop\LoL_WorkSp_win\pyLoL-_WorkSp\pyLoL-v2\data\timeline",
        help="Timeline JSONディレクトリ"
    )
    parser.add_argument(
        "--match-dir", type=str,
        default=r"c:\Users\lapis\Desktop\LoL_WorkSp_win\pyLoL-_WorkSp\pyLoL-v2\data\match",
        help="Match JSONディレクトリ"
    )
    parser.add_argument(
        "--dataset-dir", type=str,
        default=DEFAULT_DATASET_DIR,
        help="wards_matched.csv格納ディレクトリ"
    )
    parser.add_argument(
        "--output", type=str,
        default=r"c:\Users\lapis\Desktop\LoL_WorkSp_win\pyLoL-_WorkSp\pyLoL-v2\data\prediction_dataset.npz",
        help="出力先パス"
    )

    args = parser.parse_args()

    build_prediction_dataset(
        timeline_dir=Path(args.timeline_dir),
        match_dir=Path(args.match_dir),
        dataset_dir=Path(args.dataset_dir),
        output_path=Path(args.output),
    )


if __name__ == "__main__":
    main()

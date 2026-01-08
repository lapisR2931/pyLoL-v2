"""
データセット構築 - Phase 5 Task B

全試合のward_grid.npzと勝敗ラベルを統合してdataset.npzを生成する。

入力:
    - ward_grid.npz (各試合フォルダ内) - Task Aの出力
    - マッチJSON (勝敗情報)

出力:
    - vision_dataset.npz
        - X: (N, 2, 7, 32, 32) - N試合, [blue,red], 7時間帯(5分刻み), 32x32グリッド
        - y: (N,) - 勝敗ラベル (1=Blue勝利, 0=Red勝利)
        - match_ids: list - 試合IDリスト
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests


# =============================================================================
# 設定
# =============================================================================

GRID_SIZE = 32
NUM_PHASES = 7  # 5分刻み: 0-5, 5-10, 10-15, 15-20, 20-25, 25-30, 30+
DEFAULT_ROUTING = "asia"
DEFAULT_REGION_PREFIX = "JP1_"
MAX_RETRIES = 5


# =============================================================================
# 勝敗情報取得
# =============================================================================

def get_winner_from_match_json(match_json_path: Path) -> str:
    """
    マッチJSONから勝者チームを取得

    Args:
        match_json_path: マッチJSONのパス

    Returns:
        "blue" (teamId=100勝利) または "red" (teamId=200勝利)

    Raises:
        ValueError: 勝敗情報が取得できない場合
    """
    with open(match_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    teams = data.get("info", {}).get("teams", [])
    if not teams:
        raise ValueError(f"teams配列が見つかりません: {match_json_path}")

    for team in teams:
        team_id = team.get("teamId")
        win = team.get("win")

        if team_id == 100 and win is True:
            return "blue"
        elif team_id == 200 and win is True:
            return "red"

    # teamsが存在するが勝者が見つからない場合
    # 最初のチームのwinを確認
    if teams[0].get("teamId") == 100:
        return "blue" if teams[0].get("win") else "red"
    else:
        return "red" if teams[0].get("win") else "blue"


# =============================================================================
# マッチJSON自動取得
# =============================================================================

def fetch_match_json(
    api_key: str,
    match_id: str,
    output_path: Path,
    routing: str = DEFAULT_ROUTING,
    max_retries: int = MAX_RETRIES
) -> bool:
    """
    Riot APIからマッチJSONを取得して保存

    Args:
        api_key: Riot API Key
        match_id: マッチID (例: JP1_555621265)
        output_path: 保存先パス
        routing: ルーティング (asia, americas, europe, sea)
        max_retries: リトライ回数

    Returns:
        成功時True、失敗時False
    """
    # match_idのフォーマットを統一（ハイフン→アンダースコア）
    match_id_api = match_id.replace("-", "_")

    url = f"https://{routing}.api.riotgames.com/lol/match/v5/matches/{match_id_api}"
    params = {"api_key": api_key}

    for attempt in range(max_retries):
        try:
            resp = requests.get(url, params=params, timeout=10)

            # Rate limit (429): wait based on Retry-After header
            if resp.status_code == 429:
                wait = int(resp.headers.get("Retry-After", 5))
                print(f"  レート制限 - {wait + 1}秒待機...")
                time.sleep(wait + 1)
                continue

            # Server error (5xx): wait briefly and retry
            if 500 <= resp.status_code < 600:
                print(f"  サーバーエラー ({resp.status_code}) - 2秒待機...")
                time.sleep(2)
                continue

            # Client error (4xx except 429)
            if 400 <= resp.status_code < 500:
                print(f"  エラー: {resp.status_code} - {match_id_api}")
                return False

            # Success
            data = resp.json()

            # Validate response
            if "info" not in data:
                print(f"  無効なレスポンス: {match_id_api}")
                return False

            # Save to file
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            print(f"  取得完了: {match_id_api}")
            return True

        except requests.exceptions.Timeout:
            print(f"  タイムアウト - リトライ ({attempt + 1}/{max_retries})")
            time.sleep(2)
            continue
        except requests.exceptions.RequestException as e:
            print(f"  リクエストエラー: {e}")
            return False

    print(f"  リトライ上限到達: {match_id_api}")
    return False


def load_api_key(api_key: Optional[str] = None, env_file: str = "RiotAPI.env") -> Optional[str]:
    """
    APIキーを取得

    優先順位:
    1. 引数で渡されたapi_key
    2. 環境変数 RIOT_API_KEY または API_KEY
    3. RiotAPI.envファイル (RIOT_API_KEY= または API_KEY=)

    Returns:
        APIキー、または取得できない場合はNone
    """
    if api_key:
        return api_key

    # 環境変数から取得
    api_key = os.environ.get("RIOT_API_KEY") or os.environ.get("API_KEY")
    if api_key:
        return api_key

    # envファイルから取得
    env_path = Path(env_file)
    if env_path.exists():
        with open(env_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                # RIOT_API_KEY= または API_KEY= をサポート
                for prefix in ("RIOT_API_KEY=", "API_KEY="):
                    if line.startswith(prefix):
                        return line.split("=", 1)[1].strip().strip('"\'')

    return None


# =============================================================================
# DatasetBuilder クラス
# =============================================================================

class DatasetBuilder:
    """
    視界スコアデータセットを構築するメインクラス

    Attributes:
        dataset_dir: キャプチャ画像フォルダのルート (例: C:\\dataset_20260105)
        match_dir: マッチJSONフォルダ (例: data/match)
        output_path: 出力ファイルパス (例: data/vision_dataset.npz)
        api_key: Riot API Key (マッチJSON自動取得用、オプション)
    """

    def __init__(
        self,
        dataset_dir: Path,
        match_dir: Path,
        output_path: Path,
        api_key: Optional[str] = None
    ):
        self.dataset_dir = Path(dataset_dir)
        self.match_dir = Path(match_dir)
        self.output_path = Path(output_path)
        self.api_key = api_key

        # 統計情報
        self.stats = {
            "total": 0,
            "success": 0,
            "skip_no_grid": 0,
            "skip_no_match": 0,
            "skip_error": 0,
        }

    def build(self, dry_run: bool = False) -> None:
        """
        全試合を処理してデータセットを構築・保存

        Args:
            dry_run: Trueの場合、保存せずに処理のみ実行
        """
        match_ids = self._collect_match_ids()
        self.stats["total"] = len(match_ids)

        print(f"全{len(match_ids)}試合を処理します")
        print(f"データセット: {self.dataset_dir}")
        print(f"マッチJSON: {self.match_dir}")
        print(f"出力先: {self.output_path}")
        if self.api_key:
            print("APIキー: 設定済み（不足分は自動取得）")
        else:
            print("APIキー: 未設定（不足分はスキップ）")
        print()

        # データ蓄積用リスト
        X_list: List[np.ndarray] = []
        y_list: List[int] = []
        match_id_list: List[str] = []

        for match_id in match_ids:
            result = self._process_match(match_id)
            if result:
                X, y = result
                X_list.append(X)
                y_list.append(y)
                match_id_list.append(match_id)

        # 結果サマリー
        print()
        print("=" * 50)
        print("処理結果サマリー")
        print("=" * 50)
        print(f"  全試合数:          {self.stats['total']}")
        print(f"  成功:              {self.stats['success']}")
        print(f"  スキップ(grid無し): {self.stats['skip_no_grid']}")
        print(f"  スキップ(match無し): {self.stats['skip_no_match']}")
        print(f"  スキップ(エラー):   {self.stats['skip_error']}")

        # 有効な試合がない場合
        if not X_list:
            print("\nエラー: 有効な試合がありません。データセットを生成できません。")
            return

        # 配列に変換
        X = np.stack(X_list, axis=0).astype(np.float32)
        y = np.array(y_list, dtype=np.int32)

        print()
        print(f"データセット形状:")
        print(f"  X: {X.shape} (N, 2チーム, {NUM_PHASES}時間帯, 32x32)")
        print(f"  y: {y.shape} (Blue勝利={np.sum(y)}, Red勝利={len(y) - np.sum(y)})")

        if dry_run:
            print("\n[dry-run] 保存をスキップしました")
        else:
            # 出力ディレクトリ作成
            self.output_path.parent.mkdir(parents=True, exist_ok=True)

            # 保存
            np.savez(
                self.output_path,
                X=X,
                y=y,
                match_ids=match_id_list
            )
            print(f"\n保存完了: {self.output_path}")

    def _collect_match_ids(self) -> List[str]:
        """dataset_dir内の全試合IDを収集"""
        match_dirs = sorted(self.dataset_dir.glob("JP1-*"))
        return [d.name for d in match_dirs if d.is_dir()]

    def _process_match(self, match_id: str) -> Optional[Tuple[np.ndarray, int]]:
        """
        1試合を処理

        Args:
            match_id: 試合ID (例: JP1-555621265)

        Returns:
            (X, y) - X: (2, 3, 32, 32), y: 0 or 1
            処理失敗時はNone
        """
        print(f"処理中: {match_id}...", end=" ")

        match_dir = self.dataset_dir / match_id

        # ward_grid.npzの読み込み
        grid_data = self._load_grid_data(match_dir)
        if grid_data is None:
            print("スキップ (ward_grid.npz無し)")
            self.stats["skip_no_grid"] += 1
            return None

        # マッチJSONの確保
        match_json_path = self._ensure_match_json(match_id)
        if match_json_path is None:
            print("スキップ (マッチJSON取得失敗)")
            self.stats["skip_no_match"] += 1
            return None

        # 勝敗情報の取得
        try:
            winner = get_winner_from_match_json(match_json_path)
        except Exception as e:
            print(f"スキップ (勝敗取得エラー: {e})")
            self.stats["skip_error"] += 1
            return None

        # データの整形
        # X: (2, 3, 32, 32) - [blue_grid, red_grid]
        X = np.stack([grid_data["blue"], grid_data["red"]], axis=0)
        y = 1 if winner == "blue" else 0

        print(f"OK (勝者: {winner})")
        self.stats["success"] += 1

        return X, y

    def _load_grid_data(self, match_dir: Path) -> Optional[Dict]:
        """ward_grid.npzを読み込む"""
        grid_path = match_dir / "ward_grid.npz"

        if not grid_path.exists():
            return None

        try:
            data = np.load(grid_path, allow_pickle=True)

            # 形状検証
            blue = data["blue"]
            red = data["red"]

            if blue.shape != (NUM_PHASES, GRID_SIZE, GRID_SIZE):
                print(f"  警告: blue形状が不正 {blue.shape}")
                return None
            if red.shape != (NUM_PHASES, GRID_SIZE, GRID_SIZE):
                print(f"  警告: red形状が不正 {red.shape}")
                return None

            return {
                "blue": blue,
                "red": red,
                "match_id": str(data["match_id"])
            }
        except Exception as e:
            print(f"  警告: grid読み込みエラー {e}")
            return None

    def _ensure_match_json(self, match_id: str) -> Optional[Path]:
        """マッチJSONの存在を確認、なければAPI経由で取得"""
        # match_idのフォーマット変換（ハイフン→アンダースコア）
        match_id_file = match_id.replace("-", "_") + ".json"
        match_json_path = self.match_dir / match_id_file

        if match_json_path.exists():
            return match_json_path

        # APIキーがあれば取得を試行
        if self.api_key:
            print(f"マッチJSON取得中...", end=" ")
            success = fetch_match_json(
                self.api_key,
                match_id,
                match_json_path
            )
            if success:
                return match_json_path

        return None


# =============================================================================
# 公開関数
# =============================================================================

def build_dataset(
    dataset_dir: Path,
    match_dir: Path,
    output_path: Path,
    api_key: Optional[str] = None,
    dry_run: bool = False
) -> None:
    """
    全試合のグリッドデータと勝敗ラベルを統合

    Args:
        dataset_dir: キャプチャ画像フォルダのルート
        match_dir: マッチJSONフォルダ
        output_path: 出力ファイルパス
        api_key: Riot API Key (マッチJSON自動取得用)
        dry_run: Trueの場合、保存せずに処理のみ実行
    """
    builder = DatasetBuilder(
        dataset_dir=dataset_dir,
        match_dir=match_dir,
        output_path=output_path,
        api_key=api_key
    )
    builder.build(dry_run=dry_run)


def load_dataset(dataset_path: Path) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    データセットを読み込む

    Args:
        dataset_path: vision_dataset.npzのパス

    Returns:
        (X, y, match_ids)
        - X: (N, 2, 7, 32, 32) - 7時間帯(5分刻み)
        - y: (N,)
        - match_ids: list of str
    """
    data = np.load(dataset_path, allow_pickle=True)
    X = data["X"]
    y = data["y"]
    match_ids = list(data["match_ids"])
    return X, y, match_ids


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="データセット構築（Phase 5 Task B）"
    )
    parser.add_argument(
        "--dataset", type=str, default=r"C:\dataset_20260105",
        help="キャプチャ画像フォルダのルート"
    )
    parser.add_argument(
        "--match-dir", type=str, default="data/match",
        help="マッチJSONフォルダ"
    )
    parser.add_argument(
        "--output", type=str, default="data/vision_dataset.npz",
        help="出力ファイルパス"
    )
    parser.add_argument(
        "--api-key", type=str, default=None,
        help="Riot API Key (省略時は環境変数/RiotAPI.envから取得)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="保存せずに処理のみ実行"
    )
    args = parser.parse_args()

    # APIキーの取得
    api_key = load_api_key(args.api_key)

    build_dataset(
        dataset_dir=Path(args.dataset),
        match_dir=Path(args.match_dir),
        output_path=Path(args.output),
        api_key=api_key,
        dry_run=args.dry_run
    )


if __name__ == "__main__":
    main()

"""Riot APIからマッチ/タイムラインデータを取得・保存するクラス

新規コードではこのモジュールを直接使用することを推奨:
    from autoLeague.dataset.match_fetcher import MatchFetcher
"""
from __future__ import annotations

import csv
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import pandas as pd
import requests
from tqdm import tqdm

from autoLeague.base import BaseRiotClient, RiotApiConfig

logger = logging.getLogger(__name__)


class MatchFetcher(BaseRiotClient):
    """Riot APIからマッチ/タイムラインデータを取得・保存するクラス"""

    def __init__(self, api_key: str, region: str = "JP1", **kwargs) -> None:
        config = RiotApiConfig(api_key=api_key, region=region, **kwargs)
        super().__init__(config)

    # ------------------------------------------------------------------
    # backward compat properties
    # ------------------------------------------------------------------
    @property
    def api_key(self) -> str:
        return self._config.api_key

    @property
    def region(self) -> str:
        return self._config.region

    @property
    def routing(self) -> str:
        return self._config.routing

    @property
    def region_prefix(self) -> str:
        return self._config.region_prefix

    # ------------------------------------------------------------------
    # Summoner lookup
    # ------------------------------------------------------------------
    def get_puuid(self, summoner_id: str) -> Optional[str]:
        """summonerIdからpuuidを取得する。"""
        url = (
            f"{self._config.platform_url}/lol/summoner/v4/summoners/{summoner_id}"
        )
        data = self._request_with_retry(url)
        if data is None:
            return None
        return data.get("puuid")

    def get_summoner_id(self, puuid: str) -> Optional[str]:
        """puuidからsummonerIdを取得する。"""
        url = (
            f"{self._config.platform_url}/lol/summoner/v4/summoners/by-puuid/{puuid}"
        )
        data = self._request_with_retry(url)
        if data is None:
            return None
        return data.get("id")

    # ------------------------------------------------------------------
    # Match ID helpers
    # ------------------------------------------------------------------
    def get_matchids(self, full_matchids: list[str]) -> list[str]:
        """Strip region prefix with '-' (e.g., 'JP1-7650415714' -> '7650415714') and sort."""
        prefix_dash = f"{self._config.region}-"
        nums = [
            mid.replace(prefix_dash, "")
            for mid in full_matchids
            if mid.startswith(prefix_dash)
        ]
        nums.sort()
        return nums

    def get_matchids2(self, full_matchids: list[str]) -> list[str]:
        """Strip region prefix with '_' (e.g., 'JP1_7650415714' -> '7650415714') and sort."""
        nums = [
            mid.replace(self._config.region_prefix, "")
            for mid in full_matchids
            if mid.startswith(self._config.region_prefix)
        ]
        nums.sort()
        return nums

    # ------------------------------------------------------------------
    # Timestamp formatting
    # ------------------------------------------------------------------
    @staticmethod
    def _format_timestamp_ms(ts_ms: int) -> str:
        """ミリ秒タイムスタンプを 'm:ss' 形式に変換する。"""
        m, s = divmod(round(ts_ms / 1000), 60)
        return f"{m}:{s:02d}"

    # ------------------------------------------------------------------
    # Timeline data
    # ------------------------------------------------------------------
    def _get_match_timeline_data(self, match_id_num: str) -> dict:
        """Riot Timeline APIを呼び出す (_request_with_retry使用)。"""
        url = (
            f"{self._config.regional_url}/lol/match/v5/matches/"
            f"{self._config.region_prefix}{match_id_num}/timeline"
        )
        result = self._request_with_retry(url)
        if result is None:
            raise ValueError(f"[{match_id_num}] timeline call failed")
        return result

    def save_timeline_data_from_matchlist(
        self,
        matchids: list[str],
        output_dir: str,
    ) -> None:
        """matchidリストからタイムラインデータを取得し、JSONファイルとして保存する。

        Args:
            matchids: matchidリスト (例: ["JP1_123456", "KR_789012"])
            output_dir: 保存先ディレクトリ
        """
        os.makedirs(output_dir, exist_ok=True)

        nums = self.get_matchids2(matchids)
        print(f"Total match_ids: {len(nums)}")

        completed = 0
        errors = 0

        for mid in tqdm(nums, desc="Saving timeline data"):
            try:
                tl_data = self._get_match_timeline_data(mid)
                json_path = os.path.join(
                    output_dir, f"{self._config.region_prefix}{mid}.json"
                )
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(tl_data, f, ensure_ascii=False, indent=2)
                completed += 1
            except Exception as e:
                errors += 1
                print(f"  [{mid}] error: {e}")

        print(f"Completed! Saved: {completed}, Errors: {errors}")

    # ------------------------------------------------------------------
    # Match JSON data
    # ------------------------------------------------------------------
    def save_match_json_from_matchlist(
        self,
        matchids: list[str],
        output_dir: str,
    ) -> None:
        """matchidリストからマッチデータを取得し、JSONファイルとして保存する。

        Args:
            matchids: matchidリスト (例: ["JP1_123456", "JP1_789012"])
            output_dir: 保存先ディレクトリ (例: "data/match")
        """
        os.makedirs(output_dir, exist_ok=True)

        nums = self.get_matchids2(matchids)
        print(f"Total match_ids: {len(nums)}")

        # 既存ファイルをスキップ
        to_process = []
        for mid in nums:
            json_path = os.path.join(
                output_dir, f"{self._config.region_prefix}{mid}.json"
            )
            if not os.path.exists(json_path):
                to_process.append(mid)

        print(
            f"Already saved: {len(nums) - len(to_process)}, "
            f"To process: {len(to_process)}"
        )

        if not to_process:
            print("All match JSONs already exist.")
            return

        completed = 0
        errors = 0

        for mid in tqdm(to_process, desc="Saving match data"):
            url = (
                f"{self._config.regional_url}/lol/match/v5/matches/"
                f"{self._config.region_prefix}{mid}"
            )
            match_data = self._request_with_retry(url)
            if match_data and "info" in match_data:
                json_path = os.path.join(
                    output_dir, f"{self._config.region_prefix}{mid}.json"
                )
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(match_data, f, ensure_ascii=False, indent=2)
                completed += 1
            else:
                errors += 1

        print(f"Completed! Saved: {completed}, Errors: {errors}")

    # ------------------------------------------------------------------
    # Champion name fetching (internal helper)
    # ------------------------------------------------------------------
    def _fetch_champnames(self, match_id: str) -> list[str]:
        """1試合分のチャンピオン名リストを取得する (_request_with_retry使用)。"""
        matchid_api = match_id.replace("-", "_")
        url = (
            f"{self._config.regional_url}/lol/match/v5/matches/{matchid_api}"
        )
        data = self._request_with_retry(url)
        if data is None or "info" not in data:
            raise KeyError(f"Failed to fetch champion names for {matchid_api}")
        return [p["championName"] for p in data["info"]["participants"]]

    # ------------------------------------------------------------------
    # Champion names from replays
    # ------------------------------------------------------------------
    def save_champnames_from_matches_with_rofl(
        self,
        csv_save_folder_dir: str,
        csv_name: str,
        replay_dir: str,
        max_workers: int = 1,
    ) -> None:
        """replay(.rofl)ファイルが存在する場合:
        ファイル名からmatchidを抽出 -> Riot APIで10チャンピオン名取得 -> CSVに保存

        Args:
            csv_save_folder_dir: CSV保存先ディレクトリ
            csv_name: CSVファイル名
            replay_dir: リプレイファイルディレクトリ
            max_workers: 並列スレッド数 (レート制限対策で1推奨)
        """
        # -- 出力ディレクトリ準備 --
        if csv_save_folder_dir and not os.path.exists(csv_save_folder_dir):
            os.makedirs(csv_save_folder_dir, exist_ok=True)
        csv_file = os.path.join(csv_save_folder_dir, csv_name)

        # -- CSVヘッダー (初回のみ) --
        if not os.path.exists(csv_file):
            with open(csv_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    ["match_id"] + [f"champion_{i+1}" for i in range(10)]
                )

        # -- リプレイファイルリスト --
        replay_files = [
            fn for fn in os.listdir(replay_dir) if fn.lower().endswith(".rofl")
        ]
        match_ids = [os.path.splitext(fn)[0] for fn in replay_files]

        # -- 並列取得 + プログレスバー --
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(self._fetch_champnames, mid): mid
                for mid in match_ids
            }

            with open(csv_file, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)

                for fut in tqdm(
                    as_completed(futures),
                    total=len(match_ids),
                    desc="Fetching match data",
                ):
                    mid = futures[fut]
                    try:
                        champs = fut.result()
                        if len(champs) == 10:
                            writer.writerow([mid] + champs)
                        else:
                            print(f"  {mid}: champion count {len(champs)}")
                    except Exception as e:
                        print(f"  {mid} processing failed: {e}")

    # ------------------------------------------------------------------
    # Champion names from CSV (without rofl)
    # ------------------------------------------------------------------
    def save_champnames_from_matches_without_rofl(
        self,
        input_csv_path: str,
        output_csv_path: str,
        max_workers: int = 1,
        request_interval: float = 1.3,
    ) -> None:
        """download_replays.ipynbをスキップする場合:
        CSVからmatchidを読み込み、Riot APIで10チャンピオンを取得してCSVに保存。

        Args:
            input_csv_path: matchidカラムを含む入力CSVパス
            output_csv_path: チャンピオンデータの出力CSVパス
            max_workers: 並列スレッド数 (1推奨)
            request_interval: APIリクエスト間隔(秒) (デフォルト1.3s)
        """
        # -- 内部関数 (レート制限対応) --
        def get_champnames_per_match(
            matchid: str, max_retries: int = 5
        ) -> list[str]:
            time.sleep(request_interval)

            url = (
                f"{self._config.regional_url}/lol/match/v5/matches/{matchid}"
            )
            for attempt in range(max_retries):
                resp = requests.get(
                    url,
                    params={"api_key": self._config.api_key},
                    timeout=self._config.timeout,
                )

                if resp.status_code == 429:
                    wait = int(resp.headers.get("Retry-After", 5))
                    time.sleep(wait + 1)
                    continue

                if 500 <= resp.status_code < 600:
                    time.sleep(2)
                    continue

                data = resp.json()

                if "info" not in data:
                    raise KeyError(
                        f"status={resp.status_code}, response={data}"
                    )

                return [p["championName"] for p in data["info"]["participants"]]

            raise Exception(f"Max retries exceeded for {matchid}")

        # -- 出力ディレクトリ準備 --
        out_dir = os.path.dirname(output_csv_path)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)

        # -- matchidリスト読み込み --
        matchids = pd.read_csv(input_csv_path)["matchid"].tolist()

        # -- 結果CSVヘッダー (初回のみ) --
        if not os.path.exists(output_csv_path):
            with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    ["match_id"] + [f"champion_{i+1}" for i in range(10)]
                )

        # -- 並列収集 + プログレスバー --
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(get_champnames_per_match, mid): mid
                for mid in matchids
            }

            with open(
                output_csv_path, "a", newline="", encoding="utf-8"
            ) as f:
                writer = csv.writer(f)

                for fut in tqdm(
                    as_completed(futures),
                    total=len(matchids),
                    desc="Fetching match data",
                ):
                    match_id = futures[fut]
                    try:
                        champs = fut.result()
                        if len(champs) == 10:
                            writer.writerow([match_id] + champs)
                        else:
                            print(
                                f"  {match_id}: champion count {len(champs)}"
                            )
                    except Exception as e:
                        print(f"  {match_id} processing failed: {e}")

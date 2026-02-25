"""Riot API共通基盤クラス

DataGenerator と MatchFetcher が共有するHTTPリクエスト処理を提供する。
- 429 (Rate Limit) / 5xx (Server Error) の自動リトライ
- リージョン→ルーティング解決
- API認証
"""
from __future__ import annotations

import logging
import time
from abc import ABC
from dataclasses import dataclass, field
from typing import Any, Optional

import requests

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RiotApiConfig:
    """Riot API接続設定

    Attributes:
        api_key: Riot API キー
        region: プラットフォームリージョン (例: "JP1", "KR", "NA1")
        request_interval: リクエスト間隔(秒) - 100req/120s制限対策
        max_retries: 最大リトライ回数
        timeout: HTTPタイムアウト(秒)
    """

    api_key: str
    region: str = "JP1"
    request_interval: float = 1.3
    max_retries: int = 5
    timeout: int = 10

    REGION_TO_ROUTING: dict[str, str] = field(
        default_factory=lambda: {
            "JP1": "asia", "KR": "asia",
            "NA1": "americas", "BR1": "americas",
            "LA1": "americas", "LA2": "americas",
            "EUW1": "europe", "EUN1": "europe",
            "TR1": "europe", "RU": "europe",
            "OC1": "sea", "PH2": "sea", "SG2": "sea",
            "TH2": "sea", "TW2": "sea", "VN2": "sea",
        },
        repr=False,
    )

    @property
    def routing(self) -> str:
        """リージョンに対応するルーティング値 (例: JP1 -> asia)"""
        return self.REGION_TO_ROUTING.get(self.region.upper(), "asia")

    @property
    def platform_url(self) -> str:
        """プラットフォームAPI URL (例: https://jp1.api.riotgames.com)"""
        return f"https://{self.region.lower()}.api.riotgames.com"

    @property
    def regional_url(self) -> str:
        """リージョナルAPI URL (例: https://asia.api.riotgames.com)"""
        return f"https://{self.routing}.api.riotgames.com"

    @property
    def region_prefix(self) -> str:
        """マッチIDプレフィックス (例: JP1_)"""
        return f"{self.region.upper()}_"


class BaseRiotClient(ABC):
    """Riot API呼び出しの共通基盤クラス

    generator.py の api_request_with_retry() と riotapi.py の各種リトライロジックを
    統合した単一の _request_with_retry() メソッドを提供する。

    サブクラス:
        - DataGenerator: matchId収集
        - MatchFetcher: match/timelineデータ取得
    """

    def __init__(self, config: RiotApiConfig) -> None:
        self._config = config

    @property
    def config(self) -> RiotApiConfig:
        return self._config

    def _request_with_retry(
        self,
        url: str,
        params: Optional[dict[str, str]] = None,
    ) -> Optional[Any]:
        """429/5xx自動リトライ付きGETリクエスト

        Args:
            url: リクエスト先URL
            params: クエリパラメータ (api_keyは自動付与)

        Returns:
            成功時: レスポンスJSON
            失敗時: None
        """
        if params is None:
            params = {}
        params.setdefault("api_key", self._config.api_key)

        for attempt in range(self._config.max_retries):
            try:
                resp = requests.get(
                    url, params=params, timeout=self._config.timeout,
                )

                if resp.status_code == 429:
                    wait = int(resp.headers.get("Retry-After", 10))
                    logger.warning(
                        "[429] Rate limit. Waiting %ds (attempt %d/%d)",
                        wait, attempt + 1, self._config.max_retries,
                    )
                    time.sleep(wait + 1)
                    continue

                if 500 <= resp.status_code < 600:
                    logger.warning(
                        "[%d] Server error. Retrying... (attempt %d/%d)",
                        resp.status_code, attempt + 1, self._config.max_retries,
                    )
                    time.sleep(2)
                    continue

                if resp.status_code == 200:
                    time.sleep(self._config.request_interval)
                    return resp.json()

                logger.error(
                    "[%d] Unexpected status for %s", resp.status_code, url,
                )
                return None

            except requests.RequestException as e:
                logger.error("Request failed: %s", e)
                time.sleep(2)

        logger.error(
            "Max retries (%d) exceeded for %s",
            self._config.max_retries, url,
        )
        return None

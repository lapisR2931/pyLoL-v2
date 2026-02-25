"""後方互換ファサード - 既存コードからの import を維持する

旧: from autoLeague.dataset.riotapi import RiotAPI
新: from autoLeague.dataset.match_fetcher import MatchFetcher (推奨)
"""
from __future__ import annotations

from autoLeague.dataset._legacy import RiotAPILegacy
from autoLeague.dataset.match_fetcher import MatchFetcher


class RiotAPI:
    """後方互換ファサード

    全メソッドをRiotAPILegacyとMatchFetcherに委譲する。
    既存のnotebook/スクリプトはインポートを変えずに動作する。

    新規コードではMatchFetcherの直接使用を推奨:
        from autoLeague.dataset.match_fetcher import MatchFetcher
    """

    # クラス属性: 旧コードが RiotAPI.REGION_TO_ROUTING を参照する場合に対応
    REGION_TO_ROUTING = {
        "JP1": "asia", "KR": "asia",
        "NA1": "americas", "BR1": "americas",
        "LA1": "americas", "LA2": "americas",
        "EUW1": "europe", "EUN1": "europe",
        "TR1": "europe", "RU": "europe",
        "OC1": "sea", "PH2": "sea", "SG2": "sea",
        "TH2": "sea", "TW2": "sea", "VN2": "sea",
    }

    def __init__(self, api_key: str, region: str = "JP1") -> None:
        self._fetcher = MatchFetcher(api_key=api_key, region=region)
        self._legacy = RiotAPILegacy(api_key=api_key, region=region)

        # 後方互換: 旧コードが self.api_key 等を直接参照する場合に対応
        self.api_key = api_key
        self.region = region.upper()
        self.routing = self._fetcher.config.routing
        self.region_prefix = self._fetcher.config.region_prefix

    # staticmethod を明示的に公開 (RiotAPI._format_timestamp_ms 参照に対応)
    _format_timestamp_ms = staticmethod(MatchFetcher._format_timestamp_ms)
    transformList = staticmethod(RiotAPILegacy.transformList)
    subtractList = staticmethod(RiotAPILegacy.subtractList)
    filter_matches_by_excluded_champions = staticmethod(
        RiotAPILegacy.filter_matches_by_excluded_champions
    )
    find_disjoint_matches_combinations = staticmethod(
        RiotAPILegacy.find_disjoint_matches_combinations
    )
    find_disjoint_matches_combinations_with_initial = staticmethod(
        RiotAPILegacy.find_disjoint_matches_combinations_with_initial
    )

    def __getattr__(self, name: str):
        """MatchFetcher -> RiotAPILegacy の順で属性を探索する。"""
        # __getattr__ は通常の属性解決で見つからない場合のみ呼ばれる
        try:
            return getattr(self._fetcher, name)
        except AttributeError:
            return getattr(self._legacy, name)

"""DataGenerator のテスト"""
from __future__ import annotations

import responses

from autoLeague.base.riot_client import BaseRiotClient, RiotApiConfig
from autoLeague.dataset.generator import DataGenerator


class TestDataGeneratorInit:
    def test_backward_compat_init(self):
        """既存の DataGenerator(api_key, count) シグネチャが動作する"""
        dg = DataGenerator(api_key="TEST-KEY", count=50)
        assert dg.api_key == "TEST-KEY"
        assert dg.count == 50

    def test_inherits_base_riot_client(self):
        assert issubclass(DataGenerator, BaseRiotClient)

    def test_default_region(self):
        dg = DataGenerator(api_key="TEST-KEY", count=10)
        assert dg.config.region == "JP1"

    def test_custom_region(self):
        dg = DataGenerator(api_key="TEST-KEY", count=10, region="KR")
        assert dg.config.region == "KR"
        assert dg.config.routing == "asia"

    def test_config_accessible(self):
        dg = DataGenerator(api_key="TEST-KEY", count=10)
        assert isinstance(dg.config, RiotApiConfig)
        assert dg.config.api_key == "TEST-KEY"


class TestDataGeneratorMethods:
    def test_is_in_recent_patch(self):
        dg = DataGenerator(api_key="TEST", count=10)
        # 2026-06-15 12:00:00 UTC = 1781524800000 ms
        # Use dates with large margins to avoid timezone sensitivity
        ts = 1781524800000
        assert dg.is_in_recent_patch(ts, "2026.01.01") is True
        assert dg.is_in_recent_patch(ts, "2026.12.01") is False

    @responses.activate
    def test_get_puuids_empty_response(self):
        """空のレスポンスで空リストを返す"""
        responses.add(
            responses.GET,
            "https://jp1.api.riotgames.com/lol/league-exp/v4/entries/RANKED_SOLO_5x5/GOLD/I",
            json=[],
            status=200,
        )
        dg = DataGenerator(api_key="TEST", count=10, request_interval=0.0)
        result = dg.get_puuids("RANKED_SOLO_5x5", "GOLD", "I")
        assert result == []

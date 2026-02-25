"""MatchFetcher のテスト"""
from __future__ import annotations

import responses

from autoLeague.base.riot_client import BaseRiotClient, RiotApiConfig
from autoLeague.dataset.match_fetcher import MatchFetcher


class TestMatchFetcherInit:
    def test_inherits_base_riot_client(self):
        assert issubclass(MatchFetcher, BaseRiotClient)

    def test_init(self):
        mf = MatchFetcher(api_key="TEST-KEY", region="KR")
        assert mf.api_key == "TEST-KEY"
        assert mf.region == "KR"
        assert mf.routing == "asia"
        assert mf.region_prefix == "KR_"


class TestMatchFetcherMethods:
    def test_get_matchids_strips_prefix(self):
        mf = MatchFetcher(api_key="TEST", region="JP1")
        result = mf.get_matchids(["JP1-123", "JP1-456", "KR-789"])
        # Should only include JP1 matches, stripped of prefix
        assert "123" in result
        assert "456" in result
        assert "789" not in result

    def test_get_matchids2_strips_prefix(self):
        mf = MatchFetcher(api_key="TEST", region="JP1")
        result = mf.get_matchids2(["JP1_123", "JP1_456", "KR_789"])
        assert "123" in result
        assert "456" in result
        assert "789" not in result

    def test_format_timestamp_ms(self):
        assert MatchFetcher._format_timestamp_ms(0) == "0:00"
        assert MatchFetcher._format_timestamp_ms(60000) == "1:00"
        assert MatchFetcher._format_timestamp_ms(90000) == "1:30"
        assert MatchFetcher._format_timestamp_ms(600000) == "10:00"

    @responses.activate
    def test_get_puuid(self):
        responses.add(
            responses.GET,
            "https://jp1.api.riotgames.com/lol/summoner/v4/summoners/summ123",
            json={"puuid": "puuid-abc-123"},
            status=200,
        )
        mf = MatchFetcher(api_key="TEST", region="JP1", request_interval=0.0)
        result = mf.get_puuid("summ123")
        assert result == "puuid-abc-123"

    @responses.activate
    def test_fetch_timeline(self):
        responses.add(
            responses.GET,
            "https://asia.api.riotgames.com/lol/match/v5/matches/JP1_12345/timeline",
            json={"info": {"frames": []}},
            status=200,
        )
        mf = MatchFetcher(api_key="TEST", region="JP1", request_interval=0.0)
        result = mf._get_match_timeline_data("12345")
        assert "info" in result

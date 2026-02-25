"""BaseRiotClient + RiotApiConfig のテスト"""
from __future__ import annotations

import responses

from autoLeague.base.riot_client import BaseRiotClient, RiotApiConfig


class ConcreteRiotClient(BaseRiotClient):
    """テスト用の具象クラス"""
    pass


class TestRiotApiConfig:
    def test_defaults(self):
        cfg = RiotApiConfig(api_key="TEST")
        assert cfg.region == "JP1"
        assert cfg.request_interval == 1.3
        assert cfg.max_retries == 5
        assert cfg.timeout == 10

    def test_routing_jp1(self):
        cfg = RiotApiConfig(api_key="TEST", region="JP1")
        assert cfg.routing == "asia"

    def test_routing_na1(self):
        cfg = RiotApiConfig(api_key="TEST", region="NA1")
        assert cfg.routing == "americas"

    def test_routing_euw1(self):
        cfg = RiotApiConfig(api_key="TEST", region="EUW1")
        assert cfg.routing == "europe"

    def test_platform_url(self):
        cfg = RiotApiConfig(api_key="TEST", region="KR")
        assert cfg.platform_url == "https://kr.api.riotgames.com"

    def test_regional_url(self):
        cfg = RiotApiConfig(api_key="TEST", region="KR")
        assert cfg.regional_url == "https://asia.api.riotgames.com"

    def test_region_prefix(self):
        cfg = RiotApiConfig(api_key="TEST", region="JP1")
        assert cfg.region_prefix == "JP1_"

    def test_frozen(self):
        cfg = RiotApiConfig(api_key="TEST")
        try:
            cfg.api_key = "OTHER"
            assert False, "Should be frozen"
        except AttributeError:
            pass


class TestRequestWithRetry:
    @responses.activate
    def test_success_200(self, riot_config):
        responses.add(
            responses.GET, "https://example.com/api",
            json={"data": "ok"}, status=200,
        )
        client = ConcreteRiotClient(riot_config)
        result = client._request_with_retry("https://example.com/api")
        assert result == {"data": "ok"}

    @responses.activate
    def test_429_retry_then_success(self, riot_config):
        responses.add(
            responses.GET, "https://example.com/api",
            status=429, headers={"Retry-After": "0"},
        )
        responses.add(
            responses.GET, "https://example.com/api",
            json={"data": "ok"}, status=200,
        )
        client = ConcreteRiotClient(riot_config)
        result = client._request_with_retry("https://example.com/api")
        assert result == {"data": "ok"}

    @responses.activate
    def test_max_retries_exceeded(self, riot_config):
        for _ in range(riot_config.max_retries):
            responses.add(
                responses.GET, "https://example.com/api",
                status=429, headers={"Retry-After": "0"},
            )
        client = ConcreteRiotClient(riot_config)
        result = client._request_with_retry("https://example.com/api")
        assert result is None

    @responses.activate
    def test_500_retry_then_success(self, riot_config):
        responses.add(
            responses.GET, "https://example.com/api",
            status=500,
        )
        responses.add(
            responses.GET, "https://example.com/api",
            json={"data": "recovered"}, status=200,
        )
        client = ConcreteRiotClient(riot_config)
        result = client._request_with_retry("https://example.com/api")
        assert result == {"data": "recovered"}

    @responses.activate
    def test_404_returns_none(self, riot_config):
        responses.add(
            responses.GET, "https://example.com/api",
            status=404,
        )
        client = ConcreteRiotClient(riot_config)
        result = client._request_with_retry("https://example.com/api")
        assert result is None

    @responses.activate
    def test_api_key_auto_added(self, riot_config):
        responses.add(
            responses.GET, "https://example.com/api",
            json={"ok": True}, status=200,
        )
        client = ConcreteRiotClient(riot_config)
        client._request_with_retry("https://example.com/api")
        assert f"api_key={riot_config.api_key}" in responses.calls[0].request.url


class TestInheritance:
    def test_data_generator_is_riot_client(self):
        from autoLeague.dataset.generator import DataGenerator
        assert issubclass(DataGenerator, BaseRiotClient)

    def test_match_fetcher_is_riot_client(self):
        from autoLeague.dataset.match_fetcher import MatchFetcher
        assert issubclass(MatchFetcher, BaseRiotClient)

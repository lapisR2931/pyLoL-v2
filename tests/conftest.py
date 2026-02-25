"""共通テストフィクスチャ"""
from __future__ import annotations

import sys

import pytest

from autoLeague.base.lcu_client import LcuConfig
from autoLeague.base.riot_client import RiotApiConfig
from autoLeague.config import MinimapConfig, ReplayConfig

is_windows = sys.platform == "win32"
requires_windows = pytest.mark.skipif(not is_windows, reason="Windows only")
requires_lol_client = pytest.mark.skip(reason="Requires LoL client running")


@pytest.fixture
def riot_config() -> RiotApiConfig:
    """テスト用 RiotApiConfig (待機時間ゼロ)"""
    return RiotApiConfig(
        api_key="RGAPI-test-key-00000000",
        region="JP1",
        request_interval=0.0,
        max_retries=2,
        timeout=5,
    )


@pytest.fixture
def lcu_config() -> LcuConfig:
    """テスト用 LcuConfig (固定値)"""
    return LcuConfig(
        lcu_port="12345",
        lcu_token="test-token-abc",
    )


@pytest.fixture
def minimap_config() -> MinimapConfig:
    return MinimapConfig()


@pytest.fixture
def replay_config() -> ReplayConfig:
    return ReplayConfig()

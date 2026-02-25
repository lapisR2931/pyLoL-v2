"""Config dataclasses のテスト"""
from __future__ import annotations

from autoLeague.config import MinimapConfig, ReplayConfig


class TestMinimapConfig:
    def test_defaults(self):
        cfg = MinimapConfig()
        assert cfg.size == 512
        assert cfg.screen_width == 1920
        assert cfg.screen_height == 1080
        assert cfg.margin_right == 21
        assert cfg.margin_bottom == 22

    def test_top_property(self):
        cfg = MinimapConfig()
        # 1080 - 512 - 22 = 546
        assert cfg.top == 546

    def test_left_property(self):
        cfg = MinimapConfig()
        # 1920 - 512 - 21 = 1387
        assert cfg.left == 1387

    def test_monitor_dict(self):
        cfg = MinimapConfig()
        expected = {
            "top": 546,
            "left": 1387,
            "width": 512,
            "height": 512,
        }
        assert cfg.monitor == expected

    def test_custom_resolution(self):
        cfg = MinimapConfig(screen_width=2560, screen_height=1440)
        assert cfg.top == 1440 - 512 - 22
        assert cfg.left == 2560 - 512 - 21

    def test_frozen(self):
        cfg = MinimapConfig()
        try:
            cfg.size = 256
            assert False, "Should be frozen"
        except AttributeError:
            pass


class TestReplayConfig:
    def test_defaults(self):
        cfg = ReplayConfig()
        assert cfg.default_speed == 8
        assert cfg.capture_interval == 0.047
        assert cfg.process_timeout == 60
        assert cfg.api_timeout == 60
        assert cfg.max_game_duration == 45 * 60

"""autoLeague 共通設定

各モジュールに散在していたハードコード定数を集約する。
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MinimapConfig:
    """ミニマップキャプチャ座標設定

    scraper.py L653-657 に散在していた定数を集約。
    1920x1080 解像度、MinimapScale=1.8 を前提とする。
    """

    size: int = 512
    screen_width: int = 1920
    screen_height: int = 1080
    margin_right: int = 21
    margin_bottom: int = 22

    @property
    def top(self) -> int:
        return self.screen_height - self.size - self.margin_bottom

    @property
    def left(self) -> int:
        return self.screen_width - self.size - self.margin_right

    @property
    def monitor(self) -> dict[str, int]:
        """mss用キャプチャ領域辞書"""
        return {
            "top": self.top,
            "left": self.left,
            "width": self.size,
            "height": self.size,
        }


@dataclass(frozen=True)
class ReplayConfig:
    """リプレイ再生設定

    scraper.py に散在していたタイムアウト値・速度設定を集約。
    """

    default_speed: int = 8
    capture_interval: float = 0.047
    process_timeout: int = 60
    api_timeout: int = 60
    max_game_duration: int = 45 * 60

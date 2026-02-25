"""autoLeague - League of Legends データ収集・視界スコア分析ライブラリ"""

__version__ = "2.0.0"

from .base import BaseRiotClient, BaseLcuClient, RiotApiConfig, LcuConfig
from .config import MinimapConfig, ReplayConfig

__all__ = [
    "__version__",
    "BaseRiotClient",
    "BaseLcuClient",
    "RiotApiConfig",
    "LcuConfig",
    "MinimapConfig",
    "ReplayConfig",
]

"""LCU API共通基盤クラス

ReplayDownloader と ReplayScraper が共有するLCU認証取得処理を提供する。
- LeagueClientUx.exe プロセスからの認証情報自動取得
- LCU API認証ヘッダー生成
"""
from __future__ import annotations

import base64
import logging
import subprocess
from abc import ABC
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LcuConfig:
    """LCU API接続設定

    Attributes:
        lcu_port: LCUプロセスのポート番号
        lcu_token: LCU認証トークン
        replay_api_port: Replay API (ゲーム内API) のポート番号
        concurrent_downloads: 並列ダウンロード数
    """

    lcu_port: Optional[str] = None
    lcu_token: Optional[str] = None
    replay_api_port: int = 2999
    concurrent_downloads: int = 6

    @property
    def lcu_base_url(self) -> str:
        """LCU API ベースURL"""
        return f"https://127.0.0.1:{self.lcu_port}"

    @property
    def replay_api_base_url(self) -> str:
        """Replay API ベースURL (ゲーム内、port 2999)"""
        return f"https://127.0.0.1:{self.replay_api_port}"

    @property
    def auth_header(self) -> str:
        """LCU API用Basic認証ヘッダー値"""
        return "Basic " + base64.b64encode(
            f"riot:{self.lcu_token}".encode()
        ).decode()


class BaseLcuClient(ABC):
    """LoLクライアント (LCU API) 操作の共通基盤クラス

    downloader.py と scraper.py に重複していたPowerShellによる
    LCU認証情報取得ロジックを統合する。

    サブクラス:
        - ReplayDownloader: .roflファイルのダウンロード
        - ReplayScraper: リプレイ再生・ミニマップキャプチャ
    """

    def __init__(self, config: Optional[LcuConfig] = None) -> None:
        if config and config.lcu_port and config.lcu_token:
            self._config = config
        else:
            port, token = self._extract_credentials()
            self._config = LcuConfig(
                lcu_port=port,
                lcu_token=token,
                **({"replay_api_port": config.replay_api_port,
                    "concurrent_downloads": config.concurrent_downloads}
                   if config else {}),
            )

    @property
    def config(self) -> LcuConfig:
        return self._config

    @staticmethod
    def _extract_credentials() -> tuple[Optional[str], Optional[str]]:
        """LeagueClientUx.exe プロセスからLCU認証情報を取得

        PowerShell の Get-CimInstance を使用してプロセスのコマンドライン引数から
        remoting-auth-token と app-port を抽出する。

        Returns:
            (port, token) のタプル。取得失敗時は (None, None)。
        """
        try:
            command = [
                "powershell", "-Command",
                "Get-CimInstance -Query "
                "\"Select CommandLine from Win32_Process "
                "Where Name='LeagueClientUx.exe'\""
            ]
            proc = subprocess.Popen(
                command, shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
            )
            output, error = proc.communicate(timeout=10)

            if error or not output.strip():
                logger.warning("LoL client not detected.")
                return None, None

            token, port = None, None
            for segment in output.strip().split('"'):
                if "remoting-auth-token" in segment:
                    token = segment.split("=")[1]
                elif "app-port" in segment:
                    port = segment.split("=")[1]

            if token and port:
                logger.info("LCU credentials acquired (port=%s)", port)
            return port, token

        except Exception as e:
            logger.error("Failed to extract LCU credentials: %s", e)
            return None, None

from __future__ import annotations

import asyncio
import json
import ssl
from typing import Iterable, Optional

import aiohttp
import requests

from autoLeague.base import BaseLcuClient, LcuConfig


class ReplayDownloader(BaseLcuClient):

    def __init__(
        self,
        lcu_port: Optional[str] = None,
        lcu_token: Optional[str] = None,
        concurrent: int = 6,
    ) -> None:
        config = LcuConfig(
            lcu_port=lcu_port,
            lcu_token=lcu_token,
            concurrent_downloads=concurrent,
        )
        super().__init__(config)
        self.replays_dir: Optional[str] = None
        if self._config.lcu_token:
            print(f"remoting-auth-token : {self._config.lcu_token}")

    # ------------------------------------------------------------------
    # 後方互換プロパティ
    # ------------------------------------------------------------------
    @property
    def port(self) -> Optional[str]:
        return self._config.lcu_port

    @property
    def token(self) -> Optional[str]:
        return self._config.lcu_token

    # ------------------------------------------------------------------
    # リプレイダウンロード位置設定
    # ------------------------------------------------------------------
    def set_replays_dir(self, folder_dir: str) -> None:
        """リプレイダウンロード位置設定"""
        requests.packages.urllib3.disable_warnings(requests.packages.urllib3.exceptions.InsecureRequestWarning)
        requests.patch(
            f'{self._config.lcu_base_url}/lol-settings/v1/local/lol-replays',
            headers={
                'Accept': 'application/json',
                'Content-Type': 'application/json'
            },
            data=json.dumps({
                "replays-folder-path": folder_dir
            }),
            verify=False
        )

        self.replays_dir = folder_dir

    # ------------------------------------------------------------------
    # リプレイファイル(gameId.rofl)ダウンロード
    # ------------------------------------------------------------------
    def download(self, gameId: str) -> None:
        """リプレイファイル(gameId.rofl)ダウンロード"""
        requests.packages.urllib3.disable_warnings(requests.packages.urllib3.exceptions.InsecureRequestWarning)
        # リージョンプレフィックス（KR_, JP1_, NA1_など）を除去して数値IDのみ取得
        numeric_id = gameId.split("_")[-1] if "_" in gameId else gameId
        url = f'{self._config.lcu_base_url}/lol-replays/v1/rofls/{numeric_id}/download/graceful'
        requests.post(
            url,
            headers={
                'Accept': 'application/json',
                'Content-Type': 'application/json',
                'Authorization': self._config.auth_header
            },
            json={'componentType': 'string'},
            verify=False
        )

    # ────────────────────────────────────
    # 非同期(並列) API
    # ────────────────────────────────────
    async def _download_one(self, session: aiohttp.ClientSession, game_id: str) -> None:
        """内部用・単一POST"""
        # リージョンプレフィックス（KR_, JP1_, NA1_など）を除去して数値IDのみ取得
        numeric_id = game_id.split("_")[-1] if "_" in game_id else game_id
        url = (f"{self._config.lcu_base_url}"
               f"/lol-replays/v1/rofls/{numeric_id}/download/graceful")
        async with session.post(url, json={"componentType": "string"}) as resp:
            if resp.status not in (200, 202, 204):
                err = await resp.text()
                print(f"[{game_id}] 失敗 {resp.status}: {err}")

    async def download_async(
        self,
        game_ids: Iterable[str],
        concurrent: Optional[int] = None,
    ) -> None:
        """
        複数のmatchId iterableを並列でダウンロード。
        * Jupyterセル :   await rd.download_async(ids)
        * スクリプト  :   asyncio.run(rd.download_async(ids))
        """
        concurrent = concurrent or self._config.concurrent_downloads

        ssl_ctx = ssl.create_default_context()
        ssl_ctx.check_hostname = False
        ssl_ctx.verify_mode = ssl.CERT_NONE

        conn = aiohttp.TCPConnector(limit=concurrent, ssl=ssl_ctx)
        async with aiohttp.ClientSession(
            connector=conn,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
                "Authorization": self._config.auth_header,
            }) as session:

            sem = asyncio.Semaphore(concurrent)

            async def sem_task(gid: str) -> None:
                async with sem:
                    await self._download_one(session, gid)

            await asyncio.gather(*(sem_task(g) for g in game_ids))


# NOTE: Auto-instantiation removed to prevent import errors when LoL client is not running.
# Create instance explicitly when needed:
#   from autoLeague.dataset.downloader import ReplayDownloader
#   replay_downloader = ReplayDownloader()
# ─────────────────────────────────────────────────────────────

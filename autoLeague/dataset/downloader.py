import requests
import base64
import subprocess
import json
import asyncio, aiohttp, ssl

class ReplayDownloader(object):

    def __init__(self):
        self.replays_dir = None #リプレイ保存ディレクトリ
        self.token = None       #LoLクライアント認証トークン
        self.port = None        #LoLクライアントプロセスのポート番号

        # 現在Windowsのみ対応
        # wmic PROCESS WHERE name='LeagueClientUx.exe' GET commandline
        # PowerShellコマンドリストの構成
        command = [
            "powershell",
            "-Command",
            "Get-WmiObject -Query \"Select CommandLine from Win32_Process Where Name='LeagueClientUx.exe'\""
        ]
        process =  subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        output, error = process.communicate()

        if error:
            raise ProcessLookupError("⚠️ LoLクライアントが実行中か確認してください。")
        else:
            cmd = output.strip().split('"')
            for i in cmd:
                if i.find("remoting-auth-token") != -1:
                    self.token = i.split("=")[1]
                elif i.find("app-port") != -1:
                    self.port = i.split("=")[1]

        print('remoting-auth-token :', self.token)
    '''リプレイダウンロード位置設定'''
    def set_replays_dir(self,folder_dir):
        requests.packages.urllib3.disable_warnings(requests.packages.urllib3.exceptions.InsecureRequestWarning)
        requests.patch(
                        f'https://127.0.0.1:{self.port}/lol-settings/v1/local/lol-replays',
                        headers={
                            'Accept': 'application/json',
                            'Content-Type': 'application/json'
                        },
                        data = json.dumps({
                            "replays-folder-path": folder_dir
                        }),
                        verify=False
                    )
        
        self.replays_dir = folder_dir


    '''リプレイファイル(gameId.rofl)ダウンロード'''
    def download(self,gameId):
        requests.packages.urllib3.disable_warnings(requests.packages.urllib3.exceptions.InsecureRequestWarning)
        # リージョンプレフィックス（KR_, JP1_, NA1_など）を除去して数値IDのみ取得
        numeric_id = gameId.split("_")[-1] if "_" in gameId else gameId
        url = f'https://127.0.0.1:{self.port}/lol-replays/v1/rofls/{numeric_id}/download/graceful'
        auth = 'Basic ' + base64.b64encode(f'riot:{self.token}'.encode()).decode()
        requests.post(
                        url,
                        headers={
                            'Accept': 'application/json',
                            'Content-Type': 'application/json',
                            'Authorization': auth
                        },
                        json={'componentType': 'string'},
                        verify=False
                    )
        
    # ────────────────────────────────────
    # 非同期(並列) API
    # ────────────────────────────────────
    async def _download_one(self, session: aiohttp.ClientSession, game_id: str):
        """内部用・単一POST"""
        # リージョンプレフィックス（KR_, JP1_, NA1_など）を除去して数値IDのみ取得
        numeric_id = game_id.split("_")[-1] if "_" in game_id else game_id
        url = (f"https://127.0.0.1:{self.port}"
               f"/lol-replays/v1/rofls/{numeric_id}/download/graceful")
        async with session.post(url, json={"componentType": "string"}) as resp:
            if resp.status not in (200, 202, 204):
                err = await resp.text()
                print(f"[{game_id}] 失敗 {resp.status}: {err}")

    async def download_async(self, game_ids, concurrent: int = 6):
        """
        複数のmatchId iterableを並列でダウンロード。
        • Jupyterセル :   await rd.download_async(ids)
        • スクリプト  :   asyncio.run(rd.download_async(ids))
        """
        ssl_ctx = ssl.create_default_context()
        ssl_ctx.check_hostname = False
        ssl_ctx.verify_mode = ssl.CERT_NONE

        auth = "Basic " + base64.b64encode(f"riot:{self.token}".encode()).decode()
        conn = aiohttp.TCPConnector(limit=concurrent, ssl=ssl_ctx)
        async with aiohttp.ClientSession(
            connector=conn,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
                "Authorization": auth,
            }) as session:

            sem = asyncio.Semaphore(concurrent)

            async def sem_task(gid):
                async with sem:
                    await self._download_one(session, gid)

            await asyncio.gather(*(sem_task(g) for g in game_ids))


# NOTE: Auto-instantiation removed to prevent import errors when LoL client is not running.
# Create instance explicitly when needed:
#   from autoLeague.dataset.downloader import ReplayDownloader
#   replay_downloader = ReplayDownloader()
# ─────────────────────────────────────────────────────────────

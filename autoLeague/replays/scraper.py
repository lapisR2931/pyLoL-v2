import os
import shutil
import time
import json
import subprocess
import base64
import requests
import pyautogui
import pydirectinput
import cv2
import numpy as np
import re
import csv
from ftplib import FTP
import mss
import mss.tools
from threading import Thread

TEAM_HOTKEY_DICT = {'Red' : 'f2', 'Blue' : 'f1', 'All' : 'f'}


class ReplayScraper(object):
    """League of Legends replay scraper class.

    This class handles executing the League of Legends client in
    replay mode and the scraping application in the correct order.
    Args:
        game_dir: League of Legends game directory.
        replay_dir: League of Legends *.rofl replay directory.
        save_dir: JSON replay files output directory.
        replay_speed: League of Legends client replay speed multiplier.
        scraper_path: Directory of the scraper program.
    """
    def __init__(self,
            game_dir,
            replay_dir,
            save_dir,
            scraper_dir,
            replay_speed=8,
            region="JP1",
            # LCU API settings - auto-detected from LoL client process
            lcu_port=None,
            lcu_token=None,
            # FTP settings - set via environment variables or pass explicitly
            ftp_server=None,      # Override with os.environ.get('FTP_SERVER')
            ftp_username=None,    # Override with os.environ.get('FTP_USERNAME')
            ftp_password=None,    # Override with os.environ.get('FTP_PASSWORD')
            local_folder_path=r'C:\dataset',
            remote_folder_path='/home/username'):

        self.game_dir = game_dir
        self.replay_dir = replay_dir
        self.save_dir = save_dir
        self.scraper_dir = scraper_dir
        self.replay_speed = replay_speed
        self.region = region

        # LCU API認証情報を取得
        if lcu_port and lcu_token:
            self.lcu_port = lcu_port
            self.lcu_token = lcu_token
        else:
            self._init_lcu_credentials()

        self.ftp_server = ftp_server
        self.ftp_username = ftp_username
        self.ftp_password = ftp_password
        self.local_folder_path = local_folder_path #r'C:\dataset\KR-7025942524\All\1_292.38525390625.png'  # ローカルファイルパス
        self.remote_folder_path = remote_folder_path # '/home/user/image.png'  # NAS内に保存されるパスとファイル名

        files = os.listdir(self.replay_dir)
        replays = [file for file in files if file.endswith(".rofl")]
        print("Current number of replays: ",len(replays))

    def _init_lcu_credentials(self):
        """LoLクライアントからLCU API認証情報を取得"""
        try:
            command = [
                "powershell",
                "-Command",
                "Get-WmiObject -Query \"Select CommandLine from Win32_Process Where Name='LeagueClientUx.exe'\""
            ]
            process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
            output, error = process.communicate()

            if error:
                print("⚠️ LoLクライアントが実行中か確認してください。")
                self.lcu_port = None
                self.lcu_token = None
                return

            cmd = output.strip().split('"')
            for i in cmd:
                if i.find("remoting-auth-token") != -1:
                    self.lcu_token = i.split("=")[1]
                elif i.find("app-port") != -1:
                    self.lcu_port = i.split("=")[1]

            print(f"LCU API initialized - port: {self.lcu_port}")
        except Exception as e:
            print(f"LCU API initialization failed: {e}")
            self.lcu_port = None
            self.lcu_token = None

    def _is_replay_process_running(self):
        """League of Legends.exe（リプレイクライアント）が起動中か確認"""
        try:
            command = [
                "powershell", "-Command",
                "Get-Process -Name 'League of Legends' -ErrorAction SilentlyContinue"
            ]
            process = subprocess.Popen(
                command, shell=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                universal_newlines=True
            )
            output, _ = process.communicate(timeout=5)
            return bool(output.strip())
        except Exception:
            return False

    def _get_game_length(self, timeout=10, debug=False):
        """Replay APIからゲーム長（秒）を取得

        Args:
            timeout: 最大待機時間（秒）
            debug: デバッグ出力を有効にするか

        Returns:
            float: ゲーム長（秒）、取得失敗時はNone
        """
        requests.packages.urllib3.disable_warnings(requests.packages.urllib3.exceptions.InsecureRequestWarning)
        start_time = time.time()

        # 複数のエンドポイントと可能なフィールド名を試す
        endpoints = [
            ('https://127.0.0.1:2999/replay/game', ['length', 'gameLength', 'gameDuration']),
            ('https://127.0.0.1:2999/replay/playback', ['length', 'gameLength', 'gameDuration']),
        ]

        while time.time() - start_time < timeout:
            for url, field_names in endpoints:
                try:
                    resp = requests.get(url, verify=False, timeout=2)
                    if resp.status_code == 200:
                        data = resp.json()
                        if debug:
                            print(f"DEBUG {url}: {data}")
                        # 複数のフィールド名を試す
                        for field in field_names:
                            if field in data and data[field]:
                                return data[field]
                except (requests.RequestException, ConnectionError, OSError, ValueError):
                    pass
            time.sleep(1)

        return None

    def _wait_for_replay_ready(self, process_timeout=60, api_timeout=60):
        """リプレイクライアントの起動完了を待機

        2段階で確認:
        1. League of Legends.exeプロセスの起動
        2. Replay API (port 2999) の応答

        Args:
            process_timeout: プロセス起動の最大待機時間（秒）
            api_timeout: Replay API応答の最大待機時間（秒）

        Returns:
            bool: 起動完了したらTrue、タイムアウトでFalse
        """
        # Phase 1: プロセス起動待ち
        print("Waiting for replay client process...")
        start_time = time.time()
        while time.time() - start_time < process_timeout:
            if self._is_replay_process_running():
                print("Replay client process detected")
                break
            time.sleep(1)
        else:
            print("Timeout: Replay client process not found")
            return False

        # プロセス起動後、API準備までの追加待機
        print("Waiting for game to initialize...")
        time.sleep(10)

        # Phase 2: Replay API応答待ち（別タイムアウト）
        print("Waiting for Replay API...")
        requests.packages.urllib3.disable_warnings(requests.packages.urllib3.exceptions.InsecureRequestWarning)
        start_time = time.time()  # タイマーリセット
        while time.time() - start_time < api_timeout:
            try:
                resp = requests.get(
                    'https://127.0.0.1:2999/replay/playback',
                    verify=False, timeout=2
                )
                if resp.status_code == 200:
                    print("Replay API is ready")
                    return True
                else:
                    print(f"Replay API returned: {resp.status_code}")
            except (requests.RequestException, ConnectionError, OSError) as e:
                print(f"Replay API not ready: {type(e).__name__}")
            time.sleep(2)

        print("Timeout: Replay API not responding")
        return False

    def launch_replay_lcu(self, game_id):
        """LCU API経由でリプレイを起動"""
        if not self.lcu_port or not self.lcu_token:
            raise RuntimeError("LCU API credentials not available. Is LoL client running?")

        # game_idから数字のみを抽出 (JP1-551237207 → 551237207)
        numeric_id = game_id.split('-')[-1] if '-' in game_id else game_id
        numeric_id = numeric_id.split('_')[-1] if '_' in numeric_id else numeric_id

        url = f'https://127.0.0.1:{self.lcu_port}/lol-replays/v1/rofls/{numeric_id}/watch'
        auth = 'Basic ' + base64.b64encode(f'riot:{self.lcu_token}'.encode()).decode()

        requests.packages.urllib3.disable_warnings(requests.packages.urllib3.exceptions.InsecureRequestWarning)
        response = requests.post(
            url,
            headers={
                'Accept': 'application/json',
                'Content-Type': 'application/json',
                'Authorization': auth
            },
            json={"contextData": {"componentType": "replay-button_match-history"}},
            verify=False
        )

        if response.status_code in (200, 204):
            print(f"Replay launched: {game_id}")
            return True
        else:
            print(f"Failed to launch replay: {response.status_code} - {response.text}")
            return False
    # When running LoL
    def arg_list(self, replay_path):
        return [str(os.path.join(self.game_dir, "League of Legends.exe")),
                replay_path,
                "-SkipRads",
                "-SkipBuild",
                "-EnableLNP",
                "-UseNewX3D=1",
                "-UseNewX3DFramebuffers=1"]
    
    # For checking if LCU API call succeeded
    @staticmethod
    def post_initialize(paused, start, speed):
        return requests.post(
                        'https://127.0.0.1:2999/replay/playback',
                        headers={
                            'Accept': 'application/json',
                            'Content-Type': 'application/json'
                        },
                        data = json.dumps({
                            "paused": paused,
                            "seeking" : False,   
                            "time": start - 5,
                            "speed": speed
                        }),
                        verify=False
                    )
    
    @staticmethod
    def replay_view_initialize():
        requests.post(
                        'https://127.0.0.1:2999/replay/render',
                        headers={
                            'Accept': 'application/json',
                            'Content-Type': 'application/json'
                        },
                        data = json.dumps({'banners': True,
                                            'cameraAttached': False,
                                            'cameraLookSpeed': 1.0,
                                            'cameraMode': 'fps',
                                            'cameraMoveSpeed': 10000.0,
                                            'cameraPosition': {'x': 10585.7119140625, 'y': 57447.296875, 'z': 850},
                                            'cameraRotation': {'x': 347.90582275390625, 'y': 85.0, 'z': 3.0},
                                            'characters': False,
                                            'depthFogColor': {'a': 1.0, 'b': 0.0, 'g': 0.0, 'r': 0.0},
                                            'depthFogEnabled': False,
                                            'depthFogEnd': 8000.0,
                                            'depthFogIntensity': 1.0,
                                            'depthFogStart': 5000.0,
                                            'depthOfFieldCircle': 10.0,
                                            'depthOfFieldDebug': False,
                                            'depthOfFieldEnabled': False,
                                            'depthOfFieldFar': 5000.0,
                                            'depthOfFieldMid': 2000.0,
                                            'depthOfFieldNear': 0.0,
                                            'depthOfFieldWidth': 800.0,
                                            'environment': False,
                                            'farClip': 66010.0,
                                            'fieldOfView': 19.0,
                                            'floatingText': False,
                                            'fogOfWar': False, # Fog of war option is here
                                            'healthBarChampions': False,
                                            'healthBarMinions': False,
                                            'healthBarPets': False,
                                            'healthBarStructures': True,
                                            'healthBarWards': False,
                                            'heightFogColor': {'a': 1.0, 'b': 0.0, 'g': 0.0, 'r': 0.0},
                                            'heightFogEnabled': False,
                                            'heightFogEnd': -100.0,
                                            'heightFogIntensity': 1.0,
                                            'heightFogStart': 300.0,
                                            'interfaceAll': True,
                                            'interfaceAnnounce': False,
                                            'interfaceChat': False,
                                            'interfaceFrames': True,
                                            'interfaceKillCallouts': False,
                                            'interfaceMinimap': True,
                                            'interfaceNeutralTimers': False,
                                            'interfaceQuests': None,
                                            'interfaceReplay': True,
                                            'interfaceScore': True,
                                            'interfaceScoreboard': True,
                                            'interfaceTarget': False,
                                            'interfaceTimeline': False, #
                                            'navGridOffset': 0.0,
                                            'nearClip': 50.0,
                                            'outlineHover': False,
                                            'outlineSelect': True,
                                            'particles': False,
                                            'selectionName': '',
                                            'selectionOffset': {'x': 0.0, 'y': 0.0, 'z': 0.0},
                                            'skyboxOffset': 0.0,
                                            'skyboxPath': '',
                                            'skyboxRadius': 2500.0,
                                            'skyboxRotation': 0.0,
                                            'sunDirection': {'x': 0.31559720635414124,
                                            'y': -0.9467916488647461,
                                            'z': 0.06311944127082825}}),
                                            verify=False
                                            )

    

    """
    Save captured screen as image
    """
    @staticmethod
    def save_to_png(rgb, size, output):
        mss.tools.to_png(rgb, size, output=output)


    """
    Send full screenshots of 1 match to NAS
    """
    @staticmethod
    def send_to_nas(ftp_server, ftp_username, ftp_password, local_folder_path, remote_folder_path, gameId, timestamplist):
    
        # FTP server setup, start FTP session
        ftp = FTP(ftp_server)
        ftp.login(ftp_username, ftp_password)
        """
        File sorting
        """
        def list_all_files(base_dir):

            def extract_number(file_path):
                '''
                Function to extract numbers from filename
                '''
                filename = file_path.split("\\")[-1]
                number = re.search(r'\d+', filename).group()
                return int(number)
        
            file_paths = []
            for dirpath, dirnames, filenames in os.walk(base_dir):
                for filename in filenames:
                    file_path = os.path.join(dirpath, filename)
                    file_paths.append(file_path)

            sorted_file_paths = sorted(file_paths, key=extract_number)
            return sorted_file_paths

        # Load local files
        all_files = list_all_files(local_folder_path)

        print('local file size : ', len(all_files))
        # Create remote folder
        ftp.mkd(f'{remote_folder_path}/{gameId}')
        # Copy local files to remote folder
        for idx, file in enumerate(all_files):
        # Read file in binary mode and transmit
     
            with open(file, 'rb') as file:
                #ftp.storbinary(f"STOR {remote_folder_path}/{gameId}/{capture_count_list[idx]}_{timestamplist[idx]}.png", file)
                ftp.storbinary(f"STOR {remote_folder_path}/{gameId}/{timestamplist[idx]}.png", file)
        

        # Close connection
        ftp.quit()
        #print("파일 업로드 완료")

        # Delete local folder
        remain_folder_path = os.path.join(local_folder_path, os.listdir(local_folder_path)[0])
        """
        Delete local folder
        """
        def delete_folder(folder_path):
            # Check if folder exists
            if os.path.exists(folder_path):
                # Delete folder and all contents
                shutil.rmtree(folder_path)
                #print(f"The folder '{folder_path}' has been deleted successfully.")
            else:
                print("The folder does not exist.")
                
        delete_folder(remain_folder_path)

    def run_client_ver1(self, replay_path, gameId, start, end, speed, paused , team, remove_fog_of_war, use_nas=True):
        # argument setting
        args = self.arg_list(replay_path)
        # run League of Legends.exe
        
        subprocess.Popen(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,cwd=self.game_dir) 
        # Read stdout and stderr line by line

        requests.packages.urllib3.disable_warnings(requests.packages.urllib3.exceptions.InsecureRequestWarning)
        parent_dir = self.save_dir
        path = os.path.join(parent_dir, gameId, str(start//60))  
        os.makedirs(path, exist_ok=True)
        """
        The wait below is to resolve buffering issues when starting from a specific time in replay file. (Wait until file is sufficiently loaded)
        """
        time.sleep(3)         
        '''
        lcu_request_failed : Whether POST request failed.
        '''
        # Hotkey selection for choosing replay team view
        key = TEAM_HOTKEY_DICT[team]
        lcu_request_failed = True 

        # Replay execution call count threshold; if replay doesn't execute despite exceeding POST_COUNT_THRESH, move to next
        POST_COUNT_THRESH = 3
        post_count = 0
        while lcu_request_failed:
            
            try:
                time.sleep(1)
                res = self.post_initialize(False, start, speed)

                if res.status_code == 200:
                    lcu_request_failed = False
            except (requests.RequestException, ConnectionError, OSError):
                time.sleep(2)
                # Replay execution call count; add 1 every time exception occurs
                post_count = post_count + 1

                # Exit condition: if replay doesn't execute despite these requests, move to next match replay
                if post_count >= POST_COUNT_THRESH:
                    return None
                pass
       
        '''
        replay_running : Boolean for whether replay is running; ensure client runs only until end time
        capture_count : Number of images captured per replay (or current index of capturing image in replay)
        '''     
        self.replay_view_initialize()
        self.post_initialize(False, start, speed)
        replay_running = True 
        #Show Blue Team's Vision   ( RedTeam : f2 , All : f3)
        pydirectinput.press(key)
        pydirectinput.press('x')
        pydirectinput.press('f' if remove_fog_of_war else '')
        capture_count = 0  

        time.sleep(1)
        # Record recording start time
        start_time = time.time()

        # capture_count_list = []
        # timestamp_list = []

        # set replay interface_scoreboard => show curr_gold(total_gold)
        timegap_sec = end - start
        while replay_running :
            try:
                current_time = time.time()
                if (current_time - start_time) >= timegap_sec/speed:
                    replay_running  = False
             
                #capture_count_list.append(capture_count)

                with mss.mss() as sct:
                    # Original capture area: center 2160×1440
                    original_monitor = {"top": 0, "left": 1480, "width": 2160, "height": 1440}
                    # Calculate bottom-right 512×512 portion of original area:
                    sub_top = original_monitor["top"] + original_monitor["height"] - 512 - 23   # 0 + 1440 - 512 = 928
                    sub_left = original_monitor["left"] + original_monitor["width"] - 512 - 22   # 1480 + 2160 - 512 = 3128
                    monitor = {"top": sub_top, "left": sub_left, "width": 512, "height": 512}
                    
                    output = rf"{path}\{capture_count}.png"
                    sct_img = sct.grab(monitor)

                    thread = Thread(target=self.save_to_png, args=(sct_img.rgb, sct_img.size, output))
                    thread.start()
                    time.sleep(0.047)
                    
                capture_count = capture_count + 1

            except (OSError, IOError):
                pass

        ## Close client ##              
        os.system("taskkill /f /im \"League of Legends.exe\"")    

        time.sleep(3)
        # When using NAS
        if use_nas:

            ftp_server = self.ftp_server
            ftp_username = self.ftp_username
            ftp_password=self.ftp_password
            local_folder_path = self.local_folder_path
            remote_folder_path = self.remote_folder_path
            #self.send_to_nas(ftp_server, ftp_username, ftp_password, local_folder_path, remote_folder_path, gameId, capture_count_list)
            #self.send_to_nas(ftp_server, ftp_username, ftp_password, local_folder_path, remote_folder_path, gameId, capture_count_list, timestamp_list)
           
        else:
            pass
        time.sleep(1)

    def run_client_lcu(self, gameId, start, end=None, speed=8, paused=False, team="All", remove_fog_of_war=True):
        """LCU API経由でリプレイを起動し、ミニマップをキャプチャ

        Args:
            gameId: リプレイファイル名（例: JP1-551237207.rofl → JP1-551237207）
            start: 開始時間（秒）
            end: 終了時間（秒）。Noneの場合は試合終了まで自動キャプチャ
            speed: 再生速度
            paused: 一時停止状態で開始するか
            team: チーム視点（'Red', 'Blue', 'All'）
            remove_fog_of_war: Fog of Warを除去するか
        """
        # LCU API経由でリプレイを起動
        if not self.launch_replay_lcu(gameId):
            print(f"Failed to launch replay: {gameId}")
            return None

        # リプレイクライアント起動完了を待機
        if not self._wait_for_replay_ready(process_timeout=60, api_timeout=60):
            print(f"Replay client failed to start: {gameId}")
            os.system("taskkill /f /im \"League of Legends.exe\"")
            return None

        requests.packages.urllib3.disable_warnings(requests.packages.urllib3.exceptions.InsecureRequestWarning)

        # endがNoneの場合、Replay APIからゲーム長を取得
        if end is None:
            game_length = self._get_game_length(timeout=10)
            if game_length:
                end = game_length
                print(f"Game length detected: {game_length:.1f}s ({game_length/60:.1f}min)")
            else:
                # 取得失敗時はデフォルト値を使用
                end = 45 * 60  # 45分
                print(f"Failed to get game length, using default: {end}s")

        parent_dir = self.save_dir
        # フォルダ名計算修正: post_initializeで-5されるため補正
        minute = (start + 5) // 60
        path = os.path.join(parent_dir, gameId, str(minute))
        os.makedirs(path, exist_ok=True)

        # Hotkey selection for choosing replay team view
        key = TEAM_HOTKEY_DICT[team]

        # リプレイ表示設定・再生位置設定
        # NOTE: replay_view_initialize()はFPSカメラモードを設定するため、
        # 通常のリプレイ表示では呼ばない（ミニマップキャプチャのみの場合）
        # self.replay_view_initialize()
        self.post_initialize(paused, start, speed)
        replay_running = True

        # チーム視点設定
        pydirectinput.press(key)
        pydirectinput.press('x')
        if remove_fog_of_war:
            pydirectinput.press('f')
        capture_count = 0

        time.sleep(1)
        start_time = time.time()
        timegap_sec = end - start

        # フレームタイムスタンプ記録用CSVを初期化
        timestamps_csv_path = os.path.join(parent_dir, gameId, "frame_timestamps.csv")
        timestamps_file = open(timestamps_csv_path, 'w', newline='', encoding='utf-8')
        timestamps_writer = csv.writer(timestamps_file)
        timestamps_writer.writerow(['frame_number', 'game_time_ms'])

        while replay_running:
            try:
                current_time = time.time()
                elapsed = current_time - start_time

                # 方法1: 経過時間ベースでの終了判定
                if elapsed >= timegap_sec / speed:
                    replay_running = False
                    continue

                # 方法2: 定期的にReplay APIでゲーム内時間を確認（5秒ごと）
                # 試合終了を正確に検知
                if capture_count % 100 == 0:  # 約5秒ごとに確認
                    try:
                        resp = requests.get(
                            'https://127.0.0.1:2999/replay/playback',
                            verify=False, timeout=1
                        )
                        if resp.status_code == 200:
                            playback_data = resp.json()
                            current_game_time = playback_data.get('time', 0)
                            # リプレイクライアントが終了している場合（プロセス終了）
                            if current_game_time >= end - 1:
                                print(f"Game end detected at {current_game_time:.1f}s")
                                replay_running = False
                                continue
                        elif resp.status_code == 404:
                            # リプレイAPIが応答しない = クライアント終了
                            print("Replay client closed")
                            replay_running = False
                            continue
                    except (requests.RequestException, ConnectionError, OSError):
                        # 接続エラー = クライアント終了の可能性
                        pass

                # 毎フレームでゲーム内時間を取得（タイムスタンプ記録用）
                current_game_time_ms = -1
                try:
                    resp = requests.get(
                        'https://127.0.0.1:2999/replay/playback',
                        verify=False, timeout=0.1
                    )
                    if resp.status_code == 200:
                        playback_data = resp.json()
                        # time は秒単位なのでミリ秒に変換
                        current_game_time_ms = int(playback_data.get('time', 0) * 1000)
                except (requests.RequestException, ConnectionError, OSError, ValueError):
                    pass

                with mss.mss() as sct:
                    # ミニマップキャプチャ座標（1920x1080, MinimapScale=1.8用）
                    # YOLOモデル入力サイズ: 512x512
                    # ミニマップサイズ: 280 * 1.8 ≈ 504px
                    minimap_size = 512
                    screen_width = 1920
                    screen_height = 1080
                    margin_right = 21   # 右端からのマージン（0で右端ピッタリ）
                    margin_bottom = 22  # 下端からのマージン（0で下端ピッタリ）

                    monitor = {
                        "top": screen_height - minimap_size - margin_bottom,
                        "left": screen_width - minimap_size - margin_right,
                        "width": minimap_size,
                        "height": minimap_size
                    }

                    output = rf"{path}\{capture_count}.png"
                    sct_img = sct.grab(monitor)

                    thread = Thread(target=self.save_to_png, args=(sct_img.rgb, sct_img.size, output))
                    thread.start()
                    time.sleep(0.047)

                # フレーム番号とゲーム内時間をCSVに記録
                timestamps_writer.writerow([capture_count, current_game_time_ms])

                capture_count = capture_count + 1

            except (OSError, IOError):
                pass

        # フレームタイムスタンプCSVをクローズ
        timestamps_file.close()
        print(f"Frame timestamps saved to: {timestamps_csv_path}")

        # Close client
        os.system("taskkill /f /im \"League of Legends.exe\"")
        time.sleep(3)
        print(f"Captured {capture_count} frames for {gameId}")

    def get_replay_dir(self):
        return self.replay_dir
    
    @staticmethod
    def list_all_files(base_dir):

        def extract_number(file_path):
            '''
            Function to extract numbers from filename
            '''
            filename = file_path.split("\\")[-1]
            number = re.search(r'\d+', filename).group()
            return int(number)

        file_paths = []
        for dirpath, dirnames, filenames in os.walk(base_dir):
            for filename in filenames:
                file_path = os.path.join(dirpath, filename)
                file_paths.append(file_path)

        sorted_file_paths = sorted(file_paths, key=extract_number)
        return sorted_file_paths

    @staticmethod
    def delete_folder(folder_path):
        # フォルダの存在確認
        if os.path.exists(folder_path):
            # フォルダとすべての内容を削除
            shutil.rmtree(folder_path)
            print(f"The folder '{folder_path}' has been deleted successfully.")
        else:
            print("The folder does not exist.")
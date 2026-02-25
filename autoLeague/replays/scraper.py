from __future__ import annotations

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
from typing import Optional

from autoLeague.base import BaseLcuClient, LcuConfig
from autoLeague.config import MinimapConfig, ReplayConfig

TEAM_HOTKEY_DICT = {'Red' : 'f2', 'Blue' : 'f1', 'All' : 'f'}


class ReplayScraper(BaseLcuClient):
    """League of Legends replay scraper class.

    This class handles executing the League of Legends client in
    replay mode and the scraping application in the correct order.
    Args:
        game_dir: League of Legends game directory.
        replay_dir: League of Legends *.rofl replay directory.
        save_dir: JSON replay files output directory.
        replay_speed: League of Legends client replay speed multiplier.
        scraper_dir: Directory of the scraper program.
        region: Region identifier (default: "JP1").
        lcu_port: LCU API port (auto-detected if None).
        lcu_token: LCU API auth token (auto-detected if None).
        ftp_server: FTP server address (legacy, unused).
        ftp_username: FTP username (legacy, unused).
        ftp_password: FTP password (legacy, unused).
        local_folder_path: Local dataset path.
        remote_folder_path: Remote folder path (legacy, unused).
        minimap_config: Minimap capture configuration.
        replay_config: Replay playback configuration.
    """
    def __init__(self,
            game_dir: str,
            replay_dir: str,
            save_dir: str,
            scraper_dir: str,
            replay_speed: int = 8,
            region: str = "JP1",
            # LCU API settings - auto-detected from LoL client process
            lcu_port: Optional[str] = None,
            lcu_token: Optional[str] = None,
            # FTP settings - set via environment variables or pass explicitly
            ftp_server: Optional[str] = None,
            ftp_username: Optional[str] = None,
            ftp_password: Optional[str] = None,
            local_folder_path: str = r'C:\dataset',
            remote_folder_path: str = '/home/username',
            minimap_config: Optional[MinimapConfig] = None,
            replay_config: Optional[ReplayConfig] = None):

        # LCU authentication via BaseLcuClient
        config = LcuConfig(lcu_port=lcu_port, lcu_token=lcu_token)
        super().__init__(config)

        self.game_dir = game_dir
        self.replay_dir = replay_dir
        self.save_dir = save_dir
        self.scraper_dir = scraper_dir
        self.replay_speed = replay_speed
        self.region = region
        self._minimap = minimap_config or MinimapConfig()
        self._replay = replay_config or ReplayConfig()

        self.ftp_server = ftp_server
        self.ftp_username = ftp_username
        self.ftp_password = ftp_password
        self.local_folder_path = local_folder_path
        self.remote_folder_path = remote_folder_path

        files = os.listdir(self.replay_dir)
        replays = [file for file in files if file.endswith(".rofl")]
        print("Current number of replays: ",len(replays))

    # ------------------------------------------------------------------
    # Backward-compatible properties for lcu_port / lcu_token
    # ------------------------------------------------------------------

    @property
    def lcu_port(self) -> Optional[str]:
        return self._config.lcu_port

    @lcu_port.setter
    def lcu_port(self, value: Optional[str]) -> None:
        pass  # read-only, kept for compat

    @property
    def lcu_token(self) -> Optional[str]:
        return self._config.lcu_token

    @lcu_token.setter
    def lcu_token(self, value: Optional[str]) -> None:
        pass  # read-only, kept for compat

    def _is_replay_process_running(self) -> bool:
        """League of Legends.exe (replay client) is running check."""
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

    def _get_game_length(self, timeout: int = 10, debug: bool = False) -> Optional[float]:
        """Replay API game length (seconds) retrieval.

        Args:
            timeout: Max wait time (seconds).
            debug: Enable debug output.

        Returns:
            Game length in seconds, or None on failure.
        """
        requests.packages.urllib3.disable_warnings(requests.packages.urllib3.exceptions.InsecureRequestWarning)
        start_time = time.time()

        base = self._config.replay_api_base_url
        # Try multiple endpoints and possible field names
        endpoints = [
            (f'{base}/replay/game', ['length', 'gameLength', 'gameDuration']),
            (f'{base}/replay/playback', ['length', 'gameLength', 'gameDuration']),
        ]

        while time.time() - start_time < timeout:
            for url, field_names in endpoints:
                try:
                    resp = requests.get(url, verify=False, timeout=2)
                    if resp.status_code == 200:
                        data = resp.json()
                        if debug:
                            print(f"DEBUG {url}: {data}")
                        for field in field_names:
                            if field in data and data[field]:
                                return data[field]
                except (requests.RequestException, ConnectionError, OSError, ValueError):
                    pass
            time.sleep(1)

        return None

    def _wait_for_replay_ready(
        self,
        process_timeout: Optional[int] = None,
        api_timeout: Optional[int] = None,
    ) -> bool:
        """Wait for replay client startup.

        Two-phase check:
        1. League of Legends.exe process startup
        2. Replay API (port 2999) response

        Args:
            process_timeout: Max wait for process startup (seconds).
                Falls back to self._replay.process_timeout.
            api_timeout: Max wait for Replay API response (seconds).
                Falls back to self._replay.api_timeout.

        Returns:
            True if ready, False on timeout.
        """
        if process_timeout is None:
            process_timeout = self._replay.process_timeout
        if api_timeout is None:
            api_timeout = self._replay.api_timeout

        # Phase 1: Process startup wait
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

        # Additional wait after process startup for API readiness
        print("Waiting for game to initialize...")
        time.sleep(10)

        # Phase 2: Replay API response wait (separate timeout)
        print("Waiting for Replay API...")
        requests.packages.urllib3.disable_warnings(requests.packages.urllib3.exceptions.InsecureRequestWarning)
        replay_playback_url = f'{self._config.replay_api_base_url}/replay/playback'
        start_time = time.time()  # Timer reset
        while time.time() - start_time < api_timeout:
            try:
                resp = requests.get(
                    replay_playback_url,
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

    def launch_replay_lcu(self, game_id: str) -> bool:
        """Launch a replay via LCU API.

        Args:
            game_id: Match identifier (e.g. "JP1-551237207").

        Returns:
            True on success, False on failure.
        """
        if not self._config.lcu_port or not self._config.lcu_token:
            raise RuntimeError("LCU API credentials not available. Is LoL client running?")

        # Extract numeric ID from game_id (JP1-551237207 -> 551237207)
        numeric_id = game_id.split('-')[-1] if '-' in game_id else game_id
        numeric_id = numeric_id.split('_')[-1] if '_' in numeric_id else numeric_id

        url = f'{self._config.lcu_base_url}/lol-replays/v1/rofls/{numeric_id}/watch'

        requests.packages.urllib3.disable_warnings(requests.packages.urllib3.exceptions.InsecureRequestWarning)
        response = requests.post(
            url,
            headers={
                'Accept': 'application/json',
                'Content-Type': 'application/json',
                'Authorization': self._config.auth_header
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
    def arg_list(self, replay_path: str) -> list[str]:
        """Build command-line arguments for the LoL game executable.

        Args:
            replay_path: Path to the .rofl replay file.

        Returns:
            List of command-line arguments.
        """
        return [str(os.path.join(self.game_dir, "League of Legends.exe")),
                replay_path,
                "-SkipRads",
                "-SkipBuild",
                "-EnableLNP",
                "-UseNewX3D=1",
                "-UseNewX3DFramebuffers=1"]

    # For checking if LCU API call succeeded
    @staticmethod
    def post_initialize(paused: bool, start: float, speed: float) -> requests.Response:
        """Send playback initialization request to the Replay API.

        Args:
            paused: Whether to start paused.
            start: Start time in seconds (will be offset by -5).
            speed: Playback speed multiplier.

        Returns:
            requests.Response object.
        """
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
    def replay_view_initialize() -> None:
        """Initialize replay render settings (FPS camera mode, fog of war off, etc.)."""
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
    def save_to_png(rgb: bytes, size: tuple[int, int], output: str) -> None:
        """Save raw RGB data as a PNG file.

        Args:
            rgb: Raw RGB bytes from mss screen capture.
            size: (width, height) tuple.
            output: Output file path.
        """
        mss.tools.to_png(rgb, size, output=output)

    def run_client_lcu(
        self,
        gameId: str,
        start: float,
        end: Optional[float] = None,
        speed: int = 8,
        paused: bool = False,
        team: str = "All",
        remove_fog_of_war: bool = True,
    ) -> None:
        """Launch replay via LCU API and capture minimap frames.

        Args:
            gameId: Replay file name (e.g. JP1-551237207.rofl -> JP1-551237207).
            start: Start time (seconds).
            end: End time (seconds). None = auto-detect from game length.
            speed: Playback speed.
            paused: Start paused.
            team: Team perspective ('Red', 'Blue', 'All').
            remove_fog_of_war: Remove Fog of War.
        """
        # LCU API launch
        if not self.launch_replay_lcu(gameId):
            print(f"Failed to launch replay: {gameId}")
            return None

        # Wait for replay client startup
        if not self._wait_for_replay_ready():
            print(f"Replay client failed to start: {gameId}")
            os.system("taskkill /f /im \"League of Legends.exe\"")
            return None

        requests.packages.urllib3.disable_warnings(requests.packages.urllib3.exceptions.InsecureRequestWarning)

        replay_playback_url = f'{self._config.replay_api_base_url}/replay/playback'

        # If end is None, get game length from Replay API
        if end is None:
            game_length = self._get_game_length(timeout=10)
            if game_length:
                end = game_length
                print(f"Game length detected: {game_length:.1f}s ({game_length/60:.1f}min)")
            else:
                end = self._replay.max_game_duration
                print(f"Failed to get game length, using default: {end}s")

        parent_dir = self.save_dir
        # Folder name calculation fix: compensate for -5 offset in post_initialize
        minute = (start + 5) // 60
        path = os.path.join(parent_dir, gameId, str(int(minute)))
        os.makedirs(path, exist_ok=True)

        # Hotkey selection for choosing replay team view
        key = TEAM_HOTKEY_DICT[team]

        # Replay display/playback position settings
        # NOTE: replay_view_initialize() sets FPS camera mode,
        # so we don't call it for minimap-only capture.
        # self.replay_view_initialize()
        self.post_initialize(paused, start, speed)
        replay_running = True

        # Team perspective setting
        pydirectinput.press(key)
        pydirectinput.press('x')
        if remove_fog_of_war:
            pydirectinput.press('f')
        capture_count = 0

        time.sleep(1)
        start_time = time.time()
        timegap_sec = end - start

        # Initialize frame timestamp CSV
        timestamps_csv_path = os.path.join(parent_dir, gameId, "frame_timestamps.csv")
        timestamps_file = open(timestamps_csv_path, 'w', newline='', encoding='utf-8')
        timestamps_writer = csv.writer(timestamps_file)
        timestamps_writer.writerow(['frame_number', 'game_time_ms'])

        while replay_running:
            try:
                current_time = time.time()
                elapsed = current_time - start_time

                # Method 1: Elapsed-time-based end detection
                if elapsed >= timegap_sec / speed:
                    replay_running = False
                    continue

                # Method 2: Periodically check in-game time via Replay API (every ~5 sec)
                if capture_count % 100 == 0:
                    try:
                        resp = requests.get(
                            replay_playback_url,
                            verify=False, timeout=1
                        )
                        if resp.status_code == 200:
                            playback_data = resp.json()
                            current_game_time = playback_data.get('time', 0)
                            if current_game_time >= end - 1:
                                print(f"Game end detected at {current_game_time:.1f}s")
                                replay_running = False
                                continue
                        elif resp.status_code == 404:
                            print("Replay client closed")
                            replay_running = False
                            continue
                    except (requests.RequestException, ConnectionError, OSError):
                        pass

                # Get in-game time each frame for timestamp recording
                current_game_time_ms = -1
                try:
                    resp = requests.get(
                        replay_playback_url,
                        verify=False, timeout=0.1
                    )
                    if resp.status_code == 200:
                        playback_data = resp.json()
                        current_game_time_ms = int(playback_data.get('time', 0) * 1000)
                except (requests.RequestException, ConnectionError, OSError, ValueError):
                    pass

                with mss.mss() as sct:
                    monitor = self._minimap.monitor

                    output = rf"{path}\{capture_count}.png"
                    sct_img = sct.grab(monitor)

                    thread = Thread(target=self.save_to_png, args=(sct_img.rgb, sct_img.size, output))
                    thread.start()
                    time.sleep(self._replay.capture_interval)

                # Record frame number and in-game time to CSV
                timestamps_writer.writerow([capture_count, current_game_time_ms])

                capture_count = capture_count + 1

            except (OSError, IOError):
                pass

        # Close frame timestamp CSV
        timestamps_file.close()
        print(f"Frame timestamps saved to: {timestamps_csv_path}")

        # Close client
        os.system("taskkill /f /im \"League of Legends.exe\"")
        time.sleep(3)
        print(f"Captured {capture_count} frames for {gameId}")

    def get_replay_dir(self) -> str:
        """Return the replay directory path."""
        return self.replay_dir

    @staticmethod
    def list_all_files(base_dir: str) -> list[str]:
        """List all files under base_dir, sorted by numeric prefix in filename.

        Args:
            base_dir: Root directory to walk.

        Returns:
            Sorted list of absolute file paths.
        """

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
    def delete_folder(folder_path: str) -> None:
        """Delete a folder and all its contents.

        Args:
            folder_path: Path to the folder to delete.
        """
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
            print(f"The folder '{folder_path}' has been deleted successfully.")
        else:
            print("The folder does not exist.")

"""BaseLcuClient + LcuConfig のテスト"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from autoLeague.base.lcu_client import BaseLcuClient, LcuConfig


class ConcreteLcuClient(BaseLcuClient):
    """テスト用の具象クラス"""
    pass


class TestLcuConfig:
    def test_defaults(self):
        cfg = LcuConfig()
        assert cfg.lcu_port is None
        assert cfg.lcu_token is None
        assert cfg.replay_api_port == 2999
        assert cfg.concurrent_downloads == 6

    def test_lcu_base_url(self):
        cfg = LcuConfig(lcu_port="51772")
        assert cfg.lcu_base_url == "https://127.0.0.1:51772"

    def test_replay_api_base_url(self):
        cfg = LcuConfig()
        assert cfg.replay_api_base_url == "https://127.0.0.1:2999"

    def test_auth_header(self):
        cfg = LcuConfig(lcu_token="abc123")
        header = cfg.auth_header
        assert header.startswith("Basic ")
        # Decode and verify
        import base64
        decoded = base64.b64decode(header.split(" ")[1]).decode()
        assert decoded == "riot:abc123"

    def test_frozen(self):
        cfg = LcuConfig(lcu_port="1234")
        try:
            cfg.lcu_port = "5678"
            assert False, "Should be frozen"
        except AttributeError:
            pass


class TestExtractCredentials:
    @patch("subprocess.Popen")
    def test_successful_extraction(self, mock_popen):
        mock_proc = MagicMock()
        mock_proc.communicate.return_value = (
            'CommandLine\n"--remoting-auth-token=abc123" "--app-port=51772"\n',
            "",
        )
        mock_popen.return_value = mock_proc
        port, token = BaseLcuClient._extract_credentials()
        assert port == "51772"
        assert token == "abc123"

    @patch("subprocess.Popen")
    def test_client_not_running(self, mock_popen):
        mock_proc = MagicMock()
        mock_proc.communicate.return_value = ("", "error: process not found")
        mock_popen.return_value = mock_proc
        port, token = BaseLcuClient._extract_credentials()
        assert port is None
        assert token is None

    @patch("subprocess.Popen")
    def test_empty_output(self, mock_popen):
        mock_proc = MagicMock()
        mock_proc.communicate.return_value = ("", "")
        mock_popen.return_value = mock_proc
        port, token = BaseLcuClient._extract_credentials()
        assert port is None
        assert token is None

    def test_init_with_explicit_config(self):
        cfg = LcuConfig(lcu_port="9999", lcu_token="mytoken")
        client = ConcreteLcuClient(cfg)
        assert client.config.lcu_port == "9999"
        assert client.config.lcu_token == "mytoken"


class TestInheritance:
    def test_downloader_is_lcu_client(self):
        from autoLeague.dataset.downloader import ReplayDownloader
        assert issubclass(ReplayDownloader, BaseLcuClient)

    def test_scraper_is_lcu_client(self):
        from autoLeague.replays.scraper import ReplayScraper
        assert issubclass(ReplayScraper, BaseLcuClient)

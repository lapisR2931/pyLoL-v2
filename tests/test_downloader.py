"""ReplayDownloader のテスト"""
from __future__ import annotations

from unittest.mock import patch, MagicMock

from autoLeague.base.lcu_client import BaseLcuClient, LcuConfig
from autoLeague.dataset.downloader import ReplayDownloader


class TestReplayDownloaderInit:
    def test_inherits_base_lcu_client(self):
        assert issubclass(ReplayDownloader, BaseLcuClient)

    @patch.object(BaseLcuClient, "_extract_credentials", return_value=("51772", "mytoken"))
    def test_auto_detect_credentials(self, mock_extract):
        rd = ReplayDownloader()
        assert rd.config.lcu_port == "51772"
        assert rd.config.lcu_token == "mytoken"

    def test_explicit_credentials(self):
        rd = ReplayDownloader(lcu_port="9999", lcu_token="explicit-token")
        assert rd.config.lcu_port == "9999"
        assert rd.config.lcu_token == "explicit-token"

    def test_backward_compat_port_property(self):
        rd = ReplayDownloader(lcu_port="8888", lcu_token="tok")
        assert rd.port == "8888"

    def test_backward_compat_token_property(self):
        rd = ReplayDownloader(lcu_port="8888", lcu_token="tok")
        assert rd.token == "tok"

    def test_replays_dir_initially_none(self):
        rd = ReplayDownloader(lcu_port="8888", lcu_token="tok")
        assert rd.replays_dir is None

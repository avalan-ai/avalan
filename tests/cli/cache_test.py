import unittest
from unittest.mock import MagicMock, patch
from argparse import Namespace

from avalan.cli.commands import cache as cache_cmds
from avalan.model.hubs import HubAccessDeniedException
from rich.padding import Padding

class CliCacheDeleteTestCase(unittest.TestCase):
    def setUp(self):
        self.args = Namespace(model="m", delete=False, delete_revision="r")
        self.console = MagicMock()
        self.theme = MagicMock()
        self.theme._ = lambda s: s
        self.hub = MagicMock()

    def test_no_deletion(self):
        self.hub.cache_delete.return_value = (None, None)
        with patch.object(cache_cmds, "confirm") as confirm:
            cache_cmds.cache_delete(self.args, self.console, self.theme, self.hub)
        self.hub.cache_delete.assert_called_once_with("m", "r")
        self.theme.cache_delete.assert_called_once_with(None, False)
        self.console.print.assert_called_once_with(self.theme.cache_delete.return_value)
        confirm.assert_not_called()

    def test_execute_deletion_confirmed(self):
        cache_del = MagicMock()
        execute = MagicMock()
        self.hub.cache_delete.return_value = (cache_del, execute)
        self.theme.ask_delete_paths.return_value = "ask"
        with patch.object(cache_cmds, "confirm", return_value=True) as confirm:
            cache_cmds.cache_delete(self.args, self.console, self.theme, self.hub)
        self.hub.cache_delete.assert_called_once_with("m", "r")
        self.assertEqual(
            self.theme.cache_delete.call_args_list[0].args,
            (cache_del,)
        )
        self.assertEqual(
            self.theme.cache_delete.call_args_list[1].args,
            (cache_del, True)
        )
        confirm.assert_called_once_with(self.console, "ask")
        execute.assert_called_once()
        self.assertIsInstance(
            self.console.print.call_args_list[1].args[0],
            Padding
        )

    def test_deletion_cancelled(self):
        cache_del = MagicMock()
        execute = MagicMock()
        self.hub.cache_delete.return_value = (cache_del, execute)
        self.theme.ask_delete_paths.return_value = "ask"
        with patch.object(cache_cmds, "confirm", return_value=False) as confirm:
            cache_cmds.cache_delete(self.args, self.console, self.theme, self.hub)
        execute.assert_not_called()
        confirm.assert_called_once()
        self.assertEqual(len(self.theme.cache_delete.call_args_list), 1)
        self.assertEqual(self.theme.cache_delete.call_args.args, (cache_del,))
        self.assertEqual(len(self.console.print.call_args_list), 1)

class CliCacheDownloadTestCase(unittest.TestCase):
    def setUp(self):
        self.args = Namespace(model="m", skip_hub_access_check=False)
        self.console = MagicMock()
        self.theme = MagicMock()
        self.theme._ = lambda s: s
        self.theme.download_progress.return_value = ("tpl",)
        self.hub = MagicMock()
        self.hub.cache_dir = "/cache"
        self.hub.model.return_value = "model"

    def test_successful_download(self):
        self.hub.download.return_value = "/path"
        self.hub.can_access.return_value = True
        with patch.object(cache_cmds, "create_live_tqdm_class", return_value="C") as cltc:
            cache_cmds.cache_download(self.args, self.console, self.theme, self.hub)
        self.theme.model.assert_called_once_with("model", can_access=True)
        self.theme.download_start.assert_called_once_with("m")
        cltc.assert_called_once_with(("tpl",))
        self.hub.download.assert_called_once_with("m", tqdm_class="C")
        self.theme.download_finished.assert_called_once_with("m", "/path")
        self.theme.download_access_denied.assert_not_called()

    def test_access_denied(self):
        self.hub.download.side_effect = HubAccessDeniedException(Exception())
        self.hub.can_access.return_value = False
        self.hub.model_url.return_value = "url"
        with patch.object(cache_cmds, "create_live_tqdm_class", return_value="C"):
            cache_cmds.cache_download(self.args, self.console, self.theme, self.hub)
        self.theme.download_access_denied.assert_called_once_with("m", "url")
        self.theme.download_finished.assert_not_called()

class CliCacheListTestCase(unittest.TestCase):
    def setUp(self):
        self.args = Namespace(model=["m"], summary=True)
        self.console = MagicMock()
        self.theme = MagicMock()
        self.theme._ = lambda s: s
        self.hub = MagicMock()
        self.hub.cache_dir = "/cache"
        self.hub.cache_scan.return_value = ["c"]

    def test_cache_list(self):
        cache_cmds.cache_list(self.args, self.console, self.theme, self.hub)
        self.hub.cache_scan.assert_called_once_with()
        self.theme.cache_list.assert_called_once_with("/cache", ["c"], ["m"], True)
        self.console.print.assert_called_once_with(self.theme.cache_list.return_value)

if __name__ == "__main__":
    unittest.main()

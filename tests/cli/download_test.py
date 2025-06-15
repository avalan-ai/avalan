from unittest import TestCase
from unittest.mock import patch

from avalan.cli import download


class DownloadTestCase(TestCase):
    def test_create_live_tqdm_class(self):
        captured = {}

        class DummyBase:
            def __init__(self, *a, **kw):
                captured.update(kw)

        with patch.object(download, "tqdm_rich_progress", DummyBase):
            Live = download.create_live_tqdm_class(("a",))
            Live()

        self.assertEqual(captured["progress"], ("a",))

    def test_tqdm_rich_progress_disabled(self):
        prog = download.tqdm_rich_progress(disable=True)
        self.assertFalse(hasattr(prog, "_progress"))
        prog.display()
        prog.clear()
        prog.reset(1)
        prog.close()

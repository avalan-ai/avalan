import logging
from avalan.utils import _lf, _j, logger_replace
from avalan.compat import override
from avalan.cli.download import create_live_tqdm_class, tqdm_rich_progress
from unittest import TestCase
from unittest.mock import patch

class UtilsListJoinTestCase(TestCase):
    def test_lf_filters_falsy(self):
        self.assertEqual(_lf(["a", "", None, "b", 0]), ["a", "b"])

    def test_j_join_and_empty(self):
        self.assertEqual(_j(",", ["a", "", "b"]), "a,b")
        self.assertEqual(_j(",", ["", ""], empty="x"), "x")

class UtilsLoggerReplaceTestCase(TestCase):
    def test_logger_replace_copies_handlers(self):
        base_logger = logging.getLogger("base")
        handler = logging.StreamHandler()
        base_logger.addHandler(handler)
        base_logger.setLevel(logging.WARNING)
        base_logger.propagate = False

        target_logger = logging.getLogger("target")
        target_logger.handlers = []
        target_logger.setLevel(logging.NOTSET)
        target_logger.propagate = True

        logger_replace(base_logger, ["target"])

        self.assertIn(handler, target_logger.handlers)
        self.assertEqual(target_logger.level, logging.WARNING)
        self.assertFalse(target_logger.propagate)

class CompatOverrideTestCase(TestCase):
    def test_override_decorator_noop(self):
        def func():
            return 1
        decorated = override(func)
        self.assertIs(decorated, func)
        self.assertEqual(decorated(), 1)

class CliDownloadTestCase(TestCase):
    def test_create_live_tqdm_class(self):
        class DummyProgress:
            def __init__(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs
            def __enter__(self):
                return self
            def __exit__(self, exc_type, exc, tb):
                pass
            def add_task(self, desc, **fmt):
                self.desc = desc
                self.fmt = fmt
                return 1
            def update(self, *_, **__):
                pass
            def reset(self, *_, **__):
                pass

        progress_tpl = (object(),)
        with patch("avalan.cli.download.Progress", DummyProgress):
            LiveTqdm = create_live_tqdm_class(progress_tpl)
            self.assertTrue(issubclass(LiveTqdm, tqdm_rich_progress))
            bar = LiveTqdm(total=1, desc="t", leave=False, disable=False)
            self.assertIsInstance(bar._progress, DummyProgress)
            self.assertEqual(bar._progress.args, progress_tpl)
            self.assertEqual(bar._task_id, 1)


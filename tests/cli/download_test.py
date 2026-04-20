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

    def test_tqdm_rich_progress_builds_progress_and_task_options(self):
        calls: dict[str, object] = {}

        class DummyProgress:
            def __init__(self, *progress_columns, **kwargs):
                calls["progress_columns"] = progress_columns
                calls["progress_kwargs"] = kwargs

            def __enter__(self):
                calls["entered"] = True
                return self

            def __exit__(self, *_):
                calls["exited"] = True

            def add_task(self, description: str, **kwargs) -> int:
                calls["description"] = description
                calls["task_kwargs"] = kwargs
                return 1

            def update(self, *_args, **_kwargs):
                return None

            def reset(self, *_args, **_kwargs):
                return None

        with patch.object(download, "Progress", DummyProgress):
            prog = download.tqdm_rich_progress(
                total=10,
                disable=False,
                progress=("col",),
                options={"expand": True},
            )
            prog.close()

        self.assertEqual(calls["progress_columns"], ("col",))
        self.assertEqual(
            calls["progress_kwargs"],
            {"transient": False, "expand": True},
        )
        self.assertEqual(calls["description"], "")
        self.assertEqual(
            calls["task_kwargs"],
            {"total": 10.0, "completed": 0},
        )

    def test_tqdm_rich_progress_allows_unknown_total(self):
        calls: dict[str, object] = {}

        class DummyProgress:
            def __init__(self, *_progress_columns, **_kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *_):
                return None

            def add_task(self, _description: str, **kwargs) -> int:
                calls["task_kwargs"] = kwargs
                return 1

            def update(self, *_args, **_kwargs):
                return None

            def reset(self, *_args, **_kwargs):
                return None

        with patch.object(download, "Progress", DummyProgress):
            prog = download.tqdm_rich_progress(
                total=None,
                disable=False,
                progress=("col",),
            )
            prog.close()

        self.assertEqual(calls["task_kwargs"], {"total": None, "completed": 0})

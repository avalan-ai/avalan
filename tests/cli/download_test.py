from types import SimpleNamespace
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

            def reset(self, *args, **kwargs):
                calls["reset_args"] = args
                calls["reset_kwargs"] = kwargs
                return None

        with patch.object(download, "Progress", DummyProgress):
            prog = download.tqdm_rich_progress(
                total=10,
                disable=False,
                progress=("col",),
                options={"expand": True},
            )
            prog.reset(5)
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
        self.assertEqual(calls["reset_args"], (1,))
        self.assertEqual(calls["reset_kwargs"], {"total": 5})

    def test_tqdm_rich_progress_syncs_changed_total(self):
        updates: list[dict[str, object]] = []
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

            def update(self, *_args, **kwargs):
                updates.append(kwargs)
                return None

            def reset(self, *_args, **_kwargs):
                return None

        with patch.object(download, "Progress", DummyProgress):
            prog = download.tqdm_rich_progress(
                total=0,
                disable=False,
                progress=("col",),
                unit="B",
                unit_divisor=1000,
                unit_scale=True,
            )
            prog.total = 14_800_000_000
            prog.n = 6_002_666_530
            prog.display()
            prog.close()

        self.assertEqual(
            calls["task_kwargs"],
            {
                "total": 0.0,
                "completed": 0.0,
                "unit": "B",
                "unit_divisor": 1000,
                "unit_scale": True,
            },
        )
        self.assertIn(
            {
                "completed": 6_002_666_530,
                "description": "",
                "total": 14_800_000_000,
                "unit": "B",
                "unit_divisor": 1000,
                "unit_scale": True,
            },
            updates,
        )

    def test_tqdm_rich_progress_normalizes_incomplete_total_description(self):
        updates: list[dict[str, object]] = []
        calls: dict[str, object] = {}

        class DummyProgress:
            def __init__(self, *_progress_columns, **_kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *_):
                return None

            def add_task(self, description: str, **_kwargs) -> int:
                calls["description"] = description
                return 1

            def update(self, *_args, **kwargs):
                updates.append(kwargs)
                return None

            def reset(self, *_args, **_kwargs):
                return None

        with patch.object(download, "Progress", DummyProgress):
            prog = download.tqdm_rich_progress(
                total=0,
                disable=False,
                desc="Downloading (incomplete total...)",
                progress=("col",),
            )
            prog.display()
            prog.close()

        self.assertEqual(calls["description"], "Downloading")
        self.assertIn(
            {"completed": 0, "description": "Downloading", "total": 0},
            updates,
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

    def test_download_complete_column_formats_byte_tasks(self):
        column = download.DownloadCompleteColumn()
        task = SimpleNamespace(
            completed=6_002_666_530,
            total=14_800_000_000,
            fields={"unit": "B", "unit_divisor": 1000},
        )

        rendered = column.render(task)

        self.assertEqual(str(rendered), "6.0/14.8 GB")

    def test_download_complete_column_handles_unknown_byte_total(self):
        column = download.DownloadCompleteColumn()
        task = SimpleNamespace(
            completed=6_002_666_530,
            total=0,
            fields={"unit": "B", "unit_divisor": 1000},
        )

        rendered = column.render(task)

        self.assertEqual(str(rendered), "6.0/? GB")

    def test_download_complete_column_formats_count_tasks(self):
        column = download.DownloadCompleteColumn()
        task = SimpleNamespace(completed=2, total=13, fields={})

        rendered = column.render(task)

        self.assertEqual(str(rendered).strip(), "2/13")

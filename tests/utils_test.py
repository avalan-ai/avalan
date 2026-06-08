import logging
from dataclasses import dataclass
from datetime import date, datetime, time
from decimal import Decimal
from unittest import TestCase
from unittest.mock import MagicMock, patch
from uuid import UUID

from avalan.cli.download import create_live_tqdm_class, tqdm_rich_progress
from avalan.compat import override
from avalan.entities import (
    ToolCall,
    ToolCallDiagnostic,
    ToolCallDiagnosticCode,
    ToolCallDiagnosticStage,
    ToolCallError,
)
from avalan.utils import (
    _j,
    _lf,
    logger_replace,
    to_json,
    tool_call_diagnostic_payload,
    tool_call_error_payload,
)


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


@dataclass
class Dummy:
    value: Decimal


class UtilsToJsonTestCase(TestCase):
    def test_to_json_dataclass_and_decimal(self) -> None:
        self.assertEqual(
            to_json(Dummy(Decimal("1.23"))),
            '{"value": "1.23"}',
        )

    def test_to_json_temporal_and_uuid_types(self) -> None:
        self.assertEqual(
            to_json(
                {
                    "date": date(2025, 9, 19),
                    "datetime": datetime(2025, 9, 19, 12, 34, 56),
                    "time": time(12, 34, 56),
                    "id": UUID("019b7589-672b-766d-81c6-1da5efd5f49a"),
                }
            ),
            '{"date": "2025-09-19", '
            '"datetime": "2025-09-19T12:34:56", '
            '"time": "12:34:56", '
            '"id": "019b7589-672b-766d-81c6-1da5efd5f49a"}',
        )

    def test_to_json_unsupported_type(self) -> None:
        with self.assertRaises(TypeError):
            to_json(object())


class UtilsToolCallErrorPayloadTestCase(TestCase):
    def test_tool_call_diagnostic_payload_includes_canonical_name(
        self,
    ) -> None:
        diagnostic = ToolCallDiagnostic(
            id="diag",
            requested_name="lookup",
            canonical_name="lookup_weather",
            code=ToolCallDiagnosticCode.UNKNOWN_TOOL,
            stage=ToolCallDiagnosticStage.RESOLVE,
            message="Unknown tool.",
        )

        payload = tool_call_diagnostic_payload(diagnostic)

        self.assertEqual(payload["canonical_name"], "lookup_weather")

    def test_tool_call_diagnostic_payload_filters_details(self) -> None:
        diagnostic = ToolCallDiagnostic(
            id="diag-details",
            requested_name="lookup",
            canonical_name="lookup_weather",
            code=ToolCallDiagnosticCode.UNKNOWN_TOOL,
            stage=ToolCallDiagnosticStage.RESOLVE,
            message="Unknown tool.",
            details={
                "arguments": {"token": "sk-live-secret-token"},
                "candidates": ["lookup_weather"],
                "filtered_name": "lookup_weather",
                "limit": 8,
                "stream_status": {"state": "private"},
                "unsafe_path": "/private/customer/source.toml",
            },
        )

        payload = tool_call_diagnostic_payload(diagnostic)

        self.assertEqual(
            payload["details"],
            {
                "candidates": ["lookup_weather"],
                "filtered_name": "lookup_weather",
                "limit": 8,
            },
        )
        rendered = to_json(payload)
        self.assertNotIn("arguments", rendered)
        self.assertNotIn("sk-live-secret-token", rendered)
        self.assertNotIn("/private/customer/source.toml", rendered)

    def test_tool_call_error_payload_projects_exception_safely(self) -> None:
        call = ToolCall(id="call-1", name="tool", arguments={})
        error = ToolCallError(
            id="err-1",
            call=call,
            name="tool",
            arguments={},
            error=RuntimeError("secret-path"),
            message="Tool failed.",
        )

        payload = tool_call_error_payload(error)

        self.assertEqual(
            payload, {"type": "RuntimeError", "message": "Tool call failed."}
        )
        self.assertNotIn("secret-path", to_json(payload))
        self.assertNotIn("Tool failed.", to_json(payload))

    def test_tool_call_error_payload_uses_projected_error_type(self) -> None:
        call = ToolCall(id="call-2", name="tool", arguments={})
        error = ToolCallError(
            id="err-2",
            call=call,
            name="tool",
            arguments={},
            error={"type": "ValidationError", "detail": "raw value"},
            message="Arguments are invalid.",
        )

        payload = tool_call_error_payload(error)

        self.assertEqual(
            payload,
            {
                "type": "ValidationError",
                "message": "Tool call failed.",
            },
        )
        self.assertNotIn("raw value", to_json(payload))
        self.assertNotIn("Arguments are invalid.", to_json(payload))

    def test_tool_call_error_payload_omits_arguments_and_raw_message(
        self,
    ) -> None:
        call = ToolCall(
            id="call-3",
            name="tool",
            arguments={"path": "/private/customer/source.toml"},
        )
        error = ToolCallError(
            id="err-3",
            call=call,
            name="tool",
            arguments=call.arguments,
            error=RuntimeError("failed for sk-live-secret-token"),
            message="failed for sk-live-secret-token",
        )

        payload = tool_call_error_payload(error)

        rendered = to_json(payload)
        self.assertEqual(
            payload,
            {"type": "RuntimeError", "message": "Tool call failed."},
        )
        self.assertNotIn("arguments", rendered)
        self.assertNotIn("/private/customer/source.toml", rendered)
        self.assertNotIn("sk-live-secret-token", rendered)


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

    def test_tqdm_rich_progress_close_and_reset(self):
        class DummyProgress:
            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                self.exited = True

            def add_task(self, desc, **fmt):
                return 1

            def update(self, *_, **__):
                pass

            def reset(self, *_, **__):
                self.reset_total = True

        with (
            patch("avalan.cli.download.Progress", DummyProgress),
            patch("avalan.cli.download.std_tqdm.close") as super_close,
            patch("avalan.cli.download.std_tqdm.reset") as super_reset,
        ):
            LiveTqdm = create_live_tqdm_class((object(),))
            bar = LiveTqdm(total=1, desc="t", leave=False, disable=False)
            bar.display = MagicMock()
            bar._progress.__exit__ = MagicMock()
            bar._progress.reset = MagicMock()

            bar.close()
            bar.display.assert_called_once()
            bar._progress.__exit__.assert_called_once_with(None, None, None)
            super_close.assert_called_once()

            bar.reset(total=2)
            bar._progress.reset.assert_called_once_with(1, total=2)
            super_reset.assert_called_once_with(total=2)

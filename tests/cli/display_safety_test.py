from dataclasses import dataclass
from datetime import datetime, timezone
from unittest import TestCase
from unittest.mock import patch
from uuid import UUID

from avalan.cli import display_safety
from avalan.cli.display_safety import (
    MAX_SUMMARY_CHARS,
    event_type_value,
    is_sensitive_key,
    safe_data,
    safe_summary,
    safe_text,
    safe_tool_call_request_text,
    strip_terminal_controls,
    truncate_text,
    value_from,
)
from avalan.event import EventType


class BrokenString:
    def __str__(self) -> str:
        raise RuntimeError("broken")


class AttributeFailure:
    @property
    def value(self) -> object:
        raise RuntimeError("broken")


class PlainObject:
    def __str__(self) -> str:
        return "[plain]"


class UnreadableMapping(dict[object, object]):
    def items(self):  # type: ignore[no-untyped-def]
        raise RuntimeError("broken")


@dataclass
class ManyFields:
    a: int
    b: int
    c: int
    d: int
    e: int
    f: int
    g: int


@dataclass
class BrokenField:
    visible: str
    hidden: str

    def __getattribute__(self, name: str) -> object:
        if name == "hidden":
            raise RuntimeError("broken")
        return object.__getattribute__(self, name)


class DisplaySafetyTestCase(TestCase):
    def test_text_helpers_strip_escape_and_truncate(self) -> None:
        self.assertEqual(truncate_text("abc", 3), "abc")
        self.assertEqual(truncate_text("abcdef", 3), "abc")
        self.assertEqual(truncate_text("abcdef", 5), "ab...")
        self.assertEqual(strip_terminal_controls("\x1b[31mred\x1b[0m"), "red")

        text = safe_text("[red]\x00x\n\t[/red]", limit=20)

        self.assertNotIn("\x1b", text)
        self.assertIn("\\[red]", text)
        self.assertIn("\\n", text)
        self.assertIn("\\t", text)
        self.assertEqual(
            safe_text(BrokenString()),
            "<unrepresentable BrokenString>",
        )

    def test_sensitive_keys_and_safe_data_shapes(self) -> None:
        cycle: list[object] = []
        cycle.append(cycle)
        payload = {
            "api_key": "secret",
            "Authorization": "bearer",
            "bytes": b"abc",
            "cycle": cycle,
            "time": datetime(2026, 1, 2, tzinfo=timezone.utc),
            "uuid": UUID("00000000-0000-0000-0000-000000000001"),
        }

        safe = safe_data(payload)

        self.assertEqual(safe["api_key"], "<redacted>")
        self.assertEqual(safe["Authorization"], "<redacted>")
        self.assertEqual(safe["bytes"], "<bytes 3>")
        self.assertEqual(safe["cycle"], ["<cycle>"])
        self.assertEqual(safe_data([1, 2, 3, 4, 5, 6, 7])[-1], "truncated")
        large_mapping = {
            "a": 1,
            "b": 2,
            "c": 3,
            "d": 4,
            "e": 5,
            "f": 6,
            "g": 7,
        }
        self.assertEqual(safe_data(large_mapping)["..."], "truncated")
        self.assertEqual(safe_data(PlainObject()), "\\[plain]")
        self.assertEqual(
            safe_data({"c", "b", "a"}),
            ["a", "b", "c"],
        )
        self.assertEqual(
            safe_data(frozenset({"g", "f", "e", "d", "c", "b", "a"})),
            ["a", "b", "c", "d", "e", "f", "truncated"],
        )
        self.assertTrue(is_sensitive_key("api-key"))
        self.assertTrue(is_sensitive_key("[bold]Authorization[/bold]"))
        self.assertFalse(is_sensitive_key(BrokenString()))

    def test_depth_mapping_dataclass_and_summary_fallbacks(self) -> None:
        nested: object = [[[[[["leaf"]]]]]]

        self.assertEqual(safe_data(nested), [[[[["<list>"]]]]])
        self.assertEqual(
            safe_data(UnreadableMapping({"a": 1})),
            "<unreadable UnreadableMapping>",
        )

        many = safe_data(ManyFields(a=1, b=2, c=3, d=4, e=5, f=6, g=7))
        broken = safe_data(BrokenField(visible="ok", hidden="secret"))

        self.assertEqual(many["..."], "truncated")
        self.assertEqual(broken["visible"], "ok")
        self.assertEqual(broken["hidden"], "<unreadable hidden>")
        self.assertEqual(safe_data(None), None)
        self.assertEqual(safe_data(True), True)
        self.assertEqual(safe_data("x"), "x")

        with patch.object(display_safety, "dumps", side_effect=TypeError):
            fallback = safe_summary({"a": 1})

        self.assertIn("a", fallback)
        self.assertEqual(safe_summary({"b": 2, "a": 1}), '{"a": 1, "b": 2}')
        self.assertEqual(safe_summary("abcdef", limit=5), '"a...')

    def test_value_and_event_type_helpers(self) -> None:
        class Record:
            name = "value"

        self.assertEqual(value_from({"name": "mapping"}, "name"), "mapping")
        self.assertEqual(value_from(Record(), "name"), "value")
        self.assertIsNone(value_from(AttributeFailure(), "value"))
        self.assertEqual(event_type_value(EventType.START), "start")
        self.assertEqual(event_type_value(123), "123")

    def test_tool_call_request_text_is_redacted_before_display(self) -> None:
        safe_json = safe_tool_call_request_text(
            '{"query": "weather", "api_key": "secret"}'
        )
        safe_json_with_controls = safe_tool_call_request_text(
            '{"query": "\\u001b[31mweather", '
            '"nested": {"authorization": "secret"}}'
        )
        unsafe_partial = safe_tool_call_request_text(
            '{"query": "weather", "api_key": "secret"'
        )
        unsafe_controlled_partial = safe_tool_call_request_text(
            '{"query": "weather", "api\x1b[31m_key": "value"'
        )
        ordinary_partial = safe_tool_call_request_text('{"query": "weather"')
        ordinary_controlled_partial = safe_tool_call_request_text(
            '{"query": "\x1b[31mweather"'
        )

        self.assertIn("<redacted>", safe_json)
        self.assertIn("weather", safe_json)
        self.assertNotIn("secret", safe_json)
        self.assertIn("<redacted>", safe_json_with_controls)
        self.assertIn("weather", safe_json_with_controls)
        self.assertNotIn("\x1b", safe_json_with_controls)
        self.assertNotIn("secret", safe_json_with_controls)
        self.assertEqual(unsafe_partial, "<redacted>")
        self.assertEqual(unsafe_controlled_partial, "<redacted>")
        self.assertEqual(ordinary_partial, '{"query": "weather"')
        self.assertEqual(
            ordinary_controlled_partial,
            '{"query": "weather"',
        )
        self.assertEqual(safe_tool_call_request_text(""), "")

    def test_malformed_tool_call_request_text_is_bounded(self) -> None:
        ordinary = safe_tool_call_request_text("x" * (MAX_SUMMARY_CHARS + 100))
        sensitive = safe_tool_call_request_text(
            "password=" + ("x" * (MAX_SUMMARY_CHARS + 100))
        )

        self.assertEqual(len(ordinary), MAX_SUMMARY_CHARS)
        self.assertTrue(ordinary.endswith("..."))
        self.assertEqual(sensitive, "<redacted>")

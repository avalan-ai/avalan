from logging import DEBUG, INFO, Logger
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from avalan.agent.loader import OrchestratorLoader


class TestParseServeProtocols:
    def test_returns_none_for_missing_values(self) -> None:
        assert OrchestratorLoader._parse_serve_protocols(None) is None
        assert OrchestratorLoader._parse_serve_protocols([]) is None

    def test_parses_protocols_and_aliases(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "avalan.agent.loader.OrchestratorLoader._is_a2a_supported",
            staticmethod(lambda: True),
        )
        protocols = OrchestratorLoader._parse_serve_protocols(
            [
                " openai: responses,Chat ",
                "MCP",
                "a2a",
            ]
        )

        assert protocols is not None
        assert protocols["openai"] == {"responses", "completions"}
        assert protocols["mcp"] == set()
        assert protocols["a2a"] == set()

    def test_openai_without_endpoints_defaults_to_all(self) -> None:
        protocols = OrchestratorLoader._parse_serve_protocols(["openai"])

        assert protocols is not None
        assert protocols["openai"] == {"completions", "responses"}

    def test_rejects_unknown_protocol(self) -> None:
        with pytest.raises(AssertionError):
            OrchestratorLoader._parse_serve_protocols(["grpc"])

    def test_rejects_unknown_openai_endpoint(self) -> None:
        with pytest.raises(AssertionError):
            OrchestratorLoader._parse_serve_protocols(["openai:unknown"])

    def test_rejects_a2a_when_unavailable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "avalan.agent.loader.OrchestratorLoader._is_a2a_supported",
            staticmethod(lambda: False),
        )

        with pytest.raises(AssertionError):
            OrchestratorLoader._parse_serve_protocols(["a2a"])


class TestLoadServeProtocolStrings:
    def test_loads_protocols_from_toml(self, tmp_path: Path) -> None:
        config = """
[serve]
protocols = [
    " openai : Responses ",
    "mcp",
]
"""
        path = tmp_path / "config.toml"
        path.write_text(config, encoding="utf-8")

        result = OrchestratorLoader._load_serve_protocol_strings(str(path))

        assert result == ["openai : Responses", "mcp"]

    def test_missing_sections_return_none(self, tmp_path: Path) -> None:
        path = tmp_path / "config.toml"
        path.write_text("[other]\nvalue = 1\n", encoding="utf-8")

        assert (
            OrchestratorLoader._load_serve_protocol_strings(str(path)) is None
        )

    def test_missing_protocols_key_returns_none(self, tmp_path: Path) -> None:
        path = tmp_path / "config.toml"
        path.write_text("[serve]\nvalue=1\n", encoding="utf-8")

        assert (
            OrchestratorLoader._load_serve_protocol_strings(str(path)) is None
        )


class TestResolveServeProtocols:
    def test_prefers_cli_protocols(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        expected = {"openai": {"responses"}}

        monkeypatch.setattr(
            OrchestratorLoader,
            "_parse_serve_protocols",
            lambda value: expected if value is not None else None,
        )
        monkeypatch.setattr(
            OrchestratorLoader,
            "_load_serve_protocol_strings",
            lambda path: pytest.fail(
                "_load_serve_protocol_strings should not run"
            ),
        )

        result = OrchestratorLoader.resolve_serve_protocols(
            specs_path="config.toml",
            cli_protocols=["openai:responses"],
        )

        assert result is expected

    def test_reads_from_specs_file(self, tmp_path: Path) -> None:
        path = tmp_path / "config.toml"
        path.write_text('[serve]\nprotocols=["openai"]\n', encoding="utf-8")

        result = OrchestratorLoader.resolve_serve_protocols(
            specs_path=str(path),
            cli_protocols=None,
        )

        assert result == {"openai": {"completions", "responses"}}

    def test_returns_none_when_no_sources(self) -> None:
        assert (
            OrchestratorLoader.resolve_serve_protocols(
                specs_path=None,
                cli_protocols=None,
            )
            is None
        )


class TestLogWrapper:
    def test_log_wrapper_uses_logger_levels(self) -> None:
        logger = MagicMock(spec=Logger)
        wrapper = OrchestratorLoader._log_wrapper(logger)

        wrapper("Debug %s", "value", extra={"a": 1})
        wrapper(
            "Info message",
            inner_type="Lifecycle",
            is_debug=False,
            extra={"b": 2},
        )

        assert logger.log.call_args_list[0].args == (
            DEBUG,
            "<OrchestratorLoader> Debug %s",
            "value",
        )
        assert logger.log.call_args_list[0].kwargs == {"extra": {"a": 1}}

        assert logger.log.call_args_list[1].args == (
            INFO,
            "<Lifecycle @ OrchestratorLoader> Info message",
        )
        assert logger.log.call_args_list[1].kwargs == {"extra": {"b": 2}}

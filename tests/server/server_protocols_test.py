import pytest

from avalan import server as server_module
from avalan.server import _is_module_available, _normalize_protocols


class TestNormalizeProtocols:
    def test_is_module_available_basic(self) -> None:
        assert _is_module_available("math") is True
        assert _is_module_available("__this_module_wont_exist__") is False

    def test_defaults_when_protocols_missing_without_a2a(
        self, monkeypatch
    ) -> None:
        # Patch the module attribute directly to avoid import-path edge cases
        monkeypatch.setattr(
            server_module, "_is_module_available", lambda name: False
        )
        result = _normalize_protocols(None)

        assert result == {
            "openai": {"completions", "responses"},
            "mcp": set(),
        }

    def test_defaults_when_protocols_missing_with_a2a(
        self, monkeypatch
    ) -> None:
        # Patch the module attribute directly to avoid import-path edge cases
        monkeypatch.setattr(
            server_module, "_is_module_available", lambda name: name == "a2a"
        )
        result = _normalize_protocols(None)

        assert result == {
            "openai": {"completions", "responses"},
            "mcp": set(),
            "a2a": set(),
        }

    def test_normalizes_case_and_endpoints(self) -> None:
        result = _normalize_protocols({"OpenAI": {"Responses", "COMPLETIONS"}})

        assert result == {"openai": {"responses", "completions"}}

    def test_empty_openai_endpoints_expand(self) -> None:
        result = _normalize_protocols({"openai": set()})

        assert result == {"openai": {"responses", "completions"}}

    def test_non_openai_protocol_rejects_endpoints(self) -> None:
        with pytest.raises(AssertionError):
            _normalize_protocols({"mcp": {"anything"}})

    def test_non_openai_protocol_defaults_to_empty_selection(self) -> None:
        result = _normalize_protocols({"MCP": set(), "a2a": set()})

        assert result == {"mcp": set(), "a2a": set()}

    def test_unknown_protocol_rejected(self) -> None:
        with pytest.raises(AssertionError):
            _normalize_protocols({"grpc": set()})

    def test_unknown_openai_endpoint_rejected(self) -> None:
        with pytest.raises(AssertionError):
            _normalize_protocols({"openai": {"search"}})

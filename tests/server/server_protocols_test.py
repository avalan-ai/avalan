import pytest

from avalan.server import _normalize_protocols


class TestNormalizeProtocols:
    def test_defaults_when_protocols_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("avalan.server._is_a2a_supported", lambda: True)
        result = _normalize_protocols(None)

        assert result == {
            "openai": {"completions", "responses"},
            "mcp": set(),
            "a2a": set(),
        }

    def test_defaults_skip_a2a_when_unavailable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("avalan.server._is_a2a_supported", lambda: False)

        result = _normalize_protocols(None)

        assert result == {
            "openai": {"completions", "responses"},
            "mcp": set(),
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

    def test_non_openai_protocol_defaults_to_empty_selection(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("avalan.server._is_a2a_supported", lambda: True)
        result = _normalize_protocols({"MCP": set(), "a2a": set()})

        assert result == {"mcp": set(), "a2a": set()}

    def test_a2a_protocol_rejected_when_unavailable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("avalan.server._is_a2a_supported", lambda: False)

        with pytest.raises(AssertionError):
            _normalize_protocols({"a2a": set()})

    def test_unknown_protocol_rejected(self) -> None:
        with pytest.raises(AssertionError):
            _normalize_protocols({"grpc": set()})

    def test_unknown_openai_endpoint_rejected(self) -> None:
        with pytest.raises(AssertionError):
            _normalize_protocols({"openai": {"search"}})

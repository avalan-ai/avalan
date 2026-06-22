import sys

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
        result = _normalize_protocols(
            {
                "MCP": set(),
                "a2a": set(),
                "flow": set(),
            }
        )

        assert result == {"mcp": set(), "a2a": set(), "flow": set()}

    def test_unknown_protocol_rejected(self) -> None:
        with pytest.raises(AssertionError):
            _normalize_protocols({"grpc": set()})

    def test_unknown_openai_endpoint_rejected(self) -> None:
        with pytest.raises(AssertionError):
            _normalize_protocols({"openai": {"search"}})

    def test_a2a_route_install_fails_without_a2a_sdk(
        self, monkeypatch
    ) -> None:
        """Test a2a route install error when a2a-sdk is not installed."""
        import builtins

        modules_to_remove = [
            key for key in sys.modules.keys() if key.startswith("a2a")
        ]
        original_modules = {}
        for module in modules_to_remove:
            original_modules[module] = sys.modules.pop(module, None)

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name.startswith("a2a"):
                raise ImportError("No module named 'a2a'")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)

        try:
            with pytest.raises(
                ImportError,
                match="A2A router requires the a2a-sdk package",
            ):
                from avalan.server.a2a.router import install_a2a_routes

                install_a2a_routes(
                    object(),
                    prefix="/a2a",
                    name="run",
                    description=None,
                )
        finally:
            # Restore original modules
            for module, value in original_modules.items():
                if value is not None:
                    sys.modules[module] = value

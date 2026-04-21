import builtins
import importlib
from unittest.mock import patch

import pytest

from avalan.tool import ToolSet
from avalan.tool.browser import BrowserToolSettings
from avalan.tool.manager import ToolManager
from avalan.tool.parser import ToolCallParser


class _DummyImportError(ImportError):
    pass


def _reload_with_blocked_imports(
    module_name: str,
    blocked_prefixes: tuple[str, ...],
    *,
    stub_database_children: bool = False,
) -> object:
    original_import = builtins.__import__

    database_children = {
        "count": "DatabaseCountTool",
        "inspect": "DatabaseInspectTool",
        "keys": "DatabaseKeysTool",
        "kill": "DatabaseKillTool",
        "locks": "DatabaseLocksTool",
        "plan": "DatabasePlanTool",
        "relationships": "DatabaseRelationshipsTool",
        "run": "DatabaseRunTool",
        "sample": "DatabaseSampleTool",
        "size": "DatabaseSizeTool",
        "tables": "DatabaseTablesTool",
        "tasks": "DatabaseTasksTool",
        "toolset": "DatabaseToolSet",
    }

    def guarded_import(
        name: str,
        globals=None,
        locals=None,
        fromlist=(),
        level: int = 0,
    ):
        if name.startswith(blocked_prefixes):
            raise _DummyImportError(name)

        if (
            stub_database_children
            and level == 1
            and isinstance(globals, dict)
            and globals.get("__package__") == "avalan.tool.database"
            and name in database_children
        ):
            module = type(importlib)(f"avalan.tool.database.{name}")
            setattr(module, database_children[name], object)
            return module

        return original_import(name, globals, locals, fromlist, level)

    with patch("builtins.__import__", guarded_import):
        module = importlib.import_module(module_name)
        return importlib.reload(module)


def test_browser_module_import_fallbacks_are_initialized() -> None:
    module = _reload_with_blocked_imports(
        "avalan.tool.browser",
        ("faiss", "markitdown", "numpy", "playwright.async_api"),
    )

    assert module.HAS_BROWSER_DEPENDENCIES is False
    assert module.IndexFlatL2 is None
    assert module.MarkItDown is None
    assert module.async_playwright is None

    importlib.reload(module)


def test_code_module_import_fallbacks_are_initialized() -> None:
    module = _reload_with_blocked_imports(
        "avalan.tool.code",
        ("RestrictedPython",),
    )

    assert module.HAS_CODE_DEPENDENCIES is False
    assert module.RestrictingNodeTransformer is None
    assert module.compile_restricted is None
    assert module.safe_globals is None

    importlib.reload(module)


def test_database_module_import_fallbacks_and_guards() -> None:
    module = _reload_with_blocked_imports(
        "avalan.tool.database",
        ("sqlalchemy", "sqlglot"),
        stub_database_children=True,
    )

    assert module._SQLALCHEMY_AVAILABLE is False
    assert module.MetaData is None
    assert module.create_async_engine is None
    assert module._SQLGLOT_AVAILABLE is False
    assert module.parse_one is None

    with pytest.raises(ImportError):
        module.DatabaseTool._ensure_sqlalchemy_available()

    with pytest.raises(ImportError):
        module.DatabaseTool._ensure_sqlglot_available()

    importlib.reload(module)


def test_browser_tools_raise_when_dependencies_are_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from avalan.tool import browser

    monkeypatch.setattr(browser, "HAS_BROWSER_DEPENDENCIES", False)

    with pytest.raises(ImportError):
        browser.BrowserTool(BrowserToolSettings(), client=object())

    with pytest.raises(ImportError):
        browser.BrowserToolSet(BrowserToolSettings())


def test_tool_manager_ignores_nested_toolset_entries() -> None:
    nested = ToolSet(tools=[])
    manager = ToolManager.create_instance(
        available_toolsets=[ToolSet(tools=[nested])],
        enable_tools=None,
    )

    assert manager.tools is None
    assert manager.is_empty


def test_tool_call_parser_parse_tag_ignores_unexpected_literal_eval_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parser = ToolCallParser()

    def raise_json_decode_error(*_: object, **__: object):
        raise parser_module.JSONDecodeError("bad", "{}", 0)

    def raise_runtime_error(*_: object, **__: object):
        raise RuntimeError("unexpected")

    import avalan.tool.parser as parser_module

    monkeypatch.setattr(parser_module, "loads", raise_json_decode_error)
    monkeypatch.setattr(parser_module, "literal_eval", raise_runtime_error)

    assert parser._parse_tag("<tool_call>{bad}</tool_call>") is None


def test_database_schemas_uses_default_when_schema_list_is_empty() -> None:
    from avalan.tool.database import DatabaseTool

    dialect = type("Dialect", (), {"name": "sqlite"})()
    connection = type("Connection", (), {"dialect": dialect})()
    inspector = type(
        "Inspector",
        (),
        {
            "default_schema_name": "main",
            "get_schema_names": staticmethod(lambda: []),
        },
    )()

    default_schema, schemas = DatabaseTool._schemas(connection, inspector)

    assert default_schema == "main"
    assert schemas == ["main"]

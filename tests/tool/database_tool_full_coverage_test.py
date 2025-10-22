from asyncio import run
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable

import pytest
from sqlalchemy import create_engine, text

from avalan.tool.database import (
    DatabaseTool,
    DatabaseToolSet,
    DatabaseToolSettings,
)


class DummyDatabaseTool(DatabaseTool):
    async def __call__(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError


def _dummy_async_engine(dsn: str, **_: Any):
    engine = create_engine(dsn)

    class DummyAsyncConnection:
        def __init__(self, connection):
            self._connection = connection

        async def exec_driver_sql(self, sql: str, *args: Any, **kwargs: Any):
            result = self._connection.exec_driver_sql(sql, *args, **kwargs)
            if not result.returns_rows:
                self._connection.commit()
            return result

        async def execute(self, stmt):
            result = self._connection.execute(stmt)
            if not result.returns_rows:
                self._connection.commit()
            return result

        async def run_sync(self, fn, *args: Any, **kwargs: Any):
            return fn(self._connection, *args, **kwargs)

    class DummyConnectionContext:
        def __init__(self, sync_engine):
            self._engine = sync_engine
            self._connection = None

        async def __aenter__(self):
            self._connection = self._engine.connect()
            return DummyAsyncConnection(self._connection)

        async def __aexit__(self, exc_type, exc, tb):
            assert self._connection is not None
            self._connection.close()
            return False

    class DummyAsyncEngine:
        def __init__(self, sync_engine):
            self._sync_engine = sync_engine
            self.disposed = False

        def connect(self):
            return DummyConnectionContext(self._sync_engine)

        def begin(self):
            return DummyConnectionContext(self._sync_engine)

        @property
        def sync_engine(self):
            return self._sync_engine

        async def dispose(self):
            self._sync_engine.dispose()
            self.disposed = True

    return DummyAsyncEngine(engine)


def _connection(dialect: str = "sqlite") -> SimpleNamespace:
    return SimpleNamespace(dialect=SimpleNamespace(name=dialect))


def _inspector(
    *,
    default_schema: str | None = None,
    schemas: list[str | None] | None = None,
    table_names: dict[str | None, list[str]] | None = None,
) -> SimpleNamespace:
    schemas = schemas if schemas is not None else []
    table_names = table_names if table_names is not None else {}

    return SimpleNamespace(
        default_schema_name=default_schema,
        get_schema_names=lambda: list(schemas),
        get_table_names=lambda schema=None: list(table_names.get(schema, [])),
    )


def test_sqlglot_dialect_name_variants() -> None:
    no_sync = SimpleNamespace()
    assert DatabaseTool._sqlglot_dialect_name(no_sync) is None

    sqlite_engine = SimpleNamespace(
        sync_engine=SimpleNamespace(dialect=SimpleNamespace(name="sqlite"))
    )
    assert DatabaseTool._sqlglot_dialect_name(sqlite_engine) == "sqlite"

    postgres_engine = SimpleNamespace(
        sync_engine=SimpleNamespace(dialect=SimpleNamespace(name="postgresql"))
    )
    assert DatabaseTool._sqlglot_dialect_name(postgres_engine) == "postgres"

    mariadb_engine = SimpleNamespace(
        sync_engine=SimpleNamespace(dialect=SimpleNamespace(name="mariadb"))
    )
    assert DatabaseTool._sqlglot_dialect_name(mariadb_engine) == "mysql"


def test_ensure_sql_command_allowed_validates_allow_list() -> None:
    with pytest.raises(PermissionError):
        DatabaseTool._ensure_sql_command_allowed("SELECT 1", [])

    with pytest.raises(PermissionError):
        DatabaseTool._ensure_sql_command_allowed("???", ["select"])

    with pytest.raises(PermissionError):
        DatabaseTool._ensure_sql_command_allowed(
            "DELETE FROM table_name", ["select"]
        )


def test_ensure_sql_command_allowed_special_cases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parse_one_path = "avalan.tool.database.parse_one"

    with monkeypatch.context() as patch_ctx:
        patch_ctx.setattr(parse_one_path, lambda *args, **kwargs: None)
        DatabaseTool._ensure_sql_command_allowed("SELECT 1", ["select"])

    blank_key = SimpleNamespace(key="", this=None)
    with monkeypatch.context() as patch_ctx:
        patch_ctx.setattr(parse_one_path, lambda *args, **kwargs: blank_key)
        DatabaseTool._ensure_sql_command_allowed("SELECT 1", ["select"])

    with_expr = SimpleNamespace(key="with", this=SimpleNamespace(key="select"))
    with monkeypatch.context() as patch_ctx:
        patch_ctx.setattr(parse_one_path, lambda *args, **kwargs: with_expr)
        DatabaseTool._ensure_sql_command_allowed("WITH ...", ["select"])

    statement = "WITH cte AS (SELECT 1) SELECT * FROM cte"
    DatabaseTool._ensure_sql_command_allowed(statement, ["select"])


@pytest.mark.parametrize(
    "sql, expected",
    [
        ("'quoted'", True),
        ("`quoted`", True),
        ('"quoted"', True),
        ("plain", False),
    ],
)
def test_token_is_quoted(sql: str, expected: bool) -> None:
    start = 0 if sql[0].isalnum() else 1
    end = len(sql) - (0 if sql[-1].isalnum() else 1)
    assert DatabaseTool._token_is_quoted(sql, start, end) is expected


def test_rewrite_sql_with_tokens_handles_quotes_and_overlaps() -> None:
    settings = DatabaseToolSettings(dsn="sqlite://", identifier_case="lower")
    tool = DummyDatabaseTool(SimpleNamespace(), settings)

    sql = '"CamelCase" CamelCase main.CamelCase'
    replacements = {
        "camelcase": "CamelCase",
        "main.camelcase": "main.CamelCase",
    }
    rewritten = tool._rewrite_sql_with_tokens(sql, replacements)
    assert '"CamelCase"' in rewritten
    assert rewritten.endswith("main.CamelCase")

    overlapping_tool = DummyDatabaseTool(SimpleNamespace(), settings)

    class StubNormalizer:
        def iter_tokens(self, _: str) -> list[tuple[str, int, int]]:
            return [("CamelCase", 0, 9), ("CamelCase", 2, 11)]

        def normalize(self, identifier: str) -> str:
            return identifier.lower()

        def normalize_token(self, identifier: str) -> str:
            return identifier.lower()

    overlapping_tool._normalizer = StubNormalizer()
    rewritten = overlapping_tool._rewrite_sql_with_tokens(
        "CamelCaseCamelCase", {"camelcase": "CamelCase"}
    )
    assert rewritten.count("CamelCase") == 2


def test_rewrite_sql_with_tokens_no_normalizer_returns_sql() -> None:
    tool = DummyDatabaseTool(
        SimpleNamespace(), DatabaseToolSettings(dsn="sqlite://")
    )
    assert tool._rewrite_sql_with_tokens("SELECT 1", {"x": "y"}) == "SELECT 1"


def test_apply_identifier_case_token_rewrite(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = DatabaseToolSettings(dsn="sqlite://", identifier_case="lower")
    tool = DummyDatabaseTool(SimpleNamespace(), settings)
    tool._table_cache = {None: {"camelcase": "CamelCase"}}

    inspector = _inspector(
        default_schema=None,
        schemas=[None],
        table_names={None: ["CamelCase"]},
    )

    inspect_path = "avalan.tool.database.inspect"
    parse_one_path = "avalan.tool.database.parse_one"
    with monkeypatch.context() as patch_ctx:
        patch_ctx.setattr(inspect_path, lambda _: inspector)
        patch_ctx.setattr(
            parse_one_path,
            lambda *args, **kwargs: (_ for _ in ()).throw(ValueError()),
        )
        rewritten = tool._apply_identifier_case(
            _connection(),
            "SELECT * FROM camelcase",
        )

    assert rewritten.endswith("FROM CamelCase")


def test_apply_identifier_case_handles_schema_replacements(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = DatabaseToolSettings(dsn="sqlite://", identifier_case="lower")
    tool = DummyDatabaseTool(SimpleNamespace(), settings)
    tool._table_cache = {"main": {"camelcase": "CamelCase"}}

    inspector = _inspector(
        default_schema="main",
        schemas=["main"],
        table_names={"main": ["CamelCase"]},
    )

    inspect_path = "avalan.tool.database.inspect"
    with monkeypatch.context() as patch_ctx:
        patch_ctx.setattr(inspect_path, lambda _: inspector)
        rewritten = tool._apply_identifier_case(
            _connection(),
            "SELECT * FROM main.camelcase",
        )

    assert "main.CamelCase" in rewritten


def test_apply_identifier_case_parses_sql_tree(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = SimpleNamespace(
        sync_engine=SimpleNamespace(dialect=SimpleNamespace(name="sqlite"))
    )
    settings = DatabaseToolSettings(dsn="sqlite://", identifier_case="lower")
    tool = DummyDatabaseTool(engine, settings)
    tool._table_cache = {"main": {"camelcase": "CamelCase"}}

    inspector = _inspector(
        default_schema="main",
        schemas=["main"],
        table_names={"main": ["CamelCase"]},
    )

    def fail_rewrite(*_: Any, **__: Any) -> None:
        raise AssertionError("Token-based rewrite should not be used")

    inspect_path = "avalan.tool.database.inspect"
    with monkeypatch.context() as patch_ctx:
        patch_ctx.setattr(inspect_path, lambda _: inspector)
        patch_ctx.setattr(tool, "_rewrite_sql_with_tokens", fail_rewrite)
        rewritten = tool._apply_identifier_case(
            _connection(),
            "SELECT * FROM main.camelcase",
        )

    assert "main.CamelCase" in rewritten
    assert "main.camelcase" not in rewritten


def test_normalize_sql_handles_parse_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool = DummyDatabaseTool(
        SimpleNamespace(sync_engine=None),
        DatabaseToolSettings(dsn="sqlite://"),
    )
    assert tool._normalize_sql("  SELECT 1  ") == "SELECT 1"

    parse_path = "avalan.tool.database.parse"
    with monkeypatch.context() as patch_ctx:
        patch_ctx.setattr(
            parse_path,
            lambda *args, **kwargs: (_ for _ in ()).throw(ValueError()),
        )
        assert tool._normalize_sql("SELECT broken") == "SELECT broken"


def test_prepare_sql_for_execution_respects_allowed_commands(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    allow_all = DummyDatabaseTool(
        SimpleNamespace(sync_engine=None),
        DatabaseToolSettings(dsn="sqlite://", allowed_commands=None),
    )
    with monkeypatch.context() as patch_ctx:
        patch_ctx.setattr(allow_all, "_normalize_sql", lambda sql: sql.strip())
        called = False

        def forbidden(*args: Any, **kwargs: Any) -> None:
            nonlocal called
            called = True

        ensure_path = (
            "avalan.tool.database.DatabaseTool._ensure_sql_command_allowed"
        )
        patch_ctx.setattr(ensure_path, forbidden)
        assert allow_all._prepare_sql_for_execution(" SELECT 1 ") == "SELECT 1"
        assert called is False

    restrictive = DummyDatabaseTool(
        SimpleNamespace(sync_engine=None),
        DatabaseToolSettings(dsn="sqlite://", allowed_commands=["select"]),
    )
    with monkeypatch.context() as patch_ctx:
        patch_ctx.setattr(
            restrictive,
            "_normalize_sql",
            lambda sql: sql.strip(),
        )
        called = False

        def permitted(*args: Any, **kwargs: Any) -> None:
            nonlocal called
            called = True

        patch_ctx.setattr(ensure_path, permitted)
        assert (
            restrictive._prepare_sql_for_execution(" SELECT 1 ") == "SELECT 1"
        )
        assert called is True


def test_configure_read_only_engine_noops(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine_without_sync = SimpleNamespace(sync_engine=None)
    listens_for_path = "avalan.tool.database.event.listens_for"
    with monkeypatch.context() as patch_ctx:
        patch_ctx.setattr(listens_for_path, lambda *args, **kwargs: None)
        DatabaseTool._configure_read_only_engine(engine_without_sync, True)

    sqlite_dialect = SimpleNamespace(name="sqlite")
    engine = SimpleNamespace(
        sync_engine=SimpleNamespace(dialect=sqlite_dialect)
    )
    with monkeypatch.context() as patch_ctx:
        listener_called = False

        def fake_listens_for(*args: Any, **kwargs: Any) -> None:
            nonlocal listener_called
            listener_called = True

        patch_ctx.setattr(listens_for_path, fake_listens_for)
        DatabaseTool._configure_read_only_engine(engine, False)
        assert listener_called is False


def test_configure_read_only_engine_registers_statements(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sqlite_dialect = SimpleNamespace(name="sqlite")
    engine = SimpleNamespace(
        sync_engine=SimpleNamespace(dialect=sqlite_dialect)
    )

    captured: dict[str, Callable[[Any, Any], None]] = {}

    def fake_listens_for(
        target: Any, event_name: str
    ) -> Callable[[Callable[..., None]], Callable[..., None]]:
        assert event_name == "connect"

        def decorator(fn: Callable[..., None]) -> Callable[..., None]:
            captured["handler"] = fn
            return fn

        return decorator

    dialect_path = "avalan.tool.database.DatabaseTool._sqlglot_dialect_name"
    listens_for_path = "avalan.tool.database.event.listens_for"
    with monkeypatch.context() as patch_ctx:
        patch_ctx.setattr(dialect_path, lambda _: "sqlite")
        patch_ctx.setattr(listens_for_path, fake_listens_for)
        DatabaseTool._configure_read_only_engine(engine, True)

    assert "handler" in captured

    executed: list[str] = []

    @dataclass
    class DummyCursor:
        def execute(self, statement: str) -> None:
            executed.append(statement)

        def close(self) -> None:
            executed.append("closed")

    handler = captured["handler"]
    cursor_source = SimpleNamespace(cursor=lambda: DummyCursor())
    handler(cursor_source, None)
    assert executed == ["PRAGMA query_only = ON", "closed"]


def test_configure_read_only_engine_falls_back_to_sqlalchemy_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pg_dialect = SimpleNamespace(name="postgresql")
    engine = SimpleNamespace(sync_engine=SimpleNamespace(dialect=pg_dialect))

    captured: dict[str, Callable[[Any, Any], None]] = {}

    def fake_listens_for(
        target: Any, event_name: str
    ) -> Callable[[Callable[..., None]], Callable[..., None]]:
        def decorator(fn: Callable[..., None]) -> Callable[..., None]:
            captured["handler"] = fn
            return fn

        return decorator

    dialect_path = "avalan.tool.database.DatabaseTool._sqlglot_dialect_name"
    listens_for_path = "avalan.tool.database.event.listens_for"
    with monkeypatch.context() as patch_ctx:
        patch_ctx.setattr(dialect_path, lambda _: None)
        patch_ctx.setattr(listens_for_path, fake_listens_for)
        DatabaseTool._configure_read_only_engine(engine, True)

    executed: list[str] = []

    class DummyCursor:
        def execute(self, statement: str) -> None:
            executed.append(statement)

        def close(self) -> None:
            executed.append("closed")

    handler = captured["handler"]
    cursor_source = SimpleNamespace(cursor=lambda: DummyCursor())
    handler(cursor_source, None)
    assert executed[0].startswith(
        "SET SESSION CHARACTERISTICS AS TRANSACTION READ ONLY"
    )


def test_configure_read_only_engine_skips_unknown_dialect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unknown_dialect = SimpleNamespace(name="unknown")
    engine = SimpleNamespace(
        sync_engine=SimpleNamespace(dialect=unknown_dialect)
    )
    listens_for_path = "avalan.tool.database.event.listens_for"
    dialect_path = "avalan.tool.database.DatabaseTool._sqlglot_dialect_name"
    with monkeypatch.context() as patch_ctx:
        called = False

        def fake_listens_for(*args: Any, **kwargs: Any) -> None:
            nonlocal called
            called = True

        patch_ctx.setattr(dialect_path, lambda _: "unknown")
        patch_ctx.setattr(listens_for_path, fake_listens_for)
        DatabaseTool._configure_read_only_engine(engine, True)
    assert called is False


def test_toolset_with_identifier_case_normalizer(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    file_dsn = f"sqlite:///{tmp_path}/db.sqlite"
    engine = create_engine(file_dsn)
    with engine.begin() as conn:
        conn.execute(
            text("CREATE TABLE example(id INTEGER PRIMARY KEY, name TEXT)")
        )
        conn.execute(text("INSERT INTO example(name) VALUES ('item')"))
    engine.dispose()

    settings = DatabaseToolSettings(dsn=file_dsn, identifier_case="lower")

    monkeypatch.setattr(
        "avalan.tool.database.create_async_engine",
        _dummy_async_engine,
    )

    async def run_test() -> None:
        async with DatabaseToolSet(settings) as toolset:
            normalizers = {id(tool._normalizer) for tool in toolset.tools}
            assert len(normalizers) == 1
            caches = {id(tool._table_cache) for tool in toolset.tools}
            assert len(caches) == 1

            count_tool = toolset.tools[0]
            result = await count_tool("example", context=SimpleNamespace())
            assert result == 1

    run(run_test())

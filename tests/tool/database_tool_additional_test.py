from asyncio import run
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from avalan.tool.database import (
    DatabaseCountTool,
    DatabaseKillTool,
    DatabaseTask,
    DatabaseTasksTool,
    DatabaseTool,
    DatabaseToolSettings,
    IdentifierCaseNormalizer,
)


class DummyDatabaseTool(DatabaseTool):
    async def __call__(self, *args, **kwargs):
        raise NotImplementedError


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


def _result(rows):
    return SimpleNamespace(mappings=lambda: SimpleNamespace(all=lambda: list(rows)))


def test_identifier_case_normalizer_behaviour() -> None:
    lower = IdentifierCaseNormalizer("lower")
    assert lower.normalize("CamelCase") == "camelcase"
    assert lower.normalize_token("schema.Table") == "schema.table"

    upper = IdentifierCaseNormalizer("upper")
    assert upper.normalize("CamelCase") == "CAMELCASE"

    tokens = upper.iter_tokens("SELECT value FROM schema.Table")
    assert tokens[-1] == ("schema.Table", 18, 30)

    preserve = IdentifierCaseNormalizer("preserve")
    assert preserve.normalize("MiXeD") == "MiXeD"


def test_register_table_names_and_normalize_output() -> None:
    settings = DatabaseToolSettings(dsn="sqlite://", identifier_case="lower")
    tool = DummyDatabaseTool(SimpleNamespace(), settings)
    tool._register_table_names("main", ["CamelCase"])

    assert tool._table_cache["main"]["camelcase"] == "CamelCase"
    assert tool._normalize_table_for_output("CamelCase") == "camelcase"

    preserve_tool = DummyDatabaseTool(
        SimpleNamespace(), DatabaseToolSettings(dsn="sqlite://")
    )
    assert (
        preserve_tool._normalize_table_for_output("CamelCase") == "CamelCase"
    )


def test_denormalize_table_name_uses_cache_without_inspection() -> None:
    settings = DatabaseToolSettings(dsn="sqlite://", identifier_case="lower")
    tool = DummyDatabaseTool(SimpleNamespace(), settings)
    tool._table_cache = {None: {"camelcase": "CamelCase"}}

    with patch("avalan.tool.database.inspect", side_effect=AssertionError):
        actual = tool._denormalize_table_name(_connection(), None, "camelcase")

    assert actual == "CamelCase"


def test_denormalize_table_name_populates_cache_from_inspector() -> None:
    settings = DatabaseToolSettings(dsn="sqlite://", identifier_case="lower")
    tool = DummyDatabaseTool(SimpleNamespace(), settings)

    inspector = _inspector(
        default_schema=None, table_names={None: ["CamelCase"]}
    )
    with patch("avalan.tool.database.inspect", return_value=inspector):
        actual = tool._denormalize_table_name(_connection(), None, "camelcase")

    assert actual == "CamelCase"
    assert tool._table_cache[None]["camelcase"] == "CamelCase"


def test_denormalize_table_name_returns_original_when_missing() -> None:
    settings = DatabaseToolSettings(dsn="sqlite://", identifier_case="lower")
    tool = DummyDatabaseTool(SimpleNamespace(), settings)

    inspector = _inspector(default_schema=None, table_names={None: ["Other"]})
    with patch("avalan.tool.database.inspect", return_value=inspector):
        actual = tool._denormalize_table_name(_connection(), None, "unknown")

    assert actual == "unknown"


def test_apply_identifier_case_returns_sql_when_no_replacements() -> None:
    settings = DatabaseToolSettings(dsn="sqlite://", identifier_case="lower")
    tool = DummyDatabaseTool(SimpleNamespace(), settings)
    tool._table_cache = {None: {}}

    inspector = _inspector(
        default_schema=None, schemas=[None], table_names={None: []}
    )
    with patch("avalan.tool.database.inspect", return_value=inspector):
        sql = tool._apply_identifier_case(
            _connection(), "SELECT something FROM nowhere"
        )

    assert sql == "SELECT something FROM nowhere"


def test_apply_identifier_case_skips_quoted_tokens_and_unknowns() -> None:
    settings = DatabaseToolSettings(dsn="sqlite://", identifier_case="lower")
    tool = DummyDatabaseTool(SimpleNamespace(), settings)
    tool._table_cache = {
        None: {"camelcase": "CamelCase"},
        "main": {"camelcase": "CamelCase"},
    }

    inspector = _inspector(
        default_schema="main",
        schemas=["main"],
        table_names={"main": ["CamelCase"]},
    )
    sql = (
        'SELECT "CamelCase", CamelCase, CamelCase", main.CamelCase, Unknown'
        " FROM camelcase"
    )

    with patch("avalan.tool.database.inspect", return_value=inspector):
        rewritten = tool._apply_identifier_case(_connection(), sql)

    assert rewritten.endswith("FROM CamelCase")
    assert "Unknown" in rewritten
    assert '"CamelCase"' in rewritten


def test_apply_identifier_case_returns_sql_when_tokens_missing() -> None:
    settings = DatabaseToolSettings(dsn="sqlite://", identifier_case="lower")
    tool = DummyDatabaseTool(SimpleNamespace(), settings)
    tool._table_cache = {None: {"camelcase": "CamelCase"}}

    inspector = _inspector(
        default_schema=None, schemas=[None], table_names={None: []}
    )
    with patch("avalan.tool.database.inspect", return_value=inspector):
        sql = tool._apply_identifier_case(_connection(), "!! !")

    assert sql == "!! !"


def test_split_schema_and_table_without_schema() -> None:
    schema, table = DatabaseCountTool._split_schema_and_table("authors")
    assert schema is None
    assert table == "authors"


def test_database_tasks_tool_collects_postgresql_rows() -> None:
    rows = [
        {
            "id": "15",
            "user_name": "alice",
            "state": "active",
            "query": "SELECT 1",
            "duration": 120,
        },
        {
            "id": "16",
            "user_name": "bob",
            "state": "idle",
            "query": None,
        },
    ]

    calls = []

    def execute(statement, params=None):
        calls.append((statement.text, params))
        return _result(rows)

    connection = SimpleNamespace(
        dialect=SimpleNamespace(name="postgresql"), execute=execute
    )
    tool = DatabaseTasksTool(SimpleNamespace(), DatabaseToolSettings(dsn="sqlite://"))

    tasks = tool._collect(connection)

    assert len(tasks) == 1
    assert isinstance(tasks[0], DatabaseTask)
    assert tasks[0].id == "15"
    assert tasks[0].user == "alice"
    assert tasks[0].query == "SELECT 1"
    assert tasks[0].duration == 120
    assert "CAST(EXTRACT(EPOCH FROM clock_timestamp() - query_start) AS BIGINT)" in calls[0][0]


def test_database_tasks_tool_collects_mysql_rows() -> None:
    calls = []

    def scalar(statement):
        calls.append(("scalar", statement.text))
        return 2

    def execute(statement, params=None):
        calls.append(("execute", statement.text))
        if statement.text == "SHOW FULL PROCESSLIST":
            rows = [
                {"Id": 2, "Command": "Sleep", "Info": "", "Time": 3},
                {
                    "Id": 3,
                    "Command": "Query",
                    "State": "executing",
                    "Info": "SELECT 1",
                    "User": "carol",
                    "Time": 42,
                },
            ]
            return _result(rows)
        raise AssertionError("Unexpected statement")

    connection = SimpleNamespace(
        dialect=SimpleNamespace(name="mysql"),
        scalar=scalar,
        execute=execute,
    )

    tool = DatabaseTasksTool(SimpleNamespace(), DatabaseToolSettings(dsn="sqlite://"))
    tasks = tool._collect(connection)

    assert len(tasks) == 1
    task = tasks[0]
    assert task.id == "3"
    assert task.query == "SELECT 1"
    assert task.user == "carol"
    assert task.duration == 42
    assert ("scalar", "SELECT CONNECTION_ID()") in calls
    assert ("execute", "SHOW FULL PROCESSLIST") in calls


def test_database_tasks_tool_skips_mysql_rows_without_queries() -> None:
    calls = []

    def scalar(statement):
        calls.append(("scalar", statement.text))
        return None

    rows = [
        {"Id": None, "Command": "Query", "Info": "SELECT 1"},
        {"Id": 7, "Command": "Sleep", "Info": "SELECT 1"},
        {"Id": 8, "Command": "Query", "Info": "   "},
        {
            "Id": 9,
            "Command": "Query",
            "State": "running",
            "Info": "SELECT 2",
            "User": "dave",
        },
    ]

    def execute(statement, params=None):
        calls.append(("execute", statement.text))
        if statement.text == "SHOW FULL PROCESSLIST":
            return _result(rows)
        raise AssertionError("Unexpected statement")

    connection = SimpleNamespace(
        dialect=SimpleNamespace(name="mysql"),
        scalar=scalar,
        execute=execute,
    )

    tool = DatabaseTasksTool(SimpleNamespace(), DatabaseToolSettings(dsn="sqlite://"))
    tasks = tool._collect(connection)

    assert len(tasks) == 1
    task = tasks[0]
    assert task.id == "9"
    assert task.state == "running"
    assert task.query == "SELECT 2"
    assert task.user == "dave"
    assert calls.count(("scalar", "SELECT CONNECTION_ID()")) == 1
    assert calls.count(("execute", "SHOW FULL PROCESSLIST")) == 1


def test_database_tasks_tool_filters_postgresql_rows_by_running_for() -> None:
    rows = [
        {
            "id": "15",
            "user_name": "alice",
            "state": "active",
            "query": "SELECT 1",
            "duration": 120,
        },
        {
            "id": "16",
            "user_name": "bob",
            "state": "active",
            "query": "SELECT 2",
            "duration": 30,
        },
    ]

    connection = SimpleNamespace(
        dialect=SimpleNamespace(name="postgresql"),
        execute=lambda statement, params=None: _result(rows),
    )

    tool = DatabaseTasksTool(SimpleNamespace(), DatabaseToolSettings(dsn="sqlite://"))
    tasks = tool._collect_postgresql(connection, running_for=60)

    assert len(tasks) == 1
    assert tasks[0].id == "15"
    assert tasks[0].duration == 120


def test_database_tasks_tool_filters_mysql_rows_by_running_for() -> None:
    def scalar(statement):
        return None

    rows = [
        {
            "Id": 2,
            "Command": "Query",
            "State": "executing",
            "Info": "SELECT 1",
            "User": "carol",
            "Time": 120,
        },
        {
            "Id": 3,
            "Command": "Query",
            "State": "executing",
            "Info": "SELECT 2",
            "User": "dave",
            "Time": 15,
        },
    ]

    def execute(statement, params=None):
        if statement.text == "SHOW FULL PROCESSLIST":
            return _result(rows)
        raise AssertionError("Unexpected statement")

    connection = SimpleNamespace(
        dialect=SimpleNamespace(name="mysql"),
        scalar=scalar,
        execute=execute,
    )

    tool = DatabaseTasksTool(SimpleNamespace(), DatabaseToolSettings(dsn="sqlite://"))
    tasks = tool._collect_mysql(connection, running_for=60)

    assert len(tasks) == 1
    assert tasks[0].id == "2"
    assert tasks[0].duration == 120


def test_database_tasks_tool_returns_empty_for_unsupported_dialect() -> None:
    connection = SimpleNamespace(dialect=SimpleNamespace(name="sqlite"))
    tool = DatabaseTasksTool(SimpleNamespace(), DatabaseToolSettings(dsn="sqlite://"))
    assert tool._collect(connection) == []


def test_database_kill_tool_postgresql_executes_cancel() -> None:
    captured = {}

    def execute(statement, params=None):
        captured["statement"] = statement.text
        captured["params"] = params
        return SimpleNamespace(scalar=lambda: True)

    connection = SimpleNamespace(
        dialect=SimpleNamespace(name="postgresql"), execute=execute
    )

    tool = DatabaseKillTool(SimpleNamespace(), DatabaseToolSettings(dsn="sqlite://"))
    assert tool._kill(connection, task_id="7") is True
    assert captured["statement"] == "SELECT pg_cancel_backend(:pid) AS cancelled"
    assert captured["params"] == {"pid": 7}


def test_database_kill_tool_mysql_executes_kill() -> None:
    captured = {}

    def execute(statement, params=None):
        captured["statement"] = statement.text
        captured["params"] = params
        return SimpleNamespace()

    connection = SimpleNamespace(
        dialect=SimpleNamespace(name="mysql"), execute=execute
    )

    tool = DatabaseKillTool(SimpleNamespace(), DatabaseToolSettings(dsn="sqlite://"))
    assert tool._kill(connection, task_id="12") is True
    assert captured["statement"] == "KILL :pid"
    assert captured["params"] == {"pid": 12}


def test_database_kill_tool_rejects_non_integer_identifier() -> None:
    connection = SimpleNamespace(dialect=SimpleNamespace(name="postgresql"))
    tool = DatabaseKillTool(SimpleNamespace(), DatabaseToolSettings(dsn="sqlite://"))
    with pytest.raises(RuntimeError):
        tool._kill(connection, task_id="abc")


def test_database_kill_tool_rejects_negative_identifier() -> None:
    connection = SimpleNamespace(dialect=SimpleNamespace(name="postgresql"))
    tool = DatabaseKillTool(SimpleNamespace(), DatabaseToolSettings(dsn="sqlite://"))
    with pytest.raises(RuntimeError):
        tool._kill(connection, task_id="-1")


def test_database_kill_tool_unsupported_dialect() -> None:
    connection = SimpleNamespace(dialect=SimpleNamespace(name="sqlite"))
    tool = DatabaseKillTool(SimpleNamespace(), DatabaseToolSettings(dsn="sqlite://"))
    with pytest.raises(RuntimeError):
        tool._kill(connection, task_id="1")


def test_database_tool_aexit_delegates_to_parent() -> None:
    async def run_test() -> None:
        settings = DatabaseToolSettings(dsn="sqlite://")
        tool = DummyDatabaseTool(SimpleNamespace(), settings)
        with patch.object(tool, "_exit_stack") as stack_mock:
            stack_mock.__aexit__ = AsyncMock(return_value=False)
            result = await tool.__aexit__(None, None, None)

        assert result is False
        stack_mock.__aexit__.assert_called_once()

    run(run_test())

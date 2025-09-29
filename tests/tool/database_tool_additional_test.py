from asyncio import run
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from avalan.tool.database import (
    DatabaseCountTool,
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

    preserve_tool = DummyDatabaseTool(SimpleNamespace(), DatabaseToolSettings(dsn="sqlite://"))
    assert preserve_tool._normalize_table_for_output("CamelCase") == "CamelCase"


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

    inspector = _inspector(default_schema=None, table_names={None: ["CamelCase"]})
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

    inspector = _inspector(default_schema=None, schemas=[None], table_names={None: []})
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

    inspector = _inspector(default_schema="main", schemas=["main"], table_names={"main": ["CamelCase"]})
    sql = 'SELECT "CamelCase", CamelCase, CamelCase", main.CamelCase, Unknown FROM camelcase'

    with patch("avalan.tool.database.inspect", return_value=inspector):
        rewritten = tool._apply_identifier_case(_connection(), sql)

    assert rewritten.endswith("FROM CamelCase")
    assert "Unknown" in rewritten
    assert '"CamelCase"' in rewritten


def test_apply_identifier_case_returns_sql_when_tokens_missing() -> None:
    settings = DatabaseToolSettings(dsn="sqlite://", identifier_case="lower")
    tool = DummyDatabaseTool(SimpleNamespace(), settings)
    tool._table_cache = {None: {"camelcase": "CamelCase"}}

    inspector = _inspector(default_schema=None, schemas=[None], table_names={None: []})
    with patch("avalan.tool.database.inspect", return_value=inspector):
        sql = tool._apply_identifier_case(_connection(), "!! !")

    assert sql == "!! !"


def test_split_schema_and_table_without_schema() -> None:
    schema, table = DatabaseCountTool._split_schema_and_table("authors")
    assert schema is None
    assert table == "authors"


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

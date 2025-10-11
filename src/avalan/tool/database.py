from . import Tool, ToolSet
from ..compat import override
from ..entities import ToolCallContext
from abc import ABC
from asyncio import sleep
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from re import compile as regex_compile
from typing import Any, Literal
from sqlalchemy import MetaData, event, func, inspect, select, text
from sqlalchemy import Table as SATable
from sqlalchemy.engine import Connection
from sqlalchemy.engine.reflection import Inspector
from sqlalchemy.exc import NoSuchTableError
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlalchemy.sql.elements import TextClause
from sqlglot import parse, parse_one, exp


@dataclass(frozen=True, kw_only=True, slots=True)
class ForeignKey:
    field: str
    ref_table: str
    ref_field: str


@dataclass(frozen=True, kw_only=True, slots=True)
class Table:
    name: str
    columns: dict[str, str]
    foreign_keys: list[ForeignKey]


@dataclass(frozen=True, kw_only=True, slots=True)
class DatabaseTask:
    id: str
    user: str | None
    state: str | None
    query: str | None
    duration: int | None


@dataclass(frozen=True, kw_only=True, slots=True)
class DatabaseToolSettings:
    dsn: str
    delay_secs: float | None = None
    identifier_case: Literal["preserve", "lower", "upper"] = "preserve"
    read_only: bool = True
    allowed_commands: list[str] | None = field(
        default_factory=lambda: ["select"],
    )


@dataclass(frozen=True, kw_only=True, slots=True)
class QueryPlan:
    dialect: str
    steps: list[dict[str, Any]]


def _sqlglot_dialect_name(engine: AsyncEngine) -> str | None:
    sync_engine = getattr(engine, "sync_engine", None)
    if sync_engine is None:
        return None
    name = getattr(sync_engine.dialect, "name", None)
    if name == "postgresql":
        return "postgres"
    if name == "mariadb":
        return "mysql"
    return name


def _ensure_sql_command_allowed(
    statement: str, allowed: list[str], dialect_name: str | None = None
) -> None:
    normalized_allowed = {command.lower() for command in allowed}
    if not normalized_allowed:
        raise PermissionError(
            "No SQL commands are permitted by the current configuration.",
        )

    try:
        expr = (
            parse_one(statement, read=dialect_name)
            if dialect_name
            else parse_one(statement)
        )
    except Exception:
        raise PermissionError(
            "SQL could not be parsed to enforce allowed commands.",
        )

    if expr is None:
        return

    key = (expr.key or "").lower()
    if key == "with":
        inner = getattr(expr, "this", None)
        if inner is not None and getattr(inner, "key", None):
            key = inner.key.lower()

    if not key:
        return
    if key not in normalized_allowed:
        allowed_display = ", ".join(sorted(normalized_allowed))
        raise PermissionError(
            f"SQL command '{key.upper()}' is not permitted."
            f" Allowed commands: {allowed_display}.",
        )


class IdentifierCaseNormalizer:
    __slots__ = ("_mode", "_token_pattern")

    def __init__(self, mode: Literal["preserve", "lower", "upper"]) -> None:
        self._mode = mode
        self._token_pattern = regex_compile(
            r"[0-9A-Za-z_]+(?:\.[0-9A-Za-z_]+)?"
        )

    def normalize(self, identifier: str) -> str:
        if self._mode == "lower":
            return identifier.lower()
        if self._mode == "upper":
            return identifier.upper()
        return identifier

    def normalize_token(self, identifier: str) -> str:
        if "." not in identifier:
            return self.normalize(identifier)
        schema, table = identifier.split(".", 1)
        table_normalized = self.normalize(table)
        return f"{schema}.{table_normalized}"

    def iter_tokens(self, sql: str) -> list[tuple[str, int, int]]:
        return [
            (match.group(0), match.start(), match.end())
            for match in self._token_pattern.finditer(sql)
        ]


class DatabaseTool(Tool, ABC):
    _engine: AsyncEngine
    _settings: DatabaseToolSettings
    _normalizer: IdentifierCaseNormalizer | None
    _table_cache: dict[str | None, dict[str, str]]

    def __init__(
        self,
        engine: AsyncEngine,
        settings: DatabaseToolSettings,
        *,
        normalizer: IdentifierCaseNormalizer | None = None,
        table_cache: dict[str | None, dict[str, str]] | None = None,
    ) -> None:
        self._engine = engine
        self._settings = settings
        if settings.identifier_case == "preserve":
            self._normalizer = None
        else:
            self._normalizer = normalizer or IdentifierCaseNormalizer(
                settings.identifier_case
            )
        self._table_cache = table_cache if table_cache is not None else {}
        super().__init__()

    def _register_table_names(
        self, schema: str | None, table_names: list[str]
    ) -> None:
        if self._normalizer is None:
            return
        cache = self._table_cache.setdefault(schema, {})
        for name in table_names:
            cache[self._normalizer.normalize(name)] = name

    def _denormalize_table_name(
        self,
        connection: Connection,
        schema: str | None,
        table_name: str,
    ) -> str:
        if self._normalizer is None:
            return table_name

        cache = self._table_cache.get(schema)
        normalized = self._normalizer.normalize(table_name)

        if cache is None or normalized not in cache:
            inspector = inspect(connection)
            actual_tables = inspector.get_table_names(schema=schema)
            cache = {
                self._normalizer.normalize(name): name
                for name in actual_tables
            }
            self._table_cache[schema] = cache

        return cache.get(normalized, table_name)

    def _normalize_table_for_output(self, table_name: str) -> str:
        if self._normalizer is None:
            return table_name
        return self._normalizer.normalize(table_name)

    def _apply_identifier_case(self, connection: Connection, sql: str) -> str:
        if self._normalizer is None:
            return sql

        inspector = inspect(connection)
        _, schemas = self._schemas(connection, inspector)

        for schema in schemas:
            if schema in self._table_cache:
                continue
            actual_tables = inspector.get_table_names(schema=schema)
            self._register_table_names(schema, actual_tables)

        replacements: dict[str, str] = {}
        for schema, table_map in self._table_cache.items():
            for normalized, actual in table_map.items():
                replacements[normalized] = actual
                if schema is not None:
                    replacements[f"{schema}.{normalized}"] = (
                        f"{schema}.{actual}"
                    )

        if not replacements:
            return sql

        dialect = _sqlglot_dialect_name(self._engine)
        try:
            tree = parse_one(sql, read=dialect) if dialect else parse_one(sql)
        except Exception:
            return self._rewrite_sql_with_tokens(sql, replacements)

        def normalize_table(node: exp.Expression) -> exp.Expression:
            if isinstance(node, exp.Table):
                ident = node.this
                if isinstance(ident, exp.Identifier) and not ident.quoted:
                    name = ident.this
                    schema_ident = node.args.get("db")
                    schema = (
                        schema_ident.this
                        if isinstance(schema_ident, exp.Identifier)
                        else None
                    )
                    key = self._normalizer.normalize(name)
                    lookup = f"{schema}.{key}" if schema else key
                    actual = replacements.get(lookup) or replacements.get(key)
                    if actual:
                        if schema and "." in actual:
                            _, actual_name = actual.split(".", 1)
                        else:
                            actual_name = actual
                        node.set(
                            "this",
                            exp.Identifier(this=actual_name, quoted=False),
                        )
            return node

        tree = tree.transform(normalize_table)
        return tree.sql(dialect=dialect) if dialect else tree.sql()

    def _rewrite_sql_with_tokens(
        self, sql: str, replacements: dict[str, str]
    ) -> str:
        if self._normalizer is None:
            return sql

        tokens = self._normalizer.iter_tokens(sql)
        if not tokens:
            return sql

        rewritten: list[str] = []
        cursor = 0

        for token, start, end in tokens:
            if start < cursor:
                continue
            rewritten.append(sql[cursor:start])

            if self._token_is_quoted(sql, start, end):
                rewritten.append(token)
            else:
                lookup = self._normalizer.normalize_token(token)
                replacement = replacements.get(lookup)
                rewritten.append(replacement or token)

            cursor = end

        rewritten.append(sql[cursor:])
        return "".join(rewritten)

    @staticmethod
    def _token_is_quoted(sql: str, start: int, end: int) -> bool:
        if start > 0 and sql[start - 1] in {'"', "'", "`"}:
            return True
        if end < len(sql) and sql[end] in {'"', "'", "`"}:
            return True
        return False

    def _normalize_sql(self, sql: str) -> str:
        dialect = _sqlglot_dialect_name(self._engine)
        try:
            trees = parse(sql, read=dialect) if dialect else parse(sql)
            count = len([t for t in trees if t is not None])
            if count > 1:
                raise PermissionError(
                    "Multiple SQL statements are not permitted in a single"
                    " execution.",
                )
        except PermissionError:
            raise
        except Exception:
            pass
        return sql.strip()

    def _prepare_sql_for_execution(self, sql: str) -> str:
        normalized_sql = self._normalize_sql(sql)
        if self._settings.allowed_commands is None:
            return normalized_sql
        dialect = _sqlglot_dialect_name(self._engine)
        _ensure_sql_command_allowed(
            normalized_sql, self._settings.allowed_commands, dialect
        )
        return normalized_sql

    @staticmethod
    def _schemas(
        connection: Connection, inspector: Inspector
    ) -> tuple[str | None, list[str | None]]:
        default_schema = inspector.default_schema_name
        dialect = connection.dialect.name

        if dialect == "postgresql":
            sys = {"information_schema", "pg_catalog"}
            schemas = [
                s
                for s in inspector.get_schema_names()
                if s not in sys and not (s or "").startswith("pg_")
            ]
            if default_schema and default_schema not in schemas:
                schemas.append(default_schema)
            return default_schema, schemas

        all_schemas = inspector.get_schema_names() or (
            [default_schema] if default_schema is not None else [None]
        )

        sys_filters = {
            "mysql": {
                "information_schema",
                "performance_schema",
                "mysql",
                "sys",
            },
            "mariadb": {
                "information_schema",
                "performance_schema",
                "mysql",
                "sys",
            },
            "mssql": {"INFORMATION_SCHEMA", "sys"},
            "oracle": {"SYS", "SYSTEM"},
            "sqlite": set(),
        }
        sys = sys_filters.get(dialect, set())
        schemas = [s for s in all_schemas if s not in sys]

        if not schemas:
            schemas = (
                [default_schema] if default_schema is not None else [None]
            )

        seen: set[str | None] = set()
        uniq: list[str | None] = []
        for s in schemas:
            if s not in seen:
                uniq.append(s)
                seen.add(s)

        if default_schema not in seen:
            uniq.append(default_schema)

        return default_schema, uniq

    @override
    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: BaseException | None,
    ) -> bool:
        return await super().__aexit__(exc_type, exc_value, traceback)


def _configure_read_only_engine(engine: AsyncEngine, read_only: bool) -> None:
    if not read_only:
        return

    sync_engine = getattr(engine, "sync_engine", None)
    if sync_engine is None:
        return

    sg_name = _sqlglot_dialect_name(engine)
    statements_by_sqlglot: dict[str, tuple[str, ...]] = {
        "sqlite": ("PRAGMA query_only = ON",),
        "postgres": ("SET SESSION CHARACTERISTICS AS TRANSACTION READ ONLY",),
        "mysql": ("SET SESSION TRANSACTION READ ONLY",),
        "oracle": ("ALTER SESSION SET READ ONLY = TRUE",),
    }
    statements = statements_by_sqlglot.get(sg_name or "")

    if statements is None:
        dialect_name = getattr(sync_engine.dialect, "name", "")
        statements_by_sa: dict[str, tuple[str, ...]] = {
            "sqlite": ("PRAGMA query_only = ON",),
            "postgresql": (
                "SET SESSION CHARACTERISTICS AS TRANSACTION READ ONLY",
            ),
            "mysql": ("SET SESSION TRANSACTION READ ONLY",),
            "mariadb": ("SET SESSION TRANSACTION READ ONLY",),
            "oracle": ("ALTER SESSION SET READ ONLY = TRUE",),
        }
        statements = statements_by_sa.get(dialect_name)

    if statements is None:
        return

    @event.listens_for(sync_engine, "connect")
    def _set_read_only(dbapi_connection, _connection_record):  # type: ignore[arg-type]
        cursor = dbapi_connection.cursor()
        try:
            for statement in statements:
                cursor.execute(statement)
        finally:
            cursor.close()


class DatabaseCountTool(DatabaseTool):
    """Count rows in the given table.

    Args:
        table_name: Table to count rows from (optionally schema-qualified, e.g. 'public.users').

    Returns:
        Number of rows in the table.
"""

    def __init__(
        self,
        engine: AsyncEngine,
        settings: DatabaseToolSettings,
        *,
        normalizer: IdentifierCaseNormalizer | None = None,
        table_cache: dict[str | None, dict[str, str]] | None = None,
    ) -> None:
        super().__init__(
            engine,
            settings,
            normalizer=normalizer,
            table_cache=table_cache,
        )
        self.__name__ = "count"

    @staticmethod
    def _split_schema_and_table(qualified: str) -> tuple[str | None, str]:
        if "." in qualified:
            sch, tbl = qualified.split(".", 1)
            return (sch or None), tbl
        return None, qualified

    async def __call__(
        self, table_name: str, *, context: ToolCallContext
    ) -> int:
        assert table_name, "table_name must not be empty"
        if self._settings.delay_secs:
            await sleep(self._settings.delay_secs)

        async with self._engine.connect() as conn:
            schema, tbl_name = self._split_schema_and_table(table_name)
            actual_name = await conn.run_sync(
                self._denormalize_table_name, schema, tbl_name
            )
            tbl = SATable(actual_name, MetaData(), schema=schema)
            stmt = select(func.count()).select_from(tbl)

            result = await conn.execute(stmt)
            return int(result.scalar_one())


class DatabaseInspectTool(DatabaseTool):
    """Inspect tables to retrieve column schemas and foreign keys.

    Args:
        table_names: Tables to inspect.
        schema: Optional schema the tables belong to; defaults to the current schema.

    Returns:
        Schemas describing the requested tables.
"""

    def __init__(
        self,
        engine: AsyncEngine,
        settings: DatabaseToolSettings,
        *,
        normalizer: IdentifierCaseNormalizer | None = None,
        table_cache: dict[str | None, dict[str, str]] | None = None,
    ) -> None:
        super().__init__(
            engine,
            settings,
            normalizer=normalizer,
            table_cache=table_cache,
        )
        self.__name__ = "inspect"

    async def __call__(
        self,
        table_names: list[str],
        schema: str | None = None,
        *,
        context: ToolCallContext,
    ) -> list[Table]:
        assert table_names, "table_names must not be empty"
        if self._settings.delay_secs:
            await sleep(self._settings.delay_secs)

        async with self._engine.connect() as conn:
            result = await conn.run_sync(
                self._collect,
                schema=schema,
                table_names=table_names,
            )
        return result

    def _collect(
        self,
        connection: Connection,
        *,
        schema: str | None,
        table_names: list[str],
    ) -> list[Table]:
        inspector = inspect(connection)
        default_schema, _ = self._schemas(connection, inspector)
        sch = schema or default_schema

        tables: list[Table] = []
        for table_name in table_names:
            actual_table = self._denormalize_table_name(
                connection, sch, table_name
            )
            try:
                column_info = inspector.get_columns(actual_table, schema=sch)
            except NoSuchTableError:
                continue

            columns = {c["name"]: str(c["type"]) for c in column_info}

            fkeys: list[ForeignKey] = []
            try:
                fks = inspector.get_foreign_keys(actual_table, schema=sch)
            except NoSuchTableError:
                fks = []

            for fk in fks or []:
                ref_schema = fk.get("referred_schema")
                ref_table = (
                    f"{ref_schema}.{self._normalize_table_for_output(fk['referred_table'])}"
                    if ref_schema
                    else self._normalize_table_for_output(fk["referred_table"])
                )
                for source, target in zip(
                    fk.get("constrained_columns", []),
                    fk.get("referred_columns", []),
                ):
                    fkeys.append(
                        ForeignKey(
                            field=source, ref_table=ref_table, ref_field=target
                        )
                    )

            table_display = self._normalize_table_for_output(actual_table)
            name = (
                table_display
                if sch in (None, default_schema)
                else f"{sch}.{table_display}"
            )
            tables.append(
                Table(name=name, columns=columns, foreign_keys=fkeys)
            )

        return tables


class DatabaseRunTool(DatabaseTool):
    """Run the given SQL statement on the database and return result rows.

    Args:
        sql: SQL statement to execute.

    Returns:
        Rows returned by the SQL statement, if any.
"""

    def __init__(
        self,
        engine: AsyncEngine,
        settings: DatabaseToolSettings,
        *,
        normalizer: IdentifierCaseNormalizer | None = None,
        table_cache: dict[str | None, dict[str, str]] | None = None,
    ) -> None:
        super().__init__(
            engine,
            settings,
            normalizer=normalizer,
            table_cache=table_cache,
        )
        self.__name__ = "run"

    async def __call__(
        self, sql: str, *, context: ToolCallContext
    ) -> list[dict[str, Any]]:
        if self._settings.delay_secs:
            await sleep(self._settings.delay_secs)

        async with self._engine.begin() as conn:
            normalized_sql = self._prepare_sql_for_execution(sql)
            sql_to_run = await conn.run_sync(
                self._apply_identifier_case, normalized_sql
            )
            result = await conn.exec_driver_sql(sql_to_run)

            if result.returns_rows:
                return [dict(row) for row in result.mappings().all()]
        return []


class DatabasePlanTool(DatabaseTool):
    """Explain how the database will execute the provided SQL statement.

    Args:
        sql: SQL statement to analyze.

    Returns:
        Query plan describing the execution strategy for the SQL statement.
    """

    def __init__(
        self,
        engine: AsyncEngine,
        settings: DatabaseToolSettings,
        *,
        normalizer: IdentifierCaseNormalizer | None = None,
        table_cache: dict[str | None, dict[str, str]] | None = None,
    ) -> None:
        super().__init__(
            engine,
            settings,
            normalizer=normalizer,
            table_cache=table_cache,
        )
        self.__name__ = "plan"

    async def __call__(self, sql: str, *, context: ToolCallContext) -> QueryPlan:
        if self._settings.delay_secs:
            await sleep(self._settings.delay_secs)

        async with self._engine.connect() as conn:
            normalized_sql = self._prepare_sql_for_execution(sql)
            rewritten_sql = await conn.run_sync(
                self._apply_identifier_case, normalized_sql
            )
            statement, dialect = await conn.run_sync(
                self._statement_for_plan, rewritten_sql
            )
            result = await conn.execute(statement)

            if not result.returns_rows:
                return QueryPlan(dialect=dialect, steps=[])

            rows = [dict(row) for row in result.mappings().all()]
            return QueryPlan(dialect=dialect, steps=rows)

    def _statement_for_plan(
        self, connection: Connection, sql: str
    ) -> tuple[TextClause, str]:
        dialect = connection.dialect.name

        if dialect == "sqlite":
            prefix = "EXPLAIN QUERY PLAN "
        elif dialect == "postgresql":
            prefix = "EXPLAIN (FORMAT TEXT) "
        elif dialect in {"mysql", "mariadb"}:
            prefix = "EXPLAIN FORMAT=JSON "
        else:
            prefix = "EXPLAIN "

        return text(f"{prefix}{sql}"), dialect


class DatabaseTablesTool(DatabaseTool):
    """List table names available in the database grouped by schema.

    Args:
        None.

    Returns:
        Mapping of schema names to the tables they contain.
"""

    def __init__(
        self,
        engine: AsyncEngine,
        settings: DatabaseToolSettings,
        *,
        normalizer: IdentifierCaseNormalizer | None = None,
        table_cache: dict[str | None, dict[str, str]] | None = None,
    ) -> None:
        super().__init__(
            engine,
            settings,
            normalizer=normalizer,
            table_cache=table_cache,
        )
        self.__name__ = "tables"

    async def __call__(
        self, *, context: ToolCallContext
    ) -> dict[str | None, list[str]]:
        if self._settings.delay_secs:
            await sleep(self._settings.delay_secs)

        async with self._engine.connect() as conn:
            result = await conn.run_sync(self._collect)
        return result

    def _collect(self, connection: Connection) -> dict[str | None, list[str]]:
        inspector = inspect(connection)
        _, schemas = self._schemas(connection, inspector)
        result: dict[str | None, list[str]] = {}
        for schema in schemas:
            actual_tables = inspector.get_table_names(schema=schema)
            self._register_table_names(schema, actual_tables)
            result[schema] = [
                self._normalize_table_for_output(name)
                for name in actual_tables
            ]
        return result


class DatabaseTasksTool(DatabaseTool):
    """List killable database tasks from supported engines.

    Args:
        running_for: Minimum number of seconds a task must be running to be
            included. Provide ``None`` to return all tasks regardless of
            duration.

    Returns:
        Tasks that are currently running on PostgreSQL or MySQL connections.
    """

    def __init__(
        self,
        engine: AsyncEngine,
        settings: DatabaseToolSettings,
        *,
        normalizer: IdentifierCaseNormalizer | None = None,
        table_cache: dict[str | None, dict[str, str]] | None = None,
    ) -> None:
        super().__init__(
            engine,
            settings,
            normalizer=normalizer,
            table_cache=table_cache,
        )
        self.__name__ = "tasks"

    async def __call__(
        self,
        *,
        running_for: int | None = None,
        context: ToolCallContext,
    ) -> list[DatabaseTask]:
        if running_for is not None:
            assert running_for >= 0, "running_for must be zero or greater"
        if self._settings.delay_secs:
            await sleep(self._settings.delay_secs)

        async with self._engine.connect() as conn:
            result = await conn.run_sync(
                self._collect, running_for=running_for
            )
        return result

    def _collect(
        self, connection: Connection, *, running_for: int | None = None
    ) -> list[DatabaseTask]:
        dialect_name = getattr(connection.dialect, "name", None)

        if dialect_name == "postgresql":
            return self._collect_postgresql(connection, running_for=running_for)
        if dialect_name in {"mysql", "mariadb"}:
            return self._collect_mysql(connection, running_for=running_for)
        return []

    def _collect_postgresql(
        self, connection: Connection, *, running_for: int | None = None
    ) -> list[DatabaseTask]:
        statement = text(
            """
            select pid::text as id,
                   usename as user_name,
                   state,
                   query,
                   CAST(EXTRACT(EPOCH FROM clock_timestamp() - query_start) AS BIGINT)
                       AS duration
            from pg_stat_activity
            where pid <> pg_backend_pid()
              and query is not null
              and state is not null
            """
        )
        result = connection.execute(statement)
        tasks: list[DatabaseTask] = []
        for row in result.mappings().all():
            query = (row.get("query") or "").strip()
            if not query:
                continue
            duration = self._normalize_duration(row.get("duration"))
            if running_for is not None and (
                duration is None or duration < running_for
            ):
                continue
            tasks.append(
                DatabaseTask(
                    id=str(row.get("id")),
                    user=row.get("user_name"),
                    state=row.get("state"),
                    query=query,
                    duration=duration,
                )
            )
        return tasks

    def _collect_mysql(
        self, connection: Connection, *, running_for: int | None = None
    ) -> list[DatabaseTask]:
        current_id = connection.scalar(text("SELECT CONNECTION_ID()"))
        result = connection.execute(text("SHOW FULL PROCESSLIST"))
        tasks: list[DatabaseTask] = []

        for row in result.mappings().all():
            identifier = row.get("Id")
            if identifier is None:
                continue
            if current_id is not None and identifier == current_id:
                continue

            command = (row.get("Command") or "").strip().lower()
            if command == "sleep":
                continue

            info = row.get("Info")
            query = (str(info).strip() if info is not None else "")
            if not query:
                continue

            duration = self._normalize_duration(row.get("Time"))
            if running_for is not None and (
                duration is None or duration < running_for
            ):
                continue

            tasks.append(
                DatabaseTask(
                    id=str(identifier),
                    user=row.get("User"),
                    state=row.get("State") or row.get("Command"),
                    query=query,
                    duration=duration,
                )
            )
        return tasks

    @staticmethod
    def _normalize_duration(value: Any) -> int | None:
        if value is None:
            return None
        try:
            duration = int(value)
        except (TypeError, ValueError):
            return None
        if duration < 0:
            return 0
        return duration


class DatabaseKillTool(DatabaseTool):
    """Cancel a running database task by identifier.

    Args:
        task_id: Identifier of the task to cancel.

    Returns:
        True when cancellation succeeds on supported engines; otherwise False.
"""

    def __init__(
        self,
        engine: AsyncEngine,
        settings: DatabaseToolSettings,
        *,
        normalizer: IdentifierCaseNormalizer | None = None,
        table_cache: dict[str | None, dict[str, str]] | None = None,
    ) -> None:
        super().__init__(
            engine,
            settings,
            normalizer=normalizer,
            table_cache=table_cache,
        )
        self.__name__ = "kill"

    async def __call__(
        self,
        task_id: str,
        *,
        context: ToolCallContext,
    ) -> bool:
        assert task_id, "task_id must not be empty"
        if self._settings.delay_secs:
            await sleep(self._settings.delay_secs)

        async with self._engine.begin() as conn:
            return await conn.run_sync(self._kill, task_id=task_id)

    def _kill(self, connection: Connection, *, task_id: str) -> bool:
        dialect_name = getattr(connection.dialect, "name", None)

        if dialect_name == "postgresql":
            return self._kill_postgresql(connection, task_id)
        if dialect_name in {"mysql", "mariadb"}:
            return self._kill_mysql(connection, task_id)

        raise RuntimeError(
            "Killing tasks is not supported for "
            f"{dialect_name or 'unknown'} databases."
        )

    def _kill_postgresql(self, connection: Connection, task_id: str) -> bool:
        pid = self._parse_integer_task_id(task_id)
        statement = text(
            "SELECT pg_cancel_backend(:pid) AS cancelled"
        )
        result = connection.execute(statement, {"pid": pid})
        cancelled = result.scalar()
        return bool(cancelled)

    def _kill_mysql(self, connection: Connection, task_id: str) -> bool:
        pid = self._parse_integer_task_id(task_id)
        connection.execute(text("KILL :pid"), {"pid": pid})
        return True

    @staticmethod
    def _parse_integer_task_id(task_id: str) -> int:
        try:
            value = int(task_id)
        except ValueError as error:
            raise RuntimeError("Task identifier must be an integer value.") from error

        if value < 0:
            raise RuntimeError("Task identifier must be a positive integer.")
        return value


class DatabaseToolSet(ToolSet):
    _engine: AsyncEngine
    _settings: DatabaseToolSettings

    @override
    def __init__(
        self,
        settings: DatabaseToolSettings,
        *,
        exit_stack: AsyncExitStack | None = None,
        namespace: str | None = None,
    ):
        self._settings = settings
        self._engine = create_async_engine(
            self._settings.dsn, pool_pre_ping=True
        )
        _configure_read_only_engine(self._engine, self._settings.read_only)

        normalizer = (
            IdentifierCaseNormalizer(settings.identifier_case)
            if settings.identifier_case != "preserve"
            else None
        )
        table_cache: dict[str | None, dict[str, str]] = {}

        tools = [
            DatabaseCountTool(
                self._engine,
                settings,
                normalizer=normalizer,
                table_cache=table_cache,
            ),
            DatabaseInspectTool(
                self._engine,
                settings,
                normalizer=normalizer,
                table_cache=table_cache,
            ),
            DatabasePlanTool(
                self._engine,
                settings,
                normalizer=normalizer,
                table_cache=table_cache,
            ),
            DatabaseRunTool(
                self._engine,
                settings,
                normalizer=normalizer,
                table_cache=table_cache,
            ),
            DatabaseTablesTool(
                self._engine,
                settings,
                normalizer=normalizer,
                table_cache=table_cache,
            ),
            DatabaseTasksTool(
                self._engine,
                settings,
                normalizer=normalizer,
                table_cache=table_cache,
            ),
            DatabaseKillTool(
                self._engine,
                settings,
                normalizer=normalizer,
                table_cache=table_cache,
            ),
        ]
        super().__init__(
            exit_stack=exit_stack, namespace=namespace, tools=tools
        )

    @override
    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: BaseException | None,
    ) -> bool:
        try:
            if self._engine is not None:
                await self._engine.dispose()
        finally:
            return await super().__aexit__(exc_type, exc, tb)

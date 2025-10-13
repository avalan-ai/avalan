from avalan.entities import ToolCallContext
from avalan.tool.database import (
    DatabaseCountTool,
    DatabaseInspectTool,
    DatabaseKillTool,
    DatabasePlanTool,
    DatabaseRelationshipsTool,
    DatabaseRunTool,
    DatabaseTablesTool,
    DatabaseTasksTool,
    DatabaseTool,
    DatabaseToolSet,
    DatabaseToolSettings,
    TableRelationship,
    QueryPlan,
)
from sqlalchemy import create_engine, text
from sqlalchemy.exc import NoSuchTableError, OperationalError
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import AsyncMock, patch


def dummy_create_async_engine(dsn: str, **kwargs):
    engine = create_engine(dsn)

    class DummyAsyncConn:
        def __init__(self, conn):
            self.conn = conn

        async def exec_driver_sql(self, sql: str, *args, **kwargs):
            result = self.conn.exec_driver_sql(sql, *args, **kwargs)
            if not result.returns_rows:
                self.conn.commit()
            return result

        async def execute(self, stmt):
            result = self.conn.execute(stmt)
            if not result.returns_rows:
                self.conn.commit()
            return result

        async def run_sync(self, fn, *args, **kwargs):
            return fn(self.conn, *args, **kwargs)

    class DummyConnCtx:
        def __init__(self, engine):
            self.engine = engine
            self.conn = None

        async def __aenter__(self):
            self.conn = self.engine.connect()
            return DummyAsyncConn(self.conn)

        async def __aexit__(self, exc_type, exc, tb):
            self.conn.close()
            return False

    class DummyAsyncEngine:
        def __init__(self, engine):
            self.engine = engine
            self.disposed = False

        def connect(self):
            return DummyConnCtx(self.engine)

        def begin(self):
            return DummyConnCtx(self.engine)

        @property
        def sync_engine(self):
            return self.engine

        async def dispose(self):
            self.engine.dispose()
            self.disposed = True

    return DummyAsyncEngine(engine)


class DatabaseToolSetTestCase(IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.tmp = TemporaryDirectory()
        self.dsn = f"sqlite:///{self.tmp.name}/db.sqlite"
        engine = create_engine(self.dsn)
        with engine.begin() as conn:
            conn.execute(
                text("CREATE TABLE authors(id INTEGER PRIMARY KEY, name TEXT)")
            )
            conn.execute(
                text(
                    "CREATE TABLE books(id INTEGER PRIMARY KEY, author_id "
                    "INTEGER, title TEXT, FOREIGN KEY(author_id) REFERENCES "
                    "authors(id))"
                )
            )
            conn.execute(text("CREATE TABLE empty(id INTEGER PRIMARY KEY)"))
            conn.execute(text("INSERT INTO authors(name) VALUES ('Author')"))
            conn.execute(
                text("INSERT INTO books(author_id, title) VALUES (1, 'Book')")
            )
            conn.execute(
                text(
                    'CREATE TABLE "CamelCase"(id INTEGER PRIMARY KEY, value'
                    " TEXT)"
                )
            )
            conn.execute(
                text("INSERT INTO \"CamelCase\"(value) VALUES ('Item')")
            )
        engine.dispose()
        self.patcher = patch(
            "avalan.tool.database.create_async_engine",
            dummy_create_async_engine,
        )
        self.patcher.start()
        self.settings = DatabaseToolSettings(dsn=self.dsn)

    async def asyncTearDown(self) -> None:
        self.patcher.stop()
        self.tmp.cleanup()

    async def test_run_tool_returns_rows(self):
        engine = dummy_create_async_engine(self.dsn)
        tool = DatabaseRunTool(engine, self.settings)
        rows = await tool(
            "SELECT id, title FROM books", context=ToolCallContext()
        )
        self.assertEqual(rows, [{"id": 1, "title": "Book"}])
        await engine.dispose()

    async def test_run_tool_disallows_insert_by_default(self):
        engine = dummy_create_async_engine(self.dsn)
        tool = DatabaseRunTool(engine, self.settings)
        with self.assertRaises(PermissionError):
            await tool(
                "INSERT INTO authors(name) VALUES ('Other')",
                context=ToolCallContext(),
            )
        await engine.dispose()

    async def test_run_tool_disallows_multiple_statements(self):
        engine = dummy_create_async_engine(self.dsn)
        tool = DatabaseRunTool(engine, self.settings)
        with self.assertRaises(PermissionError):
            await tool(
                "SELECT * FROM authors; SELECT * FROM books",
                context=ToolCallContext(),
            )
        await engine.dispose()

    async def test_run_tool_allows_insert_when_enabled(self):
        settings = DatabaseToolSettings(
            dsn=self.dsn,
            read_only=False,
            allowed_commands=["select", "insert"],
        )
        engine = dummy_create_async_engine(self.dsn)
        tool = DatabaseRunTool(engine, settings)
        rows = await tool(
            "INSERT INTO authors(name) VALUES ('Other')",
            context=ToolCallContext(),
        )
        self.assertEqual(rows, [])

        authors = await tool(
            "SELECT name FROM authors ORDER BY id",
            context=ToolCallContext(),
        )
        self.assertEqual(
            [row["name"] for row in authors],
            ["Author", "Other"],
        )
        with engine.engine.begin() as cleanup:
            cleanup.execute(text("DELETE FROM authors WHERE name = 'Other'"))
        await engine.dispose()

    async def test_plan_tool_returns_query_plan(self):
        engine = dummy_create_async_engine(self.dsn)
        tool = DatabasePlanTool(engine, self.settings)
        plan = await tool(
            "SELECT id, title FROM books", context=ToolCallContext()
        )
        self.assertIsInstance(plan, QueryPlan)
        self.assertEqual(plan.dialect, "sqlite")
        self.assertTrue(plan.steps)
        self.assertIn("detail", plan.steps[0])
        await engine.dispose()

    async def test_plan_tool_disallows_insert_by_default(self):
        engine = dummy_create_async_engine(self.dsn)
        tool = DatabasePlanTool(engine, self.settings)
        with self.assertRaises(PermissionError):
            await tool(
                "INSERT INTO authors(name) VALUES ('Other')",
                context=ToolCallContext(),
            )
        await engine.dispose()

    async def test_plan_tool_respects_configured_delay(self):
        settings = DatabaseToolSettings(dsn=self.dsn, delay_secs=0.01)
        engine = dummy_create_async_engine(self.dsn)
        tool = DatabasePlanTool(engine, settings)

        calls: list[float] = []

        async def fake_sleep(delay: float) -> None:
            calls.append(delay)

        with patch("avalan.tool.database.sleep", new=fake_sleep):
            await tool("SELECT id FROM books", context=ToolCallContext())

        self.assertEqual(calls, [0.01])
        await engine.dispose()

    async def test_relationships_tool_returns_relationships(self):
        engine = dummy_create_async_engine(self.dsn)
        tool = DatabaseRelationshipsTool(engine, self.settings)

        books_relationships = await tool(
            "books", context=ToolCallContext()
        )
        self.assertEqual(
            books_relationships,
            [
                TableRelationship(
                    direction="outgoing",
                    local_columns=("author_id",),
                    related_table="authors",
                    related_columns=("id",),
                    constraint_name=None,
                )
            ],
        )

        authors_relationships = await tool(
            "authors", context=ToolCallContext()
        )
        self.assertEqual(
            authors_relationships,
            [
                TableRelationship(
                    direction="incoming",
                    local_columns=("id",),
                    related_table="books",
                    related_columns=("author_id",),
                    constraint_name=None,
                )
            ],
        )
        await engine.dispose()

    async def test_tables_tool_lists_tables(self):
        engine = dummy_create_async_engine(self.dsn)
        tool = DatabaseTablesTool(engine, self.settings)
        tables = await tool(context=ToolCallContext())
        self.assertIn("main", tables)
        self.assertIn("authors", tables["main"])
        self.assertIn("books", tables["main"])
        await engine.dispose()

    async def test_tasks_tool_returns_empty_for_sqlite(self):
        engine = dummy_create_async_engine(self.dsn)
        tool = DatabaseTasksTool(engine, self.settings)
        tasks = await tool(context=ToolCallContext())
        self.assertEqual(tasks, [])
        await engine.dispose()

    async def test_tasks_tool_accepts_zero_running_for(self):
        engine = dummy_create_async_engine(self.dsn)
        tool = DatabaseTasksTool(engine, self.settings)
        tasks = await tool(running_for=0, context=ToolCallContext())
        self.assertEqual(tasks, [])
        await engine.dispose()

    async def test_kill_tool_not_supported_for_sqlite(self):
        engine = dummy_create_async_engine(self.dsn)
        tool = DatabaseKillTool(engine, self.settings)
        with self.assertRaises(RuntimeError):
            await tool("1", context=ToolCallContext())
        await engine.dispose()

    async def test_inspect_tool_returns_schemas(self):
        engine = dummy_create_async_engine(self.dsn)
        tool = DatabaseInspectTool(engine, self.settings)
        tables = await tool(["books", "authors"], context=ToolCallContext())
        self.assertEqual(len(tables), 2)

        books, authors = tables
        self.assertEqual(books.name, "books")
        self.assertEqual(books.columns["id"], "INTEGER")
        self.assertEqual(books.columns["title"], "TEXT")
        self.assertEqual(len(books.foreign_keys), 1)
        fk = books.foreign_keys[0]
        self.assertEqual(fk.field, "author_id")
        self.assertEqual(fk.ref_table, "main.authors")
        self.assertEqual(fk.ref_field, "id")

        self.assertEqual(authors.name, "authors")
        self.assertIn("name", authors.columns)
        self.assertEqual(authors.columns["name"], "TEXT")
        self.assertEqual(authors.foreign_keys, [])
        await engine.dispose()

    async def test_count_tool_returns_count(self):
        engine = dummy_create_async_engine(self.dsn)
        tool = DatabaseCountTool(engine, self.settings)
        count = await tool("authors", context=ToolCallContext())
        self.assertEqual(count, 1)
        await engine.dispose()

    async def test_count_tool_empty_table(self):
        engine = dummy_create_async_engine(self.dsn)
        tool = DatabaseCountTool(engine, self.settings)
        count = await tool("empty", context=ToolCallContext())
        self.assertEqual(count, 0)
        await engine.dispose()

    async def test_count_tool_wrong_table_error(self):
        engine = dummy_create_async_engine(self.dsn)
        tool = DatabaseCountTool(engine, self.settings)
        with self.assertRaises(OperationalError):
            await tool("missing", context=ToolCallContext())
        await engine.dispose()

    async def test_tools_respect_delay_setting(self):
        settings = DatabaseToolSettings(dsn=self.dsn, delay_secs=1)
        engine = dummy_create_async_engine(self.dsn)
        with patch("avalan.tool.database.sleep", AsyncMock()) as mocked_sleep:
            count = DatabaseCountTool(engine, settings)
            await count("authors", context=ToolCallContext())

            inspect = DatabaseInspectTool(engine, settings)
            await inspect(["authors"], context=ToolCallContext())

            run = DatabaseRunTool(engine, settings)
            await run("SELECT id FROM authors", context=ToolCallContext())

            tables = DatabaseTablesTool(engine, settings)
            await tables(context=ToolCallContext())

            tasks_tool = DatabaseTasksTool(engine, settings)
            await tasks_tool(context=ToolCallContext())

            kill_tool = DatabaseKillTool(engine, settings)
            with self.assertRaises(RuntimeError):
                await kill_tool("1", context=ToolCallContext())

            self.assertEqual(mocked_sleep.await_count, 6)
        await engine.dispose()

    async def test_tables_tool_respects_identifier_case(self):
        settings = DatabaseToolSettings(dsn=self.dsn, identifier_case="lower")
        engine = dummy_create_async_engine(self.dsn)
        tool = DatabaseTablesTool(engine, settings)
        tables = await tool(context=ToolCallContext())
        self.assertIn("camelcase", tables["main"])
        self.assertNotIn("CamelCase", tables["main"])
        await engine.dispose()

    async def test_run_tool_rewrites_identifier_case(self):
        settings = DatabaseToolSettings(dsn=self.dsn, identifier_case="lower")
        engine = dummy_create_async_engine(self.dsn)
        tool = DatabaseRunTool(engine, settings)
        rows = await tool(
            "SELECT id, value FROM camelcase", context=ToolCallContext()
        )
        self.assertEqual(rows, [{"id": 1, "value": "Item"}])
        await engine.dispose()

    async def test_count_tool_with_normalized_identifier(self):
        settings = DatabaseToolSettings(dsn=self.dsn, identifier_case="lower")
        engine = dummy_create_async_engine(self.dsn)
        tool = DatabaseCountTool(engine, settings)
        count = await tool("camelcase", context=ToolCallContext())
        self.assertEqual(count, 1)
        await engine.dispose()

    async def test_inspect_tool_with_normalized_identifier(self):
        settings = DatabaseToolSettings(dsn=self.dsn, identifier_case="lower")
        engine = dummy_create_async_engine(self.dsn)
        tool = DatabaseInspectTool(engine, settings)
        tables = await tool(["camelcase"], context=ToolCallContext())
        self.assertEqual(tables[0].name, "camelcase")
        self.assertIn("value", tables[0].columns)
        await engine.dispose()

    async def test_toolset_reuses_engine_and_disposes(self):
        async with DatabaseToolSet(self.settings) as toolset:
            tools_by_name = {
                tool.__name__: tool
                for tool in toolset.tools
                if hasattr(tool, "__name__")
            }
            count_tool = tools_by_name["count"]
            inspect_tool = tools_by_name["inspect"]
            plan_tool = tools_by_name["plan"]
            run_tool = tools_by_name["run"]
            tables_tool = tools_by_name["tables"]
            tasks_tool = tools_by_name["tasks"]
            kill_tool = tools_by_name["kill"]
            count = await count_tool("authors", context=ToolCallContext())
            self.assertEqual(count, 1)
            plan = await plan_tool(
                "SELECT id FROM authors", context=ToolCallContext()
            )
            self.assertIsInstance(plan, QueryPlan)
            rows = await run_tool(
                "SELECT id FROM authors", context=ToolCallContext()
            )
            self.assertEqual(rows, [{"id": 1}])
            table = await inspect_tool(["books"], context=ToolCallContext())
            self.assertEqual(table[0].name, "books")
            tables = await tables_tool(context=ToolCallContext())
            self.assertIn("books", tables["main"])
            self.assertEqual(await tasks_tool(context=ToolCallContext()), [])
            with self.assertRaises(RuntimeError):
                await kill_tool("1", context=ToolCallContext())
        self.assertTrue(toolset._engine.disposed)

    async def test_toolset_read_only_blocks_writes(self):
        settings = DatabaseToolSettings(
            dsn=self.dsn,
            allowed_commands=["insert"],
        )
        async with DatabaseToolSet(settings) as toolset:
            tools_by_name = {
                tool.__name__: tool
                for tool in toolset.tools
                if hasattr(tool, "__name__")
            }
            run_tool = tools_by_name["run"]
            with self.assertRaises(OperationalError):
                await run_tool(
                    "INSERT INTO authors(name) VALUES ('Blocked')",
                    context=ToolCallContext(),
                )


class DatabaseInspectCollectTestCase(TestCase):
    def test_collect_prefixes_schema_name(self):
        inspector = SimpleNamespace(
            default_schema_name="public",
            get_columns=lambda table_name, schema: [
                {"name": "id", "type": "INTEGER"}
            ],
            get_foreign_keys=lambda table_name, schema: [],
        )

        tool = DatabaseInspectTool(
            SimpleNamespace(), DatabaseToolSettings(dsn="sqlite:///db.sqlite")
        )

        with (
            patch("avalan.tool.database.inspect", return_value=inspector),
            patch(
                "avalan.tool.database.DatabaseTool._schemas",
                return_value=("public", []),
            ),
        ):
            tables = tool._collect(
                SimpleNamespace(), schema="other", table_names=["t"]
            )
        self.assertEqual(tables[0].name, "other.t")

    def test_collect_skips_missing_tables_and_missing_foreign_keys(self):
        def get_columns(table_name, schema):
            if table_name == "missing":
                raise NoSuchTableError(table_name)
            return [{"name": "id", "type": "INTEGER"}]

        def get_foreign_keys(table_name, schema):
            raise NoSuchTableError(table_name)

        inspector = SimpleNamespace(
            default_schema_name="public",
            get_columns=get_columns,
            get_foreign_keys=get_foreign_keys,
        )

        tool = DatabaseInspectTool(
            SimpleNamespace(), DatabaseToolSettings(dsn="sqlite:///db.sqlite")
        )

        with (
            patch("avalan.tool.database.inspect", return_value=inspector),
            patch(
                "avalan.tool.database.DatabaseTool._schemas",
                return_value=("public", []),
            ),
        ):
            tables = tool._collect(
                SimpleNamespace(),
                schema=None,
                table_names=["missing", "present"],
            )

        self.assertEqual([t.name for t in tables], ["present"])
        self.assertEqual(tables[0].foreign_keys, [])


class DatabaseRelationshipsCollectTestCase(TestCase):
    def test_collect_incoming_relationships_include_schema_prefix(self):
        def get_table_names(schema=None):
            if schema == "public":
                return ["authors", "books"]
            if schema == "sales":
                return ["orders"]
            return []

        def get_foreign_keys(table_name, schema=None):
            if table_name == "authors":
                return []
            if table_name == "orders" and schema == "sales":
                return [
                    {
                        "name": "fk_orders_authors",
                        "constrained_columns": ["author_id"],
                        "referred_table": "authors",
                        "referred_columns": ["id"],
                        "referred_schema": "public",
                    }
                ]
            return []

        inspector = SimpleNamespace(
            default_schema_name="public",
            get_columns=lambda table_name, schema: [
                {"name": "id", "type": "INTEGER"}
            ],
            get_foreign_keys=get_foreign_keys,
            get_table_names=get_table_names,
        )

        tool = DatabaseRelationshipsTool(
            SimpleNamespace(), DatabaseToolSettings(dsn="sqlite:///db.sqlite")
        )

        with (
            patch("avalan.tool.database.inspect", return_value=inspector),
            patch(
                "avalan.tool.database.DatabaseTool._schemas",
                return_value=("public", ["public", "sales"]),
            ),
        ):
            relationships = tool._collect(
                SimpleNamespace(dialect=SimpleNamespace(name="postgresql")),
                schema="public",
                table_name="authors",
            )

        self.assertEqual(
            relationships,
            [
                TableRelationship(
                    direction="incoming",
                    local_columns=("id",),
                    related_table="sales.orders",
                    related_columns=("author_id",),
                    constraint_name="fk_orders_authors",
                )
            ],
        )

    def test_collect_handles_all_supported_vendors(self):
        vendors = [
            ("sqlite", "main"),
            ("postgresql", "public"),
            ("mysql", "test"),
            ("mariadb", "test"),
            ("mssql", "dbo"),
            ("oracle", "SYSTEM"),
        ]

        for dialect, default_schema in vendors:
            def get_table_names(schema=None):
                normalized = schema if schema is not None else default_schema
                if normalized == default_schema:
                    return ["authors", "books"]
                return []

            def get_foreign_keys(table_name, schema=None):
                if table_name == "authors":
                    return [
                        {
                            "name": "fk_authors_leads",
                            "constrained_columns": ["lead_id"],
                            "referred_table": "leads",
                            "referred_columns": ["id"],
                        }
                    ]
                if table_name == "books":
                    return [
                        {
                            "name": "fk_books_authors",
                            "constrained_columns": ["author_id"],
                            "referred_table": "authors",
                            "referred_columns": ["id"],
                        }
                    ]
                return []

            inspector = SimpleNamespace(
                default_schema_name=default_schema,
                get_columns=lambda table_name, schema: [
                    {"name": "id", "type": "INTEGER"}
                ],
                get_foreign_keys=get_foreign_keys,
                get_table_names=get_table_names,
            )

            tool = DatabaseRelationshipsTool(
                SimpleNamespace(),
                DatabaseToolSettings(dsn="sqlite:///db.sqlite"),
            )

            with (
                patch("avalan.tool.database.inspect", return_value=inspector),
                patch(
                    "avalan.tool.database.DatabaseTool._schemas",
                    return_value=(default_schema, [default_schema]),
                ),
            ):
                relationships = tool._collect(
                    SimpleNamespace(dialect=SimpleNamespace(name=dialect)),
                    schema=default_schema,
                    table_name="authors",
                )

            self.assertEqual(
                relationships,
                [
                    TableRelationship(
                        direction="outgoing",
                        local_columns=("lead_id",),
                        related_table="leads",
                        related_columns=("id",),
                        constraint_name="fk_authors_leads",
                    ),
                    TableRelationship(
                        direction="incoming",
                        local_columns=("id",),
                        related_table="books",
                        related_columns=("author_id",),
                        constraint_name="fk_books_authors",
                    ),
                ],
            )

class DatabaseSchemasTestCase(TestCase):
    def test_postgresql_schemas(self):
        connection = SimpleNamespace(
            dialect=SimpleNamespace(name="postgresql")
        )
        inspector = SimpleNamespace(
            default_schema_name="public",
            get_schema_names=lambda: [
                "information_schema",
                "pg_catalog",
                "other",
            ],
        )
        default, schemas = DatabaseTool._schemas(connection, inspector)
        self.assertEqual(default, "public")
        self.assertEqual(schemas, ["other", "public"])

    def test_non_postgresql_adds_default_schema(self):
        connection = SimpleNamespace(dialect=SimpleNamespace(name="sqlite"))
        inspector = SimpleNamespace(
            default_schema_name="main",
            get_schema_names=lambda: ["temp"],
        )
        default, schemas = DatabaseTool._schemas(connection, inspector)
        self.assertEqual(default, "main")
        self.assertEqual(schemas, ["temp", "main"])

    def test_non_postgresql_handles_only_sys_schemas(self):
        connection = SimpleNamespace(dialect=SimpleNamespace(name="mysql"))
        inspector = SimpleNamespace(
            default_schema_name="public",
            get_schema_names=lambda: ["information_schema", "sys"],
        )
        default, schemas = DatabaseTool._schemas(connection, inspector)
        self.assertEqual(default, "public")
        self.assertEqual(schemas, ["public"])


class DatabaseSplitSchemaTestCase(TestCase):
    def test_split_schema_and_table(self):
        schema, table = DatabaseCountTool._split_schema_and_table(
            "main.authors"
        )
        self.assertEqual(schema, "main")
        self.assertEqual(table, "authors")

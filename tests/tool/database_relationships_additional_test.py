import sys
from types import ModuleType, SimpleNamespace
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import AsyncMock, patch

from sqlalchemy.exc import NoSuchTableError

stub_utils = ModuleType("transformers.utils")
stub_utils.get_json_schema = lambda tool: {}
stub_transformers = ModuleType("transformers")
stub_transformers.utils = stub_utils
sys.modules.setdefault("transformers", stub_transformers)
sys.modules.setdefault("transformers.utils", stub_utils)


def _stub_callable(*args, **kwargs):
    return SimpleNamespace(args=args, kwargs=kwargs)


class _InferenceMode:
    def __call__(self, function):
        def wrapper(*args, **kwargs):
            return function(*args, **kwargs)

        return wrapper

    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


stub_torch = ModuleType("torch")
stub_torch.dtype = type("dtype", (), {})
stub_torch.Tensor = type("Tensor", (), {})
stub_torch.argmax = _stub_callable
stub_torch.softmax = _stub_callable
stub_torch.log_softmax = _stub_callable
stub_torch.topk = _stub_callable
stub_torch.gumbel_softmax = _stub_callable
stub_torch.from_numpy = _stub_callable
stub_torch.unique = _stub_callable
stub_torch.tensor = _stub_callable
stub_torch.inference_mode = _InferenceMode()
stub_torch.Generator = type("Generator", (), {})

cuda_module = ModuleType("torch.cuda")
cuda_module.device_count = lambda: 0
cuda_module.is_available = lambda: False
cuda_module.set_device = _stub_callable
sys.modules.setdefault("torch.cuda", cuda_module)

distributed_module = ModuleType("torch.distributed")
distributed_module.destroy_process_group = _stub_callable
sys.modules.setdefault("torch.distributed", distributed_module)

backends_module = ModuleType("torch.backends")
backends_module.mps = SimpleNamespace(is_available=lambda: False)
sys.modules.setdefault("torch.backends", backends_module)

nn_functional_module = ModuleType("torch.nn.functional")
nn_functional_module.gumbel_softmax = _stub_callable
sys.modules.setdefault("torch.nn.functional", nn_functional_module)

stub_torch.cuda = cuda_module
stub_torch.distributed = distributed_module
stub_torch.backends = backends_module
stub_torch.nn = SimpleNamespace(functional=nn_functional_module)

sys.modules.setdefault("torch", stub_torch)


class _DummyConnection:
    async def run_sync(self, fn, *args, **kwargs):
        return fn(SimpleNamespace(), *args, **kwargs)


class _DummyConnectionContext:
    async def __aenter__(self):
        return _DummyConnection()

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _DummyEngine:
    def connect(self):
        return _DummyConnectionContext()


from avalan.entities import ToolCallContext
from avalan.tool.database import (
    DatabaseRelationshipsTool,
    DatabaseToolSettings,
    TableRelationship,
)


class DatabaseRelationshipsEdgeCaseTestCase(TestCase):
    def setUp(self) -> None:
        self.tool = DatabaseRelationshipsTool(
            SimpleNamespace(),
            DatabaseToolSettings(dsn="sqlite:///db.sqlite"),
        )

    def test_collect_returns_empty_when_table_missing(self) -> None:
        inspector = SimpleNamespace(
            default_schema_name="public",
            get_columns=lambda table_name, schema=None: (_ for _ in ()).throw(
                NoSuchTableError(table_name)
            ),
            get_foreign_keys=lambda table_name, schema=None: [],
            get_table_names=lambda schema=None: [],
        )

        connection = SimpleNamespace(dialect=SimpleNamespace(name="postgresql"))

        with (
            patch.object(
                DatabaseRelationshipsTool,
                "_inspect_connection",
                return_value=inspector,
            ),
            patch(
                "avalan.tool.database.DatabaseTool._schemas",
                return_value=("public", ["public"]),
            ),
        ):
            relationships = self.tool._collect(
                connection,
                schema="public",
                table_name="authors",
            )

        self.assertEqual(relationships, [])

    def test_collect_outgoing_handles_missing_tables_and_relationships(self) -> None:
        inspector_missing = SimpleNamespace(
            get_foreign_keys=lambda table_name, schema=None: (_ for _ in ()).throw(
                NoSuchTableError(table_name)
            )
        )

        self.assertEqual(
            self.tool._collect_outgoing(
                inspector_missing,
                "authors",
                "public",
                "public",
            ),
            [],
        )

        inspector = SimpleNamespace(
            get_foreign_keys=lambda table_name, schema=None: [
                {"name": "fk_missing"},
                {
                    "name": "fk_valid",
                    "constrained_columns": ["author_id"],
                    "referred_table": "authors",
                    "referred_columns": ["id"],
                },
            ]
        )

        relationships = self.tool._collect_outgoing(
            inspector,
            "books",
            "public",
            "public",
        )

        self.assertEqual(
            relationships,
            [
                TableRelationship(
                    direction="outgoing",
                    local_columns=("author_id",),
                    related_table="authors",
                    related_columns=("id",),
                    constraint_name="fk_valid",
                )
            ],
        )

    def test_collect_combines_outgoing_and_incoming(self) -> None:
        def get_foreign_keys(table_name, schema=None):
            if table_name == "authors" and schema == "public":
                return [
                    {
                        "name": "fk_authors_publishers",
                        "constrained_columns": ["publisher_id"],
                        "referred_table": "publishers",
                        "referred_columns": ["id"],
                    }
                ]
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
            get_columns=lambda table_name, schema=None: [{"name": "id"}],
            get_foreign_keys=get_foreign_keys,
            get_table_names=lambda schema=None: ["orders"] if schema == "sales" else [],
        )

        connection = SimpleNamespace(dialect=SimpleNamespace(name="postgresql"))

        with (
            patch.object(
                DatabaseRelationshipsTool,
                "_inspect_connection",
                return_value=inspector,
            ),
            patch(
                "avalan.tool.database.DatabaseTool._schemas",
                return_value=("public", [None, "public", "sales"]),
            ),
        ):
            relationships = self.tool._collect(
                connection,
                schema="public",
                table_name="authors",
            )

        self.assertEqual(
            relationships,
            [
                TableRelationship(
                    direction="outgoing",
                    local_columns=("publisher_id",),
                    related_table="publishers",
                    related_columns=("id",),
                    constraint_name="fk_authors_publishers",
                ),
                TableRelationship(
                    direction="incoming",
                    local_columns=("id",),
                    related_table="sales.orders",
                    related_columns=("author_id",),
                    constraint_name="fk_orders_authors",
                ),
            ],
        )

    def test_collect_incoming_skips_invalid_foreign_keys(self) -> None:
        def get_table_names(schema=None):
            if schema == "broken":
                raise RuntimeError("broken inspector")
            if schema == "public":
                return ["authors"]
            if schema == "analytics":
                return ["events"]
            if schema == "sales":
                return ["orphans", "orders", "tickets"]
            return []

        def get_foreign_keys(table_name, schema=None):
            if schema == "analytics":
                raise NoSuchTableError(table_name)
            if schema == "sales":
                if table_name == "orphans":
                    return [
                        {
                            "name": "fk_orphans_customers",
                            "constrained_columns": ["customer_id"],
                            "referred_table": "customers",
                            "referred_columns": ["id"],
                        }
                    ]
                if table_name == "orders":
                    return [
                        {
                            "name": "fk_orders_authors_archive",
                            "constrained_columns": ["author_id"],
                            "referred_table": "authors",
                            "referred_columns": ["id"],
                            "referred_schema": "archive",
                        }
                    ]
                if table_name == "tickets":
                    return [
                        {
                            "name": "fk_tickets_missing",
                            "constrained_columns": ["author_id"],
                            "referred_columns": ["id"],
                        }
                    ]
            return []

        inspector = SimpleNamespace(
            get_table_names=get_table_names,
            get_foreign_keys=get_foreign_keys,
        )

        relationships = self.tool._collect_incoming(
            inspector,
            "authors",
            "public",
            "public",
            [None, "public", "broken", "analytics", "sales"],
        )

        self.assertEqual(relationships, [])

    def test_normalize_schema_returns_default_when_missing(self) -> None:
        self.assertEqual(
            DatabaseRelationshipsTool._normalize_schema(None, "public"),
            "public",
        )


class DatabaseRelationshipsCallTestCase(IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.tool = DatabaseRelationshipsTool(
            _DummyEngine(),
            DatabaseToolSettings(dsn="sqlite:///db.sqlite"),
        )

    async def test_call_invokes_collect(self) -> None:
        relationship = TableRelationship(
            direction="outgoing",
            local_columns=("id",),
            related_table="authors",
            related_columns=("id",),
            constraint_name=None,
        )

        with (
            patch.object(
                self.tool,
                "_sleep_if_configured",
                new=AsyncMock(),
            ) as sleep_mock,
            patch.object(
                self.tool,
                "_collect",
                return_value=[relationship],
            ) as collect_mock,
        ):
            result = await self.tool(
                "authors",
                schema="public",
                context=ToolCallContext(),
            )

        self.assertEqual(result, [relationship])
        sleep_mock.assert_awaited()
        collect_mock.assert_called_once()
        _, kwargs = collect_mock.call_args
        self.assertEqual(kwargs["schema"], "public")
        self.assertEqual(kwargs["table_name"], "authors")

    async def test_call_requires_table_name(self) -> None:
        with patch.object(
            self.tool,
            "_sleep_if_configured",
            new=AsyncMock(),
        ):
            with self.assertRaises(AssertionError):
                await self.tool("", context=ToolCallContext())

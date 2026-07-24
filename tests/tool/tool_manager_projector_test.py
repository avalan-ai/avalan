from unittest import IsolatedAsyncioTestCase, main

from avalan.entities import (
    PreparedToolCall,
    ToolCall,
    ToolCallContext,
    ToolCallDiagnostic,
    ToolCallDiagnosticCode,
    ToolCallResult,
    ToolFilterResult,
    ToolFilterResultStatus,
    ToolManagerSettings,
)
from avalan.model import ModelCapabilityCatalog
from avalan.tool import Tool, ToolSet
from avalan.tool.display import ToolDisplayProjection
from avalan.tool.manager import ToolManager


class ToolManagerProjectorTestCase(IsolatedAsyncioTestCase):
    async def test_tool_subclass_projector_exposes_descriptor_hook(
        self,
    ) -> None:
        tool = NativeProjectingTool()
        manager = ToolManager.create_instance(
            enable_tools=["native_projecting"],
            available_toolsets=[ToolSet(tools=[tool])],
            settings=ToolManagerSettings(),
        )

        descriptor = manager.describe_tool("native_projecting")

        assert descriptor is not None
        self.assertIsNotNone(descriptor.display_projector)
        projection = descriptor.project_display(
            ToolCall(
                id="call-1",
                name="native_projecting",
                arguments={"value": "hello"},
            )
        )
        self.assertIsInstance(projection, ToolDisplayProjection)
        assert isinstance(projection, ToolDisplayProjection)
        self.assertEqual(projection.action, "inspect")
        self.assertEqual(projection.target, "native_projecting")

    async def test_plain_function_without_projector_still_executes(
        self,
    ) -> None:
        manager = ToolManager.create_instance(
            enable_tools=["plain_echo"],
            available_toolsets=[ToolSet(tools=[plain_echo])],
            settings=ToolManagerSettings(),
        )
        descriptor = manager.describe_tool("plain_echo")

        outcome = await manager.execute_call(
            ToolCall(
                id="call-1",
                name="plain_echo",
                arguments={"value": "hello"},
            ),
            context=ToolCallContext(),
        )

        assert descriptor is not None
        self.assertIsNone(descriptor.display_projector)
        self.assertIsInstance(outcome, ToolCallResult)
        assert isinstance(outcome, ToolCallResult)
        self.assertEqual(outcome.result, "hello")

    async def test_plain_function_projector_is_discovered(self) -> None:
        manager = ToolManager.create_instance(
            enable_tools=["projected_echo"],
            available_toolsets=[ToolSet(tools=[projected_echo])],
            settings=ToolManagerSettings(),
        )

        descriptor = manager.describe_tool("projected_echo")

        assert descriptor is not None
        self.assertIsNotNone(descriptor.display_projector)
        projection = descriptor.project_display(
            ToolCall(
                id="call-1",
                name="projected_echo",
                arguments={"value": "hello"},
            )
        )
        self.assertIsInstance(projection, ToolDisplayProjection)
        assert isinstance(projection, ToolDisplayProjection)
        self.assertEqual(projection.target, "projected_echo")

    async def test_alias_resolution_preserves_canonical_projector_access(
        self,
    ) -> None:
        manager = ToolManager.create_instance(
            enable_tools=["aliased_projected_echo"],
            available_toolsets=[ToolSet(tools=[aliased_projected_echo])],
            settings=ToolManagerSettings(),
        )

        prepared = await manager.prepare_call(
            ToolCall(
                id="call-1",
                name="echo_alias",
                arguments={"value": "hello"},
            ),
            context=ToolCallContext(),
        )

        self.assertIsInstance(prepared, PreparedToolCall)
        assert isinstance(prepared, PreparedToolCall)
        self.assertEqual(prepared.call.name, "aliased_projected_echo")
        self.assertIs(
            prepared.descriptor,
            manager.describe_tool("echo_alias"),
        )
        projection = prepared.descriptor.project_display(prepared.call)
        self.assertIsInstance(projection, ToolDisplayProjection)
        assert isinstance(projection, ToolDisplayProjection)
        self.assertEqual(projection.target, "aliased_projected_echo")

    async def test_raising_projector_does_not_fail_preparation_or_execution(
        self,
    ) -> None:
        manager = ToolManager.create_instance(
            enable_tools=["raising_projector_echo"],
            available_toolsets=[ToolSet(tools=[raising_projector_echo])],
            settings=ToolManagerSettings(),
        )

        prepared = await manager.prepare_call(
            ToolCall(
                id="call-1",
                name="raising_projector_echo",
                arguments={"value": "hello"},
            ),
            context=ToolCallContext(),
        )

        self.assertIsInstance(prepared, PreparedToolCall)
        assert isinstance(prepared, PreparedToolCall)
        self.assertIsNone(prepared.descriptor.project_display(prepared.call))
        outcome = await manager.execute_prepared_call(prepared)
        self.assertIsInstance(outcome, ToolCallResult)
        assert isinstance(outcome, ToolCallResult)
        self.assertEqual(outcome.result, "hello")

    async def test_invalid_projector_data_is_ignored_by_safe_helper(
        self,
    ) -> None:
        manager = ToolManager.create_instance(
            enable_tools=["invalid_projector_echo"],
            available_toolsets=[ToolSet(tools=[invalid_projector_echo])],
            settings=ToolManagerSettings(),
        )
        descriptor = manager.describe_tool("invalid_projector_echo")

        assert descriptor is not None
        projection = descriptor.project_display(
            ToolCall(
                id="call-1",
                name="invalid_projector_echo",
                arguments={"value": "hello"},
            )
        )

        self.assertIsNone(projection)

    async def test_raw_projection_payload_is_ignored_by_safe_helper(
        self,
    ) -> None:
        manager = ToolManager.create_instance(
            enable_tools=["raw_payload_projector_echo"],
            available_toolsets=[ToolSet(tools=[raw_payload_projector_echo])],
            settings=ToolManagerSettings(),
        )
        descriptor = manager.describe_tool("raw_payload_projector_echo")

        assert descriptor is not None
        projection = descriptor.project_display(
            ToolCall(
                id="call-1",
                name="raw_payload_projector_echo",
                arguments={"value": "hello"},
            )
        )

        self.assertIsNone(projection)

    async def test_disabled_and_unknown_tools_do_not_access_projectors(
        self,
    ) -> None:
        manager = ToolManager.create_instance(
            enable_tools=[],
            available_toolsets=[ToolSet(tools=[ProjectorAccessGuard()])],
            settings=ToolManagerSettings(),
        )

        disabled = await manager.prepare_call(
            ToolCall(
                id="call-1",
                name="guarded",
                arguments={"value": "hello"},
            ),
            context=ToolCallContext(),
        )
        unknown = await manager.prepare_call(
            ToolCall(
                id="call-2",
                name="missing",
                arguments={"value": "hello"},
            ),
            context=ToolCallContext(),
        )

        self.assertIsNone(manager.describe_tool("guarded"))
        self.assertIsInstance(disabled, ToolCallDiagnostic)
        assert isinstance(disabled, ToolCallDiagnostic)
        self.assertIs(disabled.code, ToolCallDiagnosticCode.DISABLED_TOOL)
        self.assertIsInstance(unknown, ToolCallDiagnostic)
        assert isinstance(unknown, ToolCallDiagnostic)
        self.assertIs(unknown.code, ToolCallDiagnosticCode.UNKNOWN_TOOL)

    async def test_model_schemas_exclude_projector_metadata(
        self,
    ) -> None:
        manager = ToolManager.create_instance(
            enable_tools=["projected_echo"],
            available_toolsets=[ToolSet(tools=[projected_echo])],
            settings=ToolManagerSettings(),
        )
        descriptor = manager.describe_tool("projected_echo")
        projection = ModelCapabilityCatalog.create(
            manager.export_model_capability_seed()
        ).project()

        assert descriptor is not None
        schemas = [
            descriptor.schema,
            *projection.schemas,
        ]
        for schema in schemas:
            with self.subTest(schema=schema):
                self.assertFalse(_has_key(schema, "tool_display_projector"))
                self.assertFalse(_has_key(schema, "display_projector"))

    async def test_filter_rewrite_preserves_projector_descriptor_access(
        self,
    ) -> None:
        def rename(
            call: ToolCall,
            context: ToolCallContext,
        ) -> ToolFilterResult:
            return ToolFilterResult(
                status=ToolFilterResultStatus.MODIFY,
                call=ToolCall(
                    id=call.id,
                    name="projected_echo",
                    arguments=call.arguments,
                ),
                context=context,
            )

        manager = ToolManager.create_instance(
            enable_tools=["plain_echo", "projected_echo"],
            available_toolsets=[ToolSet(tools=[plain_echo, projected_echo])],
            settings=ToolManagerSettings(filters=[rename]),
        )

        prepared = await manager.prepare_call(
            ToolCall(
                id="call-1",
                name="plain_echo",
                arguments={"value": "hello"},
            ),
            context=ToolCallContext(),
        )

        self.assertIsInstance(prepared, PreparedToolCall)
        assert isinstance(prepared, PreparedToolCall)
        self.assertEqual(prepared.call.name, "projected_echo")
        projection = prepared.descriptor.project_display(prepared.call)
        self.assertIsInstance(projection, ToolDisplayProjection)

    async def test_projector_sees_normalized_single_input_arguments(
        self,
    ) -> None:
        manager = ToolManager.create_instance(
            enable_tools=["projected_value_echo"],
            available_toolsets=[ToolSet(tools=[projected_value_echo])],
            settings=ToolManagerSettings(),
        )

        prepared = await manager.prepare_call(
            ToolCall(
                id="call-1",
                name="projected_value_echo",
                arguments={"input": "hello"},
            ),
            context=ToolCallContext(),
        )

        self.assertIsInstance(prepared, PreparedToolCall)
        assert isinstance(prepared, PreparedToolCall)
        self.assertEqual(prepared.call.arguments, {"value": "hello"})
        projection = prepared.descriptor.project_display(prepared.call)
        self.assertIsInstance(projection, ToolDisplayProjection)
        assert isinstance(projection, ToolDisplayProjection)
        self.assertEqual(projection.summary, "hello")


class NativeProjectingTool(Tool):
    def __init__(self) -> None:
        super().__init__()
        self.__name__ = "native_projecting"

    async def __call__(self, value: str, context: ToolCallContext) -> str:
        """Return the input value.

        Args:
            value: Value to return.

        Returns:
            Input value.
        """
        return value

    def tool_display_projector(
        self,
        call: ToolCall,
    ) -> ToolDisplayProjection:
        return ToolDisplayProjection(
            action="inspect",
            label=call.name,
            target=call.name,
        )


class ProjectorAccessGuard:
    def __init__(self) -> None:
        self.__name__ = "guarded"
        self.aliases: list[str] = []

    @property
    def tool_display_projector(self) -> object:
        raise AssertionError("projector should not be accessed")

    async def __call__(self, value: str) -> str:
        """Return the input value.

        Args:
            value: Value to return.

        Returns:
            Input value.
        """
        return value


async def plain_echo(value: str) -> str:
    """Return the input value.

    Args:
        value: Value to return.

    Returns:
        Input value.
    """
    return value


async def projected_echo(value: str) -> str:
    """Return the input value.

    Args:
        value: Value to return.

    Returns:
        Input value.
    """
    return value


async def aliased_projected_echo(value: str) -> str:
    """Return the input value.

    Args:
        value: Value to return.

    Returns:
        Input value.
    """
    return value


async def raising_projector_echo(value: str) -> str:
    """Return the input value.

    Args:
        value: Value to return.

    Returns:
        Input value.
    """
    return value


async def invalid_projector_echo(value: str) -> str:
    """Return the input value.

    Args:
        value: Value to return.

    Returns:
        Input value.
    """
    return value


async def raw_payload_projector_echo(value: str) -> str:
    """Return the input value.

    Args:
        value: Value to return.

    Returns:
        Input value.
    """
    return value


async def projected_value_echo(value: str) -> str:
    """Return the input value.

    Args:
        value: Value to return.

    Returns:
        Input value.
    """
    return value


def call_projector(call: ToolCall) -> ToolDisplayProjection:
    return ToolDisplayProjection(
        action="call",
        label=call.name,
        target=call.name,
    )


def raising_projector(_call: ToolCall) -> ToolDisplayProjection:
    raise RuntimeError("projector failed")


def invalid_projector(_call: ToolCall) -> dict[str, object]:
    return {"summary": "missing action"}


def raw_payload_projector(call: ToolCall) -> dict[str, object]:
    return {
        "action": "call",
        "label": call.name,
        "target": call.name,
    }


def value_summary_projector(call: ToolCall) -> ToolDisplayProjection:
    assert call.arguments is not None
    value = call.arguments["value"]
    assert isinstance(value, str)
    return ToolDisplayProjection(
        action="call",
        label=call.name,
        target=call.name,
        summary=value,
    )


def _has_key(value: object, key: str) -> bool:
    if isinstance(value, dict):
        return key in value or any(
            _has_key(item, key) for item in value.values()
        )
    if isinstance(value, list):
        return any(_has_key(item, key) for item in value)
    return False


setattr(projected_echo, "tool_display_projector", call_projector)
setattr(aliased_projected_echo, "aliases", ["echo_alias"])
setattr(aliased_projected_echo, "tool_display_projector", call_projector)
setattr(raising_projector_echo, "tool_display_projector", raising_projector)
setattr(invalid_projector_echo, "tool_display_projector", invalid_projector)
setattr(
    raw_payload_projector_echo,
    "tool_display_projector",
    raw_payload_projector,
)
setattr(
    projected_value_echo,
    "tool_display_projector",
    value_summary_projector,
)


if __name__ == "__main__":
    main()

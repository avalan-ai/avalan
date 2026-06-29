from ast import (
    AsyncFunctionDef,
    Attribute,
    Call,
    Import,
    ImportFrom,
    keyword,
    parse,
    walk,
)
from pathlib import Path
from unittest import IsolatedAsyncioTestCase, TestCase, main

from avalan.entities import (
    ToolCall,
    ToolCallContext,
    ToolCallDiagnostic,
    ToolCallDiagnosticCode,
    ToolCallDiagnosticStage,
    ToolCallResult,
    ToolManagerSettings,
    ToolNameResolutionStatus,
)
from avalan.tool.manager import ToolManager
from avalan.tool.shell import (
    SHELL_COMMAND_IDS,
    ExecutionPolicy,
    ExecutionResult,
    ExecutionSpec,
    ShellCommandDefinition,
    ShellCommandRequest,
    ShellExecutionErrorCode,
    ShellExecutionStatus,
    ShellPolicyDenied,
    ShellToolSet,
    ShellToolSettings,
    TrustedExecutableResolver,
    unavailable_executable_lookup,
)

_EXPECTED_TOOL_NAMES = tuple(
    f"shell.{command_id}" for command_id in SHELL_COMMAND_IDS
)


class ShellToolManagerFilteringTest(TestCase):
    def test_namespace_wildcard_and_concrete_enablement(self) -> None:
        cases = (
            (["shell"], _EXPECTED_TOOL_NAMES),
            (["shell.*"], _EXPECTED_TOOL_NAMES),
            (["shell.rg"], ("shell.rg",)),
        )

        for enabled_tools, expected_names in cases:
            with self.subTest(enabled_tools=enabled_tools):
                manager = _shell_manager(enabled_tools)

                self.assertEqual(_tool_names(manager), expected_names)
                self.assertEqual(_schema_names(manager), expected_names)

    def test_negative_wildcard_does_not_enable_shell_tools(self) -> None:
        manager = _shell_manager(["shellx.*"])

        self.assertTrue(manager.is_empty)
        self.assertEqual(_tool_names(manager), ())
        self.assertEqual(_schema_names(manager), ())
        resolution = manager.resolve_tool_name("shell.rg")
        self.assertIs(resolution.status, ToolNameResolutionStatus.DISABLED)
        self.assertIs(
            resolution.diagnostic_code,
            ToolCallDiagnosticCode.DISABLED_TOOL,
        )

    def test_concrete_enablement_leaves_other_shell_tools_disabled(
        self,
    ) -> None:
        manager = _shell_manager(["shell.rg"])

        enabled = manager.resolve_tool_name("shell.rg")
        disabled = manager.resolve_tool_name("shell.cat")
        diagnostic = manager.validate_tool_call(
            ToolCall(id="call-1", name="shell.cat", arguments={})
        )

        self.assertIs(enabled.status, ToolNameResolutionStatus.EXACT)
        self.assertEqual(enabled.canonical_name, "shell.rg")
        self.assertIs(disabled.status, ToolNameResolutionStatus.DISABLED)
        self.assertEqual(disabled.candidates, ["shell.cat"])
        self.assertIs(
            disabled.diagnostic_code,
            ToolCallDiagnosticCode.DISABLED_TOOL,
        )
        assert diagnostic is not None
        self.assertIs(
            diagnostic.code,
            ToolCallDiagnosticCode.DISABLED_TOOL,
        )
        self.assertIs(diagnostic.stage, ToolCallDiagnosticStage.RESOLVE)
        self.assertEqual(diagnostic.requested_name, "shell.cat")
        self.assertIsNone(diagnostic.canonical_name)
        self.assertEqual(diagnostic.details["candidates"], ["shell.cat"])

    def test_unavailable_shell_toolset_reports_unknown_shell_tools(
        self,
    ) -> None:
        manager = ToolManager.create_instance(
            available_toolsets=[],
            enable_tools=["shell.rg"],
            settings=ToolManagerSettings(),
        )

        resolution = manager.resolve_tool_name("shell.cat")
        diagnostic = manager.validate_tool_call(
            ToolCall(id="call-1", name="shell.cat", arguments={})
        )

        self.assertIs(resolution.status, ToolNameResolutionStatus.UNKNOWN)
        self.assertIs(
            resolution.diagnostic_code,
            ToolCallDiagnosticCode.UNKNOWN_TOOL,
        )
        assert diagnostic is not None
        self.assertIs(
            diagnostic.code,
            ToolCallDiagnosticCode.UNKNOWN_TOOL,
        )
        self.assertIs(diagnostic.stage, ToolCallDiagnosticStage.RESOLVE)
        self.assertEqual(diagnostic.requested_name, "shell.cat")
        self.assertIsNone(diagnostic.canonical_name)

    def test_pipeline_tool_is_unknown_before_pipeline_opt_in(self) -> None:
        manager = _shell_manager(["shell.pipeline"])

        resolution = manager.resolve_tool_name("shell.pipeline")
        diagnostic = manager.validate_tool_call(
            ToolCall(id="call-1", name="shell.pipeline", arguments={})
        )

        self.assertTrue(manager.is_empty)
        self.assertEqual(_tool_names(manager), ())
        self.assertEqual(_schema_names(manager), ())
        self.assertIs(resolution.status, ToolNameResolutionStatus.UNKNOWN)
        self.assertIs(
            resolution.diagnostic_code,
            ToolCallDiagnosticCode.UNKNOWN_TOOL,
        )
        assert diagnostic is not None
        self.assertIs(
            diagnostic.code,
            ToolCallDiagnosticCode.UNKNOWN_TOOL,
        )
        self.assertIs(diagnostic.stage, ToolCallDiagnosticStage.RESOLVE)
        self.assertEqual(diagnostic.requested_name, "shell.pipeline")
        self.assertIsNone(diagnostic.canonical_name)

    def test_pipeline_tool_schema_is_nested_and_provider_safe(self) -> None:
        manager = _pipeline_shell_manager(["shell.pipeline"])

        self.assertEqual(_tool_names(manager), ("shell.pipeline",))
        self.assertEqual(_schema_names(manager), ("shell.pipeline",))
        descriptor = manager.describe_tool("shell.pipeline")
        assert descriptor is not None
        assert descriptor.parameter_schema is not None
        properties = descriptor.parameter_schema["properties"]
        steps_schema = properties["steps"]
        assert isinstance(steps_schema, dict)
        step_schema = steps_schema["items"]
        assert isinstance(step_schema, dict)

        self.assertEqual(step_schema["required"], ["command", "id"])
        self.assertFalse(step_schema["additionalProperties"])
        self.assertEqual(steps_schema["minItems"], 1)
        self.assertEqual(
            sorted(step_schema["properties"]),
            ["command", "cwd", "id", "options", "paths", "stdin_from"],
        )
        self.assertEqual(step_schema["properties"]["id"]["minLength"], 1)
        self.assertEqual(
            step_schema["properties"]["command"]["minLength"],
            1,
        )
        stdin_schema = step_schema["properties"]["stdin_from"]
        assert isinstance(stdin_schema, dict)
        self.assertEqual(
            stdin_schema["properties"]["step_id"]["minLength"],
            1,
        )
        self.assertEqual(
            stdin_schema["properties"]["stream"]["enum"],
            ["stdout"],
        )
        provider_name = _provider_safe_name(manager, "shell.pipeline")
        self.assertTrue(provider_name.startswith("avl_"))
        self.assertNotIn(".", provider_name)

    def test_schema_and_descriptors_use_single_shell_namespace(self) -> None:
        manager = _shell_manager(["shell"])

        self.assertEqual(_schema_names(manager), _EXPECTED_TOOL_NAMES)
        self.assertEqual(_tool_names(manager), _EXPECTED_TOOL_NAMES)
        self.assertTrue(
            all("shell.shell." not in name for name in _schema_names(manager))
        )
        self.assertTrue(
            all("shell.shell." not in name for name in _tool_names(manager))
        )
        self.assertEqual(
            {descriptor.namespace for descriptor in manager.list_tools()},
            {"shell"},
        )


class ShellToolManagerExecutionTest(IsolatedAsyncioTestCase):
    async def test_enabled_shell_tool_executes_through_manager(self) -> None:
        manager = _unavailable_shell_manager(["shell.rg"])

        outcome = await manager.execute_call(
            ToolCall(
                id="call-1",
                name="shell.rg",
                arguments={
                    "pattern": "visible",
                    "paths": ["filesystem/visible.txt"],
                },
            ),
            context=ToolCallContext(),
        )

        self.assertIsInstance(outcome, ToolCallResult)
        assert isinstance(outcome, ToolCallResult)
        self.assertEqual(outcome.call.name, "shell.rg")
        assert isinstance(outcome.result, str)
        self.assertIn("tool: shell.rg", outcome.result)
        self.assertIn(
            f"status: {ShellExecutionStatus.COMMAND_UNAVAILABLE}",
            outcome.result,
        )
        self.assertIn("error_code: command_unavailable", outcome.result)

    async def test_policy_denied_shell_tool_formats_through_manager(
        self,
    ) -> None:
        policy = _DenyingExecutionPolicy(
            ShellExecutionErrorCode.DENIED_PATH,
            "path is denied",
        )
        executor = _RecordingExecutor()
        manager = _denying_shell_manager(["shell.rg"], policy, executor)

        outcome = await manager.execute_call(
            ToolCall(
                id="call-1",
                name="shell.rg",
                arguments={
                    "pattern": "visible",
                    "paths": ["filesystem/visible.txt"],
                },
            ),
            context=ToolCallContext(),
        )

        self.assertIsInstance(outcome, ToolCallResult)
        assert isinstance(outcome, ToolCallResult)
        self.assertEqual(outcome.call.name, "shell.rg")
        assert isinstance(outcome.result, str)
        self.assertEqual(len(policy.requests), 1)
        self.assertEqual(policy.requests[0].tool_name, "shell.rg")
        self.assertEqual(policy.requests[0].command, "rg")
        self.assertEqual(executor.calls, 0)
        self.assertIn("tool: shell.rg", outcome.result)
        self.assertIn(
            f"status: {ShellExecutionStatus.POLICY_DENIED}",
            outcome.result,
        )
        self.assertIn("error_code: denied_path", outcome.result)
        self.assertIn("error_message: path is denied", outcome.result)

    async def test_provider_encoded_shell_tool_executes_through_manager(
        self,
    ) -> None:
        manager = _unavailable_shell_manager(["shell.rg"])
        provider_name = _provider_safe_name(manager, "shell.rg")

        outcome = await manager.execute_call(
            ToolCall(
                id="call-1",
                name="shell.rg",
                arguments={
                    "pattern": "visible",
                    "paths": ["filesystem/visible.txt"],
                },
                provider_name=provider_name,
                provider_name_encoded=True,
            ),
            context=ToolCallContext(),
        )

        self.assertIsInstance(outcome, ToolCallResult)
        assert isinstance(outcome, ToolCallResult)
        self.assertEqual(outcome.call.name, "shell.rg")
        self.assertEqual(outcome.provider_name, provider_name)
        self.assertTrue(outcome.provider_name_encoded)
        assert isinstance(outcome.result, str)
        self.assertIn("tool: shell.rg", outcome.result)
        self.assertIn(
            f"status: {ShellExecutionStatus.COMMAND_UNAVAILABLE}",
            outcome.result,
        )

    async def test_malformed_shell_tool_arguments_stop_before_execution(
        self,
    ) -> None:
        manager = _unavailable_shell_manager(["shell.rg"])

        outcome = await manager.execute_call(
            ToolCall(
                id="call-1",
                name="shell.rg",
                arguments={
                    "pattern": "visible",
                    "paths": "filesystem/visible.txt",
                },
            ),
            context=ToolCallContext(),
        )

        self.assertIsInstance(outcome, ToolCallDiagnostic)
        assert isinstance(outcome, ToolCallDiagnostic)
        self.assertIs(
            outcome.code,
            ToolCallDiagnosticCode.ARGUMENT_VALIDATION_FAILED,
        )
        self.assertIs(outcome.stage, ToolCallDiagnosticStage.VALIDATE)
        self.assertEqual(outcome.canonical_name, "shell.rg")
        self.assertIn("$.paths must be array", outcome.message)

    async def test_provider_encoded_shell_arguments_stop_before_execution(
        self,
    ) -> None:
        manager = _unavailable_shell_manager(["shell.rg"])
        provider_name = _provider_safe_name(manager, "shell.rg")

        outcome = await manager.execute_call(
            ToolCall(
                id="call-1",
                name="shell.rg",
                arguments={
                    "pattern": "visible",
                    "paths": "filesystem/visible.txt",
                },
                provider_name=provider_name,
                provider_name_encoded=True,
            ),
            context=ToolCallContext(),
        )

        self.assertIsInstance(outcome, ToolCallDiagnostic)
        assert isinstance(outcome, ToolCallDiagnostic)
        self.assertEqual(outcome.requested_name, "shell.rg")
        self.assertEqual(outcome.canonical_name, "shell.rg")
        self.assertIs(
            outcome.code,
            ToolCallDiagnosticCode.ARGUMENT_VALIDATION_FAILED,
        )
        self.assertIs(outcome.stage, ToolCallDiagnosticStage.VALIDATE)
        self.assertIn("$.paths must be array", outcome.message)

    async def test_provider_encoded_disabled_shell_tool_is_rejected(
        self,
    ) -> None:
        provider_name = _provider_safe_name(
            _unavailable_shell_manager(["shell"]),
            "shell.cat",
        )
        manager = _unavailable_shell_manager(["shell.rg"])

        outcome = await manager.execute_call(
            ToolCall(
                id="call-1",
                name="shell.cat",
                arguments={"path": "filesystem/visible.txt"},
                provider_name=provider_name,
                provider_name_encoded=True,
            ),
            context=ToolCallContext(),
        )

        self.assertIsInstance(outcome, ToolCallDiagnostic)
        assert isinstance(outcome, ToolCallDiagnostic)
        self.assertEqual(outcome.requested_name, provider_name)
        self.assertIsNone(outcome.canonical_name)
        self.assertIs(outcome.code, ToolCallDiagnosticCode.DISABLED_TOOL)
        self.assertIs(outcome.stage, ToolCallDiagnosticStage.RESOLVE)
        self.assertEqual(outcome.details["candidates"], ["shell.cat"])

    async def test_malformed_pipeline_arguments_stop_before_execution(
        self,
    ) -> None:
        manager = _pipeline_shell_manager(["shell.pipeline"])

        outcome = await manager.execute_call(
            ToolCall(
                id="call-1",
                name="shell.pipeline",
                arguments={
                    "steps": [
                        {
                            "id": "read",
                            "command": "cat",
                            "unexpected": True,
                        }
                    ]
                },
            ),
            context=ToolCallContext(),
        )

        self.assertIsInstance(outcome, ToolCallDiagnostic)
        assert isinstance(outcome, ToolCallDiagnostic)
        self.assertIs(
            outcome.code,
            ToolCallDiagnosticCode.ARGUMENT_VALIDATION_FAILED,
        )
        self.assertIs(outcome.stage, ToolCallDiagnosticStage.VALIDATE)
        self.assertEqual(outcome.canonical_name, "shell.pipeline")
        self.assertIn("$.steps[0].unexpected is not allowed", outcome.message)

    async def test_malformed_pipeline_stdin_ref_fails_schema_validation(
        self,
    ) -> None:
        manager = _pipeline_shell_manager(["shell.pipeline"])

        outcome = await manager.execute_call(
            ToolCall(
                id="call-1",
                name="shell.pipeline",
                arguments={
                    "steps": [
                        {"id": "read", "command": "cat"},
                        {
                            "id": "count",
                            "command": "wc",
                            "stdin_from": {
                                "step_id": "read",
                                "stream": "stderr",
                            },
                        },
                    ]
                },
            ),
            context=ToolCallContext(),
        )

        self.assertIsInstance(outcome, ToolCallDiagnostic)
        assert isinstance(outcome, ToolCallDiagnostic)
        self.assertIs(
            outcome.code,
            ToolCallDiagnosticCode.ARGUMENT_VALIDATION_FAILED,
        )
        self.assertIn(
            "$.steps[1].stdin_from.stream must be one of ['stdout']",
            outcome.message,
        )

    async def test_pipeline_empty_steps_and_blank_strings_fail_schema(
        self,
    ) -> None:
        manager = _pipeline_shell_manager(["shell.pipeline"])
        cases: tuple[tuple[dict[str, object], str], ...] = (
            (
                {"steps": []},
                "$.steps must contain at least 1 item(s).",
            ),
            (
                {"steps": [{"id": "", "command": "cat"}]},
                "$.steps[0].id must contain at least 1 character(s).",
            ),
            (
                {"steps": [{"id": "read", "command": ""}]},
                "$.steps[0].command must contain at least 1 character(s).",
            ),
        )

        for arguments, message in cases:
            with self.subTest(arguments=arguments):
                outcome = await manager.execute_call(
                    ToolCall(
                        id="call-1",
                        name="shell.pipeline",
                        arguments=arguments,
                    ),
                    context=ToolCallContext(),
                )

                self.assertIsInstance(outcome, ToolCallDiagnostic)
                assert isinstance(outcome, ToolCallDiagnostic)
                self.assertIs(
                    outcome.code,
                    ToolCallDiagnosticCode.ARGUMENT_VALIDATION_FAILED,
                )
                self.assertIs(outcome.stage, ToolCallDiagnosticStage.VALIDATE)
                self.assertEqual(outcome.canonical_name, "shell.pipeline")
                self.assertIn(message, outcome.message)


class ShellToolWrapperStaticGuardTest(TestCase):
    def test_wrapper_source_does_not_import_executor_escape_hatches(
        self,
    ) -> None:
        tree = _tools_tree()
        imported_names: set[str] = set()
        imported_modules: set[str] = set()

        for node in walk(tree):
            if isinstance(node, Import):
                for alias in node.names:
                    imported_modules.add(alias.name)
            if isinstance(node, ImportFrom):
                if node.module is not None:
                    imported_modules.add(node.module)
                for alias in node.names:
                    imported_names.add(alias.name)

        self.assertNotIn("ExecutionSpec", imported_names)
        self.assertNotIn("subprocess", imported_modules)
        self.assertNotIn("asyncio", imported_modules)
        self.assertTrue(
            all(
                not module.startswith("commands")
                for module in imported_modules
            )
        )

    def test_wrapper_calls_do_not_construct_specs_or_argv(self) -> None:
        tree = _tools_tree()
        disallowed_keywords = {"argv", "display_argv"}

        for node in walk(tree):
            if (
                not isinstance(node, AsyncFunctionDef)
                or node.name != "__call__"
            ):
                continue
            with self.subTest(line=node.lineno):
                calls = [
                    child for child in walk(node) if isinstance(child, Call)
                ]
                keywords = [
                    child for child in walk(node) if isinstance(child, keyword)
                ]
                attributes = [
                    child
                    for child in walk(node)
                    if isinstance(child, Attribute)
                ]

                self.assertFalse(
                    any(_call_name(call) == "ExecutionSpec" for call in calls)
                )
                self.assertFalse(
                    any(item.arg in disallowed_keywords for item in keywords)
                )
                self.assertFalse(
                    any(
                        attribute.attr in disallowed_keywords
                        for attribute in attributes
                    )
                )


def _shell_manager(enabled_tools: list[str]) -> ToolManager:
    return ToolManager.create_instance(
        available_toolsets=[ShellToolSet()],
        enable_tools=enabled_tools,
        settings=ToolManagerSettings(),
    )


def _pipeline_shell_manager(enabled_tools: list[str]) -> ToolManager:
    fixture_root = Path(__file__).parent / "fixtures"
    settings = ShellToolSettings(
        workspace_root=str(fixture_root),
        allow_media_tools=True,
        allow_pipelines=True,
    )
    return ToolManager.create_instance(
        available_toolsets=[
            ShellToolSet(
                settings=settings,
                policy=ExecutionPolicy(
                    settings=settings, resolver=_AllResolved()
                ),
            )
        ],
        enable_tools=enabled_tools,
        settings=ToolManagerSettings(),
    )


def _unavailable_shell_manager(enabled_tools: list[str]) -> ToolManager:
    fixture_root = Path(__file__).parent / "fixtures"
    settings = ShellToolSettings(
        workspace_root=str(fixture_root),
        allow_media_tools=True,
    )
    resolver = TrustedExecutableResolver(lookup=unavailable_executable_lookup)
    return ToolManager.create_instance(
        available_toolsets=[
            ShellToolSet(
                settings=settings,
                policy=ExecutionPolicy(settings=settings, resolver=resolver),
            )
        ],
        enable_tools=enabled_tools,
        settings=ToolManagerSettings(),
    )


def _denying_shell_manager(
    enabled_tools: list[str],
    policy: "_DenyingExecutionPolicy",
    executor: "_RecordingExecutor",
) -> ToolManager:
    return ToolManager.create_instance(
        available_toolsets=[
            ShellToolSet(
                policy=policy,
                executor=executor,
            )
        ],
        enable_tools=enabled_tools,
        settings=ToolManagerSettings(),
    )


class _DenyingExecutionPolicy(ExecutionPolicy):
    requests: list[ShellCommandRequest]
    _error_code: ShellExecutionErrorCode
    _message: str

    def __init__(
        self,
        error_code: ShellExecutionErrorCode,
        message: str,
    ) -> None:
        self.requests = []
        self._error_code = error_code
        self._message = message

    async def normalize(
        self,
        request: ShellCommandRequest,
    ) -> ExecutionSpec:
        self.requests.append(request)
        raise ShellPolicyDenied(self._error_code, self._message)


class _RecordingExecutor:
    calls: int

    def __init__(self) -> None:
        self.calls = 0

    async def execute(self, spec: ExecutionSpec) -> ExecutionResult:
        self.calls += 1
        raise AssertionError("executor should not be called")


class _AllResolved:
    async def resolve(
        self,
        command: ShellCommandDefinition,
    ) -> str | None:
        return f"/trusted/bin/{command.executable_name}"


def _tool_names(manager: ToolManager) -> tuple[str, ...]:
    return tuple(descriptor.name for descriptor in manager.list_tools())


def _schema_names(manager: ToolManager) -> tuple[str, ...]:
    schemas = manager.json_schemas()
    return tuple(schema["function"]["name"] for schema in schemas or ())


def _provider_safe_name(manager: ToolManager, name: str) -> str:
    descriptor = manager.describe_tool(name)
    assert descriptor is not None
    assert descriptor.provider_safe_schema is not None
    function = descriptor.provider_safe_schema["function"]
    assert isinstance(function, dict)
    provider_name = function["name"]
    assert isinstance(provider_name, str)
    assert provider_name.startswith("avl_")
    assert "." not in provider_name
    return provider_name


def _tools_tree():
    return parse(Path("src/avalan/tool/shell/tools.py").read_text())


def _call_name(node: Call) -> str | None:
    function = node.func
    if isinstance(function, Attribute):
        return function.attr
    return getattr(function, "id", None)


if __name__ == "__main__":
    main()

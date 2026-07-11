from asyncio import Event, create_task, gather, run, sleep, wait_for
from collections.abc import Awaitable, Callable
from pathlib import Path
from unittest import IsolatedAsyncioTestCase, TestCase, main
from unittest.mock import patch

from avalan.container import (
    ContainerBackend,
    ContainerBackendCapabilities,
    ContainerFakeBackend,
    ContainerFakeBackendScript,
    ContainerMountType,
    ContainerToolRuntimeSettings,
    trusted_container_runtime_from_mapping,
    trusted_container_source,
)
from avalan.entities import (
    ToolCallContext,
    ToolExecutionStreamEvent,
    ToolExecutionStreamKind,
)
from avalan.isolation import (
    IsolationEffectiveSettings,
    IsolationMode,
    IsolationToolRuntimeSettings,
    trusted_isolation_source,
)
from avalan.tool import Tool
from avalan.tool.shell import (
    SHELL_COMMAND_IDS,
    SHELL_GIT_DEFAULT_ALLOWED_COMMAND_IDS,
    CommandExecutor,
    ExecutionPolicy,
    ExecutionResult,
    ExecutionSpec,
    LocalCommandExecutor,
    ShellCommandDefinition,
    ShellCompositionResult,
    ShellCompositionSpec,
    ShellExecutionStatus,
    ShellExecutionStepResult,
    ShellOutputKind,
    ShellToolSet,
    ShellToolSettings,
    TrustedExecutableResolver,
    unavailable_executable_lookup,
)
from avalan.tool.shell.entities import (
    ShellFormattedCompositionResult,
    ShellFormattedResult,
)
from avalan.tool.shell.process import ShellProcessRuntime

_EXPECTED_SCHEMA_NAMES = (
    "shell.rg",
    "shell.head",
    "shell.tail",
    "shell.ls",
    "shell.cat",
    "shell.nl",
    "shell.pgrep",
    "shell.ps",
    "shell.file",
    "shell.find",
    "shell.wc",
    "shell.awk",
    "shell.sed",
    "shell.jq",
    "shell.pdfinfo",
    "shell.pdftotext",
    "shell.pdftoppm",
    "shell.reportlab",
    "shell.pdfplumber",
    "shell.pypdf",
    "shell.tesseract",
)
_EXPECTED_GIT_READ_SCHEMA_NAMES = tuple(
    f"shell.git_{command_id.replace('-', '_')}"
    for command_id in SHELL_GIT_DEFAULT_ALLOWED_COMMAND_IDS
)
_EXPECTED_SCHEMA_NAMES_WITH_READ_GIT = (
    *_EXPECTED_SCHEMA_NAMES,
    *_EXPECTED_GIT_READ_SCHEMA_NAMES,
)
_EXPECTED_SCHEMA_NAMES_WITH_PIPELINE_AND_READ_GIT = (
    *_EXPECTED_SCHEMA_NAMES,
    "shell.pipeline",
    *_EXPECTED_GIT_READ_SCHEMA_NAMES,
)
_DIGEST = "7" * 64
_IMAGE = f"ghcr.io/example/sdk-shell@sha256:{_DIGEST}"


class ShellToolSetAssemblyTest(TestCase):
    def test_toolset_registers_all_command_schemas_in_locked_order(
        self,
    ) -> None:
        toolset = ShellToolSet()

        self.assertEqual(toolset.namespace, "shell")
        self.assertEqual(
            tuple(tool.__name__ for tool in toolset.tools),
            SHELL_COMMAND_IDS,
        )
        self.assertEqual(_schema_names(toolset), _EXPECTED_SCHEMA_NAMES)

    def test_toolset_filters_by_namespace_and_concrete_tool(self) -> None:
        all_enabled = ShellToolSet().with_enabled_tools(["shell"])
        concrete_enabled = ShellToolSet().with_enabled_tools(["shell.rg"])
        disabled = ShellToolSet().with_enabled_tools(["shellx"])

        self.assertEqual(
            _schema_names(all_enabled),
            _EXPECTED_SCHEMA_NAMES_WITH_READ_GIT,
        )
        self.assertEqual(_schema_names(concrete_enabled), ("shell.rg",))
        self.assertEqual(_schema_names(disabled), ())

    def test_pipeline_selection_is_empty_without_pipeline_opt_in(self) -> None:
        pipeline_enabled = ShellToolSet().with_enabled_tools(
            ["shell.pipeline"]
        )

        self.assertEqual(_schema_names(pipeline_enabled), ())

    def test_pipeline_setting_alone_does_not_expose_pipeline(self) -> None:
        toolset = ShellToolSet(
            settings=ShellToolSettings(allow_pipelines=True)
        )

        self.assertEqual(_schema_names(toolset), _EXPECTED_SCHEMA_NAMES)

    def test_pipeline_visibility_requires_setting_and_matching_selection(
        self,
    ) -> None:
        settings = ShellToolSettings(allow_pipelines=True)
        cases = (
            (["shell"], _EXPECTED_SCHEMA_NAMES_WITH_PIPELINE_AND_READ_GIT),
            (["shell.*"], _EXPECTED_SCHEMA_NAMES_WITH_PIPELINE_AND_READ_GIT),
            (["shell.pipeline"], ("shell.pipeline",)),
            (["shell.rg"], ("shell.rg",)),
            (["shell.cat"], ("shell.cat",)),
            (["shellx.*"], ()),
        )

        for enable_tools, expected in cases:
            with self.subTest(enable_tools=enable_tools):
                toolset = ShellToolSet(settings=settings).with_enabled_tools(
                    enable_tools
                )

                self.assertEqual(_schema_names(toolset), expected)

    def test_pipeline_tool_executes_through_composition_executor(self) -> None:
        settings = ShellToolSettings(allow_pipelines=True)
        composition_executor = _RecordingCompositionExecutor()
        toolset = ShellToolSet(
            settings=settings,
            policy=ExecutionPolicy(settings=settings, resolver=_AllResolved()),
            composition_executor=composition_executor,
        ).with_enabled_tools(["shell.pipeline"])
        pipeline = toolset.tools[0]

        assert callable(pipeline)
        output = run(
            pipeline(
                steps=(
                    {"id": "list", "command": "ls"},
                    {
                        "id": "count",
                        "command": "wc",
                        "stdin_from": {
                            "step_id": "list",
                            "stream": "stdout",
                        },
                    },
                ),
                context=ToolCallContext(),
            )
        )

        self.assertIsInstance(output, ShellFormattedCompositionResult)
        self.assertIn("tool: shell.pipeline", output)
        self.assertIn("status: completed", output)
        self.assertEqual(composition_executor.modes, ["pipeline"])
        self.assertEqual(
            [step.id for step in composition_executor.specs[0].steps],
            ["list", "count"],
        )

    def test_toolset_rejects_namespaces_that_change_schema_names(
        self,
    ) -> None:
        for namespace in (None, "custom"):
            with self.subTest(namespace=namespace):
                with self.assertRaisesRegex(
                    AssertionError,
                    "namespace must be shell",
                ):
                    ShellToolSet(namespace=namespace)

    def test_missing_binary_resolution_does_not_affect_schemas(self) -> None:
        settings = ShellToolSettings()
        one_missing = ShellToolSet(
            settings=settings,
            policy=ExecutionPolicy(settings=settings, resolver=_OneMissing()),
        )
        all_missing = ShellToolSet(
            settings=settings,
            policy=ExecutionPolicy(settings=settings, resolver=_AllMissing()),
        )

        self.assertEqual(_schema_names(one_missing), _EXPECTED_SCHEMA_NAMES)
        self.assertEqual(_schema_names(all_missing), _EXPECTED_SCHEMA_NAMES)

    def test_toolset_schema_generation_does_not_resolve_binaries(self) -> None:
        settings = ShellToolSettings()
        toolset = ShellToolSet(
            settings=settings,
            policy=ExecutionPolicy(
                settings=settings,
                resolver=_UnexpectedResolve(),
            ),
        )

        self.assertEqual(_schema_names(toolset), _EXPECTED_SCHEMA_NAMES)

    def test_shell_tools_advertise_streaming_capability(self) -> None:
        toolset = ShellToolSet()

        self.assertTrue(toolset.tools)
        for tool in toolset.tools:
            with self.subTest(tool=getattr(tool, "__name__", "")):
                self.assertIs(
                    getattr(tool, "supports_streaming"),
                    getattr(tool, "__name__") not in {"pgrep", "ps"},
                )


class ShellToolSetMissingBinaryTest(IsolatedAsyncioTestCase):
    async def test_each_tool_returns_command_unavailable_when_missing(
        self,
    ) -> None:
        fixture_root = Path(__file__).parent / "fixtures"
        settings = ShellToolSettings(
            workspace_root=str(fixture_root),
            allow_media_tools=True,
            allow_process_tools=True,
        )
        resolver = TrustedExecutableResolver(
            lookup=unavailable_executable_lookup,
        )
        toolset = ShellToolSet(
            settings=settings,
            policy=ExecutionPolicy(settings=settings, resolver=resolver),
        )

        for command_id in SHELL_COMMAND_IDS:
            with self.subTest(command_id=command_id):
                tool = _tool_by_name(toolset, command_id)
                output = await _invoke_for_command(command_id, tool)

                self.assertIn(f"tool: shell.{command_id}", output)
                self.assertIn(
                    f"status: {ShellExecutionStatus.COMMAND_UNAVAILABLE}",
                    output,
                )
                self.assertIn("error_code: command_unavailable", output)
                expected_message = (
                    f"{command_id} is unavailable"
                    if command_id in {"pgrep", "ps"}
                    else "command is unavailable"
                )
                self.assertIn(f"error_message: {expected_message}", output)

    async def test_shell_tool_forwards_context_stream_callback(self) -> None:
        fixture_root = Path(__file__).parent / "fixtures"
        settings = ShellToolSettings(workspace_root=str(fixture_root))
        executor = _StreamingExecutor()
        toolset = ShellToolSet(
            settings=settings,
            policy=ExecutionPolicy(
                settings=settings,
                resolver=_OneMissing(),
            ),
            executor=executor,
        )
        events: list[ToolExecutionStreamEvent] = []

        async def record(event: ToolExecutionStreamEvent) -> None:
            events.append(event)

        tool = _tool_by_name(toolset, "cat")
        output = await tool(
            "filesystem/visible.txt",
            context=ToolCallContext(stream_event=record),
        )

        self.assertIn(f"status: {ShellExecutionStatus.COMPLETED}", output)
        self.assertEqual(executor.seen_tool_names, ["shell.cat"])
        self.assertEqual(
            [(event.kind, event.content) for event in events],
            [(ToolExecutionStreamKind.STDOUT, "live")],
        )

    async def test_toolset_process_runtime_is_shared_for_future_executors(
        self,
    ) -> None:
        settings = ShellToolSettings(max_concurrent_processes=1)
        toolset = ShellToolSet(
            settings=settings,
            policy=ExecutionPolicy(settings=settings, resolver=_AllResolved()),
        )
        tool_executor = getattr(_tool_by_name(toolset, "cat"), "_executor")
        future_executor = LocalCommandExecutor(
            settings=settings,
            process_runtime=toolset.process_runtime,
        )
        release = Event()
        tracker = _ProcessTracker(target_active=1)
        spawn_count = 0

        async def fake_create_subprocess_exec(
            *args: object,
            **kwargs: object,
        ) -> _BlockingProcess:
            nonlocal spawn_count
            spawn_count += 1
            return _BlockingProcess(release=release, tracker=tracker)

        self.assertIsInstance(toolset.process_runtime, ShellProcessRuntime)
        self.assertIsInstance(tool_executor, LocalCommandExecutor)
        self.assertIs(
            getattr(tool_executor, "_process_runtime"),
            toolset.process_runtime,
        )
        spec = _direct_execution_spec()

        with patch(
            "avalan.tool.shell.process.create_subprocess_exec",
            new=fake_create_subprocess_exec,
        ):
            tool_task = create_task(tool_executor.execute(spec))
            future_task = create_task(future_executor.execute(spec))
            try:
                await wait_for(tracker.target_reached.wait(), timeout=1)
                await sleep(0)

                self.assertEqual(spawn_count, 1)
                self.assertEqual(tracker.maximum_active, 1)
            finally:
                release.set()

            results = await wait_for(
                gather(tool_task, future_task),
                timeout=1,
            )

        self.assertEqual(spawn_count, 2)
        self.assertEqual(tracker.maximum_active, 1)
        self.assertTrue(
            all(
                result.status is ShellExecutionStatus.COMPLETED
                for result in results
            )
        )

    async def test_new_shell_commands_execute_through_toolset(self) -> None:
        fixture_root = Path(__file__).parent / "fixtures"
        settings = ShellToolSettings(
            workspace_root=str(fixture_root),
            allow_media_tools=True,
        )
        executor = _StreamingExecutor()
        toolset = ShellToolSet(
            settings=settings,
            policy=ExecutionPolicy(
                settings=settings,
                resolver=_AllResolved(),
            ),
            executor=executor,
        )

        for command_id in ("file", "find", "pdfinfo"):
            with self.subTest(command_id=command_id):
                output = await _invoke_for_command(
                    command_id,
                    _tool_by_name(toolset, command_id),
                )

                self.assertIn(f"tool: shell.{command_id}", output)
                self.assertIn(
                    f"status: {ShellExecutionStatus.COMPLETED}",
                    output,
                )

        self.assertEqual(
            executor.seen_tool_names,
            ["shell.file", "shell.find", "shell.pdfinfo"],
        )

    async def test_container_runtime_settings_inject_custom_fake_backend(
        self,
    ) -> None:
        fixture_root = Path(__file__).parent / "fixtures"
        settings = ShellToolSettings(
            backend="container",
            workspace_root=str(fixture_root),
        )
        backend = ContainerFakeBackend(
            ContainerFakeBackendScript(
                capabilities=ContainerBackendCapabilities(
                    backend=ContainerBackend.DOCKER,
                    host_os="linux",
                    guest_os="linux",
                    architecture="amd64",
                    rootless=False,
                    mount_types=(ContainerMountType.WORKSPACE,),
                    streaming_attach=True,
                ),
                stream_chunks=(),
            )
        )
        runtime = trusted_container_runtime_from_mapping(
            {
                "backend": "docker",
                "default_profile": "workspace-readonly",
                "profiles": {
                    "workspace-readonly": {
                        "image": _IMAGE,
                        "workspace_root": str(fixture_root),
                    }
                },
            },
            source=trusted_container_source("sdk"),
        )
        toolset = ShellToolSet(
            settings=settings,
            policy=ExecutionPolicy(settings=settings, resolver=_AllResolved()),
            container_runtime=ContainerToolRuntimeSettings(
                effective_settings=runtime.effective_settings,
                backend=backend,
                rootful_authorized=runtime.rootful_authorized,
            ),
        )

        output = await _call_cat(_tool_by_name(toolset, "cat"))

        self.assertIn(f"status: {ShellExecutionStatus.COMPLETED}", output)
        self.assertIsInstance(output, ShellFormattedResult)
        self.assertEqual(output.execution_result.backend, "container")

    async def test_container_runtime_uses_explicit_docker_backend(
        self,
    ) -> None:
        fixture_root = Path(__file__).parent / "fixtures"
        settings = ShellToolSettings(
            backend="container",
            workspace_root=str(fixture_root),
        )
        backend = ContainerFakeBackend(
            ContainerFakeBackendScript(
                capabilities=ContainerBackendCapabilities(
                    backend=ContainerBackend.DOCKER,
                    host_os="linux",
                    guest_os="linux",
                    architecture="amd64",
                    rootless=False,
                    mount_types=(ContainerMountType.WORKSPACE,),
                    streaming_attach=True,
                ),
                stream_chunks=(),
            )
        )
        runtime = trusted_container_runtime_from_mapping(
            {
                "backend": "docker",
                "default_profile": "workspace-readonly",
                "profiles": {
                    "workspace-readonly": {
                        "image": _IMAGE,
                        "workspace_root": str(fixture_root),
                    }
                },
            },
            source=trusted_container_source("sdk"),
        )
        toolset = ShellToolSet(
            settings=settings,
            policy=ExecutionPolicy(settings=settings, resolver=_AllResolved()),
            container_runtime=ContainerToolRuntimeSettings(
                effective_settings=runtime.effective_settings,
                backend=backend,
                rootful_authorized=runtime.rootful_authorized,
            ),
        )

        output = await _call_cat(_tool_by_name(toolset, "cat"))

        self.assertIn(f"status: {ShellExecutionStatus.COMPLETED}", output)
        self.assertIsInstance(output, ShellFormattedResult)
        self.assertEqual(output.execution_result.backend, "container")

    async def test_isolation_runtime_uses_container_settings(
        self,
    ) -> None:
        fixture_root = Path(__file__).parent / "fixtures"
        settings = ShellToolSettings(
            execution_mode="container",
            workspace_root=str(fixture_root),
        )
        backend = ContainerFakeBackend(
            ContainerFakeBackendScript(
                capabilities=ContainerBackendCapabilities(
                    backend=ContainerBackend.DOCKER,
                    host_os="linux",
                    guest_os="linux",
                    architecture="amd64",
                    rootless=True,
                    mount_types=(ContainerMountType.WORKSPACE,),
                    streaming_attach=True,
                ),
                stream_chunks=(),
            )
        )
        runtime = trusted_container_runtime_from_mapping(
            {
                "backend": "docker",
                "default_profile": "workspace-readonly",
                "profiles": {
                    "workspace-readonly": {
                        "image": _IMAGE,
                        "workspace_root": str(fixture_root),
                    }
                },
            },
            source=trusted_container_source("sdk"),
        )
        isolation_runtime = IsolationToolRuntimeSettings(
            effective_settings=IsolationEffectiveSettings(
                mode=IsolationMode.CONTAINER,
                source=trusted_isolation_source("sdk"),
                container=runtime.effective_settings,
            ),
            container_backend=backend,
        )
        toolset = ShellToolSet(
            settings=settings,
            policy=ExecutionPolicy(settings=settings, resolver=_AllResolved()),
            isolation_runtime=isolation_runtime,
        )

        output = await _call_cat(_tool_by_name(toolset, "cat"))

        self.assertIsInstance(output, ShellFormattedResult)
        self.assertEqual(output.execution_result.backend, "container")

    def test_isolation_runtime_rejects_unsupported_container_hooks(
        self,
    ) -> None:
        fixture_root = Path(__file__).parent / "fixtures"
        settings = ShellToolSettings(
            execution_mode="container",
            workspace_root=str(fixture_root),
        )
        runtime = trusted_container_runtime_from_mapping(
            {
                "backend": "docker",
                "default_profile": "workspace-readonly",
                "profiles": {
                    "workspace-readonly": {
                        "image": _IMAGE,
                        "workspace_root": str(fixture_root),
                    }
                },
            },
            source=trusted_container_source("sdk"),
        )
        isolation_runtime = IsolationToolRuntimeSettings(
            effective_settings=IsolationEffectiveSettings(
                mode=IsolationMode.CONTAINER,
                source=trusted_isolation_source("sdk"),
                container=runtime.effective_settings,
            ),
            secret_resolver=lambda name: name,
        )

        with self.assertRaisesRegex(
            AssertionError,
            "shell isolation runtime hooks are not supported",
        ):
            ShellToolSet(
                settings=settings,
                policy=ExecutionPolicy(
                    settings=settings,
                    resolver=_AllResolved(),
                ),
                isolation_runtime=isolation_runtime,
            )

    async def test_container_runtime_with_local_mode_is_rejected(
        self,
    ) -> None:
        fixture_root = Path(__file__).parent / "fixtures"
        settings = ShellToolSettings(workspace_root=str(fixture_root))
        backend = ContainerFakeBackend(
            ContainerFakeBackendScript(
                capabilities=ContainerBackendCapabilities(
                    backend=ContainerBackend.DOCKER,
                    host_os="linux",
                    guest_os="linux",
                    architecture="amd64",
                    rootless=True,
                    mount_types=(ContainerMountType.WORKSPACE,),
                    streaming_attach=True,
                ),
                stream_chunks=(),
            )
        )
        runtime = trusted_container_runtime_from_mapping(
            {
                "backend": "docker",
                "default_profile": "workspace-readonly",
                "profiles": {
                    "workspace-readonly": {
                        "image": _IMAGE,
                        "workspace_root": str(fixture_root),
                    }
                },
            },
            source=trusted_container_source("sdk"),
        )

        with self.assertRaisesRegex(
            AssertionError,
            "container runtime requires",
        ):
            ShellToolSet(
                settings=settings,
                policy=ExecutionPolicy(
                    settings=settings,
                    resolver=_AllResolved(),
                ),
                container_runtime=ContainerToolRuntimeSettings(
                    effective_settings=runtime.effective_settings,
                    backend=backend,
                ),
            )

    async def test_mixed_isolation_and_container_runtimes_are_rejected(
        self,
    ) -> None:
        fixture_root = Path(__file__).parent / "fixtures"
        settings = ShellToolSettings(
            execution_mode="container",
            workspace_root=str(fixture_root),
        )
        backend = ContainerFakeBackend(
            ContainerFakeBackendScript(
                capabilities=ContainerBackendCapabilities(
                    backend=ContainerBackend.DOCKER,
                    host_os="linux",
                    guest_os="linux",
                    architecture="amd64",
                    rootless=True,
                    mount_types=(ContainerMountType.WORKSPACE,),
                    streaming_attach=True,
                ),
                stream_chunks=(),
            )
        )
        runtime = trusted_container_runtime_from_mapping(
            {
                "backend": "docker",
                "default_profile": "workspace-readonly",
                "profiles": {
                    "workspace-readonly": {
                        "image": _IMAGE,
                        "workspace_root": str(fixture_root),
                    }
                },
            },
            source=trusted_container_source("sdk"),
        )
        isolation_runtime = IsolationToolRuntimeSettings(
            effective_settings=IsolationEffectiveSettings(
                mode=IsolationMode.CONTAINER,
                source=trusted_isolation_source("sdk"),
                container=runtime.effective_settings,
            )
        )

        with self.assertRaisesRegex(
            AssertionError,
            "cannot be combined",
        ):
            ShellToolSet(
                settings=settings,
                policy=ExecutionPolicy(
                    settings=settings,
                    resolver=_AllResolved(),
                ),
                isolation_runtime=isolation_runtime,
                container_runtime=ContainerToolRuntimeSettings(
                    effective_settings=runtime.effective_settings,
                    backend=backend,
                ),
            )

    def test_container_settings_with_local_mode_are_rejected(self) -> None:
        fixture_root = Path(__file__).parent / "fixtures"
        settings = ShellToolSettings(workspace_root=str(fixture_root))
        runtime = trusted_container_runtime_from_mapping(
            {
                "backend": "docker",
                "default_profile": "workspace-readonly",
                "profiles": {
                    "workspace-readonly": {
                        "image": _IMAGE,
                        "workspace_root": str(fixture_root),
                    }
                },
            },
            source=trusted_container_source("sdk"),
        )

        with self.assertRaisesRegex(
            AssertionError,
            "container settings require",
        ):
            ShellToolSet(
                settings=settings,
                policy=ExecutionPolicy(
                    settings=settings,
                    resolver=_AllResolved(),
                ),
                container_settings=runtime.effective_settings,
            )

    def test_container_mode_rejects_custom_executor(self) -> None:
        fixture_root = Path(__file__).parent / "fixtures"
        settings = ShellToolSettings(
            execution_mode="container",
            workspace_root=str(fixture_root),
        )

        with self.assertRaisesRegex(
            AssertionError,
            "custom shell executors require",
        ):
            ShellToolSet(
                settings=settings,
                policy=ExecutionPolicy(
                    settings=settings,
                    resolver=_AllResolved(),
                ),
                executor=LocalCommandExecutor(settings=settings),
            )

    async def test_container_backend_without_runtime_fails_closed(
        self,
    ) -> None:
        fixture_root = Path(__file__).parent / "fixtures"
        settings = ShellToolSettings(
            backend="container",
            workspace_root=str(fixture_root),
        )
        toolset = ShellToolSet(
            settings=settings,
            policy=ExecutionPolicy(settings=settings, resolver=_AllResolved()),
        )

        output = await _call_cat(_tool_by_name(toolset, "cat"))

        self.assertIn(f"status: {ShellExecutionStatus.POLICY_DENIED}", output)
        self.assertIsInstance(output, ShellFormattedResult)
        self.assertEqual(output.execution_result.backend, "container")
        self.assertIn("container execution is required", output)


def _schema_names(toolset: ShellToolSet) -> tuple[str, ...]:
    schemas = toolset.json_schemas()
    return tuple(schema["function"]["name"] for schema in schemas or ())


def _tool_by_name(toolset: ShellToolSet, command_id: str) -> Tool:
    for tool in toolset.tools:
        if getattr(tool, "__name__", "") == command_id:
            assert isinstance(tool, Tool), "shell command must be a tool"
            return tool
    raise AssertionError(f"missing shell tool {command_id}")


async def _invoke_for_command(command_id: str, tool: Tool) -> str:
    call = _TOOL_CALLS[command_id]
    return await call(tool)


async def _call_rg(tool: Tool) -> str:
    return await _call_tool(
        tool,
        "visible",
        paths=("filesystem/visible.txt",),
    )


async def _call_head(tool: Tool) -> str:
    return await _call_tool(tool, "filesystem/visible.txt")


async def _call_tail(tool: Tool) -> str:
    return await _call_tool(tool, "filesystem/visible.txt")


async def _call_ls(tool: Tool) -> str:
    return await _call_tool(tool, "filesystem")


async def _call_cat(tool: Tool) -> str:
    return await _call_tool(tool, "filesystem/visible.txt")


async def _call_nl(tool: Tool) -> str:
    return await _call_tool(tool, "filesystem/visible.txt")


async def _call_pgrep(tool: Tool) -> str:
    return await _call_tool(tool, "avalan-pgrep-missing-binary")


async def _call_ps(tool: Tool) -> str:
    return await _call_tool(tool, (1,))


async def _call_file(tool: Tool) -> str:
    return await _call_tool(tool, ("filesystem/visible.txt",))


async def _call_find(tool: Tool) -> str:
    return await _call_tool(tool, ("filesystem",), name="visible.txt")


async def _call_wc(tool: Tool) -> str:
    return await _call_tool(tool, ("filesystem/visible.txt",))


async def _call_awk(tool: Tool) -> str:
    return await _call_tool(
        tool,
        ("filters/table.csv",),
        fields=(1,),
        field_separator="comma",
        output_separator=",",
    )


async def _call_sed(tool: Tool) -> str:
    return await _call_tool(
        tool,
        ("filters/lines.txt",),
        line_ranges=("1,1",),
    )


async def _call_jq(tool: Tool) -> str:
    return await _call_tool(tool, ".", ("json/valid.json",))


async def _call_pdfinfo(tool: Tool) -> str:
    return await _call_tool(tool, "media/small.pdf")


async def _call_pdftotext(tool: Tool) -> str:
    return await _call_tool(tool, "media/small.pdf")


async def _call_pdftoppm(tool: Tool) -> str:
    return await _call_tool(tool, "media/small.pdf", last_page=1, dpi=72)


async def _call_reportlab(tool: Tool) -> str:
    return await _call_tool(tool, "hello", title="Smoke")


async def _call_pdfplumber(tool: Tool) -> str:
    return await _call_tool(tool, "media/small.pdf", last_page=1)


async def _call_pypdf(tool: Tool) -> str:
    return await _call_tool(tool, "media/small.pdf")


async def _call_tesseract(tool: Tool) -> str:
    return await _call_tool(tool, "ocr/small.pgm")


async def _call_tool(tool: Tool, *args: object, **kwargs: object) -> str:
    return await tool(*args, **kwargs, context=ToolCallContext())


_TOOL_CALLS: dict[str, Callable[[Tool], Awaitable[str]]] = {
    "rg": _call_rg,
    "head": _call_head,
    "tail": _call_tail,
    "ls": _call_ls,
    "cat": _call_cat,
    "nl": _call_nl,
    "pgrep": _call_pgrep,
    "ps": _call_ps,
    "file": _call_file,
    "find": _call_find,
    "wc": _call_wc,
    "awk": _call_awk,
    "sed": _call_sed,
    "jq": _call_jq,
    "pdfinfo": _call_pdfinfo,
    "pdftotext": _call_pdftotext,
    "pdftoppm": _call_pdftoppm,
    "reportlab": _call_reportlab,
    "pdfplumber": _call_pdfplumber,
    "pypdf": _call_pypdf,
    "tesseract": _call_tesseract,
}


class _OneMissing:
    async def resolve(
        self,
        command: ShellCommandDefinition,
    ) -> str | None:
        if command.logical_id == "rg":
            return None
        return f"/trusted/bin/{command.executable_name}"


class _AllMissing:
    async def resolve(
        self,
        command: ShellCommandDefinition,
    ) -> str | None:
        return None


class _AllResolved:
    async def resolve(
        self,
        command: ShellCommandDefinition,
    ) -> str | None:
        return f"/trusted/bin/{command.executable_name}"


class _UnexpectedResolve:
    async def resolve(
        self,
        command: ShellCommandDefinition,
    ) -> str | None:
        raise AssertionError("schema generation must not resolve binaries")


class _StreamingExecutor(CommandExecutor):
    def __init__(self) -> None:
        self.seen_tool_names: list[str] = []

    async def execute(
        self,
        spec: ExecutionSpec,
        *,
        stream: (
            Callable[[ToolExecutionStreamEvent], Awaitable[None]] | None
        ) = None,
    ) -> ExecutionResult:
        self.seen_tool_names.append(spec.tool_name)
        if stream is not None:
            await stream(
                ToolExecutionStreamEvent(
                    kind=ToolExecutionStreamKind.STDOUT,
                    content="live",
                )
            )
        return ExecutionResult(
            backend=spec.backend,
            tool_name=spec.tool_name,
            command=spec.command,
            argv=spec.argv,
            display_argv=spec.display_argv,
            cwd=spec.cwd,
            display_cwd=spec.display_cwd,
            status=ShellExecutionStatus.COMPLETED,
            exit_code=0,
            stdout="live",
            stderr="",
            stdout_media_type="text/plain",
            output_kind=ShellOutputKind.TEXT,
            stdout_bytes=4,
            stderr_bytes=0,
        )


class _RecordingCompositionExecutor:
    def __init__(self) -> None:
        self.modes: list[str] = []
        self.specs: list[ShellCompositionSpec] = []

    async def execute_composition(
        self,
        spec: ShellCompositionSpec,
        *,
        stream: (
            Callable[[ToolExecutionStreamEvent], Awaitable[None]] | None
        ) = None,
    ) -> ShellCompositionResult:
        self.modes.append(spec.mode)
        self.specs.append(spec)
        if stream is not None:
            await stream(
                ToolExecutionStreamEvent(
                    kind=ToolExecutionStreamKind.STDOUT,
                    content="2\n",
                    metadata={"stage_id": "count", "stage_index": 1},
                )
            )
        return ShellCompositionResult(
            mode=spec.mode,
            status=ShellExecutionStatus.COMPLETED,
            stdout="2\n",
            stderr="",
            steps=tuple(
                ShellExecutionStepResult(
                    id=step.id,
                    command=step.spec.command,
                    status=ShellExecutionStatus.COMPLETED,
                    exit_code=0,
                    stdout="2\n" if index == len(spec.steps) - 1 else "",
                    stderr="",
                    stdout_bytes=2 if index == len(spec.steps) - 1 else 0,
                    stderr_bytes=0,
                    stdout_truncated=False,
                    stderr_truncated=False,
                    duration_ms=1,
                )
                for index, step in enumerate(spec.steps)
            ),
            stdout_bytes=2,
            stderr_bytes=0,
            duration_ms=2,
        )


class _ProcessTracker:
    def __init__(self, *, target_active: int) -> None:
        self.active = 0
        self.maximum_active = 0
        self.target_reached = Event()
        self._target_active = target_active

    def enter(self) -> None:
        self.active += 1
        self.maximum_active = max(self.maximum_active, self.active)
        if self.active >= self._target_active:
            self.target_reached.set()

    def exit(self) -> None:
        self.active -= 1


class _BlockingProcess:
    returncode = 0
    stdin = None

    def __init__(
        self,
        *,
        release: Event,
        tracker: _ProcessTracker,
    ) -> None:
        self._release = release
        self._tracker = tracker
        self.stdout = _FakeStream(b"ok")
        self.stderr = _FakeStream(b"")

    async def wait(self) -> None:
        self._tracker.enter()
        try:
            await self._release.wait()
        finally:
            self._tracker.exit()

    def terminate(self) -> None:
        self.returncode = -15
        self._release.set()

    def kill(self) -> None:
        self.returncode = -9
        self._release.set()


class _FakeStream:
    def __init__(self, data: bytes) -> None:
        self._data = data
        self._offset = 0

    async def read(self, size: int) -> bytes:
        chunk = self._data[self._offset : self._offset + size]
        self._offset += len(chunk)
        return chunk


def _direct_execution_spec() -> ExecutionSpec:
    return ExecutionPolicy().create_execution_spec(
        backend="local",
        tool_name="shell.cat",
        command="cat",
        executable="/trusted/bin/cat",
        argv=("cat",),
        display_argv=("cat",),
        cwd=str(Path.cwd().resolve()),
        display_cwd=".",
        env={"LC_ALL": "C"},
        stdin=None,
        stdout_media_type="text/plain",
        output_kind=ShellOutputKind.TEXT,
        resource_class="standard",
        output_plan=None,
        timeout_seconds=1.0,
        max_stdout_bytes=10,
        max_stderr_bytes=10,
    )


if __name__ == "__main__":
    main()

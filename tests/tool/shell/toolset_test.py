from collections.abc import Awaitable, Callable
from pathlib import Path
from unittest import IsolatedAsyncioTestCase, TestCase, main

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
from avalan.tool import Tool
from avalan.tool.shell import (
    SHELL_COMMAND_IDS,
    CommandExecutor,
    ExecutionPolicy,
    ExecutionResult,
    ExecutionSpec,
    ShellCommandDefinition,
    ShellExecutionStatus,
    ShellOutputKind,
    ShellToolSet,
    ShellToolSettings,
    TrustedExecutableResolver,
    unavailable_executable_lookup,
)
from avalan.tool.shell.entities import ShellFormattedResult

_EXPECTED_SCHEMA_NAMES = tuple(
    f"shell.{command_id}" for command_id in SHELL_COMMAND_IDS
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

        self.assertEqual(_schema_names(all_enabled), _EXPECTED_SCHEMA_NAMES)
        self.assertEqual(_schema_names(concrete_enabled), ("shell.rg",))
        self.assertEqual(_schema_names(disabled), ())

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
                self.assertIs(getattr(tool, "supports_streaming"), True)


class ShellToolSetMissingBinaryTest(IsolatedAsyncioTestCase):
    async def test_each_tool_returns_command_unavailable_when_missing(
        self,
    ) -> None:
        fixture_root = Path(__file__).parent / "fixtures"
        settings = ShellToolSettings(
            workspace_root=str(fixture_root),
            allow_media_tools=True,
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
                self.assertIn("error_message: command is unavailable", output)

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

    async def test_container_runtime_does_not_override_local_backend(
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
        toolset = ShellToolSet(
            settings=settings,
            policy=ExecutionPolicy(settings=settings, resolver=_AllResolved()),
            container_runtime=ContainerToolRuntimeSettings(
                effective_settings=runtime.effective_settings,
                backend=backend,
            ),
        )

        output = await _call_cat(_tool_by_name(toolset, "cat"))

        self.assertIsInstance(output, ShellFormattedResult)
        self.assertEqual(output.execution_result.backend, "local")

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
    "file": _call_file,
    "find": _call_find,
    "wc": _call_wc,
    "awk": _call_awk,
    "sed": _call_sed,
    "jq": _call_jq,
    "pdfinfo": _call_pdfinfo,
    "pdftotext": _call_pdftotext,
    "pdftoppm": _call_pdftoppm,
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


if __name__ == "__main__":
    main()

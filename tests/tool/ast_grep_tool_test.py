import asyncio
from importlib import reload
from typing import cast
from unittest import IsolatedAsyncioTestCase, TestCase, main, skipIf
from unittest.mock import AsyncMock, patch

from avalan.container import (
    ContainerBackend,
    ContainerBackendCapabilities,
    ContainerBackendContainer,
    ContainerBackendDiagnosticCode,
    ContainerBackendOperation,
    ContainerBackendStream,
    ContainerBackendStreamChunk,
    ContainerBackendSupportLevel,
    ContainerDeviceClass,
    ContainerEffectiveSettings,
    ContainerExecutionScope,
    ContainerFakeBackend,
    ContainerFakeBackendScript,
    ContainerImagePolicy,
    ContainerMountType,
    ContainerNetworkMode,
    ContainerNetworkPolicy,
    ContainerProfile,
    ContainerRunPlan,
    ContainerSettingsSource,
    ContainerSurface,
    ContainerToolRuntimeSettings,
    ContainerTrustLevel,
    disabled_required_container_settings,
)
from avalan.entities import (
    ToolCall,
    ToolCallContext,
    ToolExecutionStreamEvent,
    ToolExecutionStreamKind,
)
from avalan.tool import code as code_module
from avalan.tool.code import (
    HAS_CODE_DEPENDENCIES,
    AstGrepContainerExecutionError,
    AstGrepTool,
    CodeTool,
    CodeToolSet,
)

_DIGEST = "8" * 64
_IMAGE = f"ghcr.io/example/code-search@sha256:{_DIGEST}"


class AstGrepToolTestCase(IsolatedAsyncioTestCase):
    async def test_search(self):
        tool = AstGrepTool()
        process = AsyncMock()
        process.communicate = AsyncMock(return_value=(b"out", b""))
        process.returncode = 0
        with patch(
            "avalan.tool.code.create_subprocess_exec",
            AsyncMock(return_value=process),
        ) as create:
            result = await tool("x", context=ToolCallContext(), lang="py")
        create.assert_awaited_once_with(
            "ast-grep",
            "--pattern",
            "x",
            "--lang",
            "py",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        self.assertEqual(result, "out")

    async def test_search_and_rewrite_with_paths(self):
        tool = AstGrepTool()
        process = AsyncMock()
        process.communicate = AsyncMock(return_value=(b"", b""))
        process.returncode = 0
        with patch(
            "avalan.tool.code.create_subprocess_exec",
            AsyncMock(return_value=process),
        ) as create:
            await tool(
                "p",
                context=ToolCallContext(),
                lang="ts",
                rewrite="r",
                paths=["a", "b"],
            )
        create.assert_awaited_once_with(
            "ast-grep",
            "--pattern",
            "p",
            "--lang",
            "ts",
            "--rewrite",
            "r",
            "a",
            "b",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

    async def test_error(self):
        tool = AstGrepTool()
        process = AsyncMock()
        process.communicate = AsyncMock(return_value=(b"", b"err"))
        process.returncode = 1
        with patch(
            "avalan.tool.code.create_subprocess_exec",
            AsyncMock(return_value=process),
        ):
            with self.assertRaises(RuntimeError):
                await tool("p", context=ToolCallContext(), lang="py")

    async def test_disabled_optional_container_settings_fall_back_local(
        self,
    ) -> None:
        tool = AstGrepTool(container_settings=_disabled_optional_settings())
        process = AsyncMock()
        process.communicate = AsyncMock(return_value=(b"local", b""))
        process.returncode = 0
        with patch(
            "avalan.tool.code.create_subprocess_exec",
            AsyncMock(return_value=process),
        ) as create:
            result = await tool("p", context=ToolCallContext(), lang="py")

        self.assertEqual(result, "local")
        create.assert_awaited_once()

    async def test_container_executes_ast_grep_through_lifecycle(
        self,
    ) -> None:
        backend = _RecordingContainerFakeBackend(
            ContainerFakeBackendScript(
                capabilities=_capabilities(),
                stream_chunks=(
                    ContainerBackendStreamChunk(
                        stream=ContainerBackendStream.STDOUT,
                        content=b"match\n",
                        sequence=0,
                    ),
                    ContainerBackendStreamChunk(
                        stream=ContainerBackendStream.STDERR,
                        content=b"warn\n",
                        sequence=1,
                    ),
                    ContainerBackendStreamChunk(
                        stream=ContainerBackendStream.PROGRESS,
                        content=b"attached",
                        sequence=2,
                    ),
                ),
            )
        )
        toolset = CodeToolSet(
            namespace="code",
            container_runtime=ContainerToolRuntimeSettings(
                effective_settings=_effective_settings(required=True),
                backend=backend,
            ),
        )
        events: list[ToolExecutionStreamEvent] = []

        async def record(event: ToolExecutionStreamEvent) -> None:
            events.append(event)

        tool = _ast_grep_tool(toolset)
        result = await tool(
            "call($A)",
            context=ToolCallContext(stream_event=record),
            lang="py",
            rewrite="call_new($A)",
            paths=["src/app.py", "-literal.py"],
        )

        self.assertEqual(result, "match\n")
        self.assertIn(ContainerBackendOperation.CREATE, backend.operations)
        self.assertEqual(len(backend.plans), 1)
        plan = backend.plans[0]
        self.assertEqual(plan.command.tool_name, "typed_tool:search.ast.grep")
        self.assertEqual(plan.command.command, "ast-grep")
        self.assertEqual(
            plan.command.argv,
            (
                "ast-grep",
                "--pattern",
                "call($A)",
                "--lang",
                "py",
                "--rewrite",
                "call_new($A)",
                "--",
                "src/app.py",
                "-literal.py",
            ),
        )
        self.assertEqual(plan.command.cwd, "/workspace")
        self.assertEqual(
            plan.command.scope,
            ContainerExecutionScope.SHELL_CONTAINER_EXECUTION,
        )
        self.assertEqual(
            [event.kind for event in events],
            [
                ToolExecutionStreamKind.STDOUT,
                ToolExecutionStreamKind.STDERR,
                ToolExecutionStreamKind.PROGRESS,
            ],
        )
        self.assertEqual(events[0].metadata["backend"], "container")

    async def test_container_runtime_enables_trusted_auto_backend(
        self,
    ) -> None:
        backend = _RecordingContainerFakeBackend(
            ContainerFakeBackendScript(
                capabilities=_capabilities(),
                stream_chunks=(
                    ContainerBackendStreamChunk(
                        stream=ContainerBackendStream.STDOUT,
                        content=b"match\n",
                        sequence=0,
                    ),
                ),
            )
        )
        toolset = CodeToolSet(
            namespace="code",
            container_runtime=ContainerToolRuntimeSettings(
                effective_settings=_auto_effective_settings(required=True),
                backend=backend,
            ),
        )

        result = await _ast_grep_tool(toolset)(
            "call($A)",
            context=ToolCallContext(),
            lang="py",
        )

        self.assertEqual(result, "match\n")
        self.assertIn(ContainerBackendOperation.CREATE, backend.operations)
        self.assertEqual(len(backend.plans), 1)

    async def test_container_runtime_forwards_apple_opt_in(self) -> None:
        without_opt_in = AstGrepTool(
            container_settings=_apple_effective_settings(required=True),
            container_backend=ContainerFakeBackend(
                ContainerFakeBackendScript(capabilities=_apple_capabilities())
            ),
        )
        backend = _RecordingContainerFakeBackend(
            ContainerFakeBackendScript(
                capabilities=_apple_capabilities(),
                stream_chunks=(
                    ContainerBackendStreamChunk(
                        stream=ContainerBackendStream.STDOUT,
                        content=b"match\n",
                        sequence=0,
                    ),
                ),
            )
        )
        toolset = CodeToolSet(
            namespace="code",
            container_runtime=ContainerToolRuntimeSettings(
                effective_settings=_apple_effective_settings(required=True),
                backend=backend,
                opt_in_backends=(ContainerBackend.APPLE_CONTAINER,),
            ),
        )

        with self.assertRaises(AstGrepContainerExecutionError) as raised:
            await without_opt_in(
                "call($A)",
                context=ToolCallContext(),
                lang="py",
            )
        result = await _ast_grep_tool(toolset)(
            "call($A)",
            context=ToolCallContext(),
            lang="py",
        )

        self.assertIn("capability_mismatch", raised.exception.message)
        self.assertEqual(result, "match\n")
        self.assertIn(ContainerBackendOperation.CREATE, backend.operations)

    async def test_container_stream_events_emit_each_output_kind_once(
        self,
    ) -> None:
        tool = AstGrepTool(
            container_settings=_effective_settings(required=True),
            container_backend=ContainerFakeBackend(
                ContainerFakeBackendScript(
                    capabilities=_capabilities(),
                    stream_chunks=(
                        ContainerBackendStreamChunk(
                            stream=ContainerBackendStream.STDOUT,
                            content=b"first",
                            sequence=0,
                        ),
                        ContainerBackendStreamChunk(
                            stream=ContainerBackendStream.STDOUT,
                            content=b"second",
                            sequence=1,
                        ),
                        ContainerBackendStreamChunk(
                            stream=ContainerBackendStream.STDERR,
                            content=b"warn",
                            sequence=2,
                        ),
                        ContainerBackendStreamChunk(
                            stream=ContainerBackendStream.STDERR,
                            content=b"ignored",
                            sequence=3,
                        ),
                    ),
                )
            ),
        )
        events: list[ToolExecutionStreamEvent] = []

        async def record(event: ToolExecutionStreamEvent) -> None:
            events.append(event)

        result = await tool(
            "p",
            context=ToolCallContext(stream_event=record),
            lang="py",
        )

        self.assertEqual(result, "firstsecond")
        self.assertEqual(
            [event.kind for event in events],
            [ToolExecutionStreamKind.STDOUT, ToolExecutionStreamKind.STDERR],
        )
        self.assertEqual(events[0].content, "firstsecond")
        self.assertEqual(events[1].content, "warnignored")

    async def test_required_container_without_backend_fails_closed(
        self,
    ) -> None:
        tool = AstGrepTool(
            container_settings=_effective_settings(required=True)
        )

        with patch(
            "avalan.tool.code.create_subprocess_exec",
            AsyncMock(),
        ) as create:
            with self.assertRaises(AstGrepContainerExecutionError) as raised:
                await tool("p", context=ToolCallContext(), lang="py")

        self.assertEqual(raised.exception.status, "tool_error")
        self.assertIn("no backend", str(raised.exception))
        self.assertEqual(
            raised.exception.metadata["execution_backend"],
            "container",
        )
        create.assert_not_called()

    async def test_required_container_without_profile_fails_closed(
        self,
    ) -> None:
        tool = AstGrepTool(
            container_settings=disabled_required_container_settings()
        )

        with patch(
            "avalan.tool.code.create_subprocess_exec",
            AsyncMock(),
        ) as create:
            with self.assertRaises(AstGrepContainerExecutionError) as raised:
                await tool("p", context=ToolCallContext(), lang="py")

        self.assertEqual(raised.exception.status, "policy_denied")
        self.assertIn("no profile", str(raised.exception))
        create.assert_not_called()

    async def test_container_profile_denial_happens_before_lifecycle(
        self,
    ) -> None:
        backend = _RecordingContainerFakeBackend(
            ContainerFakeBackendScript(capabilities=_capabilities())
        )
        tool = AstGrepTool(
            container_settings=_network_settings(),
            container_backend=backend,
        )

        with self.assertRaises(AstGrepContainerExecutionError) as raised:
            await tool("p", context=ToolCallContext(), lang="py")

        self.assertEqual(raised.exception.status, "tool_error")
        self.assertIn(
            "container_diagnostic_codes",
            raised.exception.metadata,
        )
        self.assertEqual(backend.plans, [])

    async def test_container_policy_denial_reports_diagnostics(self) -> None:
        backend = _RecordingContainerFakeBackend(
            ContainerFakeBackendScript(
                capabilities=_capabilities(),
                soft_operation_diagnostics={
                    ContainerBackendOperation.IMAGE_RESOLUTION: (
                        ContainerBackendDiagnosticCode.IMAGE_DENIED
                    )
                },
            )
        )
        tool = AstGrepTool(
            container_settings=_effective_settings(required=True),
            container_backend=backend,
        )

        with self.assertRaises(AstGrepContainerExecutionError) as raised:
            await tool("p", context=ToolCallContext(), lang="py")

        self.assertEqual(raised.exception.status, "policy_denied")
        codes = cast(
            tuple[str, ...],
            raised.exception.metadata["container_diagnostic_codes"],
        )
        self.assertIn(ContainerBackendDiagnosticCode.IMAGE_DENIED.value, codes)

    async def test_container_nonzero_exit_uses_stderr(self) -> None:
        tool = AstGrepTool(
            container_settings=_effective_settings(required=True),
            container_backend=ContainerFakeBackend(
                ContainerFakeBackendScript(
                    capabilities=_capabilities(),
                    stream_chunks=(
                        ContainerBackendStreamChunk(
                            stream=ContainerBackendStream.STDERR,
                            content=b"bad pattern\n",
                            sequence=0,
                        ),
                    ),
                    wait_exit_code=2,
                )
            ),
        )

        with self.assertRaises(AstGrepContainerExecutionError) as raised:
            await tool("p", context=ToolCallContext(), lang="py")

        self.assertEqual(raised.exception.status, "nonzero_exit")
        self.assertEqual(str(raised.exception), "bad pattern\n")
        self.assertEqual(raised.exception.stderr, "bad pattern\n")

    async def test_container_lifecycle_failures_are_reported(self) -> None:
        cases = (
            (
                "timeout",
                ContainerFakeBackendScript(
                    capabilities=_capabilities(),
                    wait_timed_out=True,
                ),
                "timeout",
            ),
            (
                "cancellation",
                ContainerFakeBackendScript(
                    capabilities=_capabilities(),
                    cancel_operations=(ContainerBackendOperation.WAIT,),
                ),
                "cancelled",
            ),
            (
                "cleanup failure",
                ContainerFakeBackendScript(
                    capabilities=_capabilities(),
                    cleanup_uncertain=True,
                ),
                "tool_error",
            ),
        )

        for name, script, status in cases:
            with self.subTest(name=name):
                tool = AstGrepTool(
                    container_settings=_effective_settings(required=True),
                    container_backend=ContainerFakeBackend(script),
                )

                with self.assertRaises(
                    AstGrepContainerExecutionError
                ) as raised:
                    await tool("p", context=ToolCallContext(), lang="py")

                self.assertEqual(raised.exception.status, status)
                self.assertEqual(
                    raised.exception.metadata["execution_backend"],
                    "container",
                )

    async def test_container_path_errors_do_not_echo_host_paths(self) -> None:
        tool = AstGrepTool(
            container_settings=_effective_settings(required=True),
            container_backend=ContainerFakeBackend(
                ContainerFakeBackendScript(capabilities=_capabilities())
            ),
        )
        host_path = "/Users/mariano/secret.py"

        with self.assertRaises(AstGrepContainerExecutionError) as raised:
            await tool(
                "p",
                context=ToolCallContext(),
                lang="py",
                paths=[host_path],
            )

        self.assertEqual(raised.exception.status, "tool_error")
        self.assertNotIn(host_path, str(raised.exception))

    def test_display_projectors_and_helper_fallbacks(self) -> None:
        self.assertIsNotNone(
            CodeTool().tool_display_projector(
                ToolCall(
                    id="call-code",
                    name="code.run",
                    arguments={"code": "def run(): return 1"},
                )
            )
        )
        self.assertIsNotNone(
            AstGrepTool().tool_display_projector(
                ToolCall(
                    id="call-ast-grep",
                    name="code.search.ast.grep",
                    arguments={"pattern": "p", "lang": "py"},
                )
            )
        )
        self.assertIsNone(
            code_module._ast_grep_container_plan(
                "p",
                lang="py",
                rewrite=None,
                paths=None,
                container_settings=None,
            )
        )
        self.assertEqual(
            code_module._diagnostic_summary(()),
            "container execution failed",
        )


@skipIf(not HAS_CODE_DEPENDENCIES, "RestrictedPython not installed")
class CodeToolSetTestCase(TestCase):
    def test_json_schema_includes_ast_grep(self):
        toolset = CodeToolSet(namespace="code")
        schemas = toolset.json_schemas()
        names = [s["function"]["name"] for s in schemas]
        self.assertIn("code.search.ast.grep", names)

    def test_json_schema_does_not_expose_container_authority(self) -> None:
        toolset = CodeToolSet(
            namespace="code",
            container_runtime=ContainerToolRuntimeSettings(
                effective_settings=_effective_settings(required=True),
                backend=ContainerFakeBackend(
                    ContainerFakeBackendScript(capabilities=_capabilities())
                ),
            ),
        )
        serialized_schema = str(toolset.json_schemas())

        self.assertIn("code.search.ast.grep", serialized_schema)
        for forbidden in (
            "mount",
            "secret",
            "backend",
            "profile",
            "runtime",
            "container_image",
        ):
            self.assertNotIn(forbidden, serialized_schema)

    def test_restricted_python_import_fallback(self) -> None:
        def blocked_import(
            name: str,
            *args: object,
            **kwargs: object,
        ) -> object:
            if name == "RestrictedPython":
                raise ImportError
            return original_import(name, *args, **kwargs)

        original_import = __import__
        with patch("builtins.__import__", side_effect=blocked_import):
            fallback_module = reload(code_module)
            self.assertFalse(fallback_module.HAS_CODE_DEPENDENCIES)
            self.assertIsNone(fallback_module.compile_restricted)
        reload(code_module)


class _RecordingContainerFakeBackend(ContainerFakeBackend):
    def __init__(self, script: ContainerFakeBackendScript) -> None:
        super().__init__(script)
        self.plans: list[ContainerRunPlan] = []

    async def create(
        self,
        plan: ContainerRunPlan,
    ) -> ContainerBackendContainer:
        self.plans.append(plan)
        return await super().create(plan)


def _ast_grep_tool(toolset: CodeToolSet) -> AstGrepTool:
    for tool in toolset.tools:
        if getattr(tool, "__name__", "") == "search.ast.grep":
            assert isinstance(tool, AstGrepTool)
            return tool
    raise AssertionError("ast-grep tool not found")


def _effective_settings(
    *,
    required: bool = False,
) -> ContainerEffectiveSettings:
    profile = ContainerProfile.minimal_readonly(
        name="code-search-readonly",
        image_reference=_IMAGE,
    )
    return ContainerEffectiveSettings(
        backend=ContainerBackend.DOCKER,
        required=required,
        scope=ContainerExecutionScope.SHELL_CONTAINER_EXECUTION,
        source=_source(),
        policy_version="phase19",
        profile_registry_id="code",
        profile_name=profile.name,
        profile=profile,
        allowed_profiles=(profile.name,),
    )


def _apple_effective_settings(
    *,
    required: bool = False,
) -> ContainerEffectiveSettings:
    profile = ContainerProfile.minimal_readonly(
        name="apple-code-search-readonly",
        image_reference=_IMAGE,
    )
    return ContainerEffectiveSettings(
        backend=ContainerBackend.APPLE_CONTAINER,
        required=required,
        scope=ContainerExecutionScope.SHELL_CONTAINER_EXECUTION,
        source=_source(),
        policy_version="phase19",
        profile_registry_id="code",
        profile_name=profile.name,
        profile=profile,
        allowed_profiles=(profile.name,),
    )


def _auto_effective_settings(
    *,
    required: bool = False,
) -> ContainerEffectiveSettings:
    profile = ContainerProfile.minimal_readonly(
        name="auto-code-search-readonly",
        image_reference=_IMAGE,
    )
    return ContainerEffectiveSettings(
        backend=ContainerBackend.AUTO,
        required=required,
        scope=ContainerExecutionScope.SHELL_CONTAINER_EXECUTION,
        source=_source(),
        policy_version="phase19",
        profile_registry_id="code",
        profile_name=profile.name,
        profile=profile,
        allowed_profiles=(profile.name,),
    )


def _network_settings() -> ContainerEffectiveSettings:
    profile = ContainerProfile(
        name="code-search-network",
        image=ContainerImagePolicy(reference=_IMAGE),
        network=ContainerNetworkPolicy(mode=ContainerNetworkMode.FULL),
    )
    return ContainerEffectiveSettings(
        backend=ContainerBackend.DOCKER,
        required=True,
        scope=ContainerExecutionScope.SHELL_CONTAINER_EXECUTION,
        source=_source(),
        policy_version="phase19",
        profile_registry_id="code",
        profile_name=profile.name,
        profile=profile,
        allowed_profiles=(profile.name,),
    )


def _disabled_optional_settings() -> ContainerEffectiveSettings:
    return ContainerEffectiveSettings(
        backend=ContainerBackend.NONE,
        required=False,
        scope=ContainerExecutionScope.SHELL_CONTAINER_EXECUTION,
        source=_source(),
        policy_version="phase19",
        profile_registry_id="code",
    )


def _source() -> ContainerSettingsSource:
    return ContainerSettingsSource(
        surface=ContainerSurface.SDK,
        trust_level=ContainerTrustLevel.TRUSTED_OPERATOR,
    )


def _capabilities() -> ContainerBackendCapabilities:
    return ContainerBackendCapabilities(
        backend=ContainerBackend.DOCKER,
        host_os="linux",
        guest_os="linux",
        architecture="amd64",
        rootless=True,
        network_modes=(ContainerNetworkMode.NONE,),
        mount_types=(ContainerMountType.WORKSPACE, ContainerMountType.OUTPUT),
        device_classes=(ContainerDeviceClass.CPU,),
        resource_limits=True,
        streaming_attach=True,
        stats=True,
    )


def _apple_capabilities() -> ContainerBackendCapabilities:
    return ContainerBackendCapabilities(
        backend=ContainerBackend.APPLE_CONTAINER,
        host_os="darwin",
        guest_os="linux",
        architecture="amd64",
        support_level=ContainerBackendSupportLevel.OPT_IN,
        rootless=False,
        network_modes=(ContainerNetworkMode.NONE,),
        mount_types=(ContainerMountType.WORKSPACE, ContainerMountType.OUTPUT),
        device_classes=(),
        resource_limits=True,
        per_container_vm_isolation=True,
        streaming_attach=True,
        stats=True,
    )


if __name__ == "__main__":
    main()

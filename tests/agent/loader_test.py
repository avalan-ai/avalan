import tomllib
from argparse import Namespace
from contextlib import AsyncExitStack
from logging import DEBUG, INFO, Logger
from os import chmod, devnull, environ, geteuid
from pathlib import Path
from shutil import which
from subprocess import run
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import Any, Callable, cast
from unittest import IsolatedAsyncioTestCase, main
from unittest.mock import AsyncMock, MagicMock, call, patch
from uuid import uuid4

from async_helpers import run_async
from jinja2 import Environment, FileSystemLoader

from avalan.agent.execution import AgentExecution
from avalan.agent.loader import (
    OrchestratorLoader,
    _merge_shell_tool_settings,
    _normalize_file_run_reasoning,
    _shell_tool_runtime_settings,
)
from avalan.cli.commands import agent as agent_cmds
from avalan.container import (
    ContainerProfileSelection,
    ContainerRuntimeEnvelopeKind,
    ContainerSurface,
    ContainerToolRuntimeSettings,
    trusted_container_runtime_from_mapping,
    trusted_container_source,
)
from avalan.entities import (
    EngineUri,
    Message,
    MessageContentFile,
    MessageContentText,
    MessageRole,
    OrchestratorSettings,
    PermanentMemoryStoreSettings,
    ReasoningSummaryMode,
    ToolCall,
    ToolCallContext,
    ToolCallDiagnosticCode,
    ToolCallRecoveryFormat,
    ToolCallResult,
    ToolFormat,
    ToolManagerSettings,
    ToolNamePolicyMode,
    ToolNameResolutionStatus,
    TransformerEngineSettings,
)
from avalan.event import Event, EventType
from avalan.event.manager import EventManagerMode
from avalan.isolation import (
    IsolationEffectiveSettings,
    IsolationMode,
    IsolationProfileSelection,
    IsolationSettings,
    IsolationToolRuntimeSettings,
    SandboxProfileSelection,
    trusted_isolation_source,
)
from avalan.model import ModelCapabilityCatalog
from avalan.model.capability import ProviderCapabilitySupport
from avalan.model.hubs.huggingface import HuggingfaceHub
from avalan.tool import ToolSet
from avalan.tool.browser import BrowserToolSettings
from avalan.tool.context import ToolSettingsContext
from avalan.tool.database import DatabaseToolSettings
from avalan.tool.graph_settings import GraphToolSettings
from avalan.tool.manager import ToolManager
from avalan.tool.shell import (
    ExecutionPolicy,
    ExecutionResult,
    ExecutionSpec,
    ShellCommandRequest,
    ShellExecutionErrorCode,
    ShellExecutionStatus,
    ShellGitToolSettings,
    ShellOutputKind,
    ShellPolicyDenied,
    ShellToolSet,
    ShellToolSettings,
)
from avalan.tool_cycles import UNLIMITED_TOOL_CYCLES


def _named_toolset(namespace: str) -> MagicMock:
    toolset = MagicMock()
    toolset.namespace = namespace
    return toolset


def _minimal_agent_toml() -> str:
    return """
[agent]
role = "assistant"

[engine]
uri = "ai://local/model"
"""


def _write_agent_shell_git_example_repo(root: Path, git_binary: str) -> Path:
    repo = root / "repo"
    repo.mkdir()
    _run_agent_shell_git(git_binary, "init", cwd=repo)
    _run_agent_shell_git(git_binary, "checkout", "-B", "main", cwd=repo)
    (repo / "README.md").write_text("initial\n", encoding="utf-8")
    _run_agent_shell_git(git_binary, "add", "README.md", cwd=repo)
    _run_agent_shell_git(
        git_binary,
        "-c",
        "user.name=Avalan Test",
        "-c",
        "user.email=avalan@example.test",
        "commit",
        "-m",
        "initial commit",
        cwd=repo,
    )
    (repo / "README.md").write_text(
        "initial\nworktree change\n",
        encoding="utf-8",
    )
    return repo


def _run_agent_shell_git(git_binary: str, *args: str, cwd: Path) -> None:
    git_env = dict(environ)
    git_env.update(
        {
            "GIT_CONFIG_GLOBAL": devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_TERMINAL_PROMPT": "0",
        }
    )
    run(
        (git_binary, *args),
        cwd=cwd,
        env=git_env,
        check=True,
        capture_output=True,
        text=True,
    )


def _orchestrator_settings(
    *,
    tools: list[str] | None,
) -> OrchestratorSettings:
    return OrchestratorSettings(
        agent_id=uuid4(),
        orchestrator_type=None,
        agent_config={"role": "assistant"},
        uri="ai://local/model",
        engine_config={},
        tools=tools,
        call_options=None,
        template_vars=None,
        memory_permanent_message=None,
        permanent_memory=None,
        memory_recent=False,
        sentence_model_id=OrchestratorLoader.DEFAULT_SENTENCE_MODEL_ID,
        sentence_model_engine_config=None,
        sentence_model_max_tokens=500,
        sentence_model_overlap_size=125,
        sentence_model_window_size=250,
        json_config=None,
        log_events=True,
    )


async def _from_settings_tool_manager_kwargs(
    settings: OrchestratorSettings,
    *,
    tool_settings: ToolSettingsContext | None = None,
) -> dict[str, Any]:
    hub = MagicMock(spec=HuggingfaceHub)
    logger = MagicMock(spec=Logger)
    stack = AsyncExitStack()
    memory = MagicMock()
    tool = MagicMock()
    tool.__aenter__ = AsyncMock(return_value=tool)

    with (
        patch(
            "avalan.agent.loader.MemoryManager.create_instance",
            new=AsyncMock(return_value=memory),
        ),
        patch("avalan.agent.loader.ModelManager", return_value=MagicMock()),
        patch("avalan.agent.loader.DefaultOrchestrator", return_value="orch"),
        patch(
            "avalan.agent.loader.ToolManager.create_instance",
            return_value=tool,
        ) as tm_patch,
        patch("avalan.agent.loader.EventManager", return_value=MagicMock()),
        patch("avalan.agent.loader.HAS_GRAPH_DEPENDENCIES", False),
        patch("avalan.agent.loader.HAS_CODE_DEPENDENCIES", False),
        patch("avalan.agent.loader.HAS_BROWSER_DEPENDENCIES", False),
        patch(
            "avalan.agent.loader.MathToolSet",
            side_effect=lambda *, namespace: _named_toolset(namespace),
        ),
        patch(
            "avalan.agent.loader.McpToolSet",
            side_effect=lambda *, namespace: _named_toolset(namespace),
        ),
        patch(
            "avalan.agent.loader.MemoryToolSet",
            side_effect=lambda _memory, *, namespace: _named_toolset(
                namespace
            ),
        ),
    ):
        loader = OrchestratorLoader(
            hub=hub,
            logger=logger,
            participant_id=uuid4(),
            stack=stack,
        )
        result = await loader.from_settings(
            settings,
            tool_settings=tool_settings,
        )

    await stack.aclose()
    assert result == "orch"
    return dict(tm_patch.call_args.kwargs)


async def _from_file_tool_manager_kwargs(
    config: str,
    *,
    tool_settings: ToolSettingsContext | None = None,
) -> dict[str, Any]:
    hub = MagicMock(spec=HuggingfaceHub)
    logger = MagicMock(spec=Logger)
    stack = AsyncExitStack()
    memory = MagicMock()
    tool = MagicMock()
    tool.__aenter__ = AsyncMock(return_value=tool)

    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "agent.toml"
        path.write_text(config, encoding="utf-8")
        with (
            patch(
                "avalan.agent.loader.MemoryManager.create_instance",
                new=AsyncMock(return_value=memory),
            ),
            patch(
                "avalan.agent.loader.ModelManager", return_value=MagicMock()
            ),
            patch(
                "avalan.agent.loader.DefaultOrchestrator",
                return_value="orch",
            ),
            patch(
                "avalan.agent.loader.ToolManager.create_instance",
                return_value=tool,
            ) as tm_patch,
            patch(
                "avalan.agent.loader.EventManager", return_value=MagicMock()
            ),
            patch("avalan.agent.loader.HAS_GRAPH_DEPENDENCIES", False),
            patch("avalan.agent.loader.HAS_CODE_DEPENDENCIES", False),
            patch("avalan.agent.loader.HAS_BROWSER_DEPENDENCIES", False),
            patch(
                "avalan.agent.loader.MathToolSet",
                side_effect=lambda *, namespace: _named_toolset(namespace),
            ),
            patch(
                "avalan.agent.loader.McpToolSet",
                side_effect=lambda *, namespace: _named_toolset(namespace),
            ),
            patch(
                "avalan.agent.loader.MemoryToolSet",
                side_effect=lambda _memory, *, namespace: _named_toolset(
                    namespace
                ),
            ),
        ):
            loader = OrchestratorLoader(
                hub=hub,
                logger=logger,
                participant_id=uuid4(),
                stack=stack,
            )
            result = await loader.from_file(
                str(path),
                agent_id=uuid4(),
                tool_settings=tool_settings,
            )

    await stack.aclose()
    assert result == "orch"
    return dict(tm_patch.call_args.kwargs)


async def _from_file_code_shell_toolset_kwargs(
    config: str,
    *,
    tool_settings: ToolSettingsContext | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    hub = MagicMock(spec=HuggingfaceHub)
    logger = MagicMock(spec=Logger)
    stack = AsyncExitStack()
    memory = MagicMock()
    tool = MagicMock()
    tool.__aenter__ = AsyncMock(return_value=tool)

    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "agent.toml"
        path.write_text(config, encoding="utf-8")
        with (
            patch(
                "avalan.agent.loader.MemoryManager.create_instance",
                new=AsyncMock(return_value=memory),
            ),
            patch(
                "avalan.agent.loader.ModelManager", return_value=MagicMock()
            ),
            patch(
                "avalan.agent.loader.DefaultOrchestrator",
                return_value="orch",
            ),
            patch(
                "avalan.agent.loader.ToolManager.create_instance",
                return_value=tool,
            ),
            patch(
                "avalan.agent.loader.EventManager", return_value=MagicMock()
            ),
            patch("avalan.agent.loader.HAS_GRAPH_DEPENDENCIES", False),
            patch("avalan.agent.loader.HAS_CODE_DEPENDENCIES", True),
            patch("avalan.agent.loader.HAS_BROWSER_DEPENDENCIES", False),
            patch(
                "avalan.agent.loader.MathToolSet",
                side_effect=lambda *, namespace: _named_toolset(namespace),
            ),
            patch(
                "avalan.agent.loader.McpToolSet",
                side_effect=lambda *, namespace: _named_toolset(namespace),
            ),
            patch(
                "avalan.agent.loader.MemoryToolSet",
                side_effect=lambda _memory, *, namespace: _named_toolset(
                    namespace
                ),
            ),
            patch(
                "avalan.agent.loader.CodeToolSet",
                side_effect=lambda **_kwargs: _named_toolset("code"),
            ) as code_patch,
            patch(
                "avalan.agent.loader.ShellToolSet",
                side_effect=lambda **_kwargs: _named_toolset("shell"),
            ) as shell_patch,
        ):
            loader = OrchestratorLoader(
                hub=hub,
                logger=logger,
                participant_id=uuid4(),
                stack=stack,
            )
            result = await loader.from_file(
                str(path),
                agent_id=uuid4(),
                tool_settings=tool_settings,
            )

    await stack.aclose()
    assert result == "orch"
    return (
        dict(code_patch.call_args.kwargs),
        dict(shell_patch.call_args.kwargs),
    )


def _shell_namespaces(kwargs: dict[str, Any]) -> list[str | None]:
    return [
        toolset.namespace
        for toolset in kwargs["available_toolsets"]
        if toolset.namespace == "shell"
    ]


def _mcp_namespaces(kwargs: dict[str, Any]) -> list[str | None]:
    return [
        toolset.namespace
        for toolset in kwargs["available_toolsets"]
        if toolset.namespace == "mcp"
    ]


def _a2a_namespaces(kwargs: dict[str, Any]) -> list[str | None]:
    return [
        toolset.namespace
        for toolset in kwargs["available_toolsets"]
        if toolset.namespace == "a2a"
    ]


def _toolset_namespaces(kwargs: dict[str, Any]) -> list[str | None]:
    return [toolset.namespace for toolset in kwargs["available_toolsets"]]


def _shell_only_manager(kwargs: dict[str, Any]) -> ToolManager:
    shell_toolsets = [
        toolset
        for toolset in kwargs["available_toolsets"]
        if toolset.namespace == "shell"
    ]
    return ToolManager.create_instance(
        available_toolsets=shell_toolsets,
        enable_tools=kwargs["enable_tools"],
        settings=ToolManagerSettings(),
    )


async def _loaded_agent_model_input(
    config: str, input_value: object
) -> object:
    captured_contexts: list[Any] = []

    class CapturingTemplateEngineAgent:
        def __init__(
            self,
            engine: object,
            *_args: object,
            **_kwargs: object,
        ) -> None:
            self.engine = engine
            self.input_token_count = 0
            self.last_prompt = None
            self.output = None

        async def __call__(self, context: Any) -> object:
            captured_contexts.append(context)
            self.last_prompt = (context.input, None, None, None)
            self.output = MagicMock(
                input_token_count=1,
                output_token_count=1,
                usage=None,
            )
            return self.output

        async def sync_messages(
            self,
            execution: AgentExecution | None = None,
        ) -> None:
            assert execution is None or isinstance(execution, AgentExecution)
            return None

        def acknowledge_provider_handoff(self, response: object) -> None:
            assert response is self.output

        async def drain_pending_provider_cleanups(
            self,
            execution: AgentExecution | None = None,
            *,
            abandon_unclaimed: bool = False,
        ) -> tuple[BaseException, ...]:
            assert execution is None or isinstance(execution, AgentExecution)
            assert isinstance(abandon_unclaimed, bool)
            return ()

        def __str__(self) -> str:
            return "capturing-template-engine-agent"

    class CapturingOrchestratorResponse:
        execution = None
        ownership_cleanup_complete = True

        async def aclose(self) -> None:
            return None

        async def sync_messages(self) -> None:
            return None

    fake_engine = MagicMock(model_id="m", tokenizer=None)
    fake_engine.__enter__.return_value = fake_engine
    fake_engine.__exit__.return_value = None
    fake_engine.provider_capability_support = ProviderCapabilitySupport()

    model_manager = MagicMock()
    model_manager.__enter__.return_value = model_manager
    model_manager.__exit__.return_value = None
    model_manager.parse_uri.return_value = EngineUri(
        host=None,
        port=None,
        user=None,
        password=None,
        vendor=None,
        model_id="m",
        params={},
    )
    model_manager.get_engine_settings.return_value = (
        TransformerEngineSettings()
    )
    model_manager.load_engine.return_value = fake_engine

    memory = MagicMock()
    memory.participant_id = uuid4()
    memory.permanent_message = None
    event_manager = MagicMock()
    event_manager.trigger = AsyncMock()
    event_manager.aclose = AsyncMock()
    stack = AsyncExitStack()

    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "agent.toml"
        path.write_text(config, encoding="utf-8")
        try:
            with (
                patch(
                    "avalan.agent.loader.MemoryManager.create_instance",
                    new=AsyncMock(return_value=memory),
                ),
                patch(
                    "avalan.agent.loader.ModelManager",
                    return_value=model_manager,
                ),
                patch(
                    "avalan.agent.loader.EventManager",
                    return_value=event_manager,
                ),
                patch(
                    "avalan.agent.orchestrator.TemplateEngineAgent",
                    CapturingTemplateEngineAgent,
                ),
                patch(
                    "avalan.agent.orchestrator.OrchestratorResponse",
                    return_value=CapturingOrchestratorResponse(),
                ),
                patch("avalan.agent.loader.HAS_GRAPH_DEPENDENCIES", False),
                patch("avalan.agent.loader.HAS_CODE_DEPENDENCIES", False),
                patch("avalan.agent.loader.HAS_BROWSER_DEPENDENCIES", False),
                patch(
                    "avalan.agent.loader.MathToolSet",
                    side_effect=lambda *, namespace: ToolSet(
                        namespace=namespace,
                        tools=[],
                    ),
                ),
                patch(
                    "avalan.agent.loader.McpToolSet",
                    side_effect=lambda *, namespace: ToolSet(
                        namespace=namespace,
                        tools=[],
                    ),
                ),
                patch(
                    "avalan.agent.loader.MemoryToolSet",
                    side_effect=lambda _memory, *, namespace: ToolSet(
                        namespace=namespace,
                        tools=[],
                    ),
                ),
            ):
                loader = OrchestratorLoader(
                    hub=MagicMock(spec=HuggingfaceHub),
                    logger=MagicMock(spec=Logger),
                    participant_id=uuid4(),
                    stack=stack,
                )
                agent = await loader.from_file(str(path), agent_id=uuid4())
                async with agent:
                    await agent(input_value)
        finally:
            await stack.aclose()

    assert captured_contexts
    return captured_contexts[0].input


def _input_text_blocks(input_value: object) -> list[str]:
    if isinstance(input_value, str):
        return [input_value]
    if isinstance(input_value, Message):
        return _content_text_blocks(input_value.content)
    if isinstance(input_value, list):
        texts: list[str] = []
        for item in input_value:
            texts.extend(_input_text_blocks(item))
        return texts
    return []


def _content_text_blocks(content: object) -> list[str]:
    if isinstance(content, str):
        return [content]
    if isinstance(content, MessageContentText):
        return [content.text]
    if isinstance(content, list):
        texts: list[str] = []
        for item in content:
            texts.extend(_content_text_blocks(item))
        return texts
    return []


def _input_file_blocks(input_value: object) -> list[MessageContentFile]:
    if isinstance(input_value, Message):
        return _content_file_blocks(input_value.content)
    if isinstance(input_value, list):
        files: list[MessageContentFile] = []
        for item in input_value:
            files.extend(_input_file_blocks(item))
        return files
    return []


def _content_file_blocks(content: object) -> list[MessageContentFile]:
    if isinstance(content, MessageContentFile):
        return [content]
    if isinstance(content, list):
        files: list[MessageContentFile] = []
        for item in content:
            files.extend(_content_file_blocks(item))
        return files
    return []


class _AgentShellPolicy(ExecutionPolicy):
    def __init__(
        self,
        *,
        denial: ShellPolicyDenied | None = None,
        executable: str | None = "/usr/bin/cat",
    ) -> None:
        self._denial = denial
        self._executable = executable
        self.requests: list[ShellCommandRequest] = []

    async def normalize(
        self,
        request: ShellCommandRequest,
    ) -> ExecutionSpec:
        self.requests.append(request)
        if self._denial is not None:
            raise self._denial
        return ExecutionPolicy().create_execution_spec(
            backend="local",
            tool_name=request.tool_name,
            command=request.command,
            executable=self._executable,
            argv=("cat", "--", "agent.txt"),
            display_argv=("cat", "--", "agent.txt"),
            cwd=".",
            display_cwd=".",
            env={"LC_ALL": "C"},
            stdin=None,
            stdout_media_type="text/plain",
            output_kind=ShellOutputKind.TEXT,
            resource_class="standard",
            output_plan=None,
            timeout_seconds=1.0,
            max_stdout_bytes=1024,
            max_stderr_bytes=1024,
        )


class _AgentShellExecutor:
    def __init__(self, stdout: str) -> None:
        self._stdout = stdout
        self.specs: list[ExecutionSpec] = []

    async def execute(self, spec: ExecutionSpec) -> ExecutionResult:
        self.specs.append(spec)
        if spec.executable is None:
            return ExecutionResult(
                backend=spec.backend,
                tool_name=spec.tool_name,
                command=spec.command,
                argv=spec.argv,
                display_argv=spec.display_argv,
                cwd=spec.cwd,
                display_cwd=spec.display_cwd,
                status=ShellExecutionStatus.COMMAND_UNAVAILABLE,
                exit_code=None,
                stdout="",
                stderr="",
                stdout_media_type=spec.stdout_media_type,
                output_kind=spec.output_kind,
                stdout_bytes=0,
                stderr_bytes=0,
                stdout_truncated=False,
                stderr_truncated=False,
                timed_out=False,
                cancelled=False,
                duration_ms=1,
                error_code=ShellExecutionErrorCode.COMMAND_UNAVAILABLE,
                error_message="command is unavailable",
                metadata=spec.metadata,
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
            stdout=self._stdout,
            stderr="",
            stdout_media_type=spec.stdout_media_type,
            output_kind=spec.output_kind,
            stdout_bytes=len(self._stdout.encode()),
            stderr_bytes=0,
            stdout_truncated=False,
            stderr_truncated=False,
            timed_out=False,
            cancelled=False,
            duration_ms=1,
            error_code=ShellExecutionErrorCode.COMPLETED,
            metadata=spec.metadata,
        )


async def _load_shell_agent_tool_result(
    *,
    policy: _AgentShellPolicy,
    executor: _AgentShellExecutor,
    path: str,
) -> tuple[list[str], object]:
    def shell_toolset_factory(
        *,
        container_runtime: object | None = None,
        settings: ShellToolSettings,
        namespace: str,
    ) -> ShellToolSet:
        return ShellToolSet(
            settings=settings,
            container_runtime=container_runtime,
            policy=policy,
            executor=executor,
            namespace=namespace,
        )

    with NamedTemporaryFile("w+", suffix=".toml") as tmp:
        tmp.write(
            _minimal_agent_toml()
            + '\n[tool]\nenable = ["shell.cat"]\n'
            + '\n[tool.shell]\nworkspace_root = "."\n'
        )
        tmp.flush()
        stack = AsyncExitStack()
        model_manager = MagicMock()
        model_manager.__enter__.return_value = model_manager
        model_manager.parse_uri.return_value = "uri_obj"
        model_manager.get_engine_settings.return_value = "settings_obj"

        try:
            with (
                patch(
                    "avalan.agent.loader.MemoryManager.create_instance",
                    new=AsyncMock(return_value=MagicMock()),
                ),
                patch(
                    "avalan.agent.loader.ModelManager",
                    return_value=model_manager,
                ),
                patch(
                    "avalan.agent.loader.ShellToolSet",
                    side_effect=shell_toolset_factory,
                ),
                patch("avalan.agent.loader.HAS_GRAPH_DEPENDENCIES", False),
                patch("avalan.agent.loader.HAS_CODE_DEPENDENCIES", False),
                patch(
                    "avalan.agent.loader.HAS_BROWSER_DEPENDENCIES",
                    False,
                ),
                patch(
                    "avalan.agent.loader.MathToolSet",
                    side_effect=lambda *, namespace: ToolSet(
                        namespace=namespace,
                        tools=[],
                    ),
                ),
                patch(
                    "avalan.agent.loader.McpToolSet",
                    side_effect=lambda *, namespace: ToolSet(
                        namespace=namespace,
                        tools=[],
                    ),
                ),
                patch(
                    "avalan.agent.loader.MemoryToolSet",
                    side_effect=lambda _memory, *, namespace: ToolSet(
                        namespace=namespace,
                        tools=[],
                    ),
                ),
            ):
                loader = OrchestratorLoader(
                    hub=MagicMock(spec=HuggingfaceHub),
                    logger=MagicMock(spec=Logger),
                    participant_id=uuid4(),
                    stack=stack,
                )
                agent = await loader.from_file(
                    tmp.name,
                    agent_id=uuid4(),
                )

                schema_names = [
                    descriptor.schema["function"]["name"]
                    for descriptor in agent.tool.list_tools()
                    if descriptor.schema is not None
                ]
                outcome = await agent.tool.execute_call(
                    ToolCall(
                        id="call-1",
                        name="shell.cat",
                        arguments={"path": path},
                    ),
                    context=ToolCallContext(),
                )
        finally:
            await stack.aclose()
    return schema_names, outcome


class LoaderPropertyTestCase(IsolatedAsyncioTestCase):
    async def test_hub_and_participant_id(self):
        stack = AsyncExitStack()
        hub = MagicMock(spec=HuggingfaceHub)
        logger = MagicMock(spec=Logger)
        participant_id = uuid4()
        loader = OrchestratorLoader(
            hub=hub, logger=logger, participant_id=participant_id, stack=stack
        )
        self.assertIs(loader.hub, hub)
        self.assertEqual(loader.participant_id, participant_id)
        await stack.aclose()


class LoaderFromFileTestCase(IsolatedAsyncioTestCase):
    async def test_file_not_found(self):
        stack = AsyncExitStack()
        loader = OrchestratorLoader(
            hub=MagicMock(spec=HuggingfaceHub),
            logger=MagicMock(spec=Logger),
            participant_id=uuid4(),
            stack=stack,
        )
        with self.assertRaises(FileNotFoundError):
            await loader.from_file("missing.toml", agent_id=uuid4())
        await stack.aclose()

    async def test_merge_shell_tool_settings_without_base(self) -> None:
        override = ShellToolSettings(max_head_lines=7)

        self.assertIs(_merge_shell_tool_settings(None, override), override)

    async def test_merge_shell_tool_settings_replaces_without_explicit_fields(
        self,
    ) -> None:
        base = ShellToolSettings(workspace_root="/workspace/project")
        override = ShellToolSettings()

        self.assertIs(
            _merge_shell_tool_settings(base, override),
            override,
        )

    async def test_merge_shell_tool_settings_applies_explicit_default_override(
        self,
    ) -> None:
        default = ShellToolSettings()
        base = ShellToolSettings(max_head_lines=12)

        merged = _merge_shell_tool_settings(
            base,
            default,
            explicit_fields=frozenset({"max_head_lines"}),
        )

        assert merged is not None
        self.assertEqual(merged.max_head_lines, default.max_head_lines)
        self.assertEqual(merged.workspace_root, base.workspace_root)
        self.assertIsNot(merged, default)
        self.assertIsNot(merged, base)

    async def test_merge_shell_tool_settings_applies_explicit_false_override(
        self,
    ) -> None:
        base = ShellToolSettings(input_file_manifest_enabled=True)
        override = ShellToolSettings(input_file_manifest_enabled=False)

        merged = _merge_shell_tool_settings(
            base,
            override,
            explicit_fields=frozenset({"input_file_manifest_enabled"}),
        )

        assert merged is not None
        self.assertFalse(merged.input_file_manifest_enabled)

    async def test_merge_shell_tool_settings_keeps_empty_explicit_base(
        self,
    ) -> None:
        base = ShellToolSettings(workspace_root="/workspace/project")

        self.assertIs(
            _merge_shell_tool_settings(
                base,
                ShellToolSettings(),
                explicit_fields=frozenset(),
            ),
            base,
        )

    async def test_merge_shell_tool_settings_pairs_backend_aliases(
        self,
    ) -> None:
        base = ShellToolSettings(
            backend="container",
            container=ContainerProfileSelection(required=True),
        )
        override = ShellToolSettings(backend="local")

        merged = _merge_shell_tool_settings(
            base,
            override,
            explicit_fields=frozenset({"backend"}),
        )

        assert merged is not None
        self.assertEqual(merged.backend, "local")
        self.assertEqual(merged.execution_mode, "local")
        self.assertIsNone(merged.container)

    async def test_merge_shell_tool_settings_pairs_execution_mode_aliases(
        self,
    ) -> None:
        base = ShellToolSettings(
            execution_mode="sandbox",
            sandbox=SandboxProfileSelection(required=True),
        )
        override = ShellToolSettings(execution_mode="local")

        merged = _merge_shell_tool_settings(
            base,
            override,
            explicit_fields=frozenset({"execution_mode"}),
        )

        assert merged is not None
        self.assertEqual(merged.backend, "local")
        self.assertEqual(merged.execution_mode, "local")
        self.assertIsNone(merged.sandbox)

    async def test_shell_runtime_settings_preserve_container_isolation(
        self,
    ) -> None:
        container_runtime = trusted_container_runtime_from_mapping(
            {
                "backend": "docker",
                "default_profile": "workspace-readonly",
                "profiles": {
                    "workspace-readonly": {
                        "image": "ghcr.io/example/tools@sha256:" + "4" * 64,
                        "workspace_root": ".",
                    }
                },
            },
            source=trusted_container_source("sdk"),
        )
        isolation_runtime = IsolationToolRuntimeSettings(
            effective_settings=IsolationEffectiveSettings(
                mode=IsolationMode.CONTAINER,
                source=trusted_isolation_source("sdk"),
                container=container_runtime.effective_settings,
            )
        )

        resolved_container, resolved_isolation = _shell_tool_runtime_settings(
            ShellToolSettings(execution_mode="container"),
            container_runtime,
            isolation_runtime,
        )

        self.assertIsNone(resolved_container)
        self.assertIs(resolved_isolation, isolation_runtime)

        resolved_container, resolved_isolation = _shell_tool_runtime_settings(
            ShellToolSettings(execution_mode="container"),
            container_runtime,
            None,
        )

        self.assertIs(resolved_container, container_runtime)
        self.assertIsNone(resolved_isolation)

        with self.assertRaisesRegex(
            AssertionError,
            "tool.shell backend sandbox requires tool.sandbox settings",
        ):
            _shell_tool_runtime_settings(
                ShellToolSettings(execution_mode="sandbox"),
                None,
                isolation_runtime,
            )

    async def test_empty_shell_section_enables_default_shell_settings(self):
        with NamedTemporaryFile("w+", suffix=".toml") as tmp:
            tmp.write(_minimal_agent_toml() + "\n[tool.shell]\n")
            tmp.flush()
            stack = AsyncExitStack()
            loader = OrchestratorLoader(
                hub=MagicMock(spec=HuggingfaceHub),
                logger=MagicMock(spec=Logger),
                participant_id=uuid4(),
                stack=stack,
            )

            with patch.object(
                loader,
                "from_settings",
                new=AsyncMock(return_value="orch"),
            ) as from_settings:
                result = await loader.from_file(tmp.name, agent_id=uuid4())

            self.assertEqual(result, "orch")
            tool_settings = from_settings.call_args.kwargs["tool_settings"]
            self.assertIsInstance(tool_settings.shell, ShellToolSettings)
            self.assertFalse(tool_settings.shell.allow_media_tools)
            self.assertFalse(tool_settings.shell.allow_pipelines)
            await stack.aclose()

    async def test_shell_pipeline_toml_builds_shell_settings(self):
        with NamedTemporaryFile("w+", suffix=".toml") as tmp:
            tmp.write(
                _minimal_agent_toml()
                + '\n[tool]\nenable = ["shell.pipeline"]\n'
                + "\n[tool.shell]\n"
                + "allow_pipelines = true\n"
                + "max_pipeline_stages = 3\n"
                + "max_pipeline_bytes = 1024\n"
                + "max_intermediate_bytes = 512\n"
            )
            tmp.flush()
            stack = AsyncExitStack()
            loader = OrchestratorLoader(
                hub=MagicMock(spec=HuggingfaceHub),
                logger=MagicMock(spec=Logger),
                participant_id=uuid4(),
                stack=stack,
            )

            with patch.object(
                loader,
                "from_settings",
                new=AsyncMock(return_value="orch"),
            ) as from_settings:
                result = await loader.from_file(tmp.name, agent_id=uuid4())

            self.assertEqual(result, "orch")
            settings = from_settings.call_args.args[0]
            self.assertEqual(settings.tools, ["shell.pipeline"])
            tool_settings = from_settings.call_args.kwargs["tool_settings"]
            self.assertIsInstance(tool_settings.shell, ShellToolSettings)
            self.assertTrue(tool_settings.shell.allow_pipelines)
            self.assertEqual(tool_settings.shell.max_pipeline_stages, 3)
            self.assertEqual(tool_settings.shell.max_pipeline_bytes, 1024)
            self.assertEqual(tool_settings.shell.max_intermediate_bytes, 512)
            await stack.aclose()

    async def test_shell_git_toml_builds_read_only_settings(self):
        root = Path(__file__).resolve().parents[2]
        example_path = root / "docs" / "examples" / "agent_shell_git.toml"
        with self.subTest(example=example_path.name):
            stack = AsyncExitStack()
            loader = OrchestratorLoader(
                hub=MagicMock(spec=HuggingfaceHub),
                logger=MagicMock(spec=Logger),
                participant_id=uuid4(),
                stack=stack,
            )

            with patch.object(
                loader,
                "from_settings",
                new=AsyncMock(return_value="orch"),
            ) as from_settings:
                result = await loader.from_file(
                    str(example_path),
                    agent_id=uuid4(),
                )

            self.assertEqual(result, "orch")
            settings = from_settings.call_args.args[0]
            self.assertEqual(
                settings.tools,
                ["shell.git_status", "shell.git_diff", "shell.git_log"],
            )
            tool_settings = from_settings.call_args.kwargs["tool_settings"]
            self.assertIsInstance(tool_settings.shell, ShellToolSettings)
            git_settings = tool_settings.shell.git
            self.assertIsInstance(git_settings, ShellGitToolSettings)
            assert isinstance(git_settings, ShellGitToolSettings)
            self.assertEqual(git_settings.capabilities, ("read",))
            self.assertEqual(
                git_settings.allowed_commands,
                ("status", "diff", "log"),
            )
            self.assertEqual(git_settings.default_timeout_seconds, 5.0)
            self.assertEqual(git_settings.max_timeout_seconds, 20.0)
            self.assertEqual(git_settings.max_diff_bytes, 131072)
            self.assertEqual(git_settings.max_log_count, 25)
            self.assertEqual(git_settings.max_pathspecs, 16)
            self.assertFalse(git_settings.allow_optional_locks)
            self.assertFalse(git_settings.allow_submodules)
            await stack.aclose()

    async def test_shell_git_toml_rejects_reserved_true_setting(self):
        with NamedTemporaryFile("w+", suffix=".toml") as tmp:
            tmp.write(
                _minimal_agent_toml()
                + "\n[tool.shell.git]\n"
                + "allow_textconv = true\n"
            )
            tmp.flush()
            stack = AsyncExitStack()
            loader = OrchestratorLoader(
                hub=MagicMock(spec=HuggingfaceHub),
                logger=MagicMock(spec=Logger),
                participant_id=uuid4(),
                stack=stack,
            )

            try:
                with self.assertRaisesRegex(
                    AssertionError,
                    "git.allow_textconv is reserved and must be false",
                ):
                    await loader.from_file(tmp.name, agent_id=uuid4())
            finally:
                await stack.aclose()

    async def test_merge_shell_tool_settings_applies_explicit_git_override(
        self,
    ) -> None:
        base = ShellToolSettings(
            git=ShellGitToolSettings(
                workspace_root="/workspace",
                cwd="repo",
                max_log_count=25,
                max_diff_bytes=4096,
            )
        )
        override = ShellToolSettings(
            git=ShellGitToolSettings(max_log_count=10)
        )

        merged = _merge_shell_tool_settings(
            base,
            override,
            explicit_fields=frozenset({"git.max_log_count"}),
        )

        assert merged is not None
        git_settings = merged.git
        self.assertIsInstance(git_settings, ShellGitToolSettings)
        assert isinstance(git_settings, ShellGitToolSettings)
        self.assertEqual(git_settings.workspace_root, "/workspace")
        self.assertEqual(git_settings.cwd, "repo")
        self.assertEqual(git_settings.max_log_count, 10)
        self.assertEqual(git_settings.max_diff_bytes, 4096)

    async def test_merge_shell_tool_settings_replaces_explicit_git_block(
        self,
    ) -> None:
        base = ShellToolSettings(
            git=ShellGitToolSettings(
                workspace_root="/workspace",
                cwd="repo",
                max_log_count=25,
            )
        )
        override_git = ShellGitToolSettings(
            workspace_root="/override",
            cwd="override-repo",
            max_log_count=10,
        )
        override = ShellToolSettings(git=override_git)

        merged = _merge_shell_tool_settings(
            base,
            override,
            explicit_fields=frozenset({"git"}),
        )

        assert merged is not None
        self.assertIs(merged.git, override_git)

    async def test_merge_shell_tool_settings_keeps_explicit_git_timeout_pair(
        self,
    ) -> None:
        base = ShellToolSettings(
            git=ShellGitToolSettings(
                default_timeout_seconds=10.0,
                max_timeout_seconds=20.0,
            )
        )
        override = ShellToolSettings(
            git=ShellGitToolSettings(
                default_timeout_seconds=3.0,
                max_timeout_seconds=4.0,
            )
        )

        merged = _merge_shell_tool_settings(
            base,
            override,
            explicit_fields=frozenset(
                {
                    "git.default_timeout_seconds",
                    "git.max_timeout_seconds",
                }
            ),
        )

        assert merged is not None
        git_settings = merged.git
        self.assertIsInstance(git_settings, ShellGitToolSettings)
        assert isinstance(git_settings, ShellGitToolSettings)
        self.assertEqual(git_settings.default_timeout_seconds, 3.0)
        self.assertEqual(git_settings.max_timeout_seconds, 4.0)

    async def _shell_git_settings_from_partial_cli_timeout_merge(
        self,
        *,
        toml_default_timeout_seconds: float,
        toml_max_timeout_seconds: float,
        cli_default_timeout_seconds: float | None = None,
        cli_max_timeout_seconds: float | None = None,
    ) -> ShellGitToolSettings:
        assert (
            cli_default_timeout_seconds is not None
            or cli_max_timeout_seconds is not None
        )
        args = Namespace()
        if cli_default_timeout_seconds is not None:
            args.tool_shell_git_default_timeout_seconds = (
                cli_default_timeout_seconds
            )
        if cli_max_timeout_seconds is not None:
            args.tool_shell_git_max_timeout_seconds = cli_max_timeout_seconds

        shell_override = agent_cmds.get_tool_settings(
            args,
            prefix="shell",
            settings_cls=ShellToolSettings,
        )
        assert isinstance(shell_override, ShellToolSettings)
        explicit_fields = (
            agent_cmds._tool_settings_explicit_fields_from_mapping(
                args,
                prefix="shell",
                settings_cls=ShellToolSettings,
            )
        )

        stack = AsyncExitStack()
        try:
            loader = OrchestratorLoader(
                hub=MagicMock(spec=HuggingfaceHub),
                logger=MagicMock(spec=Logger),
                participant_id=uuid4(),
                stack=stack,
            )
            with NamedTemporaryFile("w+", suffix=".toml") as tmp:
                tmp.write(
                    _minimal_agent_toml()
                    + "\n[tool.shell.git]\n"
                    + "default_timeout_seconds = "
                    f"{toml_default_timeout_seconds}\n"
                    + f"max_timeout_seconds = {toml_max_timeout_seconds}\n"
                )
                tmp.flush()

                with patch.object(
                    loader,
                    "from_settings",
                    new=AsyncMock(return_value="orch"),
                ) as from_settings:
                    result = await loader.from_file(
                        tmp.name,
                        agent_id=uuid4(),
                        tool_settings=ToolSettingsContext(
                            shell=shell_override,
                            shell_explicit_fields=explicit_fields,
                        ),
                    )

            self.assertEqual(result, "orch")
            tool_settings = from_settings.call_args.kwargs["tool_settings"]
            git_settings = tool_settings.shell.git
            self.assertIsInstance(git_settings, ShellGitToolSettings)
            assert isinstance(git_settings, ShellGitToolSettings)
            return git_settings
        finally:
            await stack.aclose()

    async def test_shell_git_cli_max_timeout_keeps_toml_default(
        self,
    ) -> None:
        git_settings = (
            await self._shell_git_settings_from_partial_cli_timeout_merge(
                toml_default_timeout_seconds=2.0,
                toml_max_timeout_seconds=4.0,
                cli_max_timeout_seconds=5.0,
            )
        )

        self.assertEqual(git_settings.default_timeout_seconds, 2.0)
        self.assertEqual(git_settings.max_timeout_seconds, 5.0)

    async def test_shell_git_cli_max_timeout_lowers_toml_default(
        self,
    ) -> None:
        git_settings = (
            await self._shell_git_settings_from_partial_cli_timeout_merge(
                toml_default_timeout_seconds=10.0,
                toml_max_timeout_seconds=20.0,
                cli_max_timeout_seconds=5.0,
            )
        )

        self.assertEqual(git_settings.default_timeout_seconds, 5.0)
        self.assertEqual(git_settings.max_timeout_seconds, 5.0)

    async def test_shell_git_cli_default_timeout_raises_toml_max(
        self,
    ) -> None:
        git_settings = (
            await self._shell_git_settings_from_partial_cli_timeout_merge(
                toml_default_timeout_seconds=10.0,
                toml_max_timeout_seconds=20.0,
                cli_default_timeout_seconds=120.0,
            )
        )

        self.assertEqual(git_settings.default_timeout_seconds, 120.0)
        self.assertEqual(git_settings.max_timeout_seconds, 120.0)

    async def test_docs_shell_pipeline_agent_example_builds_settings(self):
        example_path = (
            Path(__file__).resolve().parents[2]
            / "docs"
            / "examples"
            / "agent_shell_pipeline.toml"
        )
        stack = AsyncExitStack()
        loader = OrchestratorLoader(
            hub=MagicMock(spec=HuggingfaceHub),
            logger=MagicMock(spec=Logger),
            participant_id=uuid4(),
            stack=stack,
        )

        with patch.object(
            loader,
            "from_settings",
            new=AsyncMock(return_value="orch"),
        ) as from_settings:
            result = await loader.from_file(
                str(example_path), agent_id=uuid4()
            )

        self.assertEqual(result, "orch")
        settings = from_settings.call_args.args[0]
        self.assertEqual(settings.tools, ["shell.pipeline"])
        tool_settings = from_settings.call_args.kwargs["tool_settings"]
        self.assertIsInstance(tool_settings.shell, ShellToolSettings)
        self.assertTrue(tool_settings.shell.allow_pipelines)
        self.assertEqual(tool_settings.shell.max_pipeline_stages, 3)
        self.assertEqual(tool_settings.shell.max_pipeline_bytes, 1048576)
        self.assertEqual(tool_settings.shell.max_intermediate_bytes, 262144)
        self.assertEqual(
            tool_settings.shell.allowed_commands,
            ("rg", "head", "wc"),
        )
        await stack.aclose()

    async def test_docs_shell_git_agent_example_builds_settings(self):
        example_path = (
            Path(__file__).resolve().parents[2]
            / "docs"
            / "examples"
            / "agent_shell_git.toml"
        )
        stack = AsyncExitStack()
        loader = OrchestratorLoader(
            hub=MagicMock(spec=HuggingfaceHub),
            logger=MagicMock(spec=Logger),
            participant_id=uuid4(),
            stack=stack,
        )

        with patch.object(
            loader,
            "from_settings",
            new=AsyncMock(return_value="orch"),
        ) as from_settings:
            result = await loader.from_file(
                str(example_path), agent_id=uuid4()
            )

        self.assertEqual(result, "orch")
        settings = from_settings.call_args.args[0]
        self.assertEqual(
            settings.tools,
            ["shell.git_status", "shell.git_diff", "shell.git_log"],
        )
        tool_settings = from_settings.call_args.kwargs["tool_settings"]
        self.assertIsInstance(tool_settings.shell, ShellToolSettings)
        git_settings = tool_settings.shell.git
        self.assertIsInstance(git_settings, ShellGitToolSettings)
        assert isinstance(git_settings, ShellGitToolSettings)
        self.assertEqual(git_settings.capabilities, ("read",))
        self.assertEqual(
            git_settings.allowed_commands,
            ("status", "diff", "log"),
        )
        self.assertEqual(git_settings.max_log_count, 25)
        self.assertEqual(git_settings.max_pathspecs, 16)
        self.assertTrue(git_settings.redact_author_emails)
        await stack.aclose()

    async def test_docs_shell_git_agent_example_executes_temp_repo_status(
        self,
    ) -> None:
        git_binary = which("git")
        if git_binary is None:
            self.skipTest("git executable is required for shell Git examples")

        example_path = (
            Path(__file__).resolve().parents[2]
            / "docs"
            / "examples"
            / "agent_shell_git.toml"
        )
        with TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            repo = _write_agent_shell_git_example_repo(
                workspace,
                git_binary,
            )
            kwargs = await _from_file_tool_manager_kwargs(
                example_path.read_text(encoding="utf-8"),
                tool_settings=ToolSettingsContext(
                    shell=ShellToolSettings(
                        executable_search_paths=(
                            str(Path(git_binary).parent),
                        ),
                        git=ShellGitToolSettings(
                            workspace_root=str(workspace),
                            cwd=repo.name,
                        ),
                    ),
                    shell_explicit_fields=frozenset(
                        {
                            "executable_search_paths",
                            "git.workspace_root",
                            "git.cwd",
                        }
                    ),
                ),
            )
            manager = _shell_only_manager(kwargs)
            async with manager:
                outcome = await manager.execute_call(
                    ToolCall(
                        id="call-1",
                        name="shell.git_status",
                        arguments={
                            "mode": "porcelain_v2",
                            "include_branch": True,
                        },
                    ),
                    context=ToolCallContext(),
                )

        self.assertIsInstance(outcome, ToolCallResult)
        assert isinstance(outcome, ToolCallResult)
        result_text = str(outcome.result)
        self.assertIn("status: success", result_text)
        self.assertIn("README.md", result_text)

    async def test_from_file_pipeline_opt_in_exposes_schema_and_filter(self):
        kwargs = await _from_file_tool_manager_kwargs(
            _minimal_agent_toml()
            + '\n[tool]\nenable = ["shell.pipeline"]\n'
            + "\n[tool.shell]\n"
            + "allow_pipelines = true\n"
            + "max_pipeline_stages = 3\n"
            + "max_pipeline_bytes = 1024\n"
            + "max_intermediate_bytes = 512\n"
        )

        self.assertEqual(_shell_namespaces(kwargs), ["shell"])
        self.assertEqual(kwargs["enable_tools"], ["shell.pipeline"])
        self.assertIsNotNone(kwargs["settings"].filters)

        manager = _shell_only_manager(kwargs)
        self.assertEqual(
            [descriptor.name for descriptor in manager.list_tools()],
            ["shell.pipeline"],
        )
        descriptor = manager.describe_tool("shell.pipeline")
        assert descriptor is not None
        assert descriptor.parameter_schema is not None
        properties = descriptor.parameter_schema["properties"]
        for forbidden in (
            "allow_pipelines",
            "max_pipeline_stages",
            "max_pipeline_bytes",
            "backend",
            "execution_mode",
            "container",
            "sandbox",
        ):
            self.assertNotIn(forbidden, properties)
        self.assertIn("max_intermediate_bytes", properties)

    async def test_from_file_pipeline_selection_uses_cli_allow_override(self):
        kwargs = await _from_file_tool_manager_kwargs(
            _minimal_agent_toml() + '\n[tool]\nenable = ["shell.pipeline"]\n',
            tool_settings=ToolSettingsContext(
                shell=ShellToolSettings(allow_pipelines=True),
                shell_explicit_fields=frozenset({"allow_pipelines"}),
            ),
        )

        self.assertEqual(_shell_namespaces(kwargs), ["shell"])
        self.assertEqual(kwargs["enable_tools"], ["shell.pipeline"])
        self.assertIsNotNone(kwargs["settings"].filters)

        manager = _shell_only_manager(kwargs)
        self.assertEqual(
            [descriptor.name for descriptor in manager.list_tools()],
            ["shell.pipeline"],
        )

    async def test_from_file_pipeline_selection_is_default_denied(self):
        kwargs = await _from_file_tool_manager_kwargs(
            _minimal_agent_toml() + '\n[tool]\nenable = ["shell.pipeline"]\n'
        )

        self.assertEqual(_shell_namespaces(kwargs), ["shell"])
        self.assertEqual(kwargs["enable_tools"], ["shell.pipeline"])
        self.assertIsNone(kwargs["settings"].filters)

        manager = _shell_only_manager(kwargs)
        self.assertEqual(manager.list_tools(), [])
        self.assertTrue(
            ModelCapabilityCatalog.create(
                manager.export_model_capability_seed()
            )
            .project()
            .is_empty
        )
        self.assertIsNone(manager.describe_tool("shell.pipeline"))
        resolution = manager.resolve_tool_name("shell.pipeline")
        self.assertIs(resolution.status, ToolNameResolutionStatus.DISABLED)
        self.assertIs(
            resolution.diagnostic_code,
            ToolCallDiagnosticCode.DISABLED_TOOL,
        )

    async def test_from_file_pipeline_allow_without_enable_is_default_denied(
        self,
    ):
        kwargs = await _from_file_tool_manager_kwargs(
            _minimal_agent_toml()
            + "\n[tool.shell]\n"
            + "allow_pipelines = true\n"
        )

        self.assertEqual(_shell_namespaces(kwargs), ["shell"])
        self.assertIsNone(kwargs["enable_tools"])

        manager = _shell_only_manager(kwargs)
        tool_names = [descriptor.name for descriptor in manager.list_tools()]
        self.assertNotIn("shell.pipeline", tool_names)

    async def test_from_file_forwards_server_event_manager_mode(self):
        with NamedTemporaryFile("w+", suffix=".toml") as tmp:
            tmp.write(_minimal_agent_toml())
            tmp.flush()
            stack = AsyncExitStack()
            loader = OrchestratorLoader(
                hub=MagicMock(spec=HuggingfaceHub),
                logger=MagicMock(spec=Logger),
                participant_id=uuid4(),
                stack=stack,
            )

            with patch.object(
                loader,
                "from_settings",
                new=AsyncMock(return_value="orch"),
            ) as from_settings:
                result = await loader.from_file(
                    tmp.name,
                    agent_id=uuid4(),
                    event_manager_mode=EventManagerMode.SERVER,
                )

            self.assertEqual(result, "orch")
            self.assertIs(
                from_settings.call_args.kwargs["event_manager_mode"],
                EventManagerMode.SERVER,
            )
            await stack.aclose()

    async def test_from_file_forwards_server_mode_with_recovery_formats(
        self,
    ):
        with NamedTemporaryFile("w+", suffix=".toml") as tmp:
            tmp.write(
                _minimal_agent_toml()
                + '\n[tool]\nrecovery_formats = ["fenced"]\n'
            )
            tmp.flush()
            stack = AsyncExitStack()
            loader = OrchestratorLoader(
                hub=MagicMock(spec=HuggingfaceHub),
                logger=MagicMock(spec=Logger),
                participant_id=uuid4(),
                stack=stack,
            )

            with patch.object(
                loader,
                "from_settings",
                new=AsyncMock(return_value="orch"),
            ) as from_settings:
                result = await loader.from_file(
                    tmp.name,
                    agent_id=uuid4(),
                    event_manager_mode=EventManagerMode.SERVER,
                )

            self.assertEqual(result, "orch")
            self.assertIs(
                from_settings.call_args.kwargs["event_manager_mode"],
                EventManagerMode.SERVER,
            )
            self.assertEqual(
                from_settings.call_args.kwargs["tool_recovery_formats"],
                [ToolCallRecoveryFormat.FENCED],
            )
            await stack.aclose()

    async def test_shell_section_builds_settings_from_toml(self):
        with NamedTemporaryFile("w+", suffix=".toml") as tmp:
            tmp.write(
                _minimal_agent_toml()
                + "\n[tool.shell]\n"
                + 'workspace_root = "/tmp"\n'
                + 'cwd = "fixtures"\n'
                + 'materialized_input_files_dir = "agent-input-files"\n'
                + "input_file_manifest_enabled = false\n"
                + 'input_file_manifest_message = "Use attached paths:"\n'
                + 'input_file_manifest_path_message = "Pass them to tools."\n'
                + "max_head_lines = 12\n"
                + "allow_hidden = true\n"
                + "allow_process_tools = true\n"
                + 'allowed_commands = ["head", "cat"]\n'
            )
            tmp.flush()
            stack = AsyncExitStack()
            loader = OrchestratorLoader(
                hub=MagicMock(spec=HuggingfaceHub),
                logger=MagicMock(spec=Logger),
                participant_id=uuid4(),
                stack=stack,
            )

            with patch.object(
                loader,
                "from_settings",
                new=AsyncMock(return_value="orch"),
            ) as from_settings:
                result = await loader.from_file(tmp.name, agent_id=uuid4())

            self.assertEqual(result, "orch")
            tool_settings = from_settings.call_args.kwargs["tool_settings"]
            self.assertEqual(tool_settings.shell.workspace_root, "/tmp")
            self.assertEqual(tool_settings.shell.cwd, "fixtures")
            self.assertEqual(
                tool_settings.shell.materialized_input_files_dir,
                "agent-input-files",
            )
            self.assertFalse(tool_settings.shell.input_file_manifest_enabled)
            self.assertEqual(
                tool_settings.shell.input_file_manifest_message,
                "Use attached paths:",
            )
            self.assertEqual(
                tool_settings.shell.input_file_manifest_path_message,
                "Pass them to tools.",
            )
            self.assertEqual(tool_settings.shell.max_head_lines, 12)
            self.assertTrue(tool_settings.shell.allow_hidden)
            self.assertTrue(tool_settings.shell.allow_process_tools)
            self.assertEqual(
                tool_settings.shell.allowed_commands, ("head", "cat")
            )
            await stack.aclose()

    async def test_shell_tool_settings_merges_cli_explicit_fields_with_toml(
        self,
    ):
        with NamedTemporaryFile("w+", suffix=".toml") as tmp:
            tmp.write(
                _minimal_agent_toml()
                + "\n[tool.shell]\n"
                + 'workspace_root = "/workspace/project"\n'
                + 'materialized_input_files_dir = "agent-input-files"\n'
                + "max_head_lines = 12\n"
            )
            tmp.flush()
            stack = AsyncExitStack()
            loader = OrchestratorLoader(
                hub=MagicMock(spec=HuggingfaceHub),
                logger=MagicMock(spec=Logger),
                participant_id=uuid4(),
                stack=stack,
            )
            override = ShellToolSettings(
                allow_media_tools=True,
                max_head_lines=ShellToolSettings().max_head_lines,
            )

            with patch.object(
                loader,
                "from_settings",
                new=AsyncMock(return_value="orch"),
            ) as from_settings:
                result = await loader.from_file(
                    tmp.name,
                    agent_id=uuid4(),
                    tool_settings=ToolSettingsContext(
                        shell=override,
                        shell_explicit_fields=frozenset(
                            {
                                "allow_media_tools",
                                "max_head_lines",
                            }
                        ),
                    ),
                )

            self.assertEqual(result, "orch")
            tool_settings = from_settings.call_args.kwargs["tool_settings"]
            self.assertIsNot(tool_settings.shell, override)
            self.assertEqual(
                tool_settings.shell.workspace_root,
                "/workspace/project",
            )
            self.assertEqual(
                tool_settings.shell.materialized_input_files_dir,
                "agent-input-files",
            )
            self.assertTrue(tool_settings.shell.allow_media_tools)
            self.assertEqual(
                tool_settings.shell.max_head_lines,
                ShellToolSettings().max_head_lines,
            )
            await stack.aclose()

    async def test_shell_pipeline_settings_merge_cli_explicit_fields(
        self,
    ):
        with NamedTemporaryFile("w+", suffix=".toml") as tmp:
            tmp.write(
                _minimal_agent_toml()
                + '\n[tool]\nenable = ["shell.pipeline"]\n'
                + "\n[tool.shell]\n"
                + 'workspace_root = "/workspace/project"\n'
                + "allow_pipelines = true\n"
                + "max_pipeline_stages = 8\n"
                + "max_pipeline_bytes = 2048\n"
                + "max_intermediate_bytes = 1024\n"
            )
            tmp.flush()
            stack = AsyncExitStack()
            loader = OrchestratorLoader(
                hub=MagicMock(spec=HuggingfaceHub),
                logger=MagicMock(spec=Logger),
                participant_id=uuid4(),
                stack=stack,
            )
            override = ShellToolSettings(
                max_pipeline_stages=3,
                max_pipeline_bytes=512,
            )

            with patch.object(
                loader,
                "from_settings",
                new=AsyncMock(return_value="orch"),
            ) as from_settings:
                result = await loader.from_file(
                    tmp.name,
                    agent_id=uuid4(),
                    tool_settings=ToolSettingsContext(
                        shell=override,
                        shell_explicit_fields=frozenset(
                            {
                                "max_pipeline_stages",
                                "max_pipeline_bytes",
                            }
                        ),
                    ),
                )

            self.assertEqual(result, "orch")
            tool_settings = from_settings.call_args.kwargs["tool_settings"]
            self.assertIsNot(tool_settings.shell, override)
            self.assertEqual(
                tool_settings.shell.workspace_root,
                "/workspace/project",
            )
            self.assertTrue(tool_settings.shell.allow_pipelines)
            self.assertEqual(tool_settings.shell.max_pipeline_stages, 3)
            self.assertEqual(tool_settings.shell.max_pipeline_bytes, 512)
            self.assertEqual(tool_settings.shell.max_intermediate_bytes, 1024)
            await stack.aclose()

    async def test_container_toml_sections_load_trusted_settings(self) -> None:
        image = "ghcr.io/example/tools@sha256:" + "4" * 64
        with NamedTemporaryFile("w+", suffix=".toml") as tmp:
            tmp.write(_minimal_agent_toml() + f"""
[tool]
enable = ["shell.cat"]

[tool.container]
backend = "docker"
default_profile = "workspace-readonly"

[tool.container.profiles.workspace-readonly]
image = "{image}"
workspace_root = "."
network = "none"

[tool.shell]
backend = "container"

[tool.shell.container]
profile = "workspace-readonly"

""")
            tmp.flush()
            stack = AsyncExitStack()
            loader = OrchestratorLoader(
                hub=MagicMock(spec=HuggingfaceHub),
                logger=MagicMock(spec=Logger),
                participant_id=uuid4(),
                stack=stack,
            )

            with patch.object(
                loader,
                "from_settings",
                new=AsyncMock(return_value="orch"),
            ) as from_settings:
                result = await loader.from_file(tmp.name, agent_id=uuid4())

            self.assertEqual(result, "orch")
            tool_settings = from_settings.call_args.kwargs["tool_settings"]
            self.assertEqual(tool_settings.shell.backend, "container")
            assert tool_settings.shell.container is not None
            self.assertTrue(tool_settings.shell.container.required)
            self.assertIsNotNone(tool_settings.container)
            assert tool_settings.container is not None
            effective = tool_settings.container.effective_settings
            assert effective is not None
            self.assertEqual(effective.profile_name, "workspace-readonly")
            self.assertTrue(effective.required)
            self.assertIsNone(tool_settings.extra)
            await stack.aclose()

    async def test_cli_backend_local_overrides_toml_container_mode(
        self,
    ) -> None:
        image = "ghcr.io/example/tools@sha256:" + "4" * 64
        with NamedTemporaryFile("w+", suffix=".toml") as tmp:
            tmp.write(_minimal_agent_toml() + f"""
[tool]
enable = ["shell.cat"]

[tool.container]
backend = "docker"
default_profile = "workspace-readonly"

[tool.container.profiles.workspace-readonly]
image = "{image}"
workspace_root = "."
network = "none"

[tool.shell]
backend = "container"

[tool.shell.container]
profile = "workspace-readonly"

""")
            tmp.flush()
            stack = AsyncExitStack()
            loader = OrchestratorLoader(
                hub=MagicMock(spec=HuggingfaceHub),
                logger=MagicMock(spec=Logger),
                participant_id=uuid4(),
                stack=stack,
            )

            with patch.object(
                loader,
                "from_settings",
                new=AsyncMock(return_value="orch"),
            ) as from_settings:
                result = await loader.from_file(
                    tmp.name,
                    agent_id=uuid4(),
                    tool_settings=ToolSettingsContext(
                        shell=ShellToolSettings(backend="local"),
                        shell_explicit_fields=frozenset({"backend"}),
                    ),
                )

            self.assertEqual(result, "orch")
            tool_settings = from_settings.call_args.kwargs["tool_settings"]
            self.assertEqual(tool_settings.shell.backend, "local")
            self.assertEqual(tool_settings.shell.execution_mode, "local")
            self.assertIsNone(tool_settings.shell.container)
            self.assertIsNotNone(tool_settings.container)
            await stack.aclose()

    async def test_shell_local_override_filters_container_runtime_for_shell(
        self,
    ) -> None:
        image = "ghcr.io/example/tools@sha256:" + "4" * 64
        config = _minimal_agent_toml() + f"""
[tool]
enable = ["shell.cat"]

[tool.container]
backend = "docker"
default_profile = "workspace-readonly"

[tool.container.profiles.workspace-readonly]
image = "{image}"
workspace_root = "."
network = "none"

[tool.shell]
backend = "container"

[tool.shell.container]
profile = "workspace-readonly"

"""

        code_kwargs, shell_kwargs = await _from_file_code_shell_toolset_kwargs(
            config,
            tool_settings=ToolSettingsContext(
                shell=ShellToolSettings(backend="local"),
                shell_explicit_fields=frozenset({"backend"}),
            ),
        )

        self.assertIsNotNone(code_kwargs["container_runtime"])
        self.assertIsNone(shell_kwargs["container_runtime"])
        self.assertNotIn("isolation_runtime", shell_kwargs)
        self.assertEqual(shell_kwargs["settings"].backend, "local")

    async def test_sandbox_toml_sections_load_trusted_settings(self) -> None:
        with NamedTemporaryFile("w+", suffix=".toml") as tmp:
            tmp.write(_minimal_agent_toml() + """
[tool]
enable = ["shell.cat"]

[tool.sandbox]
backend = "seatbelt"
default_profile = "host-tools"

[tool.sandbox.profiles.host-tools]
trusted_executables = ["/bin/cat"]
read_roots = ["/tmp"]
scratch_roots = ["/tmp"]
output_roots = ["/tmp"]
child_processes = "deny"
inherited_fds = "stdio"

[tool.sandbox.profiles.host-tools.network]
mode = "none"

[tool.sandbox.profiles.host-tools.resources]
timeout_seconds = 10
pids = 16

[tool.shell]
backend = "sandbox"

[tool.shell.sandbox]
profile = "host-tools"

""")
            tmp.flush()
            stack = AsyncExitStack()
            loader = OrchestratorLoader(
                hub=MagicMock(spec=HuggingfaceHub),
                logger=MagicMock(spec=Logger),
                participant_id=uuid4(),
                stack=stack,
            )

            with patch.object(
                loader,
                "from_settings",
                new=AsyncMock(return_value="orch"),
            ) as from_settings:
                result = await loader.from_file(tmp.name, agent_id=uuid4())

            self.assertEqual(result, "orch")
            tool_settings = from_settings.call_args.kwargs["tool_settings"]
            self.assertEqual(tool_settings.shell.backend, "sandbox")
            assert tool_settings.shell.sandbox is not None
            self.assertTrue(tool_settings.shell.sandbox.required)
            self.assertIsNotNone(tool_settings.isolation)
            assert tool_settings.isolation is not None
            effective = tool_settings.isolation.effective_settings
            self.assertEqual(effective.mode, "sandbox")
            assert effective.sandbox is not None
            self.assertEqual(effective.sandbox.profile_name, "host-tools")
            self.assertTrue(effective.sandbox.required)
            self.assertIsNone(tool_settings.extra)
            await stack.aclose()

    async def test_cli_execution_mode_local_overrides_toml_sandbox_mode(
        self,
    ) -> None:
        with NamedTemporaryFile("w+", suffix=".toml") as tmp:
            tmp.write(_minimal_agent_toml() + """
[tool]
enable = ["shell.cat"]

[tool.sandbox]
backend = "seatbelt"
default_profile = "host-tools"

[tool.sandbox.profiles.host-tools]
trusted_executables = ["/bin/cat"]
read_roots = ["/tmp"]
scratch_roots = ["/tmp"]
output_roots = ["/tmp"]
child_processes = "deny"
inherited_fds = "stdio"

[tool.shell]
backend = "sandbox"

[tool.shell.sandbox]
profile = "host-tools"

""")
            tmp.flush()
            stack = AsyncExitStack()
            loader = OrchestratorLoader(
                hub=MagicMock(spec=HuggingfaceHub),
                logger=MagicMock(spec=Logger),
                participant_id=uuid4(),
                stack=stack,
            )

            with patch.object(
                loader,
                "from_settings",
                new=AsyncMock(return_value="orch"),
            ) as from_settings:
                result = await loader.from_file(
                    tmp.name,
                    agent_id=uuid4(),
                    tool_settings=ToolSettingsContext(
                        shell=ShellToolSettings(execution_mode="local"),
                        shell_explicit_fields=frozenset({"execution_mode"}),
                    ),
                )

            self.assertEqual(result, "orch")
            tool_settings = from_settings.call_args.kwargs["tool_settings"]
            self.assertEqual(tool_settings.shell.backend, "local")
            self.assertEqual(tool_settings.shell.execution_mode, "local")
            self.assertIsNone(tool_settings.shell.sandbox)
            self.assertIsNotNone(tool_settings.isolation)
            await stack.aclose()

    async def test_shell_local_override_filters_isolation_runtime_for_shell(
        self,
    ) -> None:
        config = _minimal_agent_toml() + """
[tool]
enable = ["shell.cat"]

[tool.sandbox]
backend = "seatbelt"
default_profile = "host-tools"

[tool.sandbox.profiles.host-tools]
trusted_executables = ["/bin/cat"]
read_roots = ["/tmp"]
scratch_roots = ["/tmp"]
output_roots = ["/tmp"]
child_processes = "deny"
inherited_fds = "stdio"

[tool.shell]
backend = "sandbox"

[tool.shell.sandbox]
profile = "host-tools"

"""

        (
            _code_kwargs,
            shell_kwargs,
        ) = await _from_file_code_shell_toolset_kwargs(
            config,
            tool_settings=ToolSettingsContext(
                shell=ShellToolSettings(execution_mode="local"),
                shell_explicit_fields=frozenset({"execution_mode"}),
            ),
        )

        self.assertNotIn("isolation_runtime", shell_kwargs)
        self.assertIsNone(shell_kwargs["container_runtime"])
        self.assertEqual(shell_kwargs["settings"].execution_mode, "local")

    async def test_runtime_container_requires_envelope_aware_loader(
        self,
    ) -> None:
        image = "ghcr.io/example/tools@sha256:" + "4" * 64
        OrchestratorLoader.validate_agent_config(
            {
                "agent": {"role": "assistant"},
                "engine": {"uri": "ai://local/model"},
                "tool": {
                    "container": {
                        "backend": "docker",
                        "profiles": {"runtime": {"image": image}},
                    },
                },
                "runtime": {"container": {"profile": "runtime"}},
            }
        )
        self.assertIsNone(
            OrchestratorLoader._agent_runtime_envelope_plan(
                config={
                    "agent": {"role": "assistant"},
                    "engine": {"uri": "ai://local/model"},
                },
                tool_section={},
                source=trusted_container_source(ContainerSurface.AGENT_TOML),
                path="agents/example.toml",
                agent_id=uuid4(),
            )
        )

        with NamedTemporaryFile("w+", suffix=".toml") as tmp:
            tmp.write(_minimal_agent_toml() + f"""
[tool.container]
backend = "docker"
default_profile = "runtime"

[tool.container.profiles.runtime]
image = "{image}"

[runtime.container]
profile = "runtime"
required = true
""")
            tmp.flush()
            stack = AsyncExitStack()
            loader = OrchestratorLoader(
                hub=MagicMock(spec=HuggingfaceHub),
                logger=MagicMock(spec=Logger),
                participant_id=uuid4(),
                stack=stack,
            )

            with self.assertRaisesRegex(
                AssertionError,
                "runtime.container requires an envelope-aware agent loader",
            ):
                await loader.from_file(tmp.name, agent_id=uuid4())
            await stack.aclose()

    def test_validate_agent_config_rejects_unknown_runtime_profile(
        self,
    ) -> None:
        image = "ghcr.io/example/tools@sha256:" + "4" * 64

        with self.assertRaisesRegex(
            AssertionError,
            "selected profile must be allowed",
        ):
            OrchestratorLoader.validate_agent_config(
                {
                    "agent": {"role": "assistant"},
                    "engine": {"uri": "ai://local/model"},
                    "tool": {
                        "container": {
                            "backend": "docker",
                            "profiles": {"runtime": {"image": image}},
                        },
                    },
                    "runtime": {"container": {"profile": "missing"}},
                }
            )

        with self.assertRaisesRegex(
            AssertionError,
            "runtime.container requires tool.container",
        ):
            OrchestratorLoader.validate_agent_config(
                {
                    "agent": {"role": "assistant"},
                    "engine": {"uri": "ai://local/model"},
                    "runtime": {"container": {"profile": "runtime"}},
                }
            )

    async def test_runtime_container_delegates_to_agent_envelope_loader(
        self,
    ) -> None:
        image = "ghcr.io/example/tools@sha256:" + "4" * 64

        class FakeEnvelopedAgent:
            def __init__(self) -> None:
                self.definition_locators = []

            def bind_execution_definition_locator(self, locator) -> None:
                self.definition_locators.append(locator)

        class FakeAgentEnvelopeLoader:
            trusted_runtime_envelope_runner = True

            def __init__(self) -> None:
                self.plan = None
                self.kwargs = None
                self.agent = FakeEnvelopedAgent()

            async def load_agent_runtime_envelope(self, plan, **kwargs):
                self.plan = plan
                self.kwargs = kwargs
                return self.agent

        with NamedTemporaryFile("w+", suffix=".toml") as tmp:
            tmp.write(_minimal_agent_toml() + f"""
[tool.container]
backend = "docker"
default_profile = "runtime"

[tool.container.profiles.runtime]
image = "{image}"

[tool.shell]
backend = "container"

[runtime.container]
profile = "runtime"
readiness_timeout_seconds = 12
""")
            tmp.flush()
            stack = AsyncExitStack()
            envelope_loader = FakeAgentEnvelopeLoader()
            loader = OrchestratorLoader(
                hub=MagicMock(spec=HuggingfaceHub),
                logger=MagicMock(spec=Logger),
                participant_id=uuid4(),
                stack=stack,
                runtime_envelope_loader=envelope_loader,
            )
            agent_id = uuid4()

            result = await loader.from_file(
                tmp.name,
                agent_id=agent_id,
                call_options_override={
                    "reasoning": {
                        "summary": ReasoningSummaryMode.DETAILED,
                    }
                },
                tool_settings=ToolSettingsContext(extra={"caller": "ok"}),
            )

            self.assertIs(result, envelope_loader.agent)
            assert envelope_loader.plan is not None
            assert envelope_loader.kwargs is not None
            self.assertEqual(
                envelope_loader.agent.definition_locators,
                [Path(tmp.name).resolve(strict=True).as_uri()],
            )
            self.assertEqual(
                envelope_loader.kwargs["call_options_override"],
                {
                    "reasoning": {
                        "summary": ReasoningSummaryMode.DETAILED,
                    }
                },
            )
            passed_tool_settings = envelope_loader.kwargs["tool_settings"]
            assert isinstance(passed_tool_settings, ToolSettingsContext)
            self.assertEqual(passed_tool_settings.extra, {"caller": "ok"})
            assert passed_tool_settings.container is not None
            assert (
                passed_tool_settings.container.effective_settings is not None
            )
            self.assertEqual(
                passed_tool_settings.container.effective_settings.profile_name,
                "runtime",
            )
            self.assertEqual(
                envelope_loader.plan.envelope_kind,
                ContainerRuntimeEnvelopeKind.WHOLE_AGENT,
            )
            self.assertEqual(
                envelope_loader.plan.envelope_plan.profile_name,
                "runtime",
            )
            self.assertEqual(
                envelope_loader.plan.envelope_plan.readiness_timeout_seconds,
                12,
            )
            self.assertEqual(
                envelope_loader.plan.run_plan.request.request_id,
                str(agent_id),
            )
            await stack.aclose()

    async def test_runtime_container_omission_preserves_legacy_signature(
        self,
    ) -> None:
        image = "ghcr.io/example/tools@sha256:" + "4" * 64

        class LegacyAgentEnvelopeLoader:
            trusted_runtime_envelope_runner = True

            async def load_agent_runtime_envelope(
                self,
                plan,
                *,
                path,
                agent_id,
                disable_memory,
                uri,
                tool_settings,
                event_manager_mode,
            ):
                return "legacy-enveloped"

        with NamedTemporaryFile("w+", suffix=".toml") as tmp:
            tmp.write(_minimal_agent_toml() + f"""
[tool.container]
backend = "docker"
default_profile = "runtime"

[tool.container.profiles.runtime]
image = "{image}"

[tool.shell]
backend = "container"

[runtime.container]
profile = "runtime"
""")
            tmp.flush()
            stack = AsyncExitStack()
            loader = OrchestratorLoader(
                hub=MagicMock(spec=HuggingfaceHub),
                logger=MagicMock(spec=Logger),
                participant_id=uuid4(),
                stack=stack,
                runtime_envelope_loader=LegacyAgentEnvelopeLoader(),
            )

            for override in (None, {}):
                result = await loader.from_file(
                    tmp.name,
                    agent_id=uuid4(),
                    call_options_override=override,
                )
                self.assertEqual(result, "legacy-enveloped")
            await stack.aclose()

    async def test_runtime_container_loader_must_be_trusted(self) -> None:
        class UntrustedAgentEnvelopeLoader:
            async def load_agent_runtime_envelope(self, plan, **kwargs):
                return "not-used"

        stack = AsyncExitStack()
        with self.assertRaisesRegex(
            AssertionError,
            "runtime envelope loader must be trusted",
        ):
            OrchestratorLoader(
                hub=MagicMock(spec=HuggingfaceHub),
                logger=MagicMock(spec=Logger),
                participant_id=uuid4(),
                stack=stack,
                runtime_envelope_loader=UntrustedAgentEnvelopeLoader(),
            )
        await stack.aclose()

    async def test_runtime_section_without_container_is_accepted(self) -> None:
        with NamedTemporaryFile("w+", suffix=".toml") as tmp:
            tmp.write(_minimal_agent_toml() + """
[runtime]
mode = "local"
""")
            tmp.flush()
            stack = AsyncExitStack()
            loader = OrchestratorLoader(
                hub=MagicMock(spec=HuggingfaceHub),
                logger=MagicMock(spec=Logger),
                participant_id=uuid4(),
                stack=stack,
            )

            with patch.object(
                loader,
                "from_settings",
                new=AsyncMock(return_value="orch"),
            ) as from_settings:
                result = await loader.from_file(tmp.name, agent_id=uuid4())

            self.assertEqual(result, "orch")
            tool_settings = from_settings.call_args.kwargs["tool_settings"]
            self.assertIsNone(tool_settings.extra)
            await stack.aclose()

    async def test_tool_container_registry_only_is_rejected(self) -> None:
        image = "ghcr.io/example/tools@sha256:" + "4" * 64
        with NamedTemporaryFile("w+", suffix=".toml") as tmp:
            tmp.write(_minimal_agent_toml() + f"""
[tool.container]
backend = "docker"

[tool.container.profiles.runtime]
image = "{image}"
""")
            tmp.flush()
            stack = AsyncExitStack()
            loader = OrchestratorLoader(
                hub=MagicMock(spec=HuggingfaceHub),
                logger=MagicMock(spec=Logger),
                participant_id=uuid4(),
                stack=stack,
            )

            with self.assertRaisesRegex(
                AssertionError,
                "tool.container requires tool.shell backend container",
            ):
                await loader.from_file(tmp.name, agent_id=uuid4())
            await stack.aclose()

    def test_validate_agent_config_accepts_container_syntax(self) -> None:
        OrchestratorLoader.validate_agent_config(
            {
                "agent": {"role": "assistant"},
                "engine": {"uri": "ai://local/model"},
                "tool": {
                    "container": {
                        "backend": "docker",
                        "default_profile": "workspace-readonly",
                        "profiles": {
                            "workspace-readonly": {
                                "image": (
                                    "ghcr.io/example/tools@sha256:" + "4" * 64
                                )
                            }
                        },
                    },
                    "shell": {"backend": "container"},
                },
            }
        )

    async def test_container_toml_rejects_unknown_or_invalid_fields(
        self,
    ) -> None:
        image = "ghcr.io/example/tools@sha256:" + "4" * 64
        cases = (
            (
                """
[tool.shell]
backend = "container"

[tool.container]
backend = "docker"
unknown = true
""",
                "Unknown container fields",
            ),
            (
                """
[tool.shell]
backend = "container"

[tool.container]
backend = "none"
""",
                "tool.container backend must be docker or apple-container",
            ),
            (
                f"""
[tool.shell]
backend = "container"

[tool.container]
backend = "docker"

[tool.container.profiles.workspace-readonly]
image = "{image}"
unexpected = true
""",
                "Unknown container fields",
            ),
            (
                """
[tool.shell]
backend = "container"

[tool.shell.container]
required = false
""",
                "requires required=true",
            ),
            (
                """
[tool.shell]
backend = "container"

[tool.shell.container]
profile = "missing"
""",
                "required container profile unavailable",
            ),
            (
                f"""
[tool.container]
backend = "docker"
default_profile = "workspace-readonly"

[tool.container.profiles.workspace-readonly]
image = "{image}"

[tool.shell.container]
profile = "workspace-readonly"
""",
                "tool.container requires tool.shell backend container",
            ),
            (
                """
[runtime.container]
profile = "workspace-readonly"
""",
                "runtime.container requires tool.container",
            ),
            (
                f"""
[tool.container]
backend = "docker"
default_profile = "workspace-readonly"

[tool.container.profiles.workspace-readonly]
image = "{image}"

[runtime.container]
profile = "missing"
""",
                "selected profile must be allowed",
            ),
            (
                """
[tool.shell]
execution_mode = 1
""",
                "tool.shell execution_mode must be a string",
            ),
            (
                """
[tool.shell]
execution_mode = "remote"
""",
                (
                    "tool.shell execution_mode must be local, sandbox, or"
                    " container"
                ),
            ),
            (
                """
[tool.shell]
backend = "local"
execution_mode = "sandbox"
""",
                "tool.shell backend and execution_mode must match",
            ),
            (
                """
[tool.shell]
backend = "sandbox"

[tool.sandbox]
backend = "seatbelt"
unknown = true
""",
                "Unknown isolation fields",
            ),
            (
                """
[tool.shell]
backend = "sandbox"

[tool.sandbox]
backend = "seatbelt"

[tool.sandbox.profiles.host-tools]
mounts = []
""",
                "Unknown isolation fields",
            ),
            (
                f"""
[tool.container]
backend = "docker"

[tool.container.profiles.workspace-readonly]
image = "{image}"
""",
                "tool.container requires tool.shell backend container",
            ),
            (
                """
[tool.sandbox]
backend = "seatbelt"

[tool.sandbox.profiles.host-tools]
trusted_executables = ["/bin/cat"]
""",
                "tool.sandbox requires tool.shell backend sandbox",
            ),
            (
                f"""
[tool.container]
backend = "docker"

[tool.container.profiles.workspace-readonly]
image = "{image}"

[tool.sandbox]
backend = "seatbelt"

[tool.sandbox.profiles.host-tools]
trusted_executables = ["/bin/cat"]
""",
                "tool cannot mix sandbox and container policy",
            ),
            (
                """
[tool.shell]
backend = "sandbox"

[tool.shell.sandbox]
required = false
""",
                "requires required=true",
            ),
            (
                """
[tool.shell]
backend = "sandbox"

[tool.shell.sandbox]
profile = "missing"
""",
                "required sandbox profile unavailable",
            ),
            (
                """
[tool.shell]
backend = "sandbox"

[tool.shell.container]
profile = "workspace-readonly"
""",
                "tool.shell.container requires tool.shell backend container",
            ),
            (
                """
[tool.shell]
backend = "sandbox"

[tool.shell.container]
profile = "workspace-readonly"

[tool.shell.sandbox]
profile = "host-tools"
""",
                "tool.shell cannot mix sandbox and container policy",
            ),
        )

        for config, error in cases:
            with self.subTest(error=error):
                with NamedTemporaryFile("w+", suffix=".toml") as tmp:
                    tmp.write(_minimal_agent_toml() + config)
                    tmp.flush()
                    stack = AsyncExitStack()
                    loader = OrchestratorLoader(
                        hub=MagicMock(spec=HuggingfaceHub),
                        logger=MagicMock(spec=Logger),
                        participant_id=uuid4(),
                        stack=stack,
                    )

                    with self.assertRaisesRegex(AssertionError, error):
                        await loader.from_file(tmp.name, agent_id=uuid4())
                    await stack.aclose()

    async def test_shell_section_rejects_non_mapping(self):
        with NamedTemporaryFile("w+", suffix=".toml") as tmp:
            tmp.write(_minimal_agent_toml() + '\n[tool]\nshell = "yes"\n')
            tmp.flush()
            stack = AsyncExitStack()
            loader = OrchestratorLoader(
                hub=MagicMock(spec=HuggingfaceHub),
                logger=MagicMock(spec=Logger),
                participant_id=uuid4(),
                stack=stack,
            )

            with self.assertRaises(AssertionError):
                await loader.from_file(tmp.name, agent_id=uuid4())
            await stack.aclose()

    async def test_shell_enable_wildcard_is_normalized_from_file(self):
        with NamedTemporaryFile("w+", suffix=".toml") as tmp:
            tmp.write(
                _minimal_agent_toml()
                + '\n[tool]\nenable = ["shell.*", "math.calculator"]\n'
            )
            tmp.flush()
            stack = AsyncExitStack()
            loader = OrchestratorLoader(
                hub=MagicMock(spec=HuggingfaceHub),
                logger=MagicMock(spec=Logger),
                participant_id=uuid4(),
                stack=stack,
            )

            with patch.object(
                loader,
                "from_settings",
                new=AsyncMock(return_value="orch"),
            ) as from_settings:
                result = await loader.from_file(tmp.name, agent_id=uuid4())

            self.assertEqual(result, "orch")
            settings = from_settings.call_args.args[0]
            self.assertEqual(settings.tools, ["shell", "math.calculator"])
            await stack.aclose()

    async def test_from_file_registers_shell_for_opt_in_inputs(self):
        cases = (
            ("\n[tool.shell]\n", None),
            ('\n[tool]\nenable = ["shell"]\n', ["shell"]),
            ('\n[tool]\nenable = ["shell.*"]\n', ["shell"]),
            ('\n[tool]\nenable = ["shell.rg"]\n', ["shell.rg"]),
        )

        for tool_config, expected_enable in cases:
            with self.subTest(tool_config=tool_config):
                kwargs = await _from_file_tool_manager_kwargs(
                    _minimal_agent_toml() + tool_config
                )

                self.assertEqual(_shell_namespaces(kwargs), ["shell"])
                self.assertEqual(kwargs["enable_tools"], expected_enable)

    async def test_from_file_shell_manifest_reaches_model_input_for_shell(
        self,
    ) -> None:
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            source = (
                workspace / "inputs" / "batches" / "client_docs" / "report.pdf"
            )
            source.parent.mkdir(parents=True)
            source.write_bytes(b"%PDF-1.7")
            file_content = MessageContentFile(
                type="file",
                file={
                    "filename": "report.pdf",
                    "local_path": str(source.resolve()),
                },
            )
            message = Message(
                role=MessageRole.USER,
                content=[
                    MessageContentText(
                        type="text",
                        text="summarize the attachment",
                    ),
                    file_content,
                ],
            )
            base_config = """
[agent]
role = "assistant"
user = "Review: {{ input }}"

[engine]
uri = "ai://local/model"
"""
            shell_config = base_config + f"""
[tool]
enable = ["shell.pdfinfo"]

[tool.shell]
workspace_root = "{workspace.as_posix()}"
"""

            shell_input = await _loaded_agent_model_input(
                shell_config,
                message,
            )
            plain_input = await _loaded_agent_model_input(
                base_config,
                message,
            )

        shell_text = "\n".join(_input_text_blocks(shell_input))
        plain_text = "\n".join(_input_text_blocks(plain_input))
        workspace_path = "inputs/batches/client_docs/report.pdf"

        self.assertIn("Review: summarize the attachment", shell_text)
        self.assertIn("report.pdf", shell_text)
        self.assertIn(workspace_path, shell_text)
        self.assertIn(file_content, _input_file_blocks(shell_input))

        self.assertIn("Review: summarize the attachment", plain_text)
        self.assertNotIn(workspace_path, plain_text)
        self.assertIn(file_content, _input_file_blocks(plain_input))

    async def test_from_file_shell_manifest_skips_filtered_shell_tools(
        self,
    ) -> None:
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            source = (
                workspace / "inputs" / "batches" / "client_docs" / "report.pdf"
            )
            source.parent.mkdir(parents=True)
            source.write_bytes(b"%PDF-1.7")
            file_content = MessageContentFile(
                type="file",
                file={
                    "filename": "report.pdf",
                    "local_path": str(source.resolve()),
                },
            )
            message = Message(
                role=MessageRole.USER,
                content=[
                    MessageContentText(
                        type="text",
                        text="summarize the attachment",
                    ),
                    file_content,
                ],
            )
            config = f"""
[agent]
role = "assistant"
user = "Review: {{{{ input }}}}"

[engine]
uri = "ai://local/model"

[tool]
enable = ["math.calculator"]

[tool.shell]
workspace_root = "{workspace.as_posix()}"
"""

            kwargs = await _from_file_tool_manager_kwargs(config)
            model_input = await _loaded_agent_model_input(config, message)

        settings = kwargs["settings"]
        self.assertIsInstance(settings, ToolManagerSettings)
        self.assertIsNone(settings.filters)

        text = "\n".join(_input_text_blocks(model_input))
        workspace_path = "inputs/batches/client_docs/report.pdf"

        self.assertIn("Review: summarize the attachment", text)
        self.assertNotIn(workspace_path, text)
        self.assertNotIn("Attached files available to tools", text)
        self.assertIn(file_content, _input_file_blocks(model_input))

    async def test_from_file_shell_manifest_can_be_disabled(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            source = (
                workspace / "inputs" / "batches" / "client_docs" / "report.pdf"
            )
            source.parent.mkdir(parents=True)
            source.write_bytes(b"%PDF-1.7")
            file_content = MessageContentFile(
                type="file",
                file={
                    "filename": "report.pdf",
                    "local_path": str(source.resolve()),
                },
            )
            message = Message(
                role=MessageRole.USER,
                content=[
                    MessageContentText(
                        type="text",
                        text="summarize the attachment",
                    ),
                    file_content,
                ],
            )
            config = f"""
[agent]
role = "assistant"
user = "Review: {{{{ input }}}}"

[engine]
uri = "ai://local/model"

[tool]
enable = ["shell.pdfinfo"]

[tool.shell]
workspace_root = "{workspace.as_posix()}"
input_file_manifest_enabled = false
"""

            model_input = await _loaded_agent_model_input(config, message)

        text = "\n".join(_input_text_blocks(model_input))

        self.assertIn("Review: summarize the attachment", text)
        self.assertNotIn("inputs/batches/client_docs/report.pdf", text)
        self.assertNotIn("Attached files available to tools", text)
        self.assertIn(file_content, _input_file_blocks(model_input))

    async def test_from_file_loads_tool_name_policy(self):
        kwargs = await _from_file_tool_manager_kwargs(
            _minimal_agent_toml() + """
[tool]
enable = ["math.calculator"]

[tool.name_policy]
mode = "mapped"
prefix = "tool_"
replacement = "_"
collapse_replacement = false
provider_family = "local"

[tool.name_policy.map]
"math.calculator" = "calc"
"""
        )

        settings = kwargs["settings"]
        self.assertIsInstance(settings, ToolManagerSettings)
        policy = settings.tool_name_policy
        self.assertIs(policy.mode, ToolNamePolicyMode.MAPPED)
        self.assertEqual(policy.prefix, "tool_")
        self.assertEqual(policy.replacement, "_")
        self.assertFalse(policy.collapse_replacement)
        self.assertEqual(policy.provider_family, "local")
        self.assertEqual(policy.map, {"math.calculator": "calc"})

    async def test_from_file_tool_name_policy_maps_with_fallback(self):
        kwargs = await _from_file_tool_manager_kwargs(
            _minimal_agent_toml() + """
[tool]
enable = ["shell.pdfinfo", "shell.tesseract", "shell.pdftoppm"]

[tool.name_policy]
mode = "sanitized"

[tool.name_policy.map]
"shell.pdfinfo" = "pdfinfo"
"shell.tesseract" = "tesseract"
"""
        )

        settings = kwargs["settings"]
        self.assertIsInstance(settings, ToolManagerSettings)
        policy = settings.tool_name_policy
        self.assertIs(policy.mode, ToolNamePolicyMode.SANITIZED)
        self.assertEqual(
            policy.map,
            {
                "shell.pdfinfo": "pdfinfo",
                "shell.tesseract": "tesseract",
            },
        )

        shell_toolsets = [
            toolset
            for toolset in kwargs["available_toolsets"]
            if toolset.namespace == "shell"
        ]
        manager = ToolManager.create_instance(
            available_toolsets=shell_toolsets,
            enable_tools=kwargs["enable_tools"],
            settings=settings,
        )

        projection = ModelCapabilityCatalog.create(
            manager.export_model_capability_seed()
        ).project("openai")
        provider_schemas = projection.schemas
        provider_names = [
            schema["function"]["name"] for schema in provider_schemas
        ]

        self.assertEqual(
            provider_names,
            ["pdfinfo", "shell_pdftoppm", "tesseract"],
        )
        self.assertEqual(
            projection.canonical_name("shell_pdftoppm"),
            "shell.pdftoppm",
        )

    async def test_from_file_loads_tool_name_policy_empty_prefix(self):
        kwargs = await _from_file_tool_manager_kwargs(
            _minimal_agent_toml() + """
[tool]
enable = ["shell.pdfinfo"]

[tool.name_policy]
provider_family = "openai"
mode = "mapped"
prefix = ""

[tool.name_policy.map]
"shell.pdfinfo" = "pdfinfo"
"""
        )

        settings = kwargs["settings"]
        self.assertIsInstance(settings, ToolManagerSettings)
        policy = settings.tool_name_policy
        self.assertIs(policy.mode, ToolNamePolicyMode.MAPPED)
        self.assertEqual(policy.prefix, "")
        self.assertEqual(policy.map, {"shell.pdfinfo": "pdfinfo"})

    async def test_from_file_shell_agent_invokes_tool_with_fake_executor(
        self,
    ) -> None:
        policy = _AgentShellPolicy()
        executor = _AgentShellExecutor("agent shell output\n")

        schema_names, outcome = await _load_shell_agent_tool_result(
            policy=policy,
            executor=executor,
            path="agent.txt",
        )

        self.assertEqual(schema_names, ["shell.cat"])
        self.assertIsInstance(outcome, ToolCallResult)
        assert isinstance(outcome, ToolCallResult)
        assert isinstance(outcome.result, str)
        self.assertIn("tool: shell.cat", outcome.result)
        self.assertIn("status: completed", outcome.result)
        self.assertIn("agent shell output", outcome.result)
        self.assertEqual(len(policy.requests), 1)
        self.assertEqual(policy.requests[0].paths[0].path, "agent.txt")
        self.assertEqual(
            [spec.tool_name for spec in executor.specs],
            [
                "shell.cat",
            ],
        )

    async def test_from_file_shell_agent_returns_policy_denied_result(
        self,
    ) -> None:
        policy = _AgentShellPolicy(
            denial=ShellPolicyDenied(
                ShellExecutionErrorCode.DENIED_PATH,
                "path is denied",
            )
        )
        executor = _AgentShellExecutor("unused")

        schema_names, outcome = await _load_shell_agent_tool_result(
            policy=policy,
            executor=executor,
            path="denied.txt",
        )

        self.assertEqual(schema_names, ["shell.cat"])
        self.assertIsInstance(outcome, ToolCallResult)
        assert isinstance(outcome, ToolCallResult)
        assert isinstance(outcome.result, str)
        self.assertIn("tool: shell.cat", outcome.result)
        self.assertIn("status: policy_denied", outcome.result)
        self.assertIn("error_code: denied_path", outcome.result)
        self.assertIn("error_message: path is denied", outcome.result)
        self.assertEqual(len(policy.requests), 1)
        self.assertEqual(policy.requests[0].paths[0].path, "denied.txt")
        self.assertEqual(executor.specs, [])

    async def test_from_file_shell_agent_returns_unavailable_result(
        self,
    ) -> None:
        policy = _AgentShellPolicy(executable=None)
        executor = _AgentShellExecutor("unused")

        schema_names, outcome = await _load_shell_agent_tool_result(
            policy=policy,
            executor=executor,
            path="missing.txt",
        )

        self.assertEqual(schema_names, ["shell.cat"])
        self.assertIsInstance(outcome, ToolCallResult)
        assert isinstance(outcome, ToolCallResult)
        assert isinstance(outcome.result, str)
        self.assertIn("tool: shell.cat", outcome.result)
        self.assertIn("status: command_unavailable", outcome.result)
        self.assertIn("error_code: command_unavailable", outcome.result)
        self.assertIn("error_message: command is unavailable", outcome.result)
        self.assertEqual(len(policy.requests), 1)
        self.assertEqual(policy.requests[0].paths[0].path, "missing.txt")
        self.assertEqual(len(executor.specs), 1)
        self.assertIsNone(executor.specs[0].executable)

    async def test_from_file_does_not_register_shell_without_opt_in(self):
        cases = (
            (_minimal_agent_toml(), None),
            (
                _minimal_agent_toml() + '\n[tool]\nenable = ["shellx.*"]\n',
                ["shellx.*"],
            ),
        )

        for config, expected_enable in cases:
            with self.subTest(config=config):
                kwargs = await _from_file_tool_manager_kwargs(config)

                self.assertEqual(_shell_namespaces(kwargs), [])
                self.assertEqual(kwargs["enable_tools"], expected_enable)

    async def test_permission_error(self):
        if geteuid() == 0:
            self.skipTest("Running as root; permission error won't occur")
        with NamedTemporaryFile(delete=False) as tmp:
            path = tmp.name
        chmod(path, 0)
        stack = AsyncExitStack()
        try:
            loader = OrchestratorLoader(
                hub=MagicMock(spec=HuggingfaceHub),
                logger=MagicMock(spec=Logger),
                participant_id=uuid4(),
                stack=stack,
            )
            with self.assertRaises(PermissionError):
                await loader.from_file(path, agent_id=uuid4())
        finally:
            chmod(path, 0o644)
        await stack.aclose()

    async def test_permission_error_when_access_denied(self):
        with NamedTemporaryFile() as tmp:
            path = tmp.name
        stack = AsyncExitStack()
        with (
            patch("avalan.agent.loader.exists", return_value=True),
            patch("avalan.agent.loader.access", return_value=False),
        ):
            loader = OrchestratorLoader(
                hub=MagicMock(spec=HuggingfaceHub),
                logger=MagicMock(spec=Logger),
                participant_id=uuid4(),
                stack=stack,
            )
            with self.assertRaises(PermissionError):
                await loader.from_file(path, agent_id=uuid4())
        await stack.aclose()

    def test_validate_agent_file_permission_error_when_access_denied(self):
        with NamedTemporaryFile() as tmp:
            path = tmp.name
        with (
            patch("avalan.agent.loader.exists", return_value=True),
            patch("avalan.agent.loader.access", return_value=False),
        ):
            with self.assertRaises(PermissionError):
                run_async(OrchestratorLoader.validate_agent_file(path))

    def test_validate_agent_file_returns_config_and_missing_file(self):
        with TemporaryDirectory() as tmp:
            path = f"{tmp}/agent.toml"
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("""
[agent]
role = \"assistant\"

[engine]
uri = \"ai://local/model\"
""")

            config = run_async(OrchestratorLoader.validate_agent_file(path))

        self.assertEqual(config["engine"]["uri"], "ai://local/model")
        with self.assertRaises(FileNotFoundError):
            run_async(
                OrchestratorLoader.validate_agent_file("missing-agent.toml")
            )

    def test_validate_agent_config_rejects_conflicting_user_templates(self):
        with self.assertRaises(AssertionError):
            OrchestratorLoader.validate_agent_config(
                {
                    "agent": {
                        "user": "literal",
                        "user_template": "template",
                    },
                    "engine": {"uri": "ai://local/model"},
                }
            )
        with self.assertRaises(AssertionError):
            OrchestratorLoader.validate_agent_config(
                {
                    "agent": {"type": "invalid"},
                    "engine": {"uri": "ai://local/model"},
                }
            )

    def test_validate_agent_config_rejects_non_string_direct_prompts(self):
        for key in (
            "system",
            "developer",
            "instructions",
            "goal_instructions",
            "user",
            "user_template",
        ):
            with self.subTest(key=key):
                with self.assertRaises(AssertionError):
                    OrchestratorLoader.validate_agent_config(
                        {
                            "agent": {key: 1},
                            "engine": {"uri": "ai://local/model"},
                        }
                    )

    async def test_load_default_orchestrator(self):
        config = """
[agent]
role = \"assistant\"
task = \"do\"
goal_instructions = \"how\"

[engine]
uri = \"ai://local/model\"
"""
        with TemporaryDirectory() as tmp:
            path = f"{tmp}/agent.toml"
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(config)

            hub = MagicMock(spec=HuggingfaceHub)
            logger = MagicMock(spec=Logger)
            stack = AsyncExitStack()

            sentence_model = MagicMock()
            sentence_model.__enter__.return_value = sentence_model

            model_manager = MagicMock()
            model_manager.__enter__.return_value = model_manager
            model_manager.parse_uri.return_value = "uri_obj"
            model_manager.get_engine_settings.return_value = "settings_obj"

            memory = MagicMock()
            tool = MagicMock()
            browser_toolset = MagicMock()
            event_manager = MagicMock()

            with (
                patch(
                    "avalan.agent.loader.SentenceTransformerModel",
                    return_value=sentence_model,
                ),
                patch("avalan.agent.loader.TextPartitioner"),
                patch(
                    "avalan.agent.loader.MemoryManager.create_instance",
                    new=AsyncMock(return_value=memory),
                ) as mm_patch,
                patch(
                    "avalan.agent.loader.ModelManager",
                    return_value=model_manager,
                ) as model_patch,
                patch(
                    "avalan.agent.loader.DefaultOrchestrator",
                    return_value="orch",
                ) as orch_patch,
                patch(
                    "avalan.agent.loader.ToolManager.create_instance",
                    return_value=tool,
                ),
                patch(
                    "avalan.agent.loader.BrowserToolSet",
                    return_value=browser_toolset,
                ) as bts_patch,
                patch(
                    "avalan.agent.loader.EventManager",
                    return_value=event_manager,
                ),
            ):
                loader = OrchestratorLoader(
                    hub=hub,
                    logger=logger,
                    participant_id=uuid4(),
                    stack=stack,
                )
                result = await loader.from_file(
                    path,
                    agent_id=uuid4(),
                    disable_memory=True,
                )

                self.assertEqual(result, "orch")
                orch_patch.assert_called_once()
                model_patch.assert_called_once_with(
                    hub, logger, event_manager=event_manager
                )
                mm_patch.assert_awaited_once()
                bts = bts_patch.call_args.kwargs["settings"]
                self.assertIsInstance(bts, BrowserToolSettings)
                self.assertEqual(bts.engine, "firefox")
            await stack.aclose()

    async def test_load_tool_settings(self):
        config = """
[agent]
role = \"assistant\"

[engine]
uri = \"ai://local/model\"

[tool.browser.open]
engine = \"webkit\"
debug = true
"""
        with TemporaryDirectory() as tmp:
            path = f"{tmp}/agent.toml"
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(config)

            hub = MagicMock(spec=HuggingfaceHub)
            logger = MagicMock(spec=Logger)
            stack = AsyncExitStack()

            sentence_model = MagicMock()
            sentence_model.__enter__.return_value = sentence_model

            model_manager = MagicMock()
            model_manager.__enter__.return_value = model_manager
            model_manager.parse_uri.return_value = "uri_obj"
            model_manager.get_engine_settings.return_value = "settings_obj"

            memory = MagicMock()
            tool = MagicMock()
            browser_toolset = MagicMock()
            event_manager = MagicMock()

            with (
                patch(
                    "avalan.agent.loader.SentenceTransformerModel",
                    return_value=sentence_model,
                ),
                patch("avalan.agent.loader.TextPartitioner"),
                patch(
                    "avalan.agent.loader.MemoryManager.create_instance",
                    new=AsyncMock(return_value=memory),
                ),
                patch(
                    "avalan.agent.loader.ModelManager",
                    return_value=model_manager,
                ),
                patch(
                    "avalan.agent.loader.DefaultOrchestrator",
                    return_value="orch",
                ),
                patch(
                    "avalan.agent.loader.ToolManager.create_instance",
                    return_value=tool,
                ),
                patch(
                    "avalan.agent.loader.BrowserToolSet",
                    return_value=browser_toolset,
                ) as bts_patch,
                patch(
                    "avalan.agent.loader.EventManager",
                    return_value=event_manager,
                ),
            ):
                loader = OrchestratorLoader(
                    hub=hub,
                    logger=logger,
                    participant_id=uuid4(),
                    stack=stack,
                )
                await loader.from_file(
                    path,
                    agent_id=uuid4(),
                    disable_memory=True,
                )

                bs = bts_patch.call_args.kwargs["settings"]
                self.assertIsInstance(bs, BrowserToolSettings)
                self.assertEqual(bs.engine, "webkit")
                self.assertTrue(bs.debug)
            await stack.aclose()

    async def test_load_old_tool_settings_section(self):
        config = """
[agent]
role = \"assistant\"

[engine]
uri = \"ai://local/model\"

[tool.browser]
engine = \"chromium\"
"""
        with TemporaryDirectory() as tmp:
            path = f"{tmp}/agent.toml"
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(config)

            hub = MagicMock(spec=HuggingfaceHub)
            logger = MagicMock(spec=Logger)
            stack = AsyncExitStack()

            sentence_model = MagicMock()
            sentence_model.__enter__.return_value = sentence_model

            model_manager = MagicMock()
            model_manager.__enter__.return_value = model_manager
            model_manager.parse_uri.return_value = "uri_obj"
            model_manager.get_engine_settings.return_value = "settings_obj"

            memory = MagicMock()
            tool = MagicMock()
            browser_toolset = MagicMock()
            event_manager = MagicMock()

            with (
                patch(
                    "avalan.agent.loader.SentenceTransformerModel",
                    return_value=sentence_model,
                ),
                patch("avalan.agent.loader.TextPartitioner"),
                patch(
                    "avalan.agent.loader.MemoryManager.create_instance",
                    new=AsyncMock(return_value=memory),
                ),
                patch(
                    "avalan.agent.loader.ModelManager",
                    return_value=model_manager,
                ),
                patch(
                    "avalan.agent.loader.DefaultOrchestrator",
                    return_value="orch",
                ),
                patch(
                    "avalan.agent.loader.ToolManager.create_instance",
                    return_value=tool,
                ),
                patch(
                    "avalan.agent.loader.BrowserToolSet",
                    return_value=browser_toolset,
                ) as bts_patch,
                patch(
                    "avalan.agent.loader.EventManager",
                    return_value=event_manager,
                ),
            ):
                loader = OrchestratorLoader(
                    hub=hub,
                    logger=logger,
                    participant_id=uuid4(),
                    stack=stack,
                )
                await loader.from_file(
                    path,
                    agent_id=uuid4(),
                    disable_memory=True,
                )

                bs = bts_patch.call_args.kwargs["settings"]
                self.assertIsInstance(bs, BrowserToolSettings)
                self.assertEqual(bs.engine, "chromium")
            await stack.aclose()

    async def test_load_database_tool_settings(self):
        config = """
[agent]
role = \"assistant\"

[engine]
uri = \"ai://local/model\"

[tool.database]
dsn = \"sqlite:///db.sqlite\"
"""
        with TemporaryDirectory() as tmp:
            path = f"{tmp}/agent.toml"
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(config)

            hub = MagicMock(spec=HuggingfaceHub)
            logger = MagicMock(spec=Logger)
            stack = AsyncExitStack()

            with patch.object(
                OrchestratorLoader,
                "from_settings",
                new=AsyncMock(return_value="orch"),
            ) as lfs_patch:
                loader = OrchestratorLoader(
                    hub=hub,
                    logger=logger,
                    participant_id=uuid4(),
                    stack=stack,
                )
                await loader.from_file(
                    path,
                    agent_id=uuid4(),
                )

                tool_settings = lfs_patch.call_args.kwargs["tool_settings"]
                dbs = tool_settings.database
                self.assertIsInstance(dbs, DatabaseToolSettings)
                self.assertEqual(dbs.dsn, "sqlite:///db.sqlite")
            await stack.aclose()

    async def test_load_database_tool_settings_resolves_env_dsn(self):
        config = """
[agent]
role = \"assistant\"

[engine]
uri = \"ai://local/model\"

[tool.database]
dsn = \"env:AVALAN_TEST_DATABASE_DSN\"
"""
        with TemporaryDirectory() as tmp:
            path = f"{tmp}/agent.toml"
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(config)

            hub = MagicMock(spec=HuggingfaceHub)
            logger = MagicMock(spec=Logger)
            stack = AsyncExitStack()

            with (
                patch.dict(
                    "os.environ",
                    {"AVALAN_TEST_DATABASE_DSN": "sqlite:///env.sqlite"},
                    clear=True,
                ),
                patch.object(
                    OrchestratorLoader,
                    "from_settings",
                    new=AsyncMock(return_value="orch"),
                ) as lfs_patch,
            ):
                loader = OrchestratorLoader(
                    hub=hub,
                    logger=logger,
                    participant_id=uuid4(),
                    stack=stack,
                )
                await loader.from_file(
                    path,
                    agent_id=uuid4(),
                )

                tool_settings = lfs_patch.call_args.kwargs["tool_settings"]
                dbs = tool_settings.database
                self.assertIsInstance(dbs, DatabaseToolSettings)
                self.assertEqual(dbs.dsn, "sqlite:///env.sqlite")
            await stack.aclose()

    async def test_load_database_tool_settings_rejects_invalid_env_name(self):
        config = """
[agent]
role = \"assistant\"

[engine]
uri = \"ai://local/model\"

[tool.database]
dsn = \"env:INVALID-NAME\"
"""
        with TemporaryDirectory() as tmp:
            path = f"{tmp}/agent.toml"
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(config)

            hub = MagicMock(spec=HuggingfaceHub)
            logger = MagicMock(spec=Logger)
            stack = AsyncExitStack()

            loader = OrchestratorLoader(
                hub=hub,
                logger=logger,
                participant_id=uuid4(),
                stack=stack,
            )
            with self.assertRaises(AssertionError):
                await loader.from_file(
                    path,
                    agent_id=uuid4(),
                )
            await stack.aclose()

    async def test_load_database_tool_settings_rejects_missing_env_dsn(self):
        for env_name, env_value in (
            ("AVALAN_MISSING_DATABASE_DSN", None),
            ("AVALAN_EMPTY_DATABASE_DSN", ""),
        ):
            with self.subTest(env_name=env_name):
                config = f"""
[agent]
role = \"assistant\"

[engine]
uri = \"ai://local/model\"

[tool.database]
dsn = \"env:{env_name}\"
"""
                with TemporaryDirectory() as tmp:
                    path = f"{tmp}/agent.toml"
                    with open(path, "w", encoding="utf-8") as fh:
                        fh.write(config)

                    hub = MagicMock(spec=HuggingfaceHub)
                    logger = MagicMock(spec=Logger)
                    stack = AsyncExitStack()
                    env = {} if env_value is None else {env_name: env_value}

                    with patch.dict("os.environ", env, clear=True):
                        try:
                            loader = OrchestratorLoader(
                                hub=hub,
                                logger=logger,
                                participant_id=uuid4(),
                                stack=stack,
                            )
                            with self.assertRaises(AssertionError):
                                await loader.from_file(
                                    path,
                                    agent_id=uuid4(),
                                )
                        finally:
                            await stack.aclose()

    async def test_load_database_tool_settings_normalizes_scalar_commands(
        self,
    ):
        config = """
[agent]
role = \"assistant\"

[engine]
uri = \"ai://local/model\"

[tool.database]
dsn = \"sqlite:///db.sqlite\"
allowed_commands = \"select\"
"""
        with TemporaryDirectory() as tmp:
            path = f"{tmp}/agent.toml"
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(config)

            hub = MagicMock(spec=HuggingfaceHub)
            logger = MagicMock(spec=Logger)
            stack = AsyncExitStack()

            try:
                with patch.object(
                    OrchestratorLoader,
                    "from_settings",
                    new=AsyncMock(return_value="orch"),
                ) as lfs_patch:
                    loader = OrchestratorLoader(
                        hub=hub,
                        logger=logger,
                        participant_id=uuid4(),
                        stack=stack,
                    )
                    await loader.from_file(path, agent_id=uuid4())
                    tool_settings = lfs_patch.call_args.kwargs["tool_settings"]
                    self.assertEqual(
                        tool_settings.database.allowed_commands,
                        ["select"],
                    )
            finally:
                await stack.aclose()

    async def test_load_graph_tool_settings(self):
        config = """
[agent]
role = \"assistant\"

[engine]
uri = \"ai://local/model\"

[tool.graph]
file = \"/tmp/chart.png\"
"""
        with TemporaryDirectory() as tmp:
            path = f"{tmp}/agent.toml"
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(config)

            hub = MagicMock(spec=HuggingfaceHub)
            logger = MagicMock(spec=Logger)
            stack = AsyncExitStack()

            with patch.object(
                OrchestratorLoader,
                "from_settings",
                new=AsyncMock(return_value="orch"),
            ) as lfs_patch:
                loader = OrchestratorLoader(
                    hub=hub,
                    logger=logger,
                    participant_id=uuid4(),
                    stack=stack,
                )
                await loader.from_file(path, agent_id=uuid4())

                tool_settings = lfs_patch.call_args.kwargs["tool_settings"]
                graph = tool_settings.graph
                self.assertIsInstance(graph, GraphToolSettings)
                self.assertEqual(graph.file, "/tmp/chart.png")
            await stack.aclose()

    async def test_tool_format_setting(self):
        config = """
[agent]
role = \"assistant\"

[engine]
uri = \"ai://local/model\"

[tool]
format = \"react\"
"""
        with TemporaryDirectory() as tmp:
            path = f"{tmp}/agent.toml"
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(config)

            hub = MagicMock(spec=HuggingfaceHub)
            logger = MagicMock(spec=Logger)
            stack = AsyncExitStack()

            sentence_model = MagicMock()
            sentence_model.__enter__.return_value = sentence_model

            model_manager = MagicMock()
            model_manager.__enter__.return_value = model_manager
            model_manager.parse_uri.return_value = "uri_obj"
            model_manager.get_engine_settings.return_value = "settings_obj"

            memory = MagicMock()
            tool = MagicMock()
            browser_toolset = MagicMock()
            event_manager = MagicMock()

            with (
                patch(
                    "avalan.agent.loader.SentenceTransformerModel",
                    return_value=sentence_model,
                ),
                patch("avalan.agent.loader.TextPartitioner"),
                patch(
                    "avalan.agent.loader.MemoryManager.create_instance",
                    new=AsyncMock(return_value=memory),
                ),
                patch(
                    "avalan.agent.loader.ModelManager",
                    return_value=model_manager,
                ),
                patch(
                    "avalan.agent.loader.DefaultOrchestrator",
                    return_value="orch",
                ),
                patch(
                    "avalan.agent.loader.ToolManager.create_instance",
                    return_value=tool,
                ) as tm_patch,
                patch(
                    "avalan.agent.loader.BrowserToolSet",
                    return_value=browser_toolset,
                ),
                patch(
                    "avalan.agent.loader.EventManager",
                    return_value=event_manager,
                ),
            ):
                loader = OrchestratorLoader(
                    hub=hub,
                    logger=logger,
                    participant_id=uuid4(),
                    stack=stack,
                )
                await loader.from_file(path, agent_id=uuid4())

                settings = tm_patch.call_args.kwargs["settings"]
                self.assertEqual(settings.tool_format, ToolFormat.REACT)
            await stack.aclose()

    async def test_tool_enable_accepts_list(self):
        config = """
[agent]
role = \"assistant\"

[engine]
uri = \"ai://local/model\"

[tool]
enable = [\"math.calculator\", \"browser.open\"]
"""
        with TemporaryDirectory() as tmp:
            path = f"{tmp}/agent.toml"
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(config)

            stack = AsyncExitStack()
            hub = MagicMock(spec=HuggingfaceHub)
            logger = MagicMock(spec=Logger)

            with patch.object(
                OrchestratorLoader,
                "from_settings",
                new=AsyncMock(return_value="orch"),
            ) as from_settings:
                loader = OrchestratorLoader(
                    hub=hub,
                    logger=logger,
                    participant_id=uuid4(),
                    stack=stack,
                )
                result = await loader.from_file(path, agent_id=uuid4())

                self.assertEqual(result, "orch")
                settings = from_settings.call_args.args[0]
                self.assertEqual(
                    settings.tools, ["math.calculator", "browser.open"]
                )
            await stack.aclose()

    async def test_tool_enable_accepts_string(self):
        config = """
[agent]
role = \"assistant\"

[engine]
uri = \"ai://local/model\"

[tool]
enable = \"math.calculator\"
"""
        with TemporaryDirectory() as tmp:
            path = f"{tmp}/agent.toml"
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(config)

            stack = AsyncExitStack()
            hub = MagicMock(spec=HuggingfaceHub)
            logger = MagicMock(spec=Logger)

            with patch.object(
                OrchestratorLoader,
                "from_settings",
                new=AsyncMock(return_value="orch"),
            ) as from_settings:
                loader = OrchestratorLoader(
                    hub=hub,
                    logger=logger,
                    participant_id=uuid4(),
                    stack=stack,
                )
                await loader.from_file(path, agent_id=uuid4())

                settings = from_settings.call_args.args[0]
                self.assertEqual(settings.tools, ["math.calculator"])
            await stack.aclose()

    async def test_tool_enable_invalid_type(self):
        config = """
[agent]
role = \"assistant\"

[engine]
uri = \"ai://local/model\"

[tool]
enable = 3
"""
        with TemporaryDirectory() as tmp:
            path = f"{tmp}/agent.toml"
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(config)

            stack = AsyncExitStack()
            hub = MagicMock(spec=HuggingfaceHub)
            logger = MagicMock(spec=Logger)

            loader = OrchestratorLoader(
                hub=hub,
                logger=logger,
                participant_id=uuid4(),
                stack=stack,
            )

            with self.assertRaises(AssertionError):
                await loader.from_file(path, agent_id=uuid4())

            await stack.aclose()

    async def test_engine_section_tools_option_not_supported(self):
        config = """
[agent]
role = \"assistant\"

[engine]
uri = \"ai://local/model\"
tools = [\"math.calculator\"]
"""
        with TemporaryDirectory() as tmp:
            path = f"{tmp}/agent.toml"
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(config)

            stack = AsyncExitStack()
            hub = MagicMock(spec=HuggingfaceHub)
            logger = MagicMock(spec=Logger)

            loader = OrchestratorLoader(
                hub=hub,
                logger=logger,
                participant_id=uuid4(),
                stack=stack,
            )

            with self.assertRaises(AssertionError):
                await loader.from_file(path, agent_id=uuid4())

            await stack.aclose()

    async def test_tool_format_variants(self):
        for value, expected in (
            ("react", ToolFormat.REACT),
            ("json", ToolFormat.JSON),
            ("openai", ToolFormat.OPENAI),
            ("dsml", ToolFormat.DSML),
        ):
            with self.subTest(value=value):
                config = f"""
[agent]
role = \"assistant\"

[engine]
uri = \"ai://local/model\"

[tool]
format = \"{value}\"
"""
                with TemporaryDirectory() as tmp:
                    path = f"{tmp}/agent.toml"
                    with open(path, "w", encoding="utf-8") as fh:
                        fh.write(config)

                    stack = AsyncExitStack()
                    hub = MagicMock(spec=HuggingfaceHub)
                    logger = MagicMock(spec=Logger)

                    with patch.object(
                        OrchestratorLoader,
                        "from_settings",
                        new=AsyncMock(return_value="orch"),
                    ) as from_settings:
                        loader = OrchestratorLoader(
                            hub=hub,
                            logger=logger,
                            participant_id=uuid4(),
                            stack=stack,
                        )
                        await loader.from_file(path, agent_id=uuid4())

                        tool_format = from_settings.call_args.kwargs[
                            "tool_format"
                        ]
                        self.assertEqual(tool_format, expected)
                    await stack.aclose()

    async def test_tool_recovery_format_variants(self):
        config = """
[agent]
role = \"assistant\"

[engine]
uri = \"ai://local/model\"

[tool]
recovery_formats = [\"fenced\", \"tool_call_block\"]
"""
        with TemporaryDirectory() as tmp:
            path = f"{tmp}/agent.toml"
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(config)

            stack = AsyncExitStack()
            hub = MagicMock(spec=HuggingfaceHub)
            logger = MagicMock(spec=Logger)

            with patch.object(
                OrchestratorLoader,
                "from_settings",
                new=AsyncMock(return_value="orch"),
            ) as from_settings:
                loader = OrchestratorLoader(
                    hub=hub,
                    logger=logger,
                    participant_id=uuid4(),
                    stack=stack,
                )
                await loader.from_file(path, agent_id=uuid4())

                self.assertEqual(
                    from_settings.call_args.kwargs["tool_recovery_formats"],
                    [
                        ToolCallRecoveryFormat.FENCED,
                        ToolCallRecoveryFormat.TOOL_CALL_BLOCK,
                    ],
                )
            await stack.aclose()

    async def test_tool_recovery_formats_reject_non_list(self):
        config = """
[agent]
role = \"assistant\"

[engine]
uri = \"ai://local/model\"

[tool]
recovery_formats = \"fenced\"
"""
        with TemporaryDirectory() as tmp:
            path = f"{tmp}/agent.toml"
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(config)

            stack = AsyncExitStack()
            hub = MagicMock(spec=HuggingfaceHub)
            logger = MagicMock(spec=Logger)

            loader = OrchestratorLoader(
                hub=hub,
                logger=logger,
                participant_id=uuid4(),
                stack=stack,
            )
            with self.assertRaises(AssertionError):
                await loader.from_file(path, agent_id=uuid4())
            await stack.aclose()

    async def test_tool_recovery_formats_reject_unknown_value(self):
        config = """
[agent]
role = \"assistant\"

[engine]
uri = \"ai://local/model\"

[tool]
recovery_formats = [\"unknown\"]
"""
        with TemporaryDirectory() as tmp:
            path = f"{tmp}/agent.toml"
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(config)

            stack = AsyncExitStack()
            hub = MagicMock(spec=HuggingfaceHub)
            logger = MagicMock(spec=Logger)

            loader = OrchestratorLoader(
                hub=hub,
                logger=logger,
                participant_id=uuid4(),
                stack=stack,
            )
            with self.assertRaises(ValueError):
                await loader.from_file(path, agent_id=uuid4())
            await stack.aclose()

    async def test_tool_settings_argument(self):
        config = """
[agent]
role = \"assistant\"

[engine]
uri = \"ai://local/model\"
"""
        with TemporaryDirectory() as tmp:
            path = f"{tmp}/agent.toml"
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(config)

            hub = MagicMock(spec=HuggingfaceHub)
            logger = MagicMock(spec=Logger)
            stack = AsyncExitStack()

            tool_settings = ToolSettingsContext(
                browser=BrowserToolSettings(engine="webkit"),
                database=DatabaseToolSettings(dsn="sqlite:///db.sqlite"),
                graph=GraphToolSettings(file="/tmp/chart.png"),
                extra={"x": 1},
            )

            with patch.object(
                OrchestratorLoader,
                "from_settings",
                new=AsyncMock(return_value="orch"),
            ) as fs_patch:
                loader = OrchestratorLoader(
                    hub=hub,
                    logger=logger,
                    participant_id=uuid4(),
                    stack=stack,
                )
                result = await loader.from_file(
                    path, agent_id=uuid4(), tool_settings=tool_settings
                )
                self.assertEqual(result, "orch")
                fs_patch.assert_awaited_once()
                passed = fs_patch.call_args.kwargs["tool_settings"]
                self.assertEqual(passed.browser.engine, "webkit")
                self.assertEqual(passed.database.dsn, "sqlite:///db.sqlite")
                self.assertEqual(passed.graph.file, "/tmp/chart.png")
                self.assertEqual(passed.extra, {"x": 1})
            await stack.aclose()

    async def test_load_json_orchestrator(self):
        config = """
[agent]
type = \"json\"
role = \"assistant\"
task = \"do\"
goal_instructions = \"how\"

[engine]
uri = \"ai://local/model\"

[json]
value = { type = \"string\", description = \"d\" }
"""
        with TemporaryDirectory() as tmp:
            path = f"{tmp}/agent.toml"
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(config)

            hub = MagicMock(spec=HuggingfaceHub)
            logger = MagicMock(spec=Logger)
            stack = AsyncExitStack()

            sentence_model = MagicMock()
            sentence_model.__enter__.return_value = sentence_model

            model_manager = MagicMock()
            model_manager.__enter__.return_value = model_manager
            model_manager.parse_uri.return_value = "uri_obj"
            model_manager.get_engine_settings.return_value = "settings_obj"

            memory = MagicMock()
            tool = MagicMock()
            event_manager = MagicMock()

            with (
                patch(
                    "avalan.agent.loader.SentenceTransformerModel",
                    return_value=sentence_model,
                ),
                patch("avalan.agent.loader.TextPartitioner"),
                patch(
                    "avalan.agent.loader.MemoryManager.create_instance",
                    new=AsyncMock(return_value=memory),
                ),
                patch(
                    "avalan.agent.loader.ModelManager",
                    return_value=model_manager,
                ),
                patch.object(
                    OrchestratorLoader,
                    "_load_json_orchestrator",
                    return_value="json_orch",
                ) as json_patch,
                patch(
                    "avalan.agent.loader.ToolManager.create_instance",
                    return_value=tool,
                ),
                patch(
                    "avalan.agent.loader.EventManager",
                    return_value=event_manager,
                ),
            ):
                loader = OrchestratorLoader(
                    hub=hub,
                    logger=logger,
                    participant_id=uuid4(),
                    stack=stack,
                )
                result = await loader.from_file(
                    path,
                    agent_id=uuid4(),
                    disable_memory=True,
                )

                self.assertEqual(result, "json_orch")
                json_patch.assert_called_once()
            await stack.aclose()

    async def test_unknown_type(self):
        config = """
[agent]
type = \"foo\"
role = \"assistant\"

[engine]
uri = \"ai://local/model\"
"""
        with TemporaryDirectory() as tmp:
            path = f"{tmp}/agent.toml"
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(config)

            stack = AsyncExitStack()
            loader = OrchestratorLoader(
                hub=MagicMock(spec=HuggingfaceHub),
                logger=MagicMock(spec=Logger),
                participant_id=uuid4(),
                stack=stack,
            )
            with self.assertRaises(AssertionError):
                await loader.from_file(path, agent_id=uuid4())
        await stack.aclose()

    async def test_from_file_overrides_uri(self):
        config = """
[agent]
role = \"assistant\"

[engine]
uri = \"ai://old/model\"
"""
        with TemporaryDirectory() as tmp:
            path = f"{tmp}/agent.toml"
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(config)

            hub = MagicMock(spec=HuggingfaceHub)
            logger = MagicMock(spec=Logger)
            stack = AsyncExitStack()
            loader = OrchestratorLoader(
                hub=hub,
                logger=logger,
                participant_id=uuid4(),
                stack=stack,
            )
            with patch.object(
                loader,
                "from_settings",
                AsyncMock(return_value=MagicMock()),
            ) as fs:
                await loader.from_file(
                    path, agent_id=uuid4(), uri="ai://new/model"
                )
            settings_arg = fs.await_args.args[0]
            self.assertEqual(settings_arg.uri, "ai://new/model")
        await stack.aclose()

    async def test_sentence_model_engine_config(self):
        config = """
[agent]
role = \"assistant\"

[engine]
uri = \"ai://local/model\"

["memory.engine"]
model_id = "smodel"
max_tokens = 300
overlap_size = 60
window_size = 120
backend = "onnx"
"""
        with TemporaryDirectory() as tmp:
            path = f"{tmp}/agent.toml"
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(config)

            hub = MagicMock(spec=HuggingfaceHub)
            logger = MagicMock(spec=Logger)
            stack = AsyncExitStack()

            with patch.object(
                OrchestratorLoader,
                "from_settings",
                new=AsyncMock(return_value="orch"),
            ) as lfs_patch:
                loader = OrchestratorLoader(
                    hub=hub,
                    logger=logger,
                    participant_id=uuid4(),
                    stack=stack,
                )
                result = await loader.from_file(
                    path,
                    agent_id=uuid4(),
                )

                self.assertEqual(result, "orch")
                lfs_patch.assert_awaited_once()
                settings = lfs_patch.call_args.args[0]
                self.assertEqual(settings.sentence_model_id, "smodel")
                self.assertEqual(settings.sentence_model_max_tokens, 300)
                self.assertEqual(settings.sentence_model_overlap_size, 60)
                self.assertEqual(settings.sentence_model_window_size, 120)
                self.assertEqual(
                    settings.sentence_model_engine_config, {"backend": "onnx"}
                )
            await stack.aclose()


class LoaderTomlVariantsTestCase(IsolatedAsyncioTestCase):
    async def _run_loader(self, config: str) -> dict:
        with TemporaryDirectory() as tmp:
            path = f"{tmp}/agent.toml"
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(config)

            hub = MagicMock(spec=HuggingfaceHub)
            logger = MagicMock(spec=Logger)
            stack = AsyncExitStack()

            sentence_model = MagicMock()
            sentence_model.__enter__.return_value = sentence_model

            model_manager = MagicMock()
            model_manager.__enter__.return_value = model_manager
            model_manager.parse_uri.return_value = "uri_obj"
            model_manager.get_engine_settings.return_value = "settings_obj"

            memory = MagicMock()
            tool = MagicMock()
            browser_toolset = MagicMock()
            event_manager = MagicMock()

            with (
                patch(
                    "avalan.agent.loader.SentenceTransformerModel",
                    return_value=sentence_model,
                ),
                patch("avalan.agent.loader.TextPartitioner"),
                patch(
                    "avalan.agent.loader.MemoryManager.create_instance",
                    new=AsyncMock(return_value=memory),
                ),
                patch(
                    "avalan.agent.loader.ModelManager",
                    return_value=model_manager,
                ),
                patch(
                    "avalan.agent.loader.DefaultOrchestrator",
                    return_value="orch",
                ) as orch_patch,
                patch(
                    "avalan.agent.loader.ToolManager.create_instance",
                    return_value=tool,
                ),
                patch(
                    "avalan.agent.loader.BrowserToolSet",
                    return_value=browser_toolset,
                ),
                patch(
                    "avalan.agent.loader.EventManager",
                    return_value=event_manager,
                ),
            ):
                loader = OrchestratorLoader(
                    hub=hub,
                    logger=logger,
                    participant_id=uuid4(),
                    stack=stack,
                )
                await loader.from_file(
                    path,
                    agent_id=uuid4(),
                    disable_memory=True,
                )

                kwargs = orch_patch.call_args.kwargs
            await stack.aclose()
            return kwargs

    async def test_agent_system_only(self):
        config = """
[agent]
system = \"sys\"

[engine]
uri = \"ai://local/model\"
"""
        kwargs = await self._run_loader(config)
        self.assertEqual(kwargs["system"], "sys")
        self.assertIsNone(kwargs["role"])
        self.assertIsNone(kwargs["task"])
        self.assertIsNone(kwargs["instructions"])
        self.assertIsNone(kwargs["rules"])

    async def test_agent_developer_only(self):
        config = """
[agent]
developer = \"dev\"

[engine]
uri = \"ai://local/model\"
"""
        kwargs = await self._run_loader(config)
        self.assertEqual(kwargs["developer"], "dev")
        self.assertIsNone(kwargs["system"])
        self.assertIsNone(kwargs["role"])
        self.assertIsNone(kwargs["task"])
        self.assertIsNone(kwargs["instructions"])

    async def test_agent_system_and_developer_only(self):
        config = """
[agent]
system = \"sys\"
developer = \"dev\"

[engine]
uri = \"ai://local/model\"
"""
        kwargs = await self._run_loader(config)
        self.assertEqual(kwargs["system"], "sys")
        self.assertEqual(kwargs["developer"], "dev")
        self.assertIsNone(kwargs["role"])
        self.assertIsNone(kwargs["task"])
        self.assertIsNone(kwargs["instructions"])

    async def test_agent_system_developer_and_user_prompts(self):
        config = """
[agent]
system = \"sys\"
developer = \"dev\"
user = \"prefix\"
role = \"ignored\"
task = \"ignored\"
instructions = \"provider\"

[engine]
uri = \"ai://local/model\"
"""
        kwargs = await self._run_loader(config)
        self.assertEqual(kwargs["system"], "sys")
        self.assertEqual(kwargs["developer"], "dev")
        self.assertEqual(kwargs["user"], "prefix")
        self.assertIsNone(kwargs["role"])
        self.assertIsNone(kwargs["task"])
        self.assertEqual(kwargs["instructions"], "provider")

    async def test_agent_direct_prompt_invalid_type_rejected(self):
        config = """
[agent]
system = 1

[engine]
uri = \"ai://local/model\"
"""
        with self.assertRaises(AssertionError):
            await self._run_loader(config)

    async def test_agent_goal_instructions_invalid_type_rejected(self):
        config = """
[agent]
task = \"do\"
goal_instructions = 1

[engine]
uri = \"ai://local/model\"
"""
        with self.assertRaisesRegex(
            AssertionError, "agent.goal_instructions must be a string"
        ):
            await self._run_loader(config)

    async def test_agent_role_task_only(self):
        config = """
[agent]
role = \"assistant\"
task = \"do\"

[engine]
uri = \"ai://local/model\"
"""
        kwargs = await self._run_loader(config)
        self.assertEqual(kwargs["role"], "assistant")
        self.assertEqual(kwargs["task"], "do")
        self.assertIsNone(kwargs["instructions"])
        self.assertIsNone(kwargs["system"])

    async def test_agent_full_definition(self):
        config = """
[agent]
role = \"assistant\"
task = \"do\"
goal_instructions = \"how\"
rules = [\"r1\", \"r2\"]

[engine]
uri = \"ai://local/model\"
"""
        kwargs = await self._run_loader(config)
        self.assertEqual(kwargs["role"], "assistant")
        self.assertEqual(kwargs["task"], "do")
        self.assertEqual(kwargs["goal_instructions"], "how")
        self.assertIsNone(kwargs["instructions"])
        self.assertEqual(kwargs["rules"], ["r1", "r2"])
        self.assertIsNone(kwargs["system"])

    async def test_agent_rejects_old_goal_instructions_field(self):
        config = """
[agent]
role = \"assistant\"
task = \"do\"
instructions = \"how\"

[engine]
uri = \"ai://local/model\"
"""
        with self.assertRaisesRegex(
            AssertionError,
            "agent.instructions is reserved.*agent.goal_instructions",
        ):
            await self._run_loader(config)

    async def test_agent_rejects_goal_instructions_without_task(self):
        config = """
[agent]
role = \"assistant\"
goal_instructions = \"how\"

[engine]
uri = \"ai://local/model\"
"""
        with self.assertRaisesRegex(
            AssertionError, "agent.goal_instructions requires agent.task"
        ):
            await self._run_loader(config)

    async def test_agent_user_only(self):
        config = """
[agent]
user = \"hello {{input}}\"

[engine]
uri = \"ai://local/model\"
"""
        kwargs = await self._run_loader(config)
        self.assertEqual(kwargs["user"], "hello {{input}}")
        self.assertIsNone(kwargs.get("user_template"))

    async def test_agent_user_template_only(self):
        config = """
[agent]
user_template = \"u.md\"

[engine]
uri = \"ai://local/model\"
"""
        kwargs = await self._run_loader(config)
        self.assertEqual(kwargs["user_template"], "u.md")
        self.assertIsNone(kwargs.get("user"))

    async def test_agent_provider_instructions_and_user_prompt(self):
        config = """
[agent]
instructions = \"provider\"
user = \"prefix {{input}}\"

[engine]
uri = \"ai://local/model\"
"""
        kwargs = await self._run_loader(config)
        self.assertEqual(kwargs["instructions"], "provider")
        self.assertEqual(kwargs["user"], "prefix {{input}}")
        self.assertIsNone(kwargs["role"])
        self.assertIsNone(kwargs["task"])
        self.assertIsNone(kwargs["goal_instructions"])


class LoadJsonOrchestratorVariantsTestCase(IsolatedAsyncioTestCase):
    def test_system_only(self):
        agent_id = uuid4()
        engine_uri = MagicMock()
        engine_settings = MagicMock()
        logger = MagicMock()
        model_manager = MagicMock()
        memory = MagicMock()
        tool = MagicMock()
        event_manager = MagicMock()

        config = {"json": {"value": {"type": "string", "description": "v"}}}
        agent_config = {"system": "sys"}

        with patch("avalan.agent.loader.JsonOrchestrator") as orch_patch:
            OrchestratorLoader._load_json_orchestrator(
                agent_id=agent_id,
                engine_uri=engine_uri,
                engine_settings=engine_settings,
                logger=logger,
                model_manager=model_manager,
                memory=memory,
                tool=tool,
                event_manager=event_manager,
                config=config,
                agent_config=agent_config,
                call_options=None,
                template_vars=None,
            )
            kwargs = orch_patch.call_args.kwargs
            self.assertEqual(kwargs["system"], "sys")
            self.assertIsNone(kwargs["task"])
            self.assertIsNone(kwargs["instructions"])

    async def test_browser_debug_source(self):
        with TemporaryDirectory() as tmp:
            debug_path = f"{tmp}/debug.txt"
            with open(debug_path, "w", encoding="utf-8") as fh:
                fh.write("debug")

            config = f"""
[agent]
role = \"assistant\"

[engine]
uri = \"ai://local/model\"

[tool.browser.open]
debug = true
debug_source = \"{debug_path}\"
"""
            path = f"{tmp}/agent.toml"
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(config)

            hub = MagicMock(spec=HuggingfaceHub)
            logger = MagicMock(spec=Logger)
            stack = AsyncExitStack()

            with patch.object(
                OrchestratorLoader,
                "from_settings",
                new=AsyncMock(return_value="orch"),
            ) as lfs_patch:
                loader = OrchestratorLoader(
                    hub=hub,
                    logger=logger,
                    participant_id=uuid4(),
                    stack=stack,
                )
                result = await loader.from_file(
                    path,
                    agent_id=uuid4(),
                )

                self.assertEqual(result, "orch")
                lfs_patch.assert_awaited_once()
                tool_settings = lfs_patch.call_args.kwargs["tool_settings"]
                browser_settings = tool_settings.browser
                self.assertTrue(browser_settings.debug)
                self.assertIsNotNone(browser_settings.debug_source)
                self.assertEqual(browser_settings.debug_source.read(), "debug")
                browser_settings.debug_source.close()
                dbs = tool_settings.database
                self.assertIsNone(dbs)
            await stack.aclose()

    async def test_json_settings_provided(self):
        config = """
[agent]
role = \"assistant\"
task = \"do\"
goal_instructions = \"ins\"

[engine]
uri = \"ai://local/model\"

[json]
value = { type = \"string\", description = \"d\" }
"""
        with TemporaryDirectory() as tmp:
            path = f"{tmp}/agent.toml"
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(config)

            hub = MagicMock(spec=HuggingfaceHub)
            logger = MagicMock(spec=Logger)
            stack = AsyncExitStack()

            with patch.object(
                OrchestratorLoader,
                "from_settings",
                new=AsyncMock(return_value="orch"),
            ) as lfs_patch:
                loader = OrchestratorLoader(
                    hub=hub,
                    logger=logger,
                    participant_id=uuid4(),
                    stack=stack,
                )
                result = await loader.from_file(
                    path,
                    agent_id=uuid4(),
                )

                self.assertEqual(result, "orch")
                lfs_patch.assert_awaited_once()
                settings = lfs_patch.call_args.args[0]
                self.assertEqual(
                    settings.json_config,
                    {"value": {"type": "string", "description": "d"}},
                )
        await stack.aclose()

    async def test_file_delivery_profile_hint_is_not_engine_setting(self):
        config = """
[agent]
role = \"assistant\"

[engine]
uri = \"ai://local/model\"
file_delivery_profile = \"multimodal\"
"""
        with TemporaryDirectory() as tmp:
            path = f"{tmp}/agent.toml"
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(config)

            hub = MagicMock(spec=HuggingfaceHub)
            logger = MagicMock(spec=Logger)
            stack = AsyncExitStack()

            with patch.object(
                OrchestratorLoader,
                "from_settings",
                new=AsyncMock(return_value="orch"),
            ) as lfs_patch:
                loader = OrchestratorLoader(
                    hub=hub,
                    logger=logger,
                    participant_id=uuid4(),
                    stack=stack,
                )
                result = await loader.from_file(path, agent_id=uuid4())

                self.assertEqual(result, "orch")
                settings = lfs_patch.call_args.args[0]
                self.assertEqual(settings.engine_config, {})
            await stack.aclose()

    async def test_invalid_file_delivery_profile_hint_rejects(self):
        for raw_value in ('"binary"', "42"):
            with self.subTest(raw_value=raw_value):
                config = f"""
[agent]
role = \"assistant\"

[engine]
uri = \"ai://local/model\"
file_delivery_profile = {raw_value}
"""
                with TemporaryDirectory() as tmp:
                    path = f"{tmp}/agent.toml"
                    with open(path, "w", encoding="utf-8") as fh:
                        fh.write(config)

                    hub = MagicMock(spec=HuggingfaceHub)
                    logger = MagicMock(spec=Logger)
                    stack = AsyncExitStack()
                    loader = OrchestratorLoader(
                        hub=hub,
                        logger=logger,
                        participant_id=uuid4(),
                        stack=stack,
                    )

                    with self.assertRaises(AssertionError):
                        await loader.from_file(path, agent_id=uuid4())
                    with self.assertRaises(AssertionError):
                        run_async(OrchestratorLoader.validate_agent_file(path))
                    await stack.aclose()

    async def test_hosted_file_delivery_profile_hint_rejects(self):
        config = """
[agent]
role = \"assistant\"

[engine]
uri = \"ai://openai/deployment\"
file_delivery_profile = \"multimodal\"
"""
        with TemporaryDirectory() as tmp:
            path = f"{tmp}/agent.toml"
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(config)

            hub = MagicMock(spec=HuggingfaceHub)
            logger = MagicMock(spec=Logger)
            stack = AsyncExitStack()
            loader = OrchestratorLoader(
                hub=hub,
                logger=logger,
                participant_id=uuid4(),
                stack=stack,
            )

            with self.assertRaisesRegex(AssertionError, "local models"):
                await loader.from_file(path, agent_id=uuid4())
            with self.assertRaisesRegex(AssertionError, "local models"):
                run_async(OrchestratorLoader.validate_agent_file(path))
            await stack.aclose()

    async def test_run_chat_settings_from_file(self):
        config = """
[agent]
role = \"assistant\"

[engine]
uri = \"ai://local/model\"

[run.chat]
enable_thinking = true
"""
        with TemporaryDirectory() as tmp:
            path = f"{tmp}/agent.toml"
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(config)

            hub = MagicMock(spec=HuggingfaceHub)
            logger = MagicMock(spec=Logger)
            stack = AsyncExitStack()

            with patch.object(
                OrchestratorLoader,
                "from_settings",
                new=AsyncMock(return_value="orch"),
            ) as lfs_patch:
                loader = OrchestratorLoader(
                    hub=hub,
                    logger=logger,
                    participant_id=uuid4(),
                    stack=stack,
                )
                result = await loader.from_file(
                    path,
                    agent_id=uuid4(),
                )

                self.assertEqual(result, "orch")
                lfs_patch.assert_awaited_once()
                settings = lfs_patch.call_args.args[0]
                self.assertTrue(
                    settings.call_options["chat_settings"]["enable_thinking"]
                )
            await stack.aclose()

    async def test_run_reasoning_settings_from_file(self):
        config = """
[agent]
role = \"assistant\"

[engine]
uri = \"ai://local/model\"

[run.reasoning]
effort = \"xhigh\"
summary = \"detailed\"
max_new_tokens = 77
enabled = true
stop_on_max_new_tokens = true
tag = \"channel\"
"""
        with TemporaryDirectory() as tmp:
            path = f"{tmp}/agent.toml"
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(config)

            hub = MagicMock(spec=HuggingfaceHub)
            logger = MagicMock(spec=Logger)
            stack = AsyncExitStack()

            with patch.object(
                OrchestratorLoader,
                "from_settings",
                new=AsyncMock(return_value="orch"),
            ) as lfs_patch:
                loader = OrchestratorLoader(
                    hub=hub,
                    logger=logger,
                    participant_id=uuid4(),
                    stack=stack,
                )
                result = await loader.from_file(
                    path,
                    agent_id=uuid4(),
                )

                self.assertEqual(result, "orch")
                lfs_patch.assert_awaited_once()
                settings = lfs_patch.call_args.args[0]
                self.assertEqual(
                    settings.call_options["reasoning"]["effort"], "xhigh"
                )
                self.assertEqual(
                    settings.call_options["reasoning"],
                    {
                        "effort": "xhigh",
                        "summary": ReasoningSummaryMode.DETAILED,
                        "max_new_tokens": 77,
                        "enabled": True,
                        "stop_on_max_new_tokens": True,
                        "tag": "channel",
                    },
                )
            await stack.aclose()

    async def test_run_reasoning_summary_override_preserves_toml_siblings(
        self,
    ):
        config = """
[agent]
role = \"assistant\"

[engine]
uri = \"ai://local/model\"

[run.reasoning]
effort = \"high\"
summary = \"auto\"
max_new_tokens = 45
enabled = true
stop_on_max_new_tokens = true
tag = \"think\"
"""
        with TemporaryDirectory() as tmp:
            path = f"{tmp}/agent.toml"
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(config)

            stack = AsyncExitStack()
            with patch.object(
                OrchestratorLoader,
                "from_settings",
                new=AsyncMock(return_value="orch"),
            ) as from_settings:
                loader = OrchestratorLoader(
                    hub=MagicMock(spec=HuggingfaceHub),
                    logger=MagicMock(spec=Logger),
                    participant_id=uuid4(),
                    stack=stack,
                )
                result = await loader.from_file(
                    path,
                    agent_id=uuid4(),
                    call_options_override={
                        "reasoning": {
                            "summary": ReasoningSummaryMode.CONCISE,
                        }
                    },
                )

                self.assertEqual(result, "orch")
                settings = from_settings.call_args.args[0]
                self.assertEqual(
                    settings.call_options["reasoning"],
                    {
                        "effort": "high",
                        "summary": ReasoningSummaryMode.CONCISE,
                        "max_new_tokens": 45,
                        "enabled": True,
                        "stop_on_max_new_tokens": True,
                        "tag": "think",
                    },
                )
            await stack.aclose()

    async def test_invalid_run_reasoning_summary_from_file(self):
        invalid_values = (
            '"verbose"',
            "1",
            "true",
            '{ mode = "auto" }',
        )
        for value in invalid_values:
            with self.subTest(value=value), TemporaryDirectory() as tmp:
                path = f"{tmp}/agent.toml"
                with open(path, "w", encoding="utf-8") as fh:
                    fh.write(f"""
[agent]
role = \"assistant\"

[engine]
uri = \"ai://local/model\"

[run.reasoning]
summary = {value}
""")

                stack = AsyncExitStack()
                loader = OrchestratorLoader(
                    hub=MagicMock(spec=HuggingfaceHub),
                    logger=MagicMock(spec=Logger),
                    participant_id=uuid4(),
                    stack=stack,
                )
                with self.assertRaisesRegex(
                    AssertionError,
                    "run.reasoning.summary",
                ):
                    await loader.from_file(path, agent_id=uuid4())
                await stack.aclose()

    async def test_run_reasoning_summary_file_entry_points_match(self) -> None:
        invalid_tables = (
            'summary = "verbose"',
            "summary = 1",
            "summary = true",
            'summary = { mode = "auto" }',
            'summary = "auto"\nenabled = false',
        )
        for reasoning_table in invalid_tables:
            with TemporaryDirectory() as tmp:
                path = f"{tmp}/agent.toml"
                Path(path).write_text(
                    f"""
[agent]
role = "assistant"

[engine]
uri = "ai://local/model"

[run.reasoning]
{reasoning_table}
""",
                    encoding="utf-8",
                )
                stack = AsyncExitStack()
                loader = OrchestratorLoader(
                    hub=MagicMock(spec=HuggingfaceHub),
                    logger=MagicMock(spec=Logger),
                    participant_id=uuid4(),
                    stack=stack,
                )

                with self.assertRaisesRegex(
                    AssertionError,
                    "run.reasoning.summary",
                ):
                    await OrchestratorLoader.validate_agent_file(path)
                with self.assertRaisesRegex(
                    AssertionError,
                    "run.reasoning.summary",
                ):
                    await loader.from_file(path, agent_id=uuid4())
                await stack.aclose()

        OrchestratorLoader.validate_agent_config(
            {
                "agent": {"role": "assistant"},
                "engine": {"uri": "ai://local/model"},
                "run": {"reasoning": {"summary": "auto"}},
            }
        )
        OrchestratorLoader.validate_agent_config(
            {
                "agent": {"role": "assistant"},
                "engine": {"uri": "ai://local/model"},
                "run": {"reasoning": {"effort": "high"}},
            }
        )
        reasoning_without_summary = {"run": {"reasoning": {"effort": "high"}}}
        _normalize_file_run_reasoning(reasoning_without_summary)
        self.assertEqual(
            reasoning_without_summary,
            {"run": {"reasoning": {"effort": "high"}}},
        )

    async def test_permanent_memory_from_file(self):
        config_tmpl = """
[agent]
role = \"assistant\"
task = \"do\"
goal_instructions = \"how\"

[engine]
uri = \"ai://local/model\"

[memory]
permanent = {{ {entries} }}
"""
        cases = [
            {"code": ("dsn1", None)},
            {"code": ("dsn1", None), "docs": ("dsn2", None)},
            {
                "code": ("dsn1", "Code entries"),
                "docs": ("dsn2", None),
                "more": ("dsn3", "More docs"),
            },
        ]

        for case in cases:
            with self.subTest(case=case):
                entries = ", ".join(
                    f'{k} = "{dsn}{"," + desc if desc else ""}"'
                    for k, (dsn, desc) in case.items()
                )
                config = config_tmpl.format(entries=entries)
                with TemporaryDirectory() as tmp:
                    path = f"{tmp}/agent.toml"
                    with open(path, "w", encoding="utf-8") as fh:
                        fh.write(config)

                    stack = AsyncExitStack()
                    with patch.object(
                        OrchestratorLoader,
                        "from_settings",
                        new=AsyncMock(return_value="orch"),
                    ) as lfs_patch:
                        loader = OrchestratorLoader(
                            hub=MagicMock(spec=HuggingfaceHub),
                            logger=MagicMock(spec=Logger),
                            participant_id=uuid4(),
                            stack=stack,
                        )
                        result = await loader.from_file(
                            path,
                            agent_id=uuid4(),
                        )

                        self.assertEqual(result, "orch")
                        lfs_patch.assert_awaited_once()
                        settings = lfs_patch.call_args.args[0]
                        expected = {
                            k: PermanentMemoryStoreSettings(
                                dsn=dsn, description=desc
                            )
                            for k, (dsn, desc) in case.items()
                        }
                        self.assertEqual(settings.permanent_memory, expected)
                    await stack.aclose()

    async def test_engine_only_generates_id(self):
        config = """
[agent]

[engine]
uri = \"ai://local/model\"
"""
        uid = uuid4()
        with TemporaryDirectory() as tmp:
            path = f"{tmp}/agent.toml"
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(config)

            with (
                patch.object(
                    OrchestratorLoader,
                    "from_settings",
                    new=AsyncMock(return_value="orch"),
                ) as lfs_patch,
                patch("avalan.agent.loader.uuid4", return_value=uid),
            ):
                stack = AsyncExitStack()
                loader = OrchestratorLoader(
                    hub=MagicMock(spec=HuggingfaceHub),
                    logger=MagicMock(spec=Logger),
                    participant_id=uuid4(),
                    stack=stack,
                )
                result = await loader.from_file(str(path), agent_id=None)

                self.assertEqual(result, "orch")
                lfs_patch.assert_awaited_once()
                settings = lfs_patch.call_args.args[0]
                self.assertEqual(settings.agent_id, uid)
                self.assertEqual(settings.uri, "ai://local/model")
            await stack.aclose()

    async def test_engine_with_id(self):
        uid = uuid4()
        config = f"""
[agent]
id = \"{uid}\"

[engine]
uri = \"ai://local/model\"
"""
        with TemporaryDirectory() as tmp:
            path = f"{tmp}/agent.toml"
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(config)

            with patch.object(
                OrchestratorLoader,
                "from_settings",
                new=AsyncMock(return_value="orch"),
            ) as lfs_patch:
                stack = AsyncExitStack()
                loader = OrchestratorLoader(
                    hub=MagicMock(spec=HuggingfaceHub),
                    logger=MagicMock(spec=Logger),
                    participant_id=uuid4(),
                    stack=stack,
                )
                result = await loader.from_file(path, agent_id=None)

                self.assertEqual(result, "orch")
                lfs_patch.assert_awaited_once()
                settings = lfs_patch.call_args.args[0]
                self.assertEqual(settings.agent_id, str(uid))
            await stack.aclose()

    async def test_engine_with_name(self):
        config = """
[agent]
name = \"Agent\"

[engine]
uri = \"ai://local/model\"
"""
        with TemporaryDirectory() as tmp:
            path = f"{tmp}/agent.toml"
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(config)

            with patch.object(
                OrchestratorLoader,
                "from_settings",
                new=AsyncMock(return_value="orch"),
            ) as lfs_patch:
                stack = AsyncExitStack()
                loader = OrchestratorLoader(
                    hub=MagicMock(spec=HuggingfaceHub),
                    logger=MagicMock(spec=Logger),
                    participant_id=uuid4(),
                    stack=stack,
                )
                result = await loader.from_file(path, agent_id=None)

                self.assertEqual(result, "orch")
                lfs_patch.assert_awaited_once()
                settings = lfs_patch.call_args.args[0]
                self.assertEqual(settings.agent_config.get("name"), "Agent")
            await stack.aclose()

    async def test_engine_generation_settings(self):
        config = """
[agent]

[engine]
uri = \"ai://local/model\"

[run]
temperature = 0.5
top_p = 0.9
top_k = 5
max_new_tokens = 42
openai_max_retries = 0
openai_response_failed_retries = 0
openai_response_failed_retry_delay_seconds = 0.5
openai_timeout_seconds = 30
maximum_tool_cycles = 64
block_repeated_tool_calls = true
"""
        with TemporaryDirectory() as tmp:
            path = f"{tmp}/agent.toml"
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(config)

            with patch.object(
                OrchestratorLoader,
                "from_settings",
                new=AsyncMock(return_value="orch"),
            ) as lfs_patch:
                stack = AsyncExitStack()
                loader = OrchestratorLoader(
                    hub=MagicMock(spec=HuggingfaceHub),
                    logger=MagicMock(spec=Logger),
                    participant_id=uuid4(),
                    stack=stack,
                )
                result = await loader.from_file(path, agent_id=None)

                self.assertEqual(result, "orch")
                lfs_patch.assert_awaited_once()
                settings = lfs_patch.call_args.args[0]
                self.assertEqual(settings.call_options["temperature"], 0.5)
                self.assertEqual(settings.call_options["top_p"], 0.9)
                self.assertEqual(settings.call_options["top_k"], 5)
                self.assertEqual(settings.call_options["max_new_tokens"], 42)
                self.assertEqual(
                    settings.call_options["openai_max_retries"],
                    0,
                )
                self.assertEqual(
                    settings.call_options["openai_response_failed_retries"],
                    0,
                )
                self.assertEqual(
                    settings.call_options[
                        "openai_response_failed_retry_delay_seconds"
                    ],
                    0.5,
                )
                self.assertEqual(
                    settings.call_options["openai_timeout_seconds"],
                    30,
                )
                self.assertEqual(
                    settings.call_options["maximum_tool_cycles"], 64
                )
                self.assertTrue(
                    settings.call_options["block_repeated_tool_calls"]
                )
            await stack.aclose()

    async def test_engine_generation_settings_unlimited_tool_cycles(self):
        config = """
[agent]

[engine]
uri = \"ai://local/model\"

[run]
maximum_tool_cycles = \"unlimited\"
"""
        with TemporaryDirectory() as tmp:
            path = f"{tmp}/agent.toml"
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(config)

            with patch.object(
                OrchestratorLoader,
                "from_settings",
                new=AsyncMock(return_value="orch"),
            ) as lfs_patch:
                stack = AsyncExitStack()
                loader = OrchestratorLoader(
                    hub=MagicMock(spec=HuggingfaceHub),
                    logger=MagicMock(spec=Logger),
                    participant_id=uuid4(),
                    stack=stack,
                )
                result = await loader.from_file(path, agent_id=None)

                self.assertEqual(result, "orch")
                lfs_patch.assert_awaited_once()
                settings = lfs_patch.call_args.args[0]
                self.assertEqual(
                    settings.call_options["maximum_tool_cycles"],
                    UNLIMITED_TOOL_CYCLES,
                )
            await stack.aclose()

    def test_blueprint_renders_maximum_tool_cycles(self):
        template_dir = (
            Path(__file__).resolve().parents[2]
            / "src"
            / "avalan"
            / "agent"
            / "templates"
        )
        env = Environment(
            loader=FileSystemLoader(template_dir),
            trim_blocks=True,
            lstrip_blocks=True,
        )
        env.filters["toml_value"] = agent_cmds._toml_template_value
        template = env.get_template("blueprint.toml")
        cases = [
            (64, 64, True),
            (UNLIMITED_TOOL_CYCLES, UNLIMITED_TOOL_CYCLES, False),
        ]
        retry_delay_key = "openai_response_failed_retry_delay_seconds"

        for value, expected, block_repeated_tool_calls in cases:
            with self.subTest(
                value=value,
                block_repeated_tool_calls=block_repeated_tool_calls,
            ):
                rendered = template.render(
                    orchestrator=Namespace(
                        agent_config={},
                        memory_recent=False,
                        memory_permanent_message=None,
                        sentence_model_id="sentence-model",
                        sentence_model_max_tokens=200,
                        sentence_model_overlap_size=20,
                        sentence_model_window_size=40,
                        uri="ai://local/model",
                        call_options={
                            "max_new_tokens": 42,
                            "skip_special_tokens": False,
                            "maximum_tool_cycles": value,
                            "block_repeated_tool_calls": (
                                block_repeated_tool_calls
                            ),
                            "openai_max_retries": 0,
                            "openai_response_failed_retries": 0,
                            retry_delay_key: 0.5,
                            "openai_timeout_seconds": 30,
                        },
                        tools=[],
                    ),
                    tool_format=None,
                    tool_recovery_formats=None,
                    tool_name_policy=None,
                    skills_tool=None,
                    browser_tool=None,
                    graph_tool=None,
                    database_tool=None,
                    container_tool=None,
                    sandbox_tool=None,
                    shell_tool=None,
                    shell_sandbox=None,
                    shell_container=None,
                )

                config = tomllib.loads(rendered)

                self.assertEqual(
                    config["run"]["maximum_tool_cycles"], expected
                )
                self.assertEqual(
                    config["run"]["block_repeated_tool_calls"],
                    block_repeated_tool_calls,
                )
                self.assertEqual(
                    config["run"]["openai_max_retries"],
                    0,
                )
                self.assertEqual(
                    config["run"]["openai_response_failed_retries"],
                    0,
                )
                self.assertEqual(
                    config["run"][retry_delay_key],
                    0.5,
                )
                self.assertEqual(
                    config["run"]["openai_timeout_seconds"],
                    30,
                )

    async def test_blueprint_reasoning_literal_round_trips_through_loader(
        self,
    ) -> None:
        template_dir = (
            Path(__file__).resolve().parents[2]
            / "src"
            / "avalan"
            / "agent"
            / "templates"
        )
        env = Environment(
            loader=FileSystemLoader(template_dir),
            trim_blocks=True,
            lstrip_blocks=True,
        )
        env.filters["toml_value"] = agent_cmds._toml_template_value
        template = env.get_template("blueprint.toml")

        def render(call_options: dict[str, object]) -> str:
            return template.render(
                orchestrator=Namespace(
                    agent_config={},
                    memory_recent=False,
                    memory_permanent_message=None,
                    sentence_model_id="sentence-model",
                    sentence_model_max_tokens=200,
                    sentence_model_overlap_size=20,
                    sentence_model_window_size=40,
                    uri="ai://local/model",
                    call_options=call_options,
                    tools=[],
                ),
                tool_format=None,
                tool_recovery_formats=None,
                tool_name_policy=None,
                skills_tool=None,
                browser_tool=None,
                graph_tool=None,
                database_tool=None,
                container_tool=None,
                sandbox_tool=None,
                shell_tool=None,
                shell_sandbox=None,
                shell_container=None,
            )

        expected_reasoning = {
            "effort": "xhigh",
            "summary": ReasoningSummaryMode.DETAILED,
            "max_new_tokens": 77,
            "enabled": True,
            "stop_on_max_new_tokens": True,
            "tag": "channel",
        }
        rendered = render(
            {
                "max_new_tokens": 42,
                "skip_special_tokens": False,
                "reasoning": expected_reasoning,
            }
        )
        self.assertNotIn("[run.reasoning]", render({}))

        with TemporaryDirectory() as tmp:
            path = f"{tmp}/agent.toml"
            Path(path).write_text(rendered, encoding="utf-8")

            validated = await OrchestratorLoader.validate_agent_file(path)
            self.assertEqual(
                validated["run"]["reasoning"],
                expected_reasoning,
            )
            self.assertIs(
                validated["run"]["reasoning"]["summary"],
                ReasoningSummaryMode.DETAILED,
            )

            stack = AsyncExitStack()
            with patch.object(
                OrchestratorLoader,
                "from_settings",
                new=AsyncMock(return_value="orch"),
            ) as from_settings:
                loader = OrchestratorLoader(
                    hub=MagicMock(spec=HuggingfaceHub),
                    logger=MagicMock(spec=Logger),
                    participant_id=uuid4(),
                    stack=stack,
                )
                result = await loader.from_file(path, agent_id=uuid4())

            self.assertEqual(result, "orch")
            settings = from_settings.call_args.args[0]
            self.assertEqual(
                settings.call_options["reasoning"],
                expected_reasoning,
            )
            self.assertIs(
                settings.call_options["reasoning"]["summary"],
                ReasoningSummaryMode.DETAILED,
            )
            await stack.aclose()

    async def test_run_response_format_schema_ref_is_resolved(self):
        config = """
[agent]

[engine]
uri = \"ai://local/model\"

[run.response_format]
type = \"json_schema\"
name = \"answer\"
schema_ref = \"schemas/answer.json\"
strict = true
"""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            schema_dir = root / "schemas"
            path = root / "agent.toml"
            schema_dir.mkdir()
            with open(schema_dir / "answer.json", "w", encoding="utf-8") as fh:
                fh.write('{"type": "object"}')
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(config)

            with patch.object(
                OrchestratorLoader,
                "from_settings",
                new=AsyncMock(return_value="orch"),
            ) as lfs_patch:
                stack = AsyncExitStack()
                loader = OrchestratorLoader(
                    hub=MagicMock(spec=HuggingfaceHub),
                    logger=MagicMock(spec=Logger),
                    participant_id=uuid4(),
                    stack=stack,
                )
                result = await loader.from_file(path, agent_id=None)

                self.assertEqual(result, "orch")
                settings = lfs_patch.call_args.args[0]
                response_format = settings.call_options["response_format"]
                self.assertEqual(
                    response_format["schema"],
                    {"type": "object"},
                )
                self.assertNotIn("schema_ref", response_format)
            await stack.aclose()

    async def test_chat_style_response_format_schema_ref_is_resolved(self):
        config = """
[agent]

[engine]
uri = \"ai://local/model\"

[run.response_format]
type = \"json_schema\"

[run.response_format.json_schema]
name = \"answer\"
schema_ref = \"schemas/answer.json\"
"""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            schema_dir = root / "schemas"
            path = root / "agent.toml"
            schema_dir.mkdir()
            with open(schema_dir / "answer.json", "w", encoding="utf-8") as fh:
                fh.write('{"type": "object"}')
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(config)

            with patch.object(
                OrchestratorLoader,
                "from_settings",
                new=AsyncMock(return_value="orch"),
            ) as lfs_patch:
                stack = AsyncExitStack()
                loader = OrchestratorLoader(
                    hub=MagicMock(spec=HuggingfaceHub),
                    logger=MagicMock(spec=Logger),
                    participant_id=uuid4(),
                    stack=stack,
                )
                await loader.from_file(str(path), agent_id=None)

                settings = lfs_patch.call_args.args[0]
                json_schema = settings.call_options["response_format"][
                    "json_schema"
                ]
                self.assertEqual(json_schema["schema"], {"type": "object"})
                self.assertNotIn("schema_ref", json_schema)
            await stack.aclose()

    async def test_response_format_schema_ref_rejects_escape_safely(self):
        config = """
[agent]

[engine]
uri = \"ai://local/model\"

[run.response_format]
type = \"json_schema\"
name = \"answer\"
schema_ref = \"../private/answer.json\"
"""
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "agent.toml"
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(config)
            stack = AsyncExitStack()
            loader = OrchestratorLoader(
                hub=MagicMock(spec=HuggingfaceHub),
                logger=MagicMock(spec=Logger),
                participant_id=uuid4(),
                stack=stack,
            )

            with self.assertRaises(AssertionError) as error:
                await loader.from_file(str(path), agent_id=None)

            self.assertIn(
                "run.response_format.schema_ref",
                str(error.exception),
            )
            self.assertNotIn("private/answer", str(error.exception))
            await stack.aclose()


class LoaderFromSettingsTestCase(IsolatedAsyncioTestCase):
    async def test_from_settings_rejects_invalid_event_manager_mode(self):
        loader = OrchestratorLoader(
            hub=MagicMock(spec=HuggingfaceHub),
            logger=MagicMock(spec=Logger),
            participant_id=uuid4(),
            stack=AsyncExitStack(),
        )

        with self.assertRaises(AssertionError):
            await loader.from_settings(
                MagicMock(),
                event_manager_mode=cast(Any, "server"),
            )

    async def test_shell_toolset_is_not_registered_without_shell_opt_in(self):
        kwargs = await _from_settings_tool_manager_kwargs(
            _orchestrator_settings(tools=None)
        )

        self.assertEqual(_shell_namespaces(kwargs), [])
        self.assertIsNone(kwargs["enable_tools"])

    async def test_shell_toolset_is_not_registered_for_empty_or_nonmatch(
        self,
    ):
        cases = (
            ([], []),
            (["shellx.*"], ["shellx.*"]),
        )

        for tools, expected_enable in cases:
            with self.subTest(tools=tools):
                kwargs = await _from_settings_tool_manager_kwargs(
                    _orchestrator_settings(tools=tools)
                )

                self.assertEqual(_shell_namespaces(kwargs), [])
                self.assertEqual(kwargs["enable_tools"], expected_enable)

    async def test_shell_toolset_is_registered_for_shell_selections(self):
        cases = (
            (["shell"], ["shell"]),
            (["shell.*"], ["shell"]),
            (["shell.rg"], ["shell.rg"]),
        )

        for tools, expected_enable in cases:
            with self.subTest(tools=tools):
                kwargs = await _from_settings_tool_manager_kwargs(
                    _orchestrator_settings(tools=tools)
                )

                self.assertEqual(_shell_namespaces(kwargs), ["shell"])
                self.assertEqual(kwargs["enable_tools"], expected_enable)

    async def test_shell_toolset_is_registered_for_explicit_empty_selection(
        self,
    ):
        kwargs = await _from_settings_tool_manager_kwargs(
            _orchestrator_settings(tools=[]),
            tool_settings=ToolSettingsContext(shell=ShellToolSettings()),
        )

        self.assertEqual(_shell_namespaces(kwargs), ["shell"])
        self.assertEqual(kwargs["enable_tools"], [])

    async def test_shell_toolset_is_registered_for_settings_context(self):
        kwargs = await _from_settings_tool_manager_kwargs(
            _orchestrator_settings(tools=None),
            tool_settings=ToolSettingsContext(shell=ShellToolSettings()),
        )

        self.assertEqual(_shell_namespaces(kwargs), ["shell"])
        self.assertIsNone(kwargs["enable_tools"])

    async def test_shell_toolset_uses_isolation_runtime_from_settings_context(
        self,
    ):
        isolation_settings = IsolationSettings.from_dict(
            {
                "mode": "sandbox",
                "sandbox": {
                    "backend": "seatbelt",
                    "default_profile": "host-tools",
                    "allowed_profiles": ["host-tools"],
                    "profiles": {
                        "host-tools": {
                            "trusted_executables": ["/bin/cat"],
                        },
                    },
                },
            },
            source=trusted_isolation_source("sdk"),
        )
        runtime = IsolationToolRuntimeSettings(
            effective_settings=isolation_settings.select_profile(
                IsolationProfileSelection(
                    mode="sandbox",
                    profile="host-tools",
                    required=True,
                )
            )
        )
        kwargs = await _from_settings_tool_manager_kwargs(
            _orchestrator_settings(tools=None),
            tool_settings=ToolSettingsContext(
                isolation=runtime,
                shell=ShellToolSettings(execution_mode="sandbox"),
            ),
        )

        self.assertEqual(_shell_namespaces(kwargs), ["shell"])
        self.assertIsNone(kwargs["enable_tools"])

    async def test_sandbox_shell_requires_isolation_runtime(self):
        with self.assertRaisesRegex(
            AssertionError,
            "tool.shell backend sandbox requires tool.sandbox settings",
        ):
            await _from_settings_tool_manager_kwargs(
                _orchestrator_settings(tools=["shell.pdfinfo"]),
                tool_settings=ToolSettingsContext(
                    shell=ShellToolSettings(execution_mode="sandbox"),
                ),
            )

    async def test_mcp_toolset_is_not_registered_without_mcp_opt_in(self):
        cases = (
            (None, None),
            ([], []),
            (["mcpx.*"], ["mcpx.*"]),
            (["shell.rg"], ["shell.rg"]),
        )

        for tools, expected_enable in cases:
            with self.subTest(tools=tools):
                kwargs = await _from_settings_tool_manager_kwargs(
                    _orchestrator_settings(tools=tools)
                )

                self.assertEqual(_mcp_namespaces(kwargs), [])
                self.assertEqual(kwargs["enable_tools"], expected_enable)

    async def test_mcp_toolset_is_registered_for_mcp_selections(self):
        cases = (
            (["mcp"], ["mcp"]),
            (["mcp.*"], ["mcp.*"]),
            (["mcp.call"], ["mcp.call"]),
        )

        for tools, expected_enable in cases:
            with self.subTest(tools=tools):
                kwargs = await _from_settings_tool_manager_kwargs(
                    _orchestrator_settings(tools=tools)
                )

                self.assertEqual(_mcp_namespaces(kwargs), ["mcp"])
                self.assertEqual(kwargs["enable_tools"], expected_enable)

    async def test_a2a_toolset_is_not_registered_without_a2a_opt_in(self):
        cases = (
            (None, None),
            ([], []),
            (["a2ax.*"], ["a2ax.*"]),
            (["mcp.call"], ["mcp.call"]),
        )

        for tools, expected_enable in cases:
            with self.subTest(tools=tools):
                kwargs = await _from_settings_tool_manager_kwargs(
                    _orchestrator_settings(tools=tools)
                )

                self.assertEqual(_a2a_namespaces(kwargs), [])
                self.assertEqual(kwargs["enable_tools"], expected_enable)

    async def test_a2a_toolset_is_registered_for_a2a_selections(self):
        cases = (
            (["a2a"], ["a2a"]),
            (["a2a.*"], ["a2a.*"]),
            (["a2a.call"], ["a2a.call"]),
        )

        for tools, expected_enable in cases:
            with self.subTest(tools=tools):
                kwargs = await _from_settings_tool_manager_kwargs(
                    _orchestrator_settings(tools=tools)
                )

                self.assertEqual(_a2a_namespaces(kwargs), ["a2a"])
                self.assertEqual(kwargs["enable_tools"], expected_enable)

    async def test_cli_shell_settings_preserve_default_toolsets(self):
        kwargs = await _from_settings_tool_manager_kwargs(
            _orchestrator_settings(tools=None),
            tool_settings=ToolSettingsContext(shell=ShellToolSettings()),
        )

        self.assertEqual(
            _toolset_namespaces(kwargs),
            ["math", "memory", "shell"],
        )
        self.assertIsNone(kwargs["enable_tools"])

    async def test_shell_diagnostics_distinguish_disabled_from_unknown(self):
        available_kwargs = await _from_settings_tool_manager_kwargs(
            _orchestrator_settings(tools=["shell.rg"])
        )
        available_manager = _shell_only_manager(available_kwargs)
        disabled = available_manager.resolve_tool_name("shell.cat")
        disabled_diagnostic = available_manager.validate_tool_call(
            ToolCall(id="call-1", name="shell.cat", arguments={})
        )

        self.assertIs(disabled.status, ToolNameResolutionStatus.DISABLED)
        self.assertIs(
            disabled.diagnostic_code,
            ToolCallDiagnosticCode.DISABLED_TOOL,
        )
        assert disabled_diagnostic is not None
        self.assertIs(
            disabled_diagnostic.code,
            ToolCallDiagnosticCode.DISABLED_TOOL,
        )

        unavailable_kwargs = await _from_settings_tool_manager_kwargs(
            _orchestrator_settings(tools=None)
        )
        unavailable_manager = _shell_only_manager(unavailable_kwargs)
        unknown = unavailable_manager.resolve_tool_name("shell.cat")
        unknown_diagnostic = unavailable_manager.validate_tool_call(
            ToolCall(id="call-1", name="shell.cat", arguments={})
        )

        self.assertIs(unknown.status, ToolNameResolutionStatus.UNKNOWN)
        self.assertIs(
            unknown.diagnostic_code,
            ToolCallDiagnosticCode.UNKNOWN_TOOL,
        )
        assert unknown_diagnostic is not None
        self.assertIs(
            unknown_diagnostic.code,
            ToolCallDiagnosticCode.UNKNOWN_TOOL,
        )

    async def test_load_default_orchestrator_from_settings(self):
        hub = MagicMock(spec=HuggingfaceHub)
        logger = MagicMock(spec=Logger)
        stack = AsyncExitStack()

        sentence_model = MagicMock()
        sentence_model.__enter__.return_value = sentence_model

        model_manager = MagicMock()
        model_manager.__enter__.return_value = model_manager
        model_manager.parse_uri.return_value = "uri_obj"
        model_manager.get_engine_settings.return_value = "settings_obj"

        memory = MagicMock()
        tool = MagicMock()
        event_manager = MagicMock()

        settings = OrchestratorSettings(
            agent_id=uuid4(),
            orchestrator_type=None,
            agent_config={"role": "assistant"},
            uri="ai://local/model",
            engine_config={},
            tools=None,
            call_options=None,
            template_vars=None,
            memory_permanent_message=None,
            permanent_memory=None,
            memory_recent=False,
            sentence_model_id=OrchestratorLoader.DEFAULT_SENTENCE_MODEL_ID,
            sentence_model_engine_config=None,
            sentence_model_max_tokens=500,
            sentence_model_overlap_size=125,
            sentence_model_window_size=250,
            json_config=None,
            log_events=True,
        )

        with (
            patch(
                "avalan.agent.loader.SentenceTransformerModel",
                return_value=sentence_model,
            ),
            patch("avalan.agent.loader.TextPartitioner"),
            patch(
                "avalan.agent.loader.MemoryManager.create_instance",
                new=AsyncMock(return_value=memory),
            ) as mm_patch,
            patch(
                "avalan.agent.loader.ModelManager", return_value=model_manager
            ) as model_patch,
            patch(
                "avalan.agent.loader.DefaultOrchestrator", return_value="orch"
            ) as orch_patch,
            patch(
                "avalan.agent.loader.ToolManager.create_instance",
                return_value=tool,
            ),
            patch(
                "avalan.agent.loader.EventManager", return_value=event_manager
            ),
        ):
            loader = OrchestratorLoader(
                hub=hub,
                logger=logger,
                participant_id=uuid4(),
                stack=stack,
            )
            result = await loader.from_settings(settings)

            self.assertEqual(result, "orch")
            orch_patch.assert_called_once()
            model_patch.assert_called_once_with(
                hub, logger, event_manager=event_manager
            )
            mm_patch.assert_awaited_once()
        await stack.aclose()

    async def test_permanent_memory_from_settings(self):
        base_settings = dict(
            agent_id=uuid4(),
            orchestrator_type=None,
            agent_config={"role": "assistant"},
            uri="ai://local/model",
            engine_config={},
            tools=None,
            call_options=None,
            template_vars=None,
            memory_permanent_message=None,
            memory_recent=False,
            sentence_model_id=OrchestratorLoader.DEFAULT_SENTENCE_MODEL_ID,
            sentence_model_engine_config=None,
            sentence_model_max_tokens=500,
            sentence_model_overlap_size=125,
            sentence_model_window_size=250,
            json_config=None,
            log_events=True,
        )

        cases = [
            {
                "code": PermanentMemoryStoreSettings(
                    dsn="dsn1", description=None
                )
            },
            {
                "code": PermanentMemoryStoreSettings(
                    dsn="dsn1", description="Code store"
                ),
                "docs": PermanentMemoryStoreSettings(
                    dsn="dsn2", description=None
                ),
            },
            {
                "code": PermanentMemoryStoreSettings(
                    dsn="dsn1", description="Code entries"
                ),
                "docs": PermanentMemoryStoreSettings(
                    dsn="dsn2", description="Docs store"
                ),
                "more": PermanentMemoryStoreSettings(
                    dsn="dsn3", description=None
                ),
            },
        ]

        for case in cases:
            with self.subTest(case=case):
                hub = MagicMock(spec=HuggingfaceHub)
                logger = MagicMock(spec=Logger)
                stack = AsyncExitStack()

                sentence_model = MagicMock()
                sentence_model.__enter__.return_value = sentence_model

                model_manager = MagicMock()
                model_manager.__enter__.return_value = model_manager
                model_manager.parse_uri.return_value = "uri_obj"
                model_manager.get_engine_settings.return_value = "settings_obj"

                memory = MagicMock()
                tool = MagicMock()
                event_manager = MagicMock()

                settings = OrchestratorSettings(
                    **base_settings,
                    permanent_memory=case,
                )

                store_instances = [MagicMock() for _ in case]
                pgsql_raw_memory = MagicMock()
                pgsql_raw_memory.create_instance = AsyncMock(
                    side_effect=store_instances
                )
                with (
                    patch(
                        "avalan.agent.loader.SentenceTransformerModel",
                        return_value=sentence_model,
                    ),
                    patch("avalan.agent.loader.TextPartitioner"),
                    patch(
                        "avalan.agent.loader.MemoryManager.create_instance",
                        new=AsyncMock(return_value=memory),
                    ),
                    patch(
                        "avalan.agent.loader.ModelManager",
                        return_value=model_manager,
                    ),
                    patch(
                        "avalan.agent.loader.PgsqlRawMemory",
                        pgsql_raw_memory,
                    ),
                    patch(
                        "avalan.agent.loader.DefaultOrchestrator",
                        return_value="orch",
                    ),
                    patch(
                        "avalan.agent.loader.ToolManager.create_instance",
                        return_value=tool,
                    ),
                    patch(
                        "avalan.agent.loader.EventManager",
                        return_value=event_manager,
                    ),
                ):
                    loader = OrchestratorLoader(
                        hub=hub,
                        logger=logger,
                        participant_id=uuid4(),
                        stack=stack,
                    )
                    await loader.from_settings(settings)

                    create_instance = pgsql_raw_memory.create_instance
                    self.assertEqual(create_instance.await_count, len(case))
                    expected_store_calls = [
                        call(dsn=store_settings.dsn, logger=logger)
                        for store_settings in case.values()
                    ]
                    create_instance.assert_has_awaits(expected_store_calls)
                    expected_add_calls = [
                        call(
                            namespace,
                            instance,
                            description=case[namespace].description,
                        )
                        for (namespace, instance) in zip(
                            case.keys(), store_instances
                        )
                    ]
                    self.assertEqual(
                        memory.add_permanent_memory.call_args_list,
                        expected_add_calls,
                    )
                await stack.aclose()

    async def test_database_tool_from_settings(self):
        hub = MagicMock(spec=HuggingfaceHub)
        logger = MagicMock(spec=Logger)
        stack = AsyncExitStack()

        sentence_model = MagicMock()
        sentence_model.__enter__.return_value = sentence_model

        model_manager = MagicMock()
        model_manager.__enter__.return_value = model_manager
        model_manager.parse_uri.return_value = "uri_obj"
        model_manager.get_engine_settings.return_value = "settings_obj"

        memory = MagicMock()
        tool = MagicMock()
        tool.__aenter__.return_value = tool
        event_manager = MagicMock()

        settings = OrchestratorSettings(
            agent_id=uuid4(),
            orchestrator_type=None,
            agent_config={"role": "assistant"},
            uri="ai://local/model",
            engine_config={},
            tools=None,
            call_options=None,
            template_vars=None,
            memory_permanent_message=None,
            permanent_memory=None,
            memory_recent=False,
            sentence_model_id=OrchestratorLoader.DEFAULT_SENTENCE_MODEL_ID,
            sentence_model_engine_config=None,
            sentence_model_max_tokens=500,
            sentence_model_overlap_size=125,
            sentence_model_window_size=250,
            json_config=None,
            log_events=True,
        )

        db_settings = DatabaseToolSettings(dsn="sqlite:///db.sqlite")

        browser_tool = MagicMock()
        code_tool = MagicMock()
        graph_tool = MagicMock()
        math_tool = MagicMock()
        memory_tool = MagicMock()
        db_tool = MagicMock()

        with (
            patch(
                "avalan.agent.loader.SentenceTransformerModel",
                return_value=sentence_model,
            ),
            patch("avalan.agent.loader.TextPartitioner"),
            patch(
                "avalan.agent.loader.MemoryManager.create_instance",
                new=AsyncMock(return_value=memory),
            ),
            patch(
                "avalan.agent.loader.ModelManager", return_value=model_manager
            ),
            patch(
                "avalan.agent.loader.DefaultOrchestrator", return_value="orch"
            ),
            patch(
                "avalan.agent.loader.ToolManager.create_instance",
                return_value=tool,
            ) as tm_patch,
            patch(
                "avalan.agent.loader.EventManager", return_value=event_manager
            ),
            patch(
                "avalan.agent.loader.BrowserToolSet", return_value=browser_tool
            ),
            patch("avalan.agent.loader.CodeToolSet", return_value=code_tool),
            patch(
                "avalan.agent.loader.GraphToolSet", return_value=graph_tool
            ) as graph_patch,
            patch("avalan.agent.loader.HAS_GRAPH_DEPENDENCIES", True),
            patch("avalan.agent.loader.MathToolSet", return_value=math_tool),
            patch(
                "avalan.agent.loader.MemoryToolSet", return_value=memory_tool
            ),
            patch(
                "avalan.agent.loader.DatabaseToolSet",
                return_value=db_tool,
            ) as db_patch,
        ):
            loader = OrchestratorLoader(
                hub=hub,
                logger=logger,
                participant_id=uuid4(),
                stack=stack,
            )
            result = await loader.from_settings(
                settings,
                tool_settings=ToolSettingsContext(database=db_settings),
            )

            self.assertEqual(result, "orch")
            db_patch.assert_called_once_with(
                settings=db_settings, namespace="database"
            )
            available = tm_patch.call_args.kwargs["available_toolsets"]
            graph_patch.assert_called_once_with(
                settings=GraphToolSettings(), namespace="graph"
            )
            self.assertIn(graph_tool, available)
            self.assertIn(db_tool, available)
        await stack.aclose()

    async def test_tool_settings_logged(self):
        hub = MagicMock(spec=HuggingfaceHub)
        logger = MagicMock(spec=Logger)
        stack = AsyncExitStack()

        sentence_model = MagicMock()
        sentence_model.__enter__.return_value = sentence_model

        model_manager = MagicMock()
        model_manager.__enter__.return_value = model_manager
        model_manager.parse_uri.return_value = "uri_obj"
        model_manager.get_engine_settings.return_value = "settings_obj"

        memory = MagicMock()
        tool = MagicMock()
        tool.__aenter__ = AsyncMock(return_value=tool)
        event_manager = MagicMock()

        settings = OrchestratorSettings(
            agent_id=uuid4(),
            orchestrator_type=None,
            agent_config={"role": "assistant"},
            uri="ai://local/model",
            engine_config={},
            tools=None,
            call_options=None,
            template_vars=None,
            memory_permanent_message=None,
            permanent_memory=None,
            memory_recent=False,
            sentence_model_id=OrchestratorLoader.DEFAULT_SENTENCE_MODEL_ID,
            sentence_model_engine_config=None,
            sentence_model_max_tokens=500,
            sentence_model_overlap_size=125,
            sentence_model_window_size=250,
            json_config=None,
            log_events=True,
        )

        browser_settings = BrowserToolSettings()
        db_settings = DatabaseToolSettings(dsn="sqlite:///db.sqlite")
        graph_settings = GraphToolSettings(file="/tmp/chart.png")
        shell_settings = ShellToolSettings()
        container_runtime = ContainerToolRuntimeSettings()

        with (
            patch(
                "avalan.agent.loader.SentenceTransformerModel",
                return_value=sentence_model,
            ),
            patch("avalan.agent.loader.TextPartitioner"),
            patch(
                "avalan.agent.loader.MemoryManager.create_instance",
                new=AsyncMock(return_value=memory),
            ),
            patch(
                "avalan.agent.loader.ModelManager",
                return_value=model_manager,
            ),
            patch(
                "avalan.agent.loader.DefaultOrchestrator",
                return_value="orch",
            ),
            patch(
                "avalan.agent.loader.ToolManager.create_instance",
                return_value=tool,
            ),
            patch(
                "avalan.agent.loader.EventManager", return_value=event_manager
            ),
            patch("avalan.agent.loader.BrowserToolSet"),
            patch("avalan.agent.loader.CodeToolSet") as code_patch,
            patch("avalan.agent.loader.MathToolSet"),
            patch("avalan.agent.loader.McpToolSet"),
            patch("avalan.agent.loader.MemoryToolSet"),
            patch("avalan.agent.loader.DatabaseToolSet"),
        ):
            loader = OrchestratorLoader(
                hub=hub,
                logger=logger,
                participant_id=uuid4(),
                stack=stack,
            )
            result = await loader.from_settings(
                settings,
                tool_settings=ToolSettingsContext(
                    browser=browser_settings,
                    database=db_settings,
                    graph=graph_settings,
                    shell=shell_settings,
                    container=container_runtime,
                ),
            )

            self.assertEqual(result, "orch")
            code_patch.assert_called_once_with(
                container_runtime=container_runtime,
                namespace="code",
            )
            logger.log.assert_any_call(
                DEBUG,
                "<OrchestratorLoader> Tool settings: browser=%s, database=%s,"
                " graph=%s, shell=%s",
                browser_settings,
                db_settings,
                graph_settings,
                shell_settings,
            )
        await stack.aclose()

    async def test_event_logging_listener(self):
        hub = MagicMock(spec=HuggingfaceHub)
        logger = MagicMock(spec=Logger)
        stack = AsyncExitStack()

        sentence_model = MagicMock()
        sentence_model.__enter__.return_value = sentence_model

        model_manager = MagicMock()
        model_manager.__enter__.return_value = model_manager
        model_manager.parse_uri.return_value = "uri_obj"
        model_manager.get_engine_settings.return_value = "settings_obj"

        memory = MagicMock()
        tool = MagicMock()
        tool.__aenter__ = AsyncMock(return_value=tool)
        tool.__aexit__ = AsyncMock(return_value=None)

        class DummyEventManager:
            def __init__(self) -> None:
                self.listeners: list[Callable[[Event], Any]] = []

            def add_listener(
                self,
                listener: Callable[[Event], Any],
                event_types: Any | None = None,
            ) -> None:
                self.listeners.append(listener)

        event_manager = DummyEventManager()

        settings = OrchestratorSettings(
            agent_id=uuid4(),
            orchestrator_type=None,
            agent_config={"role": "assistant"},
            uri="ai://local/model",
            engine_config={},
            tools=None,
            call_options=None,
            template_vars=None,
            memory_permanent_message=None,
            permanent_memory=None,
            memory_recent=False,
            sentence_model_id=OrchestratorLoader.DEFAULT_SENTENCE_MODEL_ID,
            sentence_model_engine_config=None,
            sentence_model_max_tokens=500,
            sentence_model_overlap_size=125,
            sentence_model_window_size=250,
            json_config=None,
            log_events=True,
        )

        with (
            patch(
                "avalan.agent.loader.SentenceTransformerModel",
                return_value=sentence_model,
            ),
            patch("avalan.agent.loader.TextPartitioner"),
            patch(
                "avalan.agent.loader.MemoryManager.create_instance",
                new=AsyncMock(return_value=memory),
            ),
            patch(
                "avalan.agent.loader.ModelManager", return_value=model_manager
            ),
            patch(
                "avalan.agent.loader.DefaultOrchestrator", return_value="orch"
            ),
            patch(
                "avalan.agent.loader.ToolManager.create_instance",
                return_value=tool,
            ),
            patch(
                "avalan.agent.loader.EventManager", return_value=event_manager
            ) as event_manager_patch,
        ):
            loader = OrchestratorLoader(
                hub=hub,
                logger=logger,
                participant_id=uuid4(),
                stack=stack,
            )
            result = await loader.from_settings(settings)

        self.assertEqual(result, "orch")
        event_manager_patch.assert_called_once_with(mode=EventManagerMode.SDK)
        self.assertEqual(len(event_manager.listeners), 1)

        listener = event_manager.listeners[0]
        logger.log.reset_mock()

        tool_event = Event(
            type=EventType.TOOL_PROCESS,
            payload={"message": "tool running"},
        )
        listener(tool_event)
        logger.log.assert_called_once()
        level, message, payload = logger.log.call_args.args
        self.assertEqual(level, INFO)
        self.assertEqual(
            message, "<Event tool_process @ OrchestratorLoader> %s"
        )
        self.assertEqual(payload, tool_event.payload)

        logger.log.reset_mock()

        debug_event = Event(
            type=EventType.START,
            payload={"message": "start"},
        )
        listener(debug_event)
        logger.log.assert_called_once()
        level, message, payload = logger.log.call_args.args
        self.assertEqual(level, DEBUG)
        self.assertEqual(message, "<Event start @ OrchestratorLoader> %s")
        self.assertEqual(payload, debug_event.payload)

        await stack.aclose()

    async def test_load_json_orchestrator_from_settings(self):
        hub = MagicMock(spec=HuggingfaceHub)
        logger = MagicMock(spec=Logger)
        stack = AsyncExitStack()

        sentence_model = MagicMock()
        sentence_model.__enter__.return_value = sentence_model

        model_manager = MagicMock()
        model_manager.__enter__.return_value = model_manager
        model_manager.parse_uri.return_value = "uri_obj"
        model_manager.get_engine_settings.return_value = "settings_obj"

        memory = MagicMock()
        tool = MagicMock()
        event_manager = MagicMock()
        shell_settings = ShellToolSettings(workspace_root="/workspace")

        settings = OrchestratorSettings(
            agent_id=uuid4(),
            orchestrator_type="json",
            agent_config={
                "role": "assistant",
                "task": "do",
                "goal_instructions": "how",
            },
            uri="ai://local/model",
            engine_config={},
            tools=["shell.pdfinfo"],
            call_options=None,
            template_vars=None,
            memory_permanent_message=None,
            permanent_memory=None,
            memory_recent=False,
            sentence_model_id=OrchestratorLoader.DEFAULT_SENTENCE_MODEL_ID,
            sentence_model_engine_config=None,
            sentence_model_max_tokens=500,
            sentence_model_overlap_size=125,
            sentence_model_window_size=250,
            json_config={"value": {"type": "string", "description": "d"}},
            log_events=True,
        )

        with (
            patch(
                "avalan.agent.loader.SentenceTransformerModel",
                return_value=sentence_model,
            ),
            patch("avalan.agent.loader.TextPartitioner"),
            patch(
                "avalan.agent.loader.MemoryManager.create_instance",
                new=AsyncMock(return_value=memory),
            ),
            patch(
                "avalan.agent.loader.ModelManager", return_value=model_manager
            ),
            patch.object(
                OrchestratorLoader,
                "_load_json_orchestrator",
                return_value="json_orch",
            ) as json_patch,
            patch(
                "avalan.agent.loader.ToolManager.create_instance",
                return_value=tool,
            ),
            patch(
                "avalan.agent.loader.EventManager", return_value=event_manager
            ),
        ):
            loader = OrchestratorLoader(
                hub=hub,
                logger=logger,
                participant_id=uuid4(),
                stack=stack,
            )
            result = await loader.from_settings(
                settings,
                tool_settings=ToolSettingsContext(shell=shell_settings),
            )

            self.assertEqual(result, "json_orch")
            json_patch.assert_called_once()
            self.assertIs(
                json_patch.call_args.kwargs["shell_input_file_settings"],
                shell_settings,
            )
        await stack.aclose()

    async def test_load_json_orchestrator_properties(self):
        agent_id = uuid4()
        engine_uri = MagicMock()
        engine_settings = MagicMock()
        logger = MagicMock()
        model_manager = MagicMock()
        memory = MagicMock()
        tool = MagicMock()
        event_manager = MagicMock()
        shell_settings = ShellToolSettings(workspace_root="/workspace")

        config = {
            "json": {
                "name": {"type": "string", "description": "n"},
                "age": {"type": "integer", "description": "a"},
            }
        }

        agent_config = {
            "role": "assistant",
            "task": "do",
            "goal_instructions": "how",
        }

        with patch("avalan.agent.loader.JsonOrchestrator") as orch_patch:
            OrchestratorLoader._load_json_orchestrator(
                agent_id=agent_id,
                engine_uri=engine_uri,
                engine_settings=engine_settings,
                logger=logger,
                model_manager=model_manager,
                memory=memory,
                tool=tool,
                event_manager=event_manager,
                config=config,
                agent_config=agent_config,
                call_options=None,
                template_vars=None,
                shell_input_file_settings=shell_settings,
            )

            orch_patch.assert_called_once()
            self.assertIs(
                orch_patch.call_args.kwargs["shell_input_file_settings"],
                shell_settings,
            )
            properties = orch_patch.call_args.args[6]
            self.assertEqual(len(properties), 2)
            self.assertEqual(properties[0].name, "name")
            self.assertEqual(properties[1].data_type, "integer")


if __name__ == "__main__":
    main()

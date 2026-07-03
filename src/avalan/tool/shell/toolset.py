from ...container import (
    ContainerAsyncBackend,
    ContainerBackend,
    ContainerEffectiveSettings,
    ContainerToolRuntimeSettings,
    disabled_required_container_settings,
)
from ...isolation import (
    IsolationMode,
    IsolationToolRuntimeSettings,
    SandboxEffectiveSettings,
)
from ...sandbox import SandboxAsyncBackend
from .. import Tool, ToolSet
from ..names import matches_tool_namespace
from .composition_executor import (
    BackendBoundaryCompositionExecutor,
    CompositionExecutor,
    LocalCompositionExecutor,
)
from .container import ShellContainerCommandExecutor
from .executor import CommandExecutor, LocalCommandExecutor
from .formatting import (
    format_shell_composition_result,
    format_shell_result,
)
from .git import (
    SHELL_GIT_COMMAND_CAPABILITIES,
    ShellGitCapability,
    ShellGitCommandName,
)
from .opt_in import (
    SHELL_TOOL_NAMESPACE,
    SHELL_TOOL_WILDCARD,
    enables_shell_pipeline,
)
from .policy import ExecutionPolicy
from .process import ShellProcessRuntime
from .sandbox import ShellSandboxCommandExecutor
from .settings import ShellGitToolSettings, ShellToolSettings
from .tools import (
    AwkTool,
    CatTool,
    FileTool,
    FindTool,
    GitAddTool,
    GitBlameTool,
    GitBranchCreateTool,
    GitBranchDeleteTool,
    GitBranchRenameTool,
    GitBranchTool,
    GitCheckoutTool,
    GitCherryPickTool,
    GitCleanTool,
    GitCloneTool,
    GitCommitTool,
    GitDescribeTool,
    GitDiffTool,
    GitFetchTool,
    GitGrepTool,
    GitLogTool,
    GitLsFilesTool,
    GitMergeTool,
    GitMvTool,
    GitPullTool,
    GitPushTool,
    GitRebaseTool,
    GitRemoteAddTool,
    GitRemoteListTool,
    GitRemoteRemoveTool,
    GitRemoteRenameTool,
    GitRemoteSetUrlTool,
    GitResetTool,
    GitRestoreTool,
    GitRevertTool,
    GitRevParseTool,
    GitRmTool,
    GitShowTool,
    GitStashApplyTool,
    GitStashDropTool,
    GitStashListTool,
    GitStashPopTool,
    GitStashPushTool,
    GitStashShowTool,
    GitStatusTool,
    GitSubmoduleUpdateTool,
    GitSwitchTool,
    GitTagCreateTool,
    GitTagDeleteTool,
    GitTagTool,
    HeadTool,
    JqTool,
    LsTool,
    NlTool,
    PdfInfoTool,
    PdfPlumberTool,
    PdfToPpmTool,
    PdfToTextTool,
    PipelineTool,
    PyPdfTool,
    ReportLabTool,
    RgTool,
    SedTool,
    ShellCompositionResultFormatter,
    ShellResultFormatter,
    TailTool,
    TesseractTool,
    WcTool,
)

from collections.abc import Callable, Sequence
from typing import Literal, cast


class ShellToolSet(ToolSet):
    _pipeline_tool: PipelineTool
    _settings: ShellToolSettings
    _process_runtime: ShellProcessRuntime

    def __init__(
        self,
        settings: ShellToolSettings | None = None,
        *,
        executor: CommandExecutor | None = None,
        composition_executor: CompositionExecutor | None = None,
        formatter: ShellResultFormatter | None = None,
        composition_formatter: ShellCompositionResultFormatter | None = None,
        namespace: str | None = "shell",
        policy: ExecutionPolicy | None = None,
        container_runtime: ContainerToolRuntimeSettings | None = None,
        container_settings: ContainerEffectiveSettings | None = None,
        container_backend: ContainerAsyncBackend | None = None,
        container_opt_in_backends: Sequence[ContainerBackend | str] = (),
        container_rootful_authorized: bool = False,
        isolation_runtime: IsolationToolRuntimeSettings | None = None,
        sandbox_settings: SandboxEffectiveSettings | None = None,
        sandbox_backend: SandboxAsyncBackend | None = None,
    ) -> None:
        assert namespace == SHELL_TOOL_NAMESPACE, "namespace must be shell"
        assert isinstance(container_rootful_authorized, bool)
        self._settings = settings or ShellToolSettings()
        self._process_runtime = ShellProcessRuntime(self._settings)
        execution_mode = self._settings.execution_mode
        if container_runtime is not None:
            assert isinstance(container_runtime, ContainerToolRuntimeSettings)
            assert not _container_runtime_hooks_configured(
                container_runtime
            ), "shell container runtime hooks are not supported"
        container_runtime_configured = (
            False
            if container_runtime is None
            else _container_runtime_configured(container_runtime)
        )
        assert not (
            isolation_runtime is not None and container_runtime is not None
        ), "isolation_runtime cannot be combined with container_runtime"
        assert not (
            execution_mode != "sandbox" and sandbox_settings is not None
        ), "sandbox settings require shell execution mode sandbox"
        assert not (
            execution_mode != "container" and container_settings is not None
        ), "container settings require shell execution mode container"
        assert not (
            execution_mode != "container" and container_runtime_configured
        ), "container runtime requires shell execution mode container"
        assert not (
            execution_mode != "local" and executor is not None
        ), "custom shell executors require shell execution mode local"
        assert not (
            execution_mode != "local" and composition_executor is not None
        ), (
            "custom shell composition executors require shell execution mode "
            "local"
        )
        if isolation_runtime is not None:
            assert isinstance(isolation_runtime, IsolationToolRuntimeSettings)
            assert not _isolation_runtime_hooks_configured(
                isolation_runtime
            ), "shell isolation runtime hooks are not supported"
            assert (
                isolation_runtime.mode.value == execution_mode
            ), "isolation runtime mode must match shell execution mode"
            if (
                isolation_runtime.mode is IsolationMode.SANDBOX
                and execution_mode == "sandbox"
            ):
                sandbox_settings = (
                    sandbox_settings or isolation_runtime.sandbox
                )
                sandbox_backend = sandbox_backend or cast(
                    SandboxAsyncBackend | None,
                    isolation_runtime.sandbox_backend,
                )
            if (
                isolation_runtime.mode is IsolationMode.CONTAINER
                and execution_mode == "container"
            ):
                container_settings = (
                    container_settings or isolation_runtime.container
                )
                container_backend = (
                    container_backend or isolation_runtime.container_backend
                )
        if container_runtime is not None and execution_mode == "container":
            container_settings = (
                container_settings or container_runtime.effective_settings
            )
            container_backend = container_backend or container_runtime.backend
            container_opt_in_backends = (
                container_opt_in_backends or container_runtime.opt_in_backends
            )
            container_rootful_authorized = (
                container_rootful_authorized
                or container_runtime.rootful_authorized
            )
        if container_settings is None and execution_mode == "container":
            container_settings = disabled_required_container_settings()
        assert not (
            execution_mode == "sandbox" and container_settings is not None
        ), "sandbox shell execution cannot carry container policy"
        assert not (
            execution_mode == "container" and sandbox_settings is not None
        ), "container shell execution cannot carry sandbox policy"
        policy = policy or ExecutionPolicy(settings=self._settings)
        if executor is None:
            local_executor = LocalCommandExecutor(
                settings=self._settings,
                process_runtime=self._process_runtime,
            )
            if execution_mode == "sandbox":
                executor = ShellSandboxCommandExecutor(
                    sandbox_settings=sandbox_settings,
                    sandbox_backend=sandbox_backend,
                )
            elif container_settings is not None:
                executor = ShellContainerCommandExecutor(
                    container_settings=container_settings,
                    container_backend=container_backend,
                    opt_in_backends=container_opt_in_backends,
                    local_executor=local_executor,
                    rootful_authorized=container_rootful_authorized,
                )
            else:
                executor = local_executor
        formatter = formatter or (
            lambda result: format_shell_result(
                result,
                settings=self._settings,
            )
        )
        composition_executor = composition_executor or (
            LocalCompositionExecutor(
                settings=self._settings,
                command_executor=executor,
                process_runtime=self._process_runtime,
            )
            if execution_mode == "local"
            else BackendBoundaryCompositionExecutor(
                backend=cast(
                    Literal["sandbox", "container"],
                    execution_mode,
                ),
                command_executor=executor,
                settings=self._settings,
            )
        )
        composition_formatter = composition_formatter or (
            lambda result: format_shell_composition_result(
                result,
                settings=self._settings,
            )
        )
        self._pipeline_tool = PipelineTool(
            settings=self._settings,
            policy=policy,
            executor=composition_executor,
            formatter=composition_formatter,
        )
        tools: list[Callable[..., object] | Tool | ToolSet] = [
            RgTool(
                settings=self._settings,
                policy=policy,
                executor=executor,
                formatter=formatter,
            ),
            HeadTool(
                settings=self._settings,
                policy=policy,
                executor=executor,
                formatter=formatter,
            ),
            TailTool(
                settings=self._settings,
                policy=policy,
                executor=executor,
                formatter=formatter,
            ),
            LsTool(
                settings=self._settings,
                policy=policy,
                executor=executor,
                formatter=formatter,
            ),
            CatTool(
                settings=self._settings,
                policy=policy,
                executor=executor,
                formatter=formatter,
            ),
            NlTool(
                settings=self._settings,
                policy=policy,
                executor=executor,
                formatter=formatter,
            ),
            FileTool(
                settings=self._settings,
                policy=policy,
                executor=executor,
                formatter=formatter,
            ),
            FindTool(
                settings=self._settings,
                policy=policy,
                executor=executor,
                formatter=formatter,
            ),
            WcTool(
                settings=self._settings,
                policy=policy,
                executor=executor,
                formatter=formatter,
            ),
            AwkTool(
                settings=self._settings,
                policy=policy,
                executor=executor,
                formatter=formatter,
            ),
            SedTool(
                settings=self._settings,
                policy=policy,
                executor=executor,
                formatter=formatter,
            ),
            JqTool(
                settings=self._settings,
                policy=policy,
                executor=executor,
                formatter=formatter,
            ),
            PdfInfoTool(
                settings=self._settings,
                policy=policy,
                executor=executor,
                formatter=formatter,
            ),
            PdfToTextTool(
                settings=self._settings,
                policy=policy,
                executor=executor,
                formatter=formatter,
            ),
            PdfToPpmTool(
                settings=self._settings,
                policy=policy,
                executor=executor,
                formatter=formatter,
            ),
            ReportLabTool(
                settings=self._settings,
                policy=policy,
                executor=executor,
                formatter=formatter,
            ),
            PdfPlumberTool(
                settings=self._settings,
                policy=policy,
                executor=executor,
                formatter=formatter,
            ),
            PyPdfTool(
                settings=self._settings,
                policy=policy,
                executor=executor,
                formatter=formatter,
            ),
            TesseractTool(
                settings=self._settings,
                policy=policy,
                executor=executor,
                formatter=formatter,
            ),
        ]
        super().__init__(namespace=namespace, tools=tools)

    @property
    def process_runtime(self) -> ShellProcessRuntime:
        return self._process_runtime

    @property
    def available_tools(self) -> list[Callable[..., object] | Tool | ToolSet]:
        tools: list[Callable[..., object] | Tool | ToolSet] = list(self.tools)
        if not _has_pipeline_tool(tools):
            tools.append(self._pipeline_tool)
        return tools

    def available_tools_for_enabled_tools(
        self,
        enable_tools: Sequence[str],
    ) -> list[Callable[..., object] | Tool | ToolSet]:
        tools = self.available_tools
        _append_missing_tools(
            tools,
            _available_git_tools_for_selection(self._settings, enable_tools),
        )
        return tools

    def with_enabled_tools(self, enable_tools: list[str]) -> "ShellToolSet":
        if enables_shell_pipeline(
            enable_tools, self._settings
        ) and not _has_pipeline_tool(self.tools):
            self.tools.append(self._pipeline_tool)
        if _enables_shell_git_tools(enable_tools):
            _append_missing_tools(
                self.tools,
                _authorized_git_tools(self._settings),
            )
        return cast(ShellToolSet, super().with_enabled_tools(enable_tools))


def _container_runtime_configured(
    runtime: ContainerToolRuntimeSettings,
) -> bool:
    assert isinstance(runtime, ContainerToolRuntimeSettings)
    return (
        runtime.effective_settings is not None
        or runtime.backend is not None
        or bool(runtime.opt_in_backends)
        or runtime.rootful_authorized
        or runtime.authorization_provider is not None
        or runtime.secret_resolver is not None
        or bool(runtime.audit_listeners)
    )


def _container_runtime_hooks_configured(
    runtime: ContainerToolRuntimeSettings,
) -> bool:
    assert isinstance(runtime, ContainerToolRuntimeSettings)
    return (
        runtime.authorization_provider is not None
        or runtime.secret_resolver is not None
        or bool(runtime.audit_listeners)
    )


def _has_pipeline_tool(
    tools: Sequence[Callable[..., object] | Tool | ToolSet],
) -> bool:
    return any(getattr(tool, "__name__", "") == "pipeline" for tool in tools)


def _append_missing_tools(
    tools: list[Callable[..., object] | Tool | ToolSet],
    candidates: Sequence[Callable[..., object] | Tool | ToolSet],
) -> None:
    names = {getattr(tool, "__name__", "") for tool in tools}
    for candidate in candidates:
        name = getattr(candidate, "__name__", "")
        if name not in names:
            tools.append(candidate)
            names.add(name)


def _enables_shell_git_tools(enable_tools: Sequence[str]) -> bool:
    return any(
        matches_tool_namespace("shell.git_status", enabled)
        or enabled.startswith("shell.git_")
        for enabled in enable_tools
    )


def _available_git_tools_for_selection(
    settings: ShellToolSettings,
    enable_tools: Sequence[str],
) -> list[Tool]:
    if _enables_shell_namespace(enable_tools):
        return _authorized_git_tools(settings)

    explicit_tool_names = {
        enabled
        for enabled in enable_tools
        if enabled.startswith(f"{SHELL_TOOL_NAMESPACE}.git_")
    }
    if not explicit_tool_names:
        return []

    return [
        tool
        for tool in _all_git_tools(settings)
        if f"{SHELL_TOOL_NAMESPACE}.{getattr(tool, '__name__', '')}"
        in explicit_tool_names
    ]


def _enables_shell_namespace(enable_tools: Sequence[str]) -> bool:
    return any(
        enabled in (SHELL_TOOL_NAMESPACE, SHELL_TOOL_WILDCARD)
        for enabled in enable_tools
    )


def _authorized_git_tools(settings: ShellToolSettings) -> list[Tool]:
    git_settings = settings.git
    assert isinstance(
        git_settings,
        ShellGitToolSettings,
    ), "git must be shell Git tool settings"
    return [
        tool
        for tool in _all_git_tools(settings)
        if _git_tool_allowed(tool, git_settings)
    ]


def _git_tool_allowed(
    tool: Tool,
    settings: ShellGitToolSettings,
) -> bool:
    command = _git_tool_command(tool)
    capability = SHELL_GIT_COMMAND_CAPABILITIES[command]
    if command.value not in settings.allowed_commands:
        return False
    if capability.value not in settings.capabilities:
        return False
    if capability is ShellGitCapability.REMOTE:
        return _git_remote_policy_allows_tool(command, settings)
    return True


def _git_remote_policy_allows_tool(
    command: ShellGitCommandName,
    settings: ShellGitToolSettings,
) -> bool:
    if (
        not settings.allowed_remote_protocols
        or not settings.allowed_remote_hosts
    ):
        return False
    if (
        command is ShellGitCommandName.SUBMODULE_UPDATE
        and not settings.allow_submodule_update
    ):
        return False
    return True


def _git_tool_command(tool: Tool) -> ShellGitCommandName:
    command = getattr(tool, "_command", None)
    assert isinstance(
        command,
        ShellGitCommandName,
    ), "Git tool must declare a shell Git command"
    return command


def _all_git_tools(settings: ShellToolSettings) -> list[Tool]:
    return [
        GitStatusTool(settings=settings),
        GitRevParseTool(settings=settings),
        GitBranchTool(settings=settings),
        GitTagTool(settings=settings),
        GitDescribeTool(settings=settings),
        GitLsFilesTool(settings=settings),
        GitLogTool(settings=settings),
        GitDiffTool(settings=settings),
        GitShowTool(settings=settings),
        GitBlameTool(settings=settings),
        GitGrepTool(settings=settings),
        GitStashListTool(settings=settings),
        GitStashShowTool(settings=settings),
        GitAddTool(settings=settings),
        GitRestoreTool(settings=settings),
        GitCheckoutTool(settings=settings),
        GitSwitchTool(settings=settings),
        GitResetTool(settings=settings),
        GitRmTool(settings=settings),
        GitMvTool(settings=settings),
        GitStashPushTool(settings=settings),
        GitStashApplyTool(settings=settings),
        GitCommitTool(settings=settings),
        GitBranchCreateTool(settings=settings),
        GitBranchDeleteTool(settings=settings),
        GitBranchRenameTool(settings=settings),
        GitTagCreateTool(settings=settings),
        GitTagDeleteTool(settings=settings),
        GitMergeTool(settings=settings),
        GitRebaseTool(settings=settings),
        GitCherryPickTool(settings=settings),
        GitRevertTool(settings=settings),
        GitCleanTool(settings=settings),
        GitStashPopTool(settings=settings),
        GitStashDropTool(settings=settings),
        GitFetchTool(settings=settings),
        GitPullTool(settings=settings),
        GitPushTool(settings=settings),
        GitCloneTool(settings=settings),
        GitRemoteListTool(settings=settings),
        GitRemoteAddTool(settings=settings),
        GitRemoteSetUrlTool(settings=settings),
        GitRemoteRemoveTool(settings=settings),
        GitRemoteRenameTool(settings=settings),
        GitSubmoduleUpdateTool(settings=settings),
    ]


def _isolation_runtime_hooks_configured(
    runtime: IsolationToolRuntimeSettings,
) -> bool:
    assert isinstance(runtime, IsolationToolRuntimeSettings)
    return (
        runtime.authorization_provider is not None
        or runtime.secret_resolver is not None
        or bool(runtime.audit_listeners)
    )

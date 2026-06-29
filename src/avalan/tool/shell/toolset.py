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
from .container import ShellContainerCommandExecutor
from .executor import CommandExecutor, LocalCommandExecutor
from .formatting import format_shell_result
from .opt_in import SHELL_TOOL_NAMESPACE, enables_shell_pipeline
from .policy import ExecutionPolicy
from .process import ShellProcessRuntime
from .sandbox import ShellSandboxCommandExecutor
from .settings import ShellToolSettings
from .tools import (
    AwkTool,
    CatTool,
    FileTool,
    FindTool,
    HeadTool,
    JqTool,
    LsTool,
    NlTool,
    PdfInfoTool,
    PdfToPpmTool,
    PdfToTextTool,
    RgTool,
    SedTool,
    ShellResultFormatter,
    TailTool,
    TesseractTool,
    WcTool,
)

from collections.abc import Callable, Sequence
from typing import cast


async def _pipeline_tool_unavailable(**arguments: object) -> str:
    """Reject shell pipeline calls before runtime support exists.

    Args:
        arguments: Structured pipeline arguments accepted by later phases.

    Returns:
        This stub never returns.
    """
    assert isinstance(arguments, dict)
    raise AssertionError("shell.pipeline runtime is not implemented")


_pipeline_tool_unavailable.__name__ = "pipeline"


class ShellToolSet(ToolSet):
    _settings: ShellToolSettings
    _process_runtime: ShellProcessRuntime

    def __init__(
        self,
        settings: ShellToolSettings | None = None,
        *,
        executor: CommandExecutor | None = None,
        formatter: ShellResultFormatter | None = None,
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

    def with_enabled_tools(self, enable_tools: list[str]) -> "ShellToolSet":
        if enables_shell_pipeline(
            enable_tools, self._settings
        ) and not _has_pipeline_tool(self.tools):
            self.tools.append(_pipeline_tool_unavailable)
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


def _isolation_runtime_hooks_configured(
    runtime: IsolationToolRuntimeSettings,
) -> bool:
    assert isinstance(runtime, IsolationToolRuntimeSettings)
    return (
        runtime.authorization_provider is not None
        or runtime.secret_resolver is not None
        or bool(runtime.audit_listeners)
    )

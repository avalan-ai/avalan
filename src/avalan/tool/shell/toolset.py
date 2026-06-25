from ...container import (
    ContainerAsyncBackend,
    ContainerBackend,
    ContainerEffectiveSettings,
    ContainerToolRuntimeSettings,
    disabled_required_container_settings,
)
from .. import ToolSet
from .container import ShellContainerCommandExecutor
from .executor import CommandExecutor, LocalCommandExecutor
from .formatting import format_shell_result
from .opt_in import SHELL_TOOL_NAMESPACE
from .policy import ExecutionPolicy
from .settings import ShellToolSettings
from .tools import (
    AwkTool,
    CatTool,
    FileTool,
    FindTool,
    HeadTool,
    JqTool,
    LsTool,
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

from collections.abc import Sequence


class ShellToolSet(ToolSet):
    _settings: ShellToolSettings

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
    ) -> None:
        assert namespace == SHELL_TOOL_NAMESPACE, "namespace must be shell"
        assert isinstance(container_rootful_authorized, bool)
        self._settings = settings or ShellToolSettings()
        if (
            container_runtime is not None
            and self._settings.backend == "container"
        ):
            assert isinstance(container_runtime, ContainerToolRuntimeSettings)
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
        if (
            container_settings is None
            and self._settings.backend == "container"
        ):
            container_settings = disabled_required_container_settings()
        policy = policy or ExecutionPolicy(settings=self._settings)
        if executor is None:
            local_executor = LocalCommandExecutor(settings=self._settings)
            executor = (
                local_executor
                if container_settings is None
                else ShellContainerCommandExecutor(
                    container_settings=container_settings,
                    container_backend=container_backend,
                    opt_in_backends=container_opt_in_backends,
                    local_executor=local_executor,
                    rootful_authorized=container_rootful_authorized,
                )
            )
        formatter = formatter or (
            lambda result: format_shell_result(
                result,
                settings=self._settings,
            )
        )
        tools = [
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

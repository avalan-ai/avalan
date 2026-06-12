from .. import ToolSet
from .executor import CommandExecutor, LocalCommandExecutor
from .formatting import format_shell_result
from .opt_in import SHELL_TOOL_NAMESPACE
from .policy import ExecutionPolicy
from .settings import ShellToolSettings
from .tools import (
    AwkTool,
    CatTool,
    HeadTool,
    JqTool,
    LsTool,
    PdfToPpmTool,
    PdfToTextTool,
    RgTool,
    SedTool,
    ShellResultFormatter,
    TailTool,
    TesseractTool,
    WcTool,
)


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
    ) -> None:
        assert namespace == SHELL_TOOL_NAMESPACE, "namespace must be shell"
        self._settings = settings or ShellToolSettings()
        policy = policy or ExecutionPolicy(settings=self._settings)
        executor = executor or LocalCommandExecutor(settings=self._settings)
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

from .. import ToolSet
from .settings import ShellToolSettings


class ShellToolSet(ToolSet):
    _settings: ShellToolSettings

    def __init__(
        self,
        settings: ShellToolSettings | None = None,
        *,
        namespace: str | None = "shell",
    ) -> None:
        self._settings = settings or ShellToolSettings()
        super().__init__(namespace=namespace, tools=[])

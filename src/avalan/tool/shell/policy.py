from .entities import (
    ExecutionSpec,
    ShellCommandRequest,
    ShellExecutionErrorCode,
)
from .resolver import ExecutableResolver, TrustedExecutableResolver
from .settings import ShellToolSettings


class ShellPolicyDenied(Exception):
    error_code: ShellExecutionErrorCode

    def __init__(
        self,
        error_code: ShellExecutionErrorCode,
        message: str,
    ) -> None:
        super().__init__(message)
        self.error_code = error_code


class ExecutionPolicy:
    _resolver: ExecutableResolver
    _settings: ShellToolSettings

    def __init__(
        self,
        settings: ShellToolSettings | None = None,
        resolver: ExecutableResolver | None = None,
    ) -> None:
        self._settings = settings or ShellToolSettings()
        self._resolver = resolver or TrustedExecutableResolver(
            executable_paths=self._settings.executable_paths,
            executable_search_paths=self._settings.executable_search_paths,
        )

    async def normalize(self, request: ShellCommandRequest) -> ExecutionSpec:
        raise NotImplementedError("shell execution policy is not implemented")

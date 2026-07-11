from .commands import (
    NormalizedPath as _NormalizedPath,
)
from .commands import (
    NormalizedWorkspace as _NormalizedWorkspace,
)
from .commands import (
    ShellCommandDefinition,
    ShellCommandPolicyContext,
)
from .commands.helpers import (
    _contains_traversal,
    path_matches_sensitive_denylist,
)
from .commands.helpers import policy_denied as _policy_denied
from .commands.nl import (
    RESERVED_SECTION_DELIMITER_SEQUENCES as _NL_SECTION_DELIMITER_SEQUENCES,
)
from .commands.pdftoppm import (
    _page_size_raster_dpi_cap,
)
from .entities import (
    ExecutionSpec,
    GeneratedOutputPlan,
    PathOperand,
    ShellCommandRequest,
    ShellCommandStepRequest,
    ShellCompositionMode,
    ShellCompositionRequest,
    ShellCompositionSpec,
    ShellExecutionErrorCode,
    ShellExecutionModeValue,
    ShellExecutionStepSpec,
    ShellOutputKind,
    ShellPathKind,
    ShellStreamRef,
    _create_execution_spec_from_policy,
)
from .filesystem import (
    ShellPathMetadata,
    inspect_path,
    probe_image_dimensions,
    probe_pdf_page_boxes,
    read_pdf_signature,
    read_signature,
    resolve_policy_path,
    sniff_binary,
)
from .registry import SHELL_COMMAND_DEFINITIONS
from .resolver import ExecutableResolver, TrustedExecutableResolver
from .settings import ShellToolSettings

from collections.abc import Mapping, Sequence
from math import isfinite
from os import environ
from pathlib import Path, PurePosixPath
from typing import Literal, cast

_DEFAULT_CHILD_ENVIRONMENT = {
    "LC_ALL": "C",
    "LANG": "C",
    "TERM": "dumb",
    "NO_COLOR": "1",
    "CLICOLOR": "0",
    "CLICOLOR_FORCE": "0",
}
_SAFE_EMPTY_ENVIRONMENT_PATH = "/nonexistent"
_SAFE_HOME_ENVIRONMENT = {
    "HOME": _SAFE_EMPTY_ENVIRONMENT_PATH,
    "XDG_CONFIG_HOME": _SAFE_EMPTY_ENVIRONMENT_PATH,
    "XDG_CACHE_HOME": _SAFE_EMPTY_ENVIRONMENT_PATH,
}
_SCRUBBED_ENVIRONMENT_NAMES = {
    "PATH",
    "LS_COLORS",
    "RIPGREP_CONFIG_PATH",
    "AWKPATH",
    "AWKLIBPATH",
    "JQ_LIBRARY_PATH",
    "TESSDATA_PREFIX",
    "POSIXLY_CORRECT",
}
_SECRET_ENVIRONMENT_MARKERS = (
    "TOKEN",
    "SECRET",
    "PASSWORD",
    "KEY",
    "CREDENTIAL",
)
_VIRTUAL_FILESYSTEM_ROOTS = ("dev", "proc", "sys")
_PDF_PAGE_BOX_METADATA_KEY = "_pdf_page_box_points"
_COMPOSITION_MODES: tuple[ShellCompositionMode, ...] = (
    "pipeline",
    "serial",
    "parallel",
)
_COMPOSITION_PATH_KINDS: Mapping[str, ShellPathKind] = {
    "awk": "text_file",
    "cat": "text_file",
    "file": "file",
    "find": "any",
    "head": "text_file",
    "jq": "json_file",
    "ls": "any",
    "nl": "text_file",
    "pdfinfo": "pdf_file",
    "pdfplumber": "pdf_file",
    "pdftoppm": "pdf_file",
    "pdftotext": "pdf_file",
    "pypdf": "pdf_file",
    "rg": "any",
    "reportlab": "any",
    "sed": "text_file",
    "tail": "text_file",
    "tesseract": "image_file",
    "wc": "text_file",
}


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
        assert isinstance(
            request,
            ShellCommandRequest,
        ), "request must be a shell command request"
        if request.stdin is not None:
            raise _policy_denied(
                ShellExecutionErrorCode.STDIN_DENIED,
                "stdin is disabled",
            )
        return await self._normalize_request(
            request,
            stdin_mode=False,
            stdin_media_type=None,
            stdin_output_kind=None,
        )

    async def normalize_composition(
        self,
        request: ShellCompositionRequest,
    ) -> ShellCompositionSpec:
        assert isinstance(
            request,
            ShellCompositionRequest,
        ), "request must be a shell composition request"
        process_commands = {"kill", "lsof", "pgrep", "ps"}
        if any(step.command in process_commands for step in request.steps):
            raise _policy_denied(
                ShellExecutionErrorCode.DENIED_COMMAND,
                "process tools are disabled in shell compositions",
            )
        if not self._settings.allow_pipelines:
            raise _policy_denied(
                ShellExecutionErrorCode.POLICY_DENIED,
                "shell pipelines are disabled",
            )
        mode = _validated_composition_mode(request.mode)
        steps = _validated_composition_steps(request.steps)
        _validate_composition_stage_count(
            steps,
            max_pipeline_stages=self._settings.max_pipeline_stages,
        )
        stdin_refs = _composition_stdin_refs(mode, steps)
        composition_metadata: dict[str, object] = {}
        timeout_seconds = _normalized_timeout(
            request.timeout_seconds,
            self._settings.default_timeout_seconds,
            self._settings.max_timeout_seconds,
            composition_metadata,
        )
        max_stdout_bytes = _normalized_byte_budget(
            request.max_stdout_bytes,
            self._settings.max_pipeline_bytes,
            "max_stdout_bytes",
            composition_metadata,
        )
        max_stderr_bytes = _normalized_byte_budget(
            request.max_stderr_bytes,
            self._settings.max_stderr_bytes,
            "max_stderr_bytes",
            composition_metadata,
        )
        max_intermediate_bytes = _normalized_byte_budget(
            request.max_intermediate_bytes,
            self._settings.max_intermediate_bytes,
            "max_intermediate_bytes",
            composition_metadata,
        )

        normalized_steps: list[ShellExecutionStepSpec] = []
        specs_by_id: dict[str, ExecutionSpec] = {}
        for step, stdin_from in zip(steps, stdin_refs, strict=True):
            command_request = _composition_command_request(step)
            if stdin_from is None:
                spec = await self.normalize(command_request)
            else:
                producer = specs_by_id[stdin_from.step_id]
                spec = await self._normalize_stdin(
                    command_request,
                    producer_media_type=producer.stdout_media_type,
                    producer_output_kind=producer.output_kind,
                )
            if spec.output_plan is not None:
                raise _policy_denied(
                    ShellExecutionErrorCode.DENIED_COMMAND,
                    "generated outputs are disabled in compositions",
                )
            specs_by_id[step.id] = spec
            normalized_steps.append(
                ShellExecutionStepSpec(
                    id=step.id,
                    spec=spec,
                    stdin_from=stdin_from,
                )
            )

        return ShellCompositionSpec(
            mode=mode,
            steps=tuple(normalized_steps),
            timeout_seconds=timeout_seconds,
            max_stdout_bytes=max_stdout_bytes,
            max_stderr_bytes=max_stderr_bytes,
            max_intermediate_bytes=max_intermediate_bytes,
        )

    async def _normalize_stdin(
        self,
        request: ShellCommandRequest,
        *,
        producer_media_type: str,
        producer_output_kind: ShellOutputKind,
    ) -> ExecutionSpec:
        assert isinstance(
            request,
            ShellCommandRequest,
        ), "request must be a shell command request"
        if request.stdin is not None:
            raise _policy_denied(
                ShellExecutionErrorCode.STDIN_DENIED,
                "stdin is disabled",
            )
        return await self._normalize_request(
            request,
            stdin_mode=True,
            stdin_media_type=producer_media_type,
            stdin_output_kind=producer_output_kind,
        )

    async def _normalize_request(
        self,
        request: ShellCommandRequest,
        *,
        stdin_mode: bool,
        stdin_media_type: str | None,
        stdin_output_kind: ShellOutputKind | None,
    ) -> ExecutionSpec:
        if any(path.access != "read" for path in request.paths):
            raise _policy_denied(
                ShellExecutionErrorCode.WRITE_DENIED,
                "write access is disabled",
            )
        if _requests_shell_evaluation(request.options):
            raise _policy_denied(
                ShellExecutionErrorCode.SHELL_DENIED,
                "shell evaluation is disabled",
            )
        if request.tool_name != f"shell.{request.command}":
            raise _policy_denied(
                ShellExecutionErrorCode.DENIED_COMMAND,
                "tool name does not match command",
            )
        command_definition = SHELL_COMMAND_DEFINITIONS.get(request.command)
        if (
            command_definition is None
            or request.command not in self._settings.allowed_commands
        ):
            raise _policy_denied(
                ShellExecutionErrorCode.DENIED_COMMAND,
                "command is not allowed",
            )
        if (
            command_definition.media_risk
            and not self._settings.allow_media_tools
        ):
            raise _policy_denied(
                ShellExecutionErrorCode.DENIED_COMMAND,
                "media tools are disabled",
            )
        if (
            command_definition.process_risk
            and not self._settings.allow_process_tools
        ):
            raise _policy_denied(
                ShellExecutionErrorCode.DENIED_COMMAND,
                "process tools are disabled",
            )
        if (
            request.command == "kill"
            and not self._settings.allow_process_control
        ):
            raise _policy_denied(
                ShellExecutionErrorCode.DENIED_COMMAND,
                "process control tools are disabled",
            )
        if (
            request.command == "kill"
            and self._settings.execution_mode != "local"
        ):
            raise _policy_denied(
                ShellExecutionErrorCode.DENIED_COMMAND,
                "process control requires local execution",
            )
        if (
            request.command in {"kill", "lsof", "pgrep", "ps"}
            and request.paths
        ):
            raise _policy_denied(
                ShellExecutionErrorCode.INVALID_OPTION,
                "process tools do not accept paths",
            )
        if stdin_mode:
            _validate_stdin_mode(
                request,
                command_definition,
                stdin_media_type=stdin_media_type,
                stdin_output_kind=stdin_output_kind,
            )
        _validate_argument_budgets(request, self._settings)
        metadata = _with_local_audit_metadata(
            _validated_metadata(request.metadata),
            self._settings,
        )
        workspace = await _normalized_workspace(
            self._settings,
            request.cwd,
        )
        normalized_paths = await _normalized_paths(
            request.paths,
            workspace=workspace,
            settings=self._settings,
        )
        await _annotate_pdf_raster_metadata(
            request,
            normalized_paths,
            metadata,
            settings=self._settings,
        )
        default_timeout_seconds, max_timeout_seconds = _timeout_limits(
            request.command,
            self._settings,
        )
        timeout_seconds = _normalized_timeout(
            request.timeout_seconds,
            default_timeout_seconds,
            max_timeout_seconds,
            metadata,
        )
        max_stdout_bytes = _normalized_byte_budget(
            request.max_stdout_bytes,
            self._settings.max_stdout_bytes,
            "max_stdout_bytes",
            metadata,
        )
        max_stderr_bytes = _normalized_byte_budget(
            request.max_stderr_bytes,
            self._settings.max_stderr_bytes,
            "max_stderr_bytes",
            metadata,
        )
        argv, display_argv, output_plan = command_definition.argv_builder(
            ShellCommandPolicyContext(
                executable_name=command_definition.executable_name,
                request=request,
                paths=normalized_paths,
                workspace=workspace,
                settings=self._settings,
                metadata=metadata,
                stdin_mode=stdin_mode,
                stdin_media_type=stdin_media_type,
                stdin_output_kind=stdin_output_kind,
            )
        )
        _validate_argv_budgets(argv, display_argv, self._settings)
        await _enforce_content_policy(
            request,
            normalized_paths,
            settings=self._settings,
        )
        executable = await self._resolver.resolve(command_definition)
        env = _child_environment(self._settings)

        stdout_media_type, output_kind = command_definition.output_contract(
            request
        )
        return self.create_execution_spec(
            backend=cast(
                ShellExecutionModeValue, self._settings.execution_mode
            ),
            tool_name=request.tool_name,
            command=request.command,
            executable=executable,
            argv=argv,
            display_argv=display_argv,
            cwd=str(workspace.cwd),
            display_cwd=workspace.display_cwd,
            env=env,
            stdin=None,
            stdout_media_type=stdout_media_type,
            output_kind=output_kind,
            resource_class=(
                "heavy" if command_definition.media_risk else "standard"
            ),
            output_plan=output_plan,
            timeout_seconds=timeout_seconds,
            max_stdout_bytes=max_stdout_bytes,
            max_stderr_bytes=max_stderr_bytes,
            metadata=metadata,
        )

    def create_execution_spec(
        self,
        *,
        backend: ShellExecutionModeValue,
        tool_name: str,
        command: str,
        executable: str | None,
        argv: tuple[str, ...],
        display_argv: tuple[str, ...],
        cwd: str,
        display_cwd: str,
        env: dict[str, str],
        stdin: bytes | None,
        stdout_media_type: str,
        output_kind: ShellOutputKind,
        resource_class: Literal["standard", "heavy"],
        output_plan: GeneratedOutputPlan | None,
        timeout_seconds: float,
        max_stdout_bytes: int,
        max_stderr_bytes: int,
        metadata: dict[str, object] | None = None,
    ) -> ExecutionSpec:
        return _create_execution_spec_from_policy(
            backend=backend,
            tool_name=tool_name,
            command=command,
            executable=executable,
            argv=argv,
            display_argv=display_argv,
            cwd=cwd,
            display_cwd=display_cwd,
            env=env,
            stdin=stdin,
            stdout_media_type=stdout_media_type,
            output_kind=output_kind,
            resource_class=resource_class,
            output_plan=output_plan,
            timeout_seconds=timeout_seconds,
            max_stdout_bytes=max_stdout_bytes,
            max_stderr_bytes=max_stderr_bytes,
            metadata=metadata,
        )


def _validated_composition_mode(
    mode: object,
) -> ShellCompositionMode:
    if mode not in _COMPOSITION_MODES:
        raise _policy_denied(
            ShellExecutionErrorCode.INVALID_OPTION,
            "composition mode is unsupported",
        )
    return mode


def _validated_composition_steps(
    steps: tuple[ShellCommandStepRequest, ...],
) -> tuple[ShellCommandStepRequest, ...]:
    if not steps:
        raise _policy_denied(
            ShellExecutionErrorCode.INVALID_OPTION,
            "composition requires at least one step",
        )
    step_ids: set[str] = set()
    for step in steps:
        if step.id in step_ids:
            raise _policy_denied(
                ShellExecutionErrorCode.INVALID_OPTION,
                "composition step ids must be unique",
            )
        step_ids.add(step.id)
    return tuple(steps)


def _validate_composition_stage_count(
    steps: tuple[ShellCommandStepRequest, ...],
    *,
    max_pipeline_stages: int,
) -> None:
    if len(steps) > max_pipeline_stages:
        raise _policy_denied(
            ShellExecutionErrorCode.INVALID_OPTION,
            "composition has too many steps",
        )


def _composition_stdin_refs(
    mode: ShellCompositionMode,
    steps: tuple[ShellCommandStepRequest, ...],
) -> tuple[ShellStreamRef | None, ...]:
    step_indexes = {step.id: index for index, step in enumerate(steps)}
    if mode == "pipeline":
        return _pipeline_stdin_refs(steps)
    if mode == "serial":
        return _serial_stdin_refs(steps, step_indexes)
    return _parallel_stdin_refs(steps)


def _pipeline_stdin_refs(
    steps: tuple[ShellCommandStepRequest, ...],
) -> tuple[ShellStreamRef | None, ...]:
    stdin_refs: list[ShellStreamRef | None] = []
    for index, step in enumerate(steps):
        if index == 0:
            if step.stdin_from is not None:
                raise _policy_denied(
                    ShellExecutionErrorCode.INVALID_OPTION,
                    "first pipeline step cannot read stdin",
                )
            stdin_refs.append(None)
            continue
        expected_ref = ShellStreamRef(
            step_id=steps[index - 1].id,
            stream="stdout",
        )
        if step.stdin_from is not None:
            _validate_composition_ref(step.stdin_from)
        if step.stdin_from is not None and step.stdin_from != expected_ref:
            raise _policy_denied(
                ShellExecutionErrorCode.INVALID_OPTION,
                "pipeline stdin_from must reference the previous step",
            )
        stdin_refs.append(expected_ref)
    return tuple(stdin_refs)


def _serial_stdin_refs(
    steps: tuple[ShellCommandStepRequest, ...],
    step_indexes: dict[str, int],
) -> tuple[ShellStreamRef | None, ...]:
    stdin_refs: list[ShellStreamRef | None] = []
    for index, step in enumerate(steps):
        stdin_from = step.stdin_from
        if stdin_from is None:
            stdin_refs.append(None)
            continue
        _validate_composition_ref(stdin_from)
        producer_index = step_indexes.get(stdin_from.step_id)
        if producer_index is None:
            raise _policy_denied(
                ShellExecutionErrorCode.INVALID_OPTION,
                "stdin_from references an unknown step",
            )
        if producer_index >= index:
            raise _policy_denied(
                ShellExecutionErrorCode.INVALID_OPTION,
                "serial stdin_from must reference an earlier step",
            )
        stdin_refs.append(stdin_from)
    return tuple(stdin_refs)


def _parallel_stdin_refs(
    steps: tuple[ShellCommandStepRequest, ...],
) -> tuple[None, ...]:
    for step in steps:
        if step.stdin_from is not None:
            raise _policy_denied(
                ShellExecutionErrorCode.INVALID_OPTION,
                "parallel composition steps cannot read stdin",
            )
    return tuple(None for _ in steps)


def _validate_composition_ref(stdin_from: ShellStreamRef) -> None:
    if stdin_from.stream != "stdout":
        raise _policy_denied(
            ShellExecutionErrorCode.INVALID_OPTION,
            "stdin_from stream must be stdout",
        )


def _composition_command_request(
    step: ShellCommandStepRequest,
) -> ShellCommandRequest:
    return ShellCommandRequest(
        tool_name=f"shell.{step.command}",
        command=step.command,
        options=step.options,
        paths=_composition_path_operands(step),
        cwd=step.cwd,
    )


def _composition_path_operands(
    step: ShellCommandStepRequest,
) -> tuple[PathOperand, ...]:
    return tuple(
        PathOperand(
            name=f"path_{index}",
            path=path,
            kind=_composition_path_kind(step.command, path),
            access="read",
        )
        for index, path in enumerate(step.paths)
    )


def _composition_path_kind(
    command: str,
    path: str,
) -> ShellPathKind:
    if command == "cat" and PurePosixPath(path).suffix.lower() == ".json":
        return "json_file"
    return _COMPOSITION_PATH_KINDS.get(command, "any")


async def _normalized_workspace(
    settings: ShellToolSettings,
    request_cwd: str | None,
) -> _NormalizedWorkspace:
    workspace_root = await resolve_policy_path(settings.workspace_root)
    cwd_value = request_cwd if request_cwd is not None else settings.cwd
    cwd_candidate = _candidate_path_value(
        cwd_value,
        root=workspace_root,
        cwd=workspace_root,
        allow_absolute=settings.allow_absolute_paths,
        error_code=ShellExecutionErrorCode.INVALID_CWD,
        field_name="cwd",
    )
    metadata = await _inspect_existing_policy_path(
        cwd_candidate,
        root=workspace_root,
        allow_symlinks=settings.allow_symlinks,
    )
    cwd_path = await _resolve_workspace_path(
        cwd_candidate,
        root=workspace_root,
        error_code=ShellExecutionErrorCode.INVALID_CWD,
        field_name="cwd",
    )
    display_cwd = _display_path(workspace_root, cwd_path)
    await _enforce_path_policy(
        resolved_path=cwd_path,
        display_path=display_cwd,
        settings=settings,
        metadata=metadata,
    )
    return _NormalizedWorkspace(
        root=workspace_root,
        cwd=cwd_path,
        display_cwd=display_cwd,
    )


async def _normalized_paths(
    paths: tuple[PathOperand, ...],
    *,
    workspace: _NormalizedWorkspace,
    settings: ShellToolSettings,
) -> tuple[_NormalizedPath, ...]:
    if len(paths) > settings.max_path_count:
        raise _policy_denied(
            ShellExecutionErrorCode.DENIED_PATH,
            "request has too many paths",
        )
    normalized_paths: list[_NormalizedPath] = []
    for operand in paths:
        candidate_path = _candidate_path_value(
            operand.path,
            root=workspace.root,
            cwd=workspace.cwd,
            allow_absolute=settings.allow_absolute_paths,
            error_code=ShellExecutionErrorCode.DENIED_PATH,
            field_name=operand.name,
            reject_dash=True,
        )
        metadata = await _inspect_existing_policy_path(
            candidate_path,
            root=workspace.root,
            allow_symlinks=settings.allow_symlinks,
        )
        path = await _resolve_workspace_path(
            candidate_path,
            root=workspace.root,
            error_code=ShellExecutionErrorCode.DENIED_PATH,
            field_name=operand.name,
        )
        display_path = _display_path(workspace.root, path)
        await _enforce_path_policy(
            resolved_path=path,
            display_path=display_path,
            settings=settings,
            metadata=metadata,
        )
        normalized_paths.append(
            _NormalizedPath(
                operand=operand,
                path=path,
                display_path=display_path,
                metadata=metadata,
            )
        )
    return tuple(normalized_paths)


def _candidate_path_value(
    value: str,
    *,
    root: Path,
    cwd: Path,
    allow_absolute: bool,
    error_code: ShellExecutionErrorCode,
    field_name: str,
    reject_dash: bool = False,
) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise _policy_denied(error_code, f"{field_name} must not be empty")
    if "\x00" in value:
        raise _policy_denied(error_code, f"{field_name} contains NUL")
    if reject_dash and value == "-":
        raise _policy_denied(error_code, f"{field_name} cannot read stdin")
    if value.startswith("~"):
        raise _policy_denied(error_code, f"{field_name} cannot expand user")
    if _contains_environment_expansion(value):
        raise _policy_denied(
            error_code,
            f"{field_name} cannot expand environment",
        )
    if _contains_traversal(value):
        raise _policy_denied(
            ShellExecutionErrorCode.TRAVERSAL,
            f"{field_name} contains traversal",
        )
    source_path = Path(value)
    if source_path.is_absolute():
        if not allow_absolute:
            raise _policy_denied(
                error_code,
                f"{field_name} must be workspace-relative",
            )
        return source_path
    return cwd / source_path


async def _resolve_workspace_path(
    candidate: Path,
    *,
    root: Path,
    error_code: ShellExecutionErrorCode,
    field_name: str,
) -> Path:
    normalized_path = await resolve_policy_path(candidate)
    if not _is_relative_to(normalized_path, root):
        raise _policy_denied(error_code, f"{field_name} is outside workspace")
    return normalized_path


async def _enforce_path_policy(
    *,
    resolved_path: Path,
    display_path: str,
    settings: ShellToolSettings,
    metadata: ShellPathMetadata | None,
) -> None:
    if not settings.allow_hidden and _has_hidden_component(display_path):
        raise _policy_denied(
            ShellExecutionErrorCode.HIDDEN_PATH,
            "hidden paths are disabled",
        )
    if path_matches_sensitive_denylist(display_path):
        raise _policy_denied(
            ShellExecutionErrorCode.SENSITIVE_PATH,
            "path is denied",
        )
    if _is_virtual_filesystem_path(resolved_path):
        raise _policy_denied(
            ShellExecutionErrorCode.SPECIAL_FILE,
            "special files are disabled",
        )
    if metadata is not None and metadata.is_special_file:
        raise _policy_denied(
            ShellExecutionErrorCode.SPECIAL_FILE,
            "special files are disabled",
        )


async def _inspect_existing_policy_path(
    path: Path,
    *,
    root: Path,
    allow_symlinks: bool,
) -> ShellPathMetadata | None:
    metadata: ShellPathMetadata | None = None
    for component_path in _path_component_candidates(root, path):
        try:
            metadata = await inspect_path(component_path)
        except FileNotFoundError:
            return None
        except PermissionError:
            raise _policy_denied(
                ShellExecutionErrorCode.DENIED_PATH,
                "path metadata is unavailable",
            ) from None
        if metadata.is_symlink and not allow_symlinks:
            raise _policy_denied(
                ShellExecutionErrorCode.SYMLINK,
                "symlinks are disabled",
            )
    return metadata


def _path_component_candidates(root: Path, path: Path) -> tuple[Path, ...]:
    try:
        relative_parts = path.relative_to(root).parts
    except ValueError:
        return ()
    candidates: list[Path] = []
    current_path = root
    for part in relative_parts:
        current_path /= part
        candidates.append(current_path)
    return tuple(candidates)


def _has_hidden_component(display_path: str) -> bool:
    return any(
        part not in ("", ".", "..") and part.startswith(".")
        for part in PurePosixPath(display_path).parts
    )


def _is_virtual_filesystem_path(path: Path) -> bool:
    parts = path.parts
    return len(parts) > 1 and parts[1] in _VIRTUAL_FILESYSTEM_ROOTS


async def _enforce_content_policy(
    request: ShellCommandRequest,
    paths: tuple[_NormalizedPath, ...],
    *,
    settings: ShellToolSettings,
) -> None:
    if not paths:
        return
    match request.command:
        case "cat" | "nl":
            await _enforce_text_inputs(
                paths,
                max_bytes=settings.max_full_file_bytes,
                forbidden_sequences=(
                    _NL_SECTION_DELIMITER_SEQUENCES
                    if request.command == "nl"
                    else ()
                ),
            )
        case "head" | "tail":
            await _enforce_text_inputs(paths)
        case "wc":
            if _wc_reads_text(request.options):
                await _enforce_text_inputs(
                    paths,
                    max_bytes=settings.max_full_file_bytes,
                )
            else:
                await _enforce_regular_file_inputs(paths)
        case "file":
            await _enforce_regular_file_inputs(paths)
        case "ls":
            await _enforce_listing_inputs(paths)
        case "find":
            await _enforce_listing_inputs(paths)
        case "awk" | "sed":
            await _enforce_text_inputs(
                paths,
                max_total_bytes=settings.max_text_filter_input_bytes,
            )
        case "jq":
            await _enforce_json_inputs(paths, settings=settings)
        case "pdfinfo" | "pdftotext" | "pdftoppm" | "pdfplumber" | "pypdf":
            await _enforce_pdf_inputs(paths, settings=settings)
        case "tesseract":
            await _enforce_image_inputs(paths, settings=settings)


def _wc_reads_text(options: Mapping[str, object]) -> bool:
    lines = options.get("lines")
    words = options.get("words")
    count_bytes = options.get("count_bytes")
    if lines is True or words is True:
        return True
    return count_bytes is not True


def _validate_stdin_mode(
    request: ShellCommandRequest,
    command_definition: ShellCommandDefinition,
    *,
    stdin_media_type: str | None,
    stdin_output_kind: ShellOutputKind | None,
) -> None:
    assert stdin_media_type is not None, "stdin_media_type is required"
    assert stdin_output_kind is not None, "stdin_output_kind is required"
    if request.paths:
        raise _policy_denied(
            ShellExecutionErrorCode.INVALID_OPTION,
            "stdin mode does not accept paths",
        )
    if not command_definition.stdin_contract.supports_stdin:
        raise _policy_denied(
            ShellExecutionErrorCode.DENIED_COMMAND,
            "command cannot read stdin",
        )
    if request.command == "wc" and request.options.get("count_bytes") is True:
        raise _policy_denied(
            ShellExecutionErrorCode.INVALID_OPTION,
            "wc count_bytes cannot read stdin",
        )
    if not command_definition.stdin_contract.accepts(
        media_type=stdin_media_type,
        output_kind=stdin_output_kind,
    ):
        raise _policy_denied(
            ShellExecutionErrorCode.UNSUPPORTED_MEDIA_SIGNATURE,
            "unsupported stdin stream",
        )


async def _annotate_pdf_raster_metadata(
    request: ShellCommandRequest,
    paths: tuple[_NormalizedPath, ...],
    metadata: dict[str, object],
    *,
    settings: ShellToolSettings,
) -> None:
    if request.command != "pdftoppm" or len(paths) != 1:
        return
    path = paths[0]
    if path.metadata is None or not path.metadata.is_file:
        return
    boxes = await probe_pdf_page_boxes(path.path)
    if not boxes:
        return
    metadata[_PDF_PAGE_BOX_METADATA_KEY] = min(
        boxes,
        key=lambda box: _page_size_raster_dpi_cap(box, settings),
    )


async def _enforce_text_inputs(
    paths: tuple[_NormalizedPath, ...],
    *,
    max_bytes: int | None = None,
    max_total_bytes: int | None = None,
    forbidden_sequences: tuple[bytes, ...] = (),
) -> None:
    for sequence in forbidden_sequences:
        assert isinstance(sequence, bytes), "forbidden sequences must be bytes"
    total_bytes = 0
    for path in paths:
        metadata = await _required_file_metadata(path)
        total_bytes += metadata.size
        if max_bytes is not None and metadata.size > max_bytes:
            raise _policy_denied(
                ShellExecutionErrorCode.TOO_LARGE,
                "input file is too large",
            )
        if max_total_bytes is not None and total_bytes > max_total_bytes:
            raise _policy_denied(
                ShellExecutionErrorCode.TOO_LARGE,
                "input files are too large",
            )
        if await sniff_binary(path.path):
            raise _policy_denied(
                ShellExecutionErrorCode.BINARY_CONTENT,
                "binary input is disabled",
            )
        if forbidden_sequences:
            content = await read_signature(path.path, max_bytes=metadata.size)
            if any(sequence in content for sequence in forbidden_sequences):
                raise _policy_denied(
                    ShellExecutionErrorCode.INVALID_OPTION,
                    "input contains a reserved command delimiter",
                )


async def _enforce_regular_file_inputs(
    paths: tuple[_NormalizedPath, ...],
) -> None:
    for path in paths:
        await _required_file_metadata(path)


async def _enforce_listing_inputs(
    paths: tuple[_NormalizedPath, ...],
) -> None:
    for path in paths:
        try:
            metadata = await inspect_path(path.path)
        except FileNotFoundError:
            raise _policy_denied(
                ShellExecutionErrorCode.DENIED_PATH,
                "listed path is unavailable",
            ) from None
        except PermissionError:
            raise _policy_denied(
                ShellExecutionErrorCode.DENIED_PATH,
                "listed path metadata is unavailable",
            ) from None
        if not (metadata.is_file or metadata.is_directory):
            raise _policy_denied(
                ShellExecutionErrorCode.DENIED_PATH,
                "listed path must be a file or directory",
            )


async def _enforce_json_inputs(
    paths: tuple[_NormalizedPath, ...],
    *,
    settings: ShellToolSettings,
) -> None:
    for path in paths:
        metadata = await _required_file_metadata(path)
        if metadata.size > settings.max_json_input_bytes:
            raise _policy_denied(
                ShellExecutionErrorCode.TOO_LARGE,
                "JSON input file is too large",
            )
        if await sniff_binary(path.path):
            raise _policy_denied(
                ShellExecutionErrorCode.BINARY_CONTENT,
                "binary JSON input is disabled",
            )


async def _enforce_pdf_inputs(
    paths: tuple[_NormalizedPath, ...],
    *,
    settings: ShellToolSettings,
) -> None:
    for path in paths:
        metadata = await _required_file_metadata(path)
        if metadata.size > settings.max_pdf_input_bytes:
            raise _policy_denied(
                ShellExecutionErrorCode.TOO_LARGE,
                "PDF input file is too large",
            )
        if await read_pdf_signature(path.path) != b"%PDF-":
            raise _policy_denied(
                ShellExecutionErrorCode.UNSUPPORTED_MEDIA_SIGNATURE,
                "unsupported PDF signature",
            )


async def _enforce_image_inputs(
    paths: tuple[_NormalizedPath, ...],
    *,
    settings: ShellToolSettings,
) -> None:
    for path in paths:
        metadata = await _required_file_metadata(path)
        if metadata.size > settings.max_ocr_input_bytes:
            raise _policy_denied(
                ShellExecutionErrorCode.TOO_LARGE,
                "OCR input file is too large",
            )
        dimensions = await probe_image_dimensions(path.path)
        if dimensions is None:
            raise _policy_denied(
                ShellExecutionErrorCode.UNSUPPORTED_MEDIA_SIGNATURE,
                "unsupported image signature",
            )
        width, height = dimensions
        if (
            width > settings.max_raster_long_edge_pixels
            or height > settings.max_raster_long_edge_pixels
            or width * height > settings.max_ocr_pixels
        ):
            raise _policy_denied(
                ShellExecutionErrorCode.TOO_LARGE,
                "OCR input image is too large",
            )


async def _required_file_metadata(
    path: _NormalizedPath,
) -> ShellPathMetadata:
    try:
        metadata = await inspect_path(path.path)
    except FileNotFoundError:
        raise _policy_denied(
            ShellExecutionErrorCode.DENIED_PATH,
            "input file is unavailable",
        ) from None
    except PermissionError:
        raise _policy_denied(
            ShellExecutionErrorCode.DENIED_PATH,
            "input file metadata is unavailable",
        ) from None
    if not metadata.is_file:
        raise _policy_denied(
            ShellExecutionErrorCode.DENIED_PATH,
            "input path must be a regular file",
        )
    return metadata


def _contains_environment_expansion(value: str) -> bool:
    return "$" in value or "%" in value


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _display_path(root: Path, path: Path) -> str:
    relative_path = path.relative_to(root)
    display_path = relative_path.as_posix()
    return display_path if display_path else "."


def _child_environment(settings: ShellToolSettings) -> dict[str, str]:
    env: dict[str, str] = {}
    _inject_environment(env, _DEFAULT_CHILD_ENVIRONMENT)
    for name in settings.environment_allowlist:
        value = environ.get(name)
        if value:
            _inject_environment(env, {name: value})
    _inject_environment(env, settings.environment)
    _inject_environment(env, _DEFAULT_CHILD_ENVIRONMENT)
    _inject_environment(env, _SAFE_HOME_ENVIRONMENT)
    env["OMP_THREAD_LIMIT"] = str(settings.tesseract_thread_limit)
    return env


def _inject_environment(
    target: dict[str, str],
    values: Mapping[str, str],
) -> None:
    for name, value in values.items():
        if _is_scrubbed_environment_name(name):
            continue
        target[name] = value


def _is_scrubbed_environment_name(name: str) -> bool:
    normalized_name = name.upper()
    if normalized_name in _SCRUBBED_ENVIRONMENT_NAMES:
        return True
    return any(
        marker in normalized_name for marker in _SECRET_ENVIRONMENT_MARKERS
    )


def _requests_shell_evaluation(options: Mapping[str, object]) -> bool:
    for key in ("shell", "shell_eval", "use_shell"):
        if options.get(key) is True:
            return True
    return False


def _timeout_limits(
    command: str,
    settings: ShellToolSettings,
) -> tuple[float, float]:
    if command in (
        "pdfinfo",
        "pdftotext",
        "pdftoppm",
        "reportlab",
        "pdfplumber",
        "pypdf",
    ):
        return (
            settings.default_pdf_timeout_seconds,
            settings.max_pdf_timeout_seconds,
        )
    if command == "tesseract":
        return (
            settings.default_ocr_timeout_seconds,
            settings.max_ocr_timeout_seconds,
        )
    return settings.default_timeout_seconds, settings.max_timeout_seconds


def _normalized_timeout(
    value: object | None,
    default_value: float,
    max_value: float,
    metadata: dict[str, object],
) -> float:
    if value is None:
        return default_value
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise _policy_denied(
            ShellExecutionErrorCode.INVALID_OPTION,
            "timeout_seconds must be numeric",
        )
    if not isfinite(value) or value <= 0:
        raise _policy_denied(
            ShellExecutionErrorCode.INVALID_OPTION,
            "timeout_seconds must be positive",
        )
    if value > max_value:
        _record_budget_clamp(metadata, "timeout_seconds", value, max_value)
        return max_value
    return float(value)


def _normalized_byte_budget(
    value: object | None,
    max_value: int,
    field_name: str,
    metadata: dict[str, object],
) -> int:
    if value is None:
        return max_value
    if not isinstance(value, int) or isinstance(value, bool):
        raise _policy_denied(
            ShellExecutionErrorCode.INVALID_OPTION,
            f"{field_name} must be an integer",
        )
    if value <= 0:
        raise _policy_denied(
            ShellExecutionErrorCode.INVALID_OPTION,
            f"{field_name} must be positive",
        )
    if value > max_value:
        _record_budget_clamp(metadata, field_name, value, max_value)
        return max_value
    return value


def _record_budget_clamp(
    metadata: dict[str, object],
    field_name: str,
    requested: int | float,
    applied: int | float,
) -> None:
    existing = metadata.setdefault("budget_clamps", {})
    if not isinstance(existing, dict):
        raise _policy_denied(
            ShellExecutionErrorCode.INVALID_OPTION,
            "budget_clamps metadata must be a dictionary",
        )
    existing[field_name] = {"requested": requested, "applied": applied}


def _validated_metadata(value: Mapping[str, object]) -> dict[str, object]:
    metadata: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key.strip():
            raise _policy_denied(
                ShellExecutionErrorCode.INVALID_OPTION,
                "metadata keys must be non-empty strings",
            )
        metadata[key] = item
    return metadata


def _with_local_audit_metadata(
    metadata: dict[str, object],
    settings: ShellToolSettings,
) -> dict[str, object]:
    if settings.execution_mode != "local":
        return metadata
    audited = dict(metadata)
    audited["local_host_approval"] = "required"
    return audited


def _validate_argument_budgets(
    request: ShellCommandRequest,
    settings: ShellToolSettings,
) -> None:
    _validate_argument_fragments(
        tuple(_request_argument_fragments(request)),
        settings,
        "request",
    )


def _validate_argv_budgets(
    argv: tuple[str, ...],
    display_argv: tuple[str, ...],
    settings: ShellToolSettings,
) -> None:
    _validate_argument_fragments(
        argv,
        settings,
        "normalized command",
        trusted_argument_count=_trusted_internal_argument_count(
            argv, display_argv
        ),
    )


def _validate_argument_fragments(
    fragments: tuple[str, ...],
    settings: ShellToolSettings,
    label: str,
    *,
    trusted_argument_count: int = 0,
) -> None:
    assert isinstance(
        trusted_argument_count, int
    ), "trusted_argument_count must be an integer"
    assert (
        trusted_argument_count >= 0
    ), "trusted_argument_count must be non-negative"
    counted_fragments = max(len(fragments) - trusted_argument_count, 0)
    if counted_fragments > settings.max_arguments:
        raise _policy_denied(
            ShellExecutionErrorCode.TOO_MANY_ARGUMENTS,
            f"{label} has too many arguments",
        )
    try:
        encoded_fragments = tuple(
            fragment.encode("utf-8") for fragment in fragments
        )
    except UnicodeEncodeError as error:
        raise _policy_denied(
            ShellExecutionErrorCode.INVALID_OPTION,
            f"{label} must contain valid Unicode",
        ) from error
    if any(
        len(encoded_fragment) > settings.max_argument_bytes
        for encoded_fragment in encoded_fragments
    ):
        raise _policy_denied(
            ShellExecutionErrorCode.ARGUMENT_TOO_LARGE,
            f"{label} argument is too large",
        )
    command_bytes = sum(len(fragment) for fragment in encoded_fragments)
    command_bytes += max(len(encoded_fragments) - 1, 0)
    if command_bytes > settings.max_command_bytes:
        raise _policy_denied(
            ShellExecutionErrorCode.COMMAND_TOO_LARGE,
            f"{label} is too large",
        )


def _trusted_internal_argument_count(
    argv: tuple[str, ...],
    display_argv: tuple[str, ...],
) -> int:
    assert isinstance(argv, tuple), "argv must be a tuple"
    assert isinstance(display_argv, tuple), "display_argv must be a tuple"
    return max(len(argv) - len(display_argv), 0)


def _request_argument_fragments(
    request: ShellCommandRequest,
) -> tuple[str, ...]:
    fragments = [request.command]
    if request.cwd is not None:
        fragments.append(request.cwd)
    for key, value in request.options.items():
        if not isinstance(key, str) or not key.strip():
            raise _policy_denied(
                ShellExecutionErrorCode.INVALID_OPTION,
                "option keys must be non-empty strings",
            )
        fragments.append(key)
        fragments.extend(_option_fragments(value, f"options.{key}"))
    for path in request.paths:
        fragments.extend((path.name, path.path, path.kind, path.access))
    return tuple(fragments)


def _option_fragments(value: object, field_name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, bool):
        return ("true" if value else "false",)
    if isinstance(value, int):
        return (str(value),)
    if isinstance(value, float):
        if not isfinite(value):
            raise _policy_denied(
                ShellExecutionErrorCode.INVALID_OPTION,
                f"{field_name} must be finite",
            )
        return (str(value),)
    if isinstance(value, str):
        if value == "":
            raise _policy_denied(
                ShellExecutionErrorCode.INVALID_OPTION,
                f"{field_name} must not be empty",
            )
        return (value,)
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        fragments: list[str] = []
        for index, item in enumerate(value):
            fragments.extend(_option_fragments(item, f"{field_name}[{index}]"))
        return tuple(fragments)
    raise _policy_denied(
        ShellExecutionErrorCode.INVALID_OPTION,
        f"{field_name} has unsupported type",
    )

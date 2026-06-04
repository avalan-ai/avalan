from ...cli.theme import Theme
from ...flow import (
    FlowDefinition,
    FlowDefinitionLoader,
    FlowInputType,
    FlowLoadIssue,
    FlowLoadIssueCategory,
    FlowOutputType,
    flow_input_binding,
)
from ...task import (
    TaskDefinition,
    TaskExecutionTarget,
    TaskInputContract,
    TaskMetadata,
    TaskOutputContract,
    TaskTargetType,
    TaskValidationCategory,
    TaskValidationIssue,
    validate_task_input,
    validate_task_output,
)
from .task import (
    TaskCliInputError,
    _format_task_cli_value,
    _print_issues,
    _print_task_cli_input_error,
    _print_task_command_error,
    _run_awaitable,
    _task_diagnostic_console,
    _task_run_json_output,
    _task_run_quiet,
    _task_run_structured_output_requested,
    _validate_task_run_output_path,
    _write_task_run_structured_output,
    task_cli_input,
)

from argparse import Namespace
from collections.abc import Mapping
from logging import Logger
from os import strerror
from pathlib import Path

from rich.console import Console


def flow_run(
    args: Namespace,
    console: Console,
    theme: Theme,
    hub: object | None = None,
    logger: Logger | None = None,
) -> bool:
    """Run a native flow definition."""
    _ = theme, hub, logger
    return _run_awaitable(_flow_run(args, console))


async def _flow_run(args: Namespace, console: Console) -> bool:
    diagnostic_console = _task_diagnostic_console(args, console)
    flow_path = Path(args.flow)
    try:
        load_result = FlowDefinitionLoader().load_result(flow_path)
    except OSError as exc:
        message = strerror(exc.errno) if exc.errno else "Unable to read file."
        diagnostic_console.print(
            "Flow definition could not be read.",
            markup=False,
        )
        diagnostic_console.print(f"error file.read {message}", markup=False)
        return False
    if load_result.definition is None or load_result.flow is None:
        _print_issues(
            diagnostic_console,
            "Flow definition could not be loaded.",
            _flow_load_task_issues(load_result.issues),
        )
        return False
    definition = load_result.definition
    task_definition = _flow_task_definition(definition, flow_path)
    try:
        flow_input = task_cli_input(args, task_definition)
    except TaskCliInputError as exc:
        _print_task_cli_input_error(diagnostic_console, exc)
        return False
    input_issues = validate_task_input(task_definition, flow_input.value)
    if input_issues:
        _print_issues(
            diagnostic_console, "Flow input is invalid.", input_issues
        )
        return False
    if not _validate_flow_local_files(flow_input.value, diagnostic_console):
        return False
    if not _validate_task_run_output_path(args, diagnostic_console):
        return False
    try:
        result = await load_result.flow.execute_async(
            initial_node=definition.entrypoint,
            initial_inputs=flow_input_binding(
                definition.input,
                flow_input.value,
            ),
        )
    except BaseException:
        _print_task_command_error(
            diagnostic_console,
            "Flow run failed.",
            "flow.execution",
            "Fix the flow definition or node configuration and retry.",
        )
        return False
    output_issues = validate_task_output(task_definition, result)
    if output_issues:
        _print_issues(
            diagnostic_console,
            "Flow output is invalid.",
            output_issues,
        )
        return False
    if _task_run_structured_output_requested(args):
        output_written = _write_task_run_structured_output(
            args,
            console,
            diagnostic_console,
            result,
        )
        if not output_written:
            return False
    if not _task_run_json_output(args) and not _task_run_quiet(args):
        console.print("Flow run completed.", markup=False)
        console.print(f"output {_format_task_cli_value(result)}", markup=False)
    return True


def _flow_task_definition(
    definition: FlowDefinition,
    flow_path: Path,
) -> TaskDefinition:
    return TaskDefinition(
        task=TaskMetadata(
            name=definition.name, version=definition.version or "1"
        ),
        input=_flow_task_input(definition),
        output=_flow_task_output(definition),
        execution=TaskExecutionTarget(
            type=TaskTargetType.FLOW,
            ref=str(flow_path),
        ),
        definition_base=flow_path,
    )


def _flow_task_input(definition: FlowDefinition) -> TaskInputContract:
    input_definition = definition.input
    if input_definition is None:
        return TaskInputContract.object()
    match input_definition.type:
        case FlowInputType.STRING:
            return TaskInputContract.string()
        case FlowInputType.INTEGER:
            return TaskInputContract.integer()
        case FlowInputType.NUMBER:
            return TaskInputContract.number()
        case FlowInputType.BOOLEAN:
            return TaskInputContract.boolean()
        case FlowInputType.OBJECT:
            return TaskInputContract.object(
                _plain_mapping(input_definition.schema) or {"type": "object"},
            )
        case FlowInputType.ARRAY:
            return TaskInputContract.array(
                _plain_mapping(input_definition.schema) or {"type": "array"}
            )
        case FlowInputType.FILE:
            return TaskInputContract.file(
                mime_types=input_definition.mime_types,
            )
        case FlowInputType.FILE_ARRAY:
            return TaskInputContract.file_array(
                mime_types=input_definition.mime_types,
            )
    raise AssertionError("unsupported flow input type")  # pragma: no cover


def _flow_task_output(definition: FlowDefinition) -> TaskOutputContract:
    output_definition = definition.output
    if output_definition is None:
        return TaskOutputContract.json({})
    schema = _plain_mapping(output_definition.schema)
    match output_definition.type:
        case FlowOutputType.TEXT:
            return TaskOutputContract.text()
        case FlowOutputType.JSON:
            return TaskOutputContract.json(schema or {})
        case FlowOutputType.OBJECT:
            return TaskOutputContract.object(schema or {"type": "object"})
        case FlowOutputType.ARRAY:
            return TaskOutputContract.array(schema or {"type": "array"})
        case FlowOutputType.FILE:
            return TaskOutputContract.file()
        case FlowOutputType.FILE_ARRAY:
            return TaskOutputContract.file_array()
    raise AssertionError("unsupported flow output type")  # pragma: no cover


def _flow_load_task_issues(
    issues: tuple[FlowLoadIssue, ...],
) -> tuple[TaskValidationIssue, ...]:
    return tuple(
        TaskValidationIssue(
            code=issue.code,
            path=issue.path,
            message=issue.message,
            hint=issue.hint,
            category=_flow_task_validation_category(issue.category),
        )
        for issue in issues
    )


def _flow_task_validation_category(
    category: FlowLoadIssueCategory,
) -> TaskValidationCategory:
    match category:
        case FlowLoadIssueCategory.PARSE | FlowLoadIssueCategory.STRUCTURE:
            return TaskValidationCategory.STRUCTURE
        case FlowLoadIssueCategory.VALUE:
            return TaskValidationCategory.VALUE
        case FlowLoadIssueCategory.UNSUPPORTED:
            return TaskValidationCategory.UNSUPPORTED
        case FlowLoadIssueCategory.PRIVACY:
            return TaskValidationCategory.PRIVACY
    raise AssertionError("unsupported flow issue category")  # pragma: no cover


def _plain_mapping(
    value: Mapping[str, object] | None,
) -> Mapping[str, object] | None:
    if value is None:
        return None
    return {str(key): _plain_value(item) for key, item in value.items()}


def _plain_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _plain_value(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_plain_value(item) for item in value]
    return value


def _validate_flow_local_files(value: object, console: Console) -> bool:
    for descriptor in _flow_local_file_descriptors(value):
        reference = descriptor.get("reference")
        if not isinstance(reference, str) or not Path(reference).is_file():
            _print_task_command_error(
                console,
                "Flow input file could not be read.",
                "input.file_missing",
                "Pass a readable local file path.",
            )
            return False
    return True


def _flow_local_file_descriptors(
    value: object,
) -> tuple[Mapping[str, object], ...]:
    descriptors: list[Mapping[str, object]] = []
    _collect_flow_local_file_descriptors(value, descriptors)
    return tuple(descriptors)


def _collect_flow_local_file_descriptors(
    value: object,
    descriptors: list[Mapping[str, object]],
) -> None:
    if isinstance(value, Mapping):
        if value.get("source_kind") == "local_path":
            descriptors.append(value)
            return
        for item in value.values():
            _collect_flow_local_file_descriptors(item, descriptors)
    elif isinstance(value, list | tuple):
        for item in value:
            _collect_flow_local_file_descriptors(item, descriptors)

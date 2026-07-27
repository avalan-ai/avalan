"""Render canonical task input through an injected CLI control channel."""

from ..interaction import error
from ..interaction.entities import (
    AnswerProvenance,
    Choice,
    ChoiceValue,
    ConfirmationAnswer,
    ConfirmationQuestion,
    FreeFormOther,
    InputAnswer,
    InputQuestion,
    InputTimedOutResult,
    MultilineTextAnswer,
    MultilineTextQuestion,
    MultipleSelectionAnswer,
    MultipleSelectionQuestion,
    SelectedChoice,
    SingleSelectionAnswer,
    SingleSelectionQuestion,
    TextAnswer,
    TextQuestion,
)
from ..sdk import (
    AgentRunCancelled,
    AgentRunCompleted,
    AgentRunInputRequired,
    AttachedInputContext,
    AttachedInputDisconnected,
    AttachedInputDisconnectReason,
    AttachedInputOutcome,
    InputAnswerSubmission,
    InputDeclineSubmission,
    InputResolutionAccepted,
)
from .display_safety import strip_terminal_controls
from .interaction_channel import CliInteractionChannelProtocol

from asyncio.exceptions import CancelledError
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal, Protocol, TypeAlias, final
from unicodedata import category

_CONTROL_HELP = (
    ":decline decline this request; :cancel cancel only this input; "
    ":cancel-run cancel the containing run; :steer TEXT send steering; "
    ":help show help. Prefix control-looking text with an extra ':'."
)


class CliInteractionCommandKind(StrEnum):
    """Identify controls that must remain distinct from input submissions."""

    INPUT_CANCELLED = "input_cancelled"
    RUN_CANCELLED = "run_cancelled"
    STEERING = "steering"


class CliInteractionCommandDisposition(StrEnum):
    """Identify whether the owning CLI accepted a non-input command."""

    ACCEPTED = "accepted"
    UNAVAILABLE = "unavailable"


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class CliInteractionResult:
    """Describe one stable CLI control-channel result."""

    envelope_id: str
    payload: dict[str, object]
    exit_code: int | None


@final
class CliInteractionExit(SystemExit):
    """Terminate the CLI with one typed interaction result."""

    def __init__(self, result: CliInteractionResult) -> None:
        assert result.exit_code is not None
        self.result = result
        super().__init__(result.exit_code)


class CliRunCancelled(CancelledError):
    """Signal explicit structured containing-run cancellation."""


def _cli_result(
    name: str, payload: dict[str, object], exit_code: int | None
) -> CliInteractionResult:
    return CliInteractionResult(
        envelope_id=f"cli.{name}.v1",
        payload={**payload, "channel": "control"},
        exit_code=exit_code,
    )


def cli_input_required_result(
    request_id: str, continuation_id: str
) -> CliInteractionResult:
    """Return the frozen resumable CLI input-required result."""
    return _cli_result(
        "input_required",
        {
            "kind": "input_required",
            "request_id": request_id,
            "continuation_id": continuation_id,
            "detached_resumption_available": True,
        },
        75,
    )


def cli_interaction_result(value: object) -> CliInteractionResult:
    """Project one public interaction outcome onto the CLI control channel."""
    if isinstance(value, AgentRunCompleted):
        payload = {"kind": "completed", "value": value.to_json()}
        return _cli_result("completed", payload, 0)
    if isinstance(value, AgentRunInputRequired):
        assert value.request_id and value.continuation_id
        assert value.detached_resumption_available
        return cli_input_required_result(
            str(value.request_id), str(value.continuation_id)
        )
    if isinstance(value, AgentRunCancelled):
        return _cli_result("cancelled", {"kind": "cancelled"}, 130)
    if isinstance(value, InputResolutionAccepted):
        assert value.interaction_state == "answered" and value.idempotent
        payload = dict(
            kind=value.kind,
            interaction_state=value.interaction_state,
            idempotent=value.idempotent,
        )
        return _cli_result("resolution_accepted", payload, 0)
    if isinstance(value, InputTimedOutResult):
        payload = dict(kind="timed_out", interaction_state="timed_out")
        return _cli_result("timed_out", payload, 0)
    if isinstance(value, error.InputContractError):
        result = cli_input_error_result(value)
        if result is None:
            raise TypeError("unsupported public interaction failure")
        return result
    raise TypeError("value must be a public interaction outcome")


def cli_input_error_result(
    value: error.InputContractError,
) -> CliInteractionResult | None:
    """Project a supported public input failure, or return no result."""
    payload: dict[str, object]
    if isinstance(value, error.InputValidationError):
        payload = {"error": "input.validation", "interaction_state": "pending"}
        return _cli_result("validation_error", payload, None)
    result = _CLI_INPUT_ERROR_RESULTS.get(value.code)
    if result is None:
        return None
    name, payload, exit_code = result
    return _cli_result(name, payload, exit_code)


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class CliInputCancellationCommand:
    """Cancel only the active input interaction."""

    kind: Literal[CliInteractionCommandKind.INPUT_CANCELLED] = field(
        init=False,
        default=CliInteractionCommandKind.INPUT_CANCELLED,
    )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class CliRunCancellationCommand:
    """Cancel the run containing the active interaction."""

    kind: Literal[CliInteractionCommandKind.RUN_CANCELLED] = field(
        init=False,
        default=CliInteractionCommandKind.RUN_CANCELLED,
    )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class CliSteeringCommand:
    """Carry unsolicited steering separately from task-input answers."""

    text: str
    kind: Literal[CliInteractionCommandKind.STEERING] = field(
        init=False,
        default=CliInteractionCommandKind.STEERING,
    )

    def __post_init__(self) -> None:
        assert isinstance(self.text, str)
        assert self.text.strip()
        assert "\r" not in self.text and "\n" not in self.text


CliInteractionCommand: TypeAlias = (
    CliInputCancellationCommand
    | CliRunCancellationCommand
    | CliSteeringCommand
)


class CliInteractionCommandHandler(Protocol):
    """Dispatch run cancellation and steering outside input resolution."""

    async def __call__(
        self,
        command: CliRunCancellationCommand | CliSteeringCommand,
    ) -> CliInteractionCommandDisposition:
        """Return whether one non-input command was accepted."""
        ...


_CLI_INPUT_ERROR_RESULTS: dict[
    error.InputErrorCode,
    tuple[str, dict[str, object], int],
] = {
    error.InputErrorCode.UNAVAILABLE: (
        "unavailable",
        {"kind": "unavailable", "detached_resumption_available": False},
        69,
    ),
    error.InputErrorCode.ALREADY_RESOLVED: (
        "already_resolved",
        {"error": "input.already_resolved", "interaction_state": "answered"},
        73,
    ),
    error.InputErrorCode.EXPIRED: (
        "expired",
        {"error": "input.expired", "interaction_state": "expired"},
        74,
    ),
    error.InputErrorCode.SUPERSEDED: (
        "superseded",
        {"error": "input.superseded", "interaction_state": "superseded"},
        75,
    ),
}


_EarlyOutcome: TypeAlias = InputDeclineSubmission | AttachedInputDisconnected
_QuestionResult: TypeAlias = InputAnswer | _EarlyOutcome | None
_LineResult: TypeAlias = str | _EarlyOutcome | CliInteractionCommand | None


@final
class CliInteractionRenderer:
    """Collect canonical answers with an accessible async text renderer."""

    _channel: CliInteractionChannelProtocol
    _command_handler: CliInteractionCommandHandler | None

    def __init__(
        self,
        channel: CliInteractionChannelProtocol,
        *,
        command_handler: CliInteractionCommandHandler | None = None,
    ) -> None:
        assert channel is not None
        self._channel = channel
        self._command_handler = command_handler

    async def __call__(
        self,
        context: AttachedInputContext,
    ) -> AttachedInputOutcome:
        """Render one attached interaction and return its typed outcome."""
        return await self.render(context)

    async def render(
        self,
        context: AttachedInputContext,
    ) -> AttachedInputOutcome:
        """Render one attached interaction and return its typed outcome."""
        assert type(context) is AttachedInputContext
        request = context.request
        assert 1 <= len(request.questions) <= 3
        await self._write_line("Input required")
        if request.context_label is not None:
            await self._write_line(
                f"Context: {_literal_text(request.context_label)}"
            )
        await self._write_line(f"Reason: {_literal_text(request.reason)}")
        await self._write_line(f"Controls: {_CONTROL_HELP}")
        if context.validation_error is not None:
            feedback = context.validation_error
            await self._write_line(
                "Previous input was not accepted: "
                f"{_literal_text(feedback.message)} "
                f"({_literal_text(feedback.code.value)} at "
                f"{_literal_text(feedback.path)})."
            )

        answers: list[InputAnswer] = []
        total = len(request.questions)
        for index, question in enumerate(request.questions, start=1):
            result = await self._ask_question(question, index, total)
            if isinstance(
                result,
                (
                    InputDeclineSubmission,
                    AttachedInputDisconnected,
                ),
            ):
                return result
            if result is None:
                continue
            assert isinstance(result, InputAnswer)
            answers.append(result)
        return InputAnswerSubmission(
            answers=tuple(answers),
            provenance=AnswerProvenance.HUMAN,
        )

    async def _ask_question(
        self,
        question: InputQuestion,
        index: int,
        total: int,
    ) -> _QuestionResult:
        heading = (
            question.header
            if question.header is not None
            else question.kind.value.replace("_", " ").title()
        )
        await self._write_line("")
        await self._write_line(
            f"Question {index}/{total} - {_literal_text(heading)}"
        )
        await self._write_line(_literal_text(question.prompt))
        if question.help_text is not None:
            await self._write_line(
                f"Help: {_literal_text(question.help_text)}"
            )

        match question:
            case ConfirmationQuestion():
                return await self._ask_confirmation(question)
            case TextQuestion():
                return await self._ask_text(question)
            case MultilineTextQuestion():
                return await self._ask_multiline(question)
            case SingleSelectionQuestion():
                return await self._ask_single_selection(question)
            case MultipleSelectionQuestion():
                return await self._ask_multiple_selection(question)
        raise AssertionError("unsupported canonical question variant")

    async def _ask_confirmation(
        self,
        question: ConfirmationQuestion,
    ) -> _QuestionResult:
        if question.default_value is None:
            default = ""
        else:
            default = (
                " [default: yes]"
                if question.default_value
                else " [default: no]"
            )
        while True:
            await self._write_line(f"Answer yes or no{default}:")
            line = await self._read_controlled_line(question)
            if not isinstance(line, str):
                return line
            normalized = line.strip().casefold()
            if not normalized:
                if question.default_value is not None:
                    value = question.default_value
                elif not question.required:
                    return None
                else:
                    await self._invalid("Enter yes or no.")
                    continue
            elif normalized in {"y", "yes"}:
                value = True
            elif normalized in {"n", "no"}:
                value = False
            else:
                await self._invalid("Enter yes or no.")
                continue
            return ConfirmationAnswer(
                question_id=question.question_id,
                provenance=AnswerProvenance.HUMAN,
                value=value,
            )

    async def _ask_text(self, question: TextQuestion) -> _QuestionResult:
        if question.default_value is not None:
            await self._write_line(
                f"Default: {_literal_text(question.default_value)}"
            )
        while True:
            await self._write_line("Enter one line:")
            line = await self._read_controlled_line(question)
            if not isinstance(line, str):
                return line
            if not line:
                if question.default_value is not None:
                    line = question.default_value
                elif not question.required:
                    return None
            minimum = max(
                question.constraints.minimum_length,
                int(question.required),
            )
            if (
                len(line) < minimum
                or len(line) > question.constraints.maximum_length
            ):
                await self._invalid(
                    _length_message(
                        minimum,
                        question.constraints.maximum_length,
                    )
                )
                continue
            try:
                return TextAnswer(
                    question_id=question.question_id,
                    provenance=AnswerProvenance.HUMAN,
                    value=line,
                )
            except error.InputValidationError as exc:
                await self._invalid(exc.safe_message)

    async def _ask_multiline(
        self,
        question: MultilineTextQuestion,
    ) -> _QuestionResult:
        if question.default_value is not None:
            await self._write_line(
                f"Default: {_literal_text(question.default_value)}"
            )
        while True:
            await self._write_line(
                "Enter text; finish with a line containing only '.'. "
                "Enter '..' for a literal '.'."
            )
            lines: list[str] = []
            while True:
                line = await self._read_controlled_line(question)
                if not isinstance(line, str):
                    return line
                if line == ".":
                    break
                lines.append("." if line == ".." else line)
            value = "\n".join(lines)
            if not value:
                if question.default_value is not None:
                    value = question.default_value
                elif not question.required:
                    return None
            minimum = max(
                question.constraints.minimum_length,
                int(question.required),
            )
            if (
                len(value) < minimum
                or len(value) > question.constraints.maximum_length
            ):
                await self._invalid(
                    _length_message(
                        minimum,
                        question.constraints.maximum_length,
                    )
                )
                continue
            try:
                return MultilineTextAnswer(
                    question_id=question.question_id,
                    provenance=AnswerProvenance.HUMAN,
                    value=value,
                )
            except error.InputValidationError as exc:
                await self._invalid(exc.safe_message)

    async def _ask_single_selection(
        self,
        question: SingleSelectionQuestion,
    ) -> _QuestionResult:
        await self._render_choices(question)
        while True:
            await self._write_line("Select one by number:")
            line = await self._read_controlled_line(question)
            if not isinstance(line, str):
                return line
            value = line.strip()
            if not value:
                if question.default_value is not None:
                    return SingleSelectionAnswer(
                        question_id=question.question_id,
                        provenance=AnswerProvenance.HUMAN,
                        value=SelectedChoice(value=question.default_value),
                    )
                if not question.required:
                    return None
                await self._invalid("Select one option.")
                continue
            selection = _selection_at(question.choices, value)
            if selection is not None:
                return SingleSelectionAnswer(
                    question_id=question.question_id,
                    provenance=AnswerProvenance.HUMAN,
                    value=SelectedChoice(value=selection.value),
                )
            if question.allow_other and _is_other_index(
                question.choices,
                value,
            ):
                other = await self._ask_other(question)
                if isinstance(other, str):
                    return SingleSelectionAnswer(
                        question_id=question.question_id,
                        provenance=AnswerProvenance.HUMAN,
                        value=FreeFormOther(text=other),
                    )
                return other
            await self._invalid("Select a listed option number.")

    async def _ask_multiple_selection(
        self,
        question: MultipleSelectionQuestion,
    ) -> _QuestionResult:
        await self._render_choices(question)
        while True:
            await self._write_line(
                "Select numbers separated by commas, or enter 'none':"
            )
            line = await self._read_controlled_line(question)
            if not isinstance(line, str):
                return line
            raw = line.strip()
            if not raw:
                if question.default_value is not None:
                    values = tuple(
                        SelectedChoice(value=value)
                        for value in question.default_value
                    )
                    return MultipleSelectionAnswer(
                        question_id=question.question_id,
                        provenance=AnswerProvenance.HUMAN,
                        values=values,
                    )
                if not question.required:
                    return None
                await self._invalid("Select at least one option.")
                continue

            tokens = (
                ()
                if raw.casefold() == "none"
                else tuple(raw.replace(",", " ").split())
            )
            if len(tokens) != len(set(tokens)):
                await self._invalid("Do not select an option more than once.")
                continue
            selections: list[SelectedChoice | FreeFormOther] = []
            selected_values: set[ChoiceValue] = set()
            other_selected = False
            duplicate = False
            invalid = False
            for token in tokens:
                choice = _selection_at(question.choices, token)
                if choice is not None:
                    if choice.value in selected_values:
                        duplicate = True
                        break
                    selected_values.add(choice.value)
                    selections.append(SelectedChoice(value=choice.value))
                elif question.allow_other and _is_other_index(
                    question.choices,
                    token,
                ):
                    if other_selected:
                        duplicate = True
                        break
                    other_selected = True
                else:
                    invalid = True
                    break
            if duplicate:
                await self._invalid("Do not select an option more than once.")
                continue
            if invalid:
                await self._invalid("Select only listed option numbers.")
                continue
            minimum = max(question.constraints.minimum, int(question.required))
            count = len(selections) + int(other_selected)
            if count < minimum or count > question.constraints.maximum:
                await self._invalid(
                    _selection_count_message(
                        minimum,
                        question.constraints.maximum,
                    )
                )
                continue
            if other_selected:
                other = await self._ask_other(question)
                if not isinstance(other, str):
                    return other
                selections.append(FreeFormOther(text=other))
            return MultipleSelectionAnswer(
                question_id=question.question_id,
                provenance=AnswerProvenance.HUMAN,
                values=tuple(selections),
            )

    async def _render_choices(
        self,
        question: SingleSelectionQuestion | MultipleSelectionQuestion,
    ) -> None:
        defaults = (
            frozenset((question.default_value,))
            if isinstance(question, SingleSelectionQuestion)
            and question.default_value is not None
            else frozenset(
                question.default_value
                if isinstance(question, MultipleSelectionQuestion)
                and question.default_value is not None
                else ()
            )
        )
        for index, choice in enumerate(question.choices, start=1):
            markers: list[str] = []
            if choice.value == question.recommended_choice:
                markers.append("recommended")
            if choice.value in defaults:
                markers.append("default")
            marker = f" [{', '.join(markers)}]" if markers else ""
            description = (
                ""
                if choice.description is None
                else f" - {_literal_text(choice.description)}"
            )
            await self._write_line(
                f"  {index}. {_literal_text(choice.label)}"
                f"{description}{marker}"
            )
        if question.allow_other:
            await self._write_line(f"  {len(question.choices) + 1}. Other")

    async def _ask_other(
        self,
        question: SingleSelectionQuestion | MultipleSelectionQuestion,
    ) -> str | _EarlyOutcome:
        while True:
            await self._write_line("Enter the Other value:")
            line = await self._read_controlled_line(question)
            if not isinstance(line, str):
                return line
            if not line.strip():
                await self._invalid("Other must not be empty.")
                continue
            try:
                return FreeFormOther(text=line).text
            except error.InputValidationError as exc:
                await self._invalid(exc.safe_message)

    async def _read_controlled_line(
        self,
        question: InputQuestion,
    ) -> str | _EarlyOutcome:
        while True:
            line = await self._channel.read_line()
            if line is None:
                return AttachedInputDisconnected(
                    reason=(
                        AttachedInputDisconnectReason.CONTROL_CHANNEL_CLOSED
                    )
                )
            control = _parse_control(line)
            if control is None:
                help_text = (
                    "No additional question help is available."
                    if question.help_text is None
                    else _literal_text(question.help_text)
                )
                await self._write_line(f"Help: {help_text}")
                await self._write_line(f"Controls: {_CONTROL_HELP}")
                continue
            if isinstance(control, str):
                return control
            if isinstance(control, CliInputCancellationCommand):
                return AttachedInputDisconnected(
                    reason=AttachedInputDisconnectReason.HANDLER_CANCELLED
                )
            if isinstance(
                control,
                (CliRunCancellationCommand, CliSteeringCommand),
            ):
                disposition = (
                    CliInteractionCommandDisposition.UNAVAILABLE
                    if self._command_handler is None
                    else await self._command_handler(control)
                )
                assert isinstance(
                    disposition,
                    CliInteractionCommandDisposition,
                )
                if disposition is CliInteractionCommandDisposition.UNAVAILABLE:
                    control_name = (
                        "Containing-run cancellation"
                        if isinstance(control, CliRunCancellationCommand)
                        else "Steering"
                    )
                    await self._invalid(
                        f"{control_name} is unavailable in this CLI session."
                    )
                    continue
                if isinstance(control, CliRunCancellationCommand):
                    return AttachedInputDisconnected(
                        reason=(
                            AttachedInputDisconnectReason.HANDLER_CANCELLED
                        )
                    )
                await self._write_line(
                    "Steering sent; this input request is still pending."
                )
                continue
            return control

    async def _invalid(self, message: str) -> None:
        await self._write_line(f"Invalid input: {_literal_text(message)}")

    async def _write_line(self, text: str) -> None:
        await self._channel.write(f"{text}\n")


def _parse_control(line: str) -> _LineResult:
    if line.startswith("::"):
        return line[1:]
    normalized = line.strip()
    folded = normalized.casefold()
    if folded in {"?", ":help"}:
        return None
    if folded == ":decline":
        return InputDeclineSubmission(provenance=AnswerProvenance.HUMAN)
    if folded in {":cancel", ":cancel-input"}:
        return CliInputCancellationCommand()
    if folded == ":cancel-run":
        return CliRunCancellationCommand()
    if folded == ":steer":
        return None
    if folded.startswith(":steer "):
        steering = normalized[len(":steer ") :].strip()
        return CliSteeringCommand(text=steering)
    return line


def _selection_at(choices: tuple[Choice, ...], token: str) -> Choice | None:
    index = _selection_index(token, len(choices))
    if index is None:
        return None
    return choices[index - 1]


def _is_other_index(choices: tuple[Choice, ...], token: str) -> bool:
    other_index = len(choices) + 1
    return _selection_index(token, other_index) == other_index


def _selection_index(token: str, maximum: int) -> int | None:
    assert isinstance(token, str)
    assert isinstance(maximum, int)
    assert maximum >= 1
    if not token.isdecimal():
        return None
    normalized = token.lstrip("0") or "0"
    if len(normalized) > len(str(maximum)):
        return None
    index = int(normalized)
    return index if 1 <= index <= maximum else None


def _length_message(minimum: int, maximum: int) -> str:
    if minimum == maximum:
        return f"Enter exactly {minimum} characters."
    return f"Enter between {minimum} and {maximum} characters."


def _selection_count_message(minimum: int, maximum: int) -> str:
    if minimum == maximum:
        return f"Select exactly {minimum} options."
    return f"Select between {minimum} and {maximum} options."


def _literal_text(text: str) -> str:
    """Return inert single-line terminal text without changing its meaning."""
    assert isinstance(text, str)
    stripped = strip_terminal_controls(text)
    rendered: list[str] = []
    for character in stripped:
        codepoint = ord(character)
        if character == "\n":
            rendered.append("\\n")
        elif character == "\r":
            rendered.append("\\r")
        elif character == "\t":
            rendered.append("\\t")
        elif category(character) in {"Cc", "Cf", "Cs"}:
            rendered.append(
                f"\\u{codepoint:04x}"
                if codepoint <= 0xFFFF
                else f"\\U{codepoint:08x}"
            )
        else:
            rendered.append(character)
    return "".join(rendered)

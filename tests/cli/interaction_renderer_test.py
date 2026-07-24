"""Exercise the async semantic CLI interaction renderer."""

from asyncio import CancelledError, Event, create_task, sleep
from collections import deque
from contextlib import redirect_stderr, redirect_stdout
from datetime import UTC, datetime
from io import StringIO
from pathlib import Path
from typing import cast
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import patch

from avalan.cli.interaction_renderer import (
    CliInputCancellationCommand,
    CliInteractionCommand,
    CliInteractionCommandDisposition,
    CliInteractionCommandKind,
    CliInteractionRenderer,
    CliRunCancellationCommand,
    CliSteeringCommand,
    _length_message,
    _literal_text,
    _selection_count_message,
)
from avalan.interaction import (
    AnswerProvenance,
    Choice,
    ChoiceValue,
    ConfirmationAnswer,
    ConfirmationQuestion,
    FreeFormOther,
    InputErrorCode,
    InputQuestion,
    InputValidationError,
    MultilineTextAnswer,
    MultilineTextQuestion,
    MultipleSelectionAnswer,
    MultipleSelectionQuestion,
    PresentationHint,
    QuestionId,
    QuestionType,
    RequestState,
    RequirementMode,
    SelectedChoice,
    SelectionValidationConstraints,
    SingleSelectionAnswer,
    SingleSelectionQuestion,
    StateRevision,
    TextAnswer,
    TextQuestion,
    TextValidationConstraints,
)
from avalan.sdk import (
    AttachedInputContext,
    AttachedInputDisconnected,
    AttachedInputDisconnectReason,
    InputAnswerSubmission,
    InputDeclineSubmission,
    InputRequestView,
    InputValidationFeedback,
)

_CREATED_AT = datetime(2026, 7, 24, 12, 0, tzinfo=UTC)


class _FakeChannel:
    """Record control output and serve deterministic async input."""

    def __init__(self, *lines: str | None | BaseException) -> None:
        self.lines = deque(lines)
        self.read_count = 0
        self.writes: list[str] = []
        self.closed = False

    async def write(self, text: str) -> None:
        self.writes.append(text)

    async def read_line(self) -> str | None:
        self.read_count += 1
        if not self.lines:
            return None
        item = self.lines.popleft()
        if isinstance(item, BaseException):
            raise item
        return item

    async def aclose(self) -> None:
        self.closed = True

    @property
    def output(self) -> str:
        return "".join(self.writes)


class _BlockingChannel(_FakeChannel):
    """Block one read until its caller is cancelled."""

    def __init__(self) -> None:
        super().__init__()
        self.reading = Event()
        self.release = Event()

    async def read_line(self) -> str | None:
        self.read_count += 1
        self.reading.set()
        await self.release.wait()
        return "yes"


class _CommandRecorder:
    """Record renderer-level run and steering commands."""

    def __init__(
        self,
        disposition: CliInteractionCommandDisposition = (
            CliInteractionCommandDisposition.ACCEPTED
        ),
    ) -> None:
        self.commands: list[CliRunCancellationCommand | CliSteeringCommand] = (
            []
        )
        self.disposition = disposition

    async def __call__(
        self,
        command: CliRunCancellationCommand | CliSteeringCommand,
    ) -> CliInteractionCommandDisposition:
        self.commands.append(command)
        return self.disposition


class _UnsupportedQuestion:
    """Provide enough presentation fields to reach the closed variant check."""

    header = None
    help_text = None
    kind = QuestionType.TEXT
    prompt = "Unsupported."


def _context(
    *questions: (
        ConfirmationQuestion
        | TextQuestion
        | MultilineTextQuestion
        | SingleSelectionQuestion
        | MultipleSelectionQuestion
    ),
    reason: str = "The agent needs more information.",
    validation_error: InputValidationFeedback | None = None,
) -> AttachedInputContext:
    return AttachedInputContext(
        request=InputRequestView(
            mode=RequirementMode.REQUIRED,
            reason=reason,
            questions=questions,
            created_at=_CREATED_AT,
            state=RequestState.PENDING,
            state_revision=StateRevision(1),
        ),
        validation_error=validation_error,
    )


def _confirmation(
    *,
    required: bool = True,
    default: bool | None = None,
    hint: PresentationHint | None = None,
    prompt: str = "Continue?",
    header: str | None = None,
    help_text: str | None = None,
) -> ConfirmationQuestion:
    return ConfirmationQuestion(
        question_id=QuestionId("confirm"),
        prompt=prompt,
        required=required,
        default_value=default,
        presentation_hint=hint,
        header=header,
        help_text=help_text,
    )


def _choices(
    *,
    first_label: str = "Fast",
    first_description: str | None = "Finish sooner.",
) -> tuple[Choice, ...]:
    return (
        Choice(
            value=ChoiceValue("stable-fast"),
            label=first_label,
            description=first_description,
        ),
        Choice(
            value=ChoiceValue("stable-safe"),
            label="Careful",
            description="Run every check.",
        ),
    )


def _answer(
    result: object,
    answer_type: type[
        ConfirmationAnswer
        | TextAnswer
        | MultilineTextAnswer
        | SingleSelectionAnswer
        | MultipleSelectionAnswer
    ],
) -> (
    ConfirmationAnswer
    | TextAnswer
    | MultilineTextAnswer
    | SingleSelectionAnswer
    | MultipleSelectionAnswer
):
    assert isinstance(result, InputAnswerSubmission)
    assert result.provenance is AnswerProvenance.HUMAN
    assert len(result.answers) == 1
    answer = result.answers[0]
    assert isinstance(answer, answer_type)
    assert answer.provenance is AnswerProvenance.HUMAN
    return answer


class CliInteractionRendererTestCase(IsolatedAsyncioTestCase):
    async def test_confirmation_reprompts_defaults_and_skips_optional(
        self,
    ) -> None:
        invalid = _FakeChannel("maybe", "NO")
        result = await CliInteractionRenderer(invalid).render(
            _context(_confirmation())
        )

        answer = cast(
            ConfirmationAnswer,
            _answer(result, ConfirmationAnswer),
        )
        self.assertFalse(answer.value)
        self.assertIn("Invalid input: Enter yes or no.", invalid.output)
        self.assertEqual(invalid.read_count, 2)

        default = _FakeChannel("")
        default_result = await CliInteractionRenderer(default)(
            _context(_confirmation(default=True))
        )
        default_answer = cast(
            ConfirmationAnswer,
            _answer(default_result, ConfirmationAnswer),
        )
        self.assertTrue(default_answer.value)
        self.assertIn("[default: yes]", default.output)

        optional = _FakeChannel("")
        optional_result = await CliInteractionRenderer(optional).render(
            _context(_confirmation(required=False))
        )
        self.assertIsInstance(optional_result, InputAnswerSubmission)
        assert isinstance(optional_result, InputAnswerSubmission)
        self.assertEqual(optional_result.answers, ())

        required_empty = _FakeChannel("", "yes")
        required_empty_result = await CliInteractionRenderer(
            required_empty
        ).render(_context(_confirmation()))
        required_empty_answer = cast(
            ConfirmationAnswer,
            _answer(required_empty_result, ConfirmationAnswer),
        )
        self.assertTrue(required_empty_answer.value)
        self.assertIn("Invalid input: Enter yes or no.", required_empty.output)

    async def test_text_reprompts_accepts_default_and_escapes_controls(
        self,
    ) -> None:
        question = TextQuestion(
            question_id=QuestionId("name"),
            prompt="Enter a name.",
            required=True,
            constraints=TextValidationConstraints(
                minimum_length=2,
                maximum_length=4,
            ),
        )
        channel = _FakeChannel("x", "okay")

        result = await CliInteractionRenderer(channel).render(
            _context(question)
        )

        answer = cast(TextAnswer, _answer(result, TextAnswer))
        self.assertEqual(answer.value, "okay")
        self.assertIn("between 2 and 4 characters", channel.output)

        default_question = TextQuestion(
            question_id=QuestionId("name"),
            prompt="Enter a name.",
            required=True,
            default_value="[guest]\x1b[31m",
        )
        default_channel = _FakeChannel("")
        default_result = await CliInteractionRenderer(default_channel).render(
            _context(default_question)
        )
        default_answer = cast(
            TextAnswer,
            _answer(default_result, TextAnswer),
        )
        self.assertEqual(default_answer.value, "[guest]\x1b[31m")
        self.assertIn("[guest]", default_channel.output)
        self.assertNotIn("\x1b", default_channel.output)

        escaped = _FakeChannel("::decline")
        escaped_question = TextQuestion(
            question_id=QuestionId("name"),
            prompt="Enter a name.",
            required=True,
        )
        escaped_result = await CliInteractionRenderer(escaped).render(
            _context(escaped_question)
        )
        escaped_answer = cast(
            TextAnswer,
            _answer(escaped_result, TextAnswer),
        )
        self.assertEqual(escaped_answer.value, ":decline")

        optional_question = TextQuestion(
            question_id=QuestionId("note"),
            prompt="Optional note.",
            required=False,
        )
        skipped = await CliInteractionRenderer(_FakeChannel("")).render(
            _context(optional_question)
        )
        assert isinstance(skipped, InputAnswerSubmission)
        self.assertEqual(skipped.answers, ())

        cancelled = await CliInteractionRenderer(
            _FakeChannel(":cancel")
        ).render(_context(question))
        self.assertEqual(
            cancelled,
            AttachedInputDisconnected(
                reason=AttachedInputDisconnectReason.HANDLER_CANCELLED
            ),
        )

        canonical_question = TextQuestion(
            question_id=QuestionId("canonical"),
            prompt="Canonical text.",
            required=True,
        )
        canonical_channel = _FakeChannel("bad\nvalue", "valid")
        canonical_result = await CliInteractionRenderer(
            canonical_channel
        ).render(_context(canonical_question))
        canonical_answer = cast(
            TextAnswer,
            _answer(canonical_result, TextAnswer),
        )
        self.assertEqual(canonical_answer.value, "valid")
        self.assertIn(
            "value must not contain a newline",
            canonical_channel.output,
        )

        required_blank = _FakeChannel("", "valid")
        blank_result = await CliInteractionRenderer(required_blank).render(
            _context(canonical_question)
        )
        blank_answer = cast(TextAnswer, _answer(blank_result, TextAnswer))
        self.assertEqual(blank_answer.value, "valid")
        self.assertIn(
            "Invalid input: Enter between 1 and 4096 characters.",
            required_blank.output,
        )

    async def test_multiline_reprompts_escapes_dot_and_uses_default(
        self,
    ) -> None:
        question = MultilineTextQuestion(
            question_id=QuestionId("notes"),
            prompt="Enter notes.",
            required=True,
            constraints=TextValidationConstraints(
                minimum_length=3,
                maximum_length=20,
            ),
        )
        channel = _FakeChannel("a", ".", "first", "..", "last", ".")

        result = await CliInteractionRenderer(channel).render(
            _context(question)
        )

        answer = cast(
            MultilineTextAnswer,
            _answer(result, MultilineTextAnswer),
        )
        self.assertEqual(answer.value, "first\n.\nlast")
        self.assertIn("between 3 and 20 characters", channel.output)
        self.assertEqual(channel.read_count, 6)

        default_question = MultilineTextQuestion(
            question_id=QuestionId("notes"),
            prompt="Enter notes.",
            required=True,
            default_value="first\nsecond",
        )
        default_channel = _FakeChannel(".")
        default_result = await CliInteractionRenderer(default_channel).render(
            _context(default_question)
        )
        default_answer = cast(
            MultilineTextAnswer,
            _answer(default_result, MultilineTextAnswer),
        )
        self.assertEqual(default_answer.value, "first\nsecond")
        self.assertIn(r"first\nsecond", default_channel.output)

        optional_question = MultilineTextQuestion(
            question_id=QuestionId("notes"),
            prompt="Optional notes.",
            required=False,
        )
        skipped = await CliInteractionRenderer(_FakeChannel(".")).render(
            _context(optional_question)
        )
        assert isinstance(skipped, InputAnswerSubmission)
        self.assertEqual(skipped.answers, ())

        scalar_question = MultilineTextQuestion(
            question_id=QuestionId("scalar"),
            prompt="Enter scalar-safe text.",
            required=True,
        )
        invalid_scalar = _FakeChannel("\ud800", ".", "valid", ".")
        scalar_result = await CliInteractionRenderer(invalid_scalar).render(
            _context(scalar_question)
        )
        scalar_answer = cast(
            MultilineTextAnswer,
            _answer(scalar_result, MultilineTextAnswer),
        )
        self.assertEqual(scalar_answer.value, "valid")
        self.assertIn(
            "invalid Unicode scalar",
            invalid_scalar.output,
        )

        required_blank = _FakeChannel(".", "valid", ".")
        blank_result = await CliInteractionRenderer(required_blank).render(
            _context(scalar_question)
        )
        blank_answer = cast(
            MultilineTextAnswer,
            _answer(blank_result, MultilineTextAnswer),
        )
        self.assertEqual(blank_answer.value, "valid")
        self.assertIn(
            "Invalid input: Enter between 1 and 65536 characters.",
            required_blank.output,
        )

    async def test_single_selection_returns_stable_values_and_other(
        self,
    ) -> None:
        question = SingleSelectionQuestion(
            question_id=QuestionId("strategy"),
            prompt="Choose a strategy.",
            required=True,
            choices=_choices(),
            allow_other=True,
            recommended_choice=ChoiceValue("stable-safe"),
            default_value=ChoiceValue("stable-safe"),
            presentation_hint=PresentationHint.RADIO,
        )
        selected = await CliInteractionRenderer(_FakeChannel("1")).render(
            _context(question)
        )
        selected_answer = cast(
            SingleSelectionAnswer,
            _answer(selected, SingleSelectionAnswer),
        )
        self.assertEqual(
            selected_answer.value,
            SelectedChoice(value=ChoiceValue("stable-fast")),
        )

        default_channel = _FakeChannel("")
        default = await CliInteractionRenderer(default_channel).render(
            _context(question)
        )
        default_answer = cast(
            SingleSelectionAnswer,
            _answer(default, SingleSelectionAnswer),
        )
        self.assertEqual(
            default_answer.value,
            SelectedChoice(value=ChoiceValue("stable-safe")),
        )
        self.assertIn("[recommended, default]", default_channel.output)

        other_channel = _FakeChannel("3", "", "custom")
        other = await CliInteractionRenderer(other_channel).render(
            _context(question)
        )
        other_answer = cast(
            SingleSelectionAnswer,
            _answer(other, SingleSelectionAnswer),
        )
        self.assertEqual(other_answer.value, FreeFormOther(text="custom"))
        self.assertIn("Other must not be empty", other_channel.output)

        invalid_channel = _FakeChannel("Fast", "9", "2")
        invalid = await CliInteractionRenderer(invalid_channel).render(
            _context(question)
        )
        invalid_answer = cast(
            SingleSelectionAnswer,
            _answer(invalid, SingleSelectionAnswer),
        )
        self.assertEqual(
            invalid_answer.value,
            SelectedChoice(value=ChoiceValue("stable-safe")),
        )
        self.assertEqual(
            invalid_channel.output.count(
                "Invalid input: Select a listed option number."
            ),
            2,
        )

        optional = SingleSelectionQuestion(
            question_id=QuestionId("strategy"),
            prompt="Optional strategy.",
            required=False,
            choices=_choices(),
        )
        skipped = await CliInteractionRenderer(_FakeChannel("")).render(
            _context(optional)
        )
        assert isinstance(skipped, InputAnswerSubmission)
        self.assertEqual(skipped.answers, ())

        required_no_default = SingleSelectionQuestion(
            question_id=QuestionId("required"),
            prompt="Required strategy.",
            required=True,
            choices=_choices(),
        )
        required_empty = _FakeChannel("", "1")
        required_empty_result = await CliInteractionRenderer(
            required_empty
        ).render(_context(required_no_default))
        self.assertIsInstance(required_empty_result, InputAnswerSubmission)
        self.assertIn(
            "Invalid input: Select one option.",
            required_empty.output,
        )

        cancelled = await CliInteractionRenderer(
            _FakeChannel(":cancel")
        ).render(_context(question))
        self.assertEqual(
            cancelled,
            AttachedInputDisconnected(
                reason=AttachedInputDisconnectReason.HANDLER_CANCELLED
            ),
        )

        other_invalid = _FakeChannel("3", "bad\nother", "custom")
        other_invalid_result = await CliInteractionRenderer(
            other_invalid
        ).render(_context(question))
        other_invalid_answer = cast(
            SingleSelectionAnswer,
            _answer(other_invalid_result, SingleSelectionAnswer),
        )
        self.assertEqual(
            other_invalid_answer.value,
            FreeFormOther(text="custom"),
        )
        self.assertIn(
            "value must not contain a newline",
            other_invalid.output,
        )

    async def test_multiple_selection_validates_and_preserves_order(
        self,
    ) -> None:
        question = MultipleSelectionQuestion(
            question_id=QuestionId("checks"),
            prompt="Choose checks.",
            required=True,
            choices=_choices(),
            allow_other=True,
            recommended_choice=ChoiceValue("stable-safe"),
            constraints=SelectionValidationConstraints(
                minimum=1,
                maximum=3,
            ),
            presentation_hint=PresentationHint.CHECKBOX,
        )
        channel = _FakeChannel(
            "1,1",
            "4",
            "none",
            "2,3",
            "",
            "security",
        )

        result = await CliInteractionRenderer(channel).render(
            _context(question)
        )

        answer = cast(
            MultipleSelectionAnswer,
            _answer(result, MultipleSelectionAnswer),
        )
        self.assertEqual(
            answer.values,
            (
                SelectedChoice(value=ChoiceValue("stable-safe")),
                FreeFormOther(text="security"),
            ),
        )
        self.assertIn("Do not select an option more than once", channel.output)
        self.assertIn("Select only listed option numbers", channel.output)
        self.assertIn("Select between 1 and 3 options", channel.output)
        self.assertIn("Other must not be empty", channel.output)

        alias_channel = _FakeChannel("1,01", "2")
        alias_result = await CliInteractionRenderer(alias_channel).render(
            _context(question)
        )
        alias_answer = cast(
            MultipleSelectionAnswer,
            _answer(alias_result, MultipleSelectionAnswer),
        )
        self.assertEqual(
            alias_answer.values,
            (SelectedChoice(value=ChoiceValue("stable-safe")),),
        )
        self.assertIn(
            "Do not select an option more than once",
            alias_channel.output,
        )

        other_alias_channel = _FakeChannel("3,03", "1")
        other_alias_result = await CliInteractionRenderer(
            other_alias_channel
        ).render(_context(question))
        self.assertIsInstance(other_alias_result, InputAnswerSubmission)
        self.assertIn(
            "Do not select an option more than once",
            other_alias_channel.output,
        )

        oversized_channel = _FakeChannel("9" * 10_000, "1")
        oversized_result = await CliInteractionRenderer(
            oversized_channel
        ).render(_context(question))
        oversized_answer = cast(
            MultipleSelectionAnswer,
            _answer(oversized_result, MultipleSelectionAnswer),
        )
        self.assertEqual(
            oversized_answer.values,
            (SelectedChoice(value=ChoiceValue("stable-fast")),),
        )
        self.assertIn(
            "Select only listed option numbers",
            oversized_channel.output,
        )

        default_question = MultipleSelectionQuestion(
            question_id=QuestionId("checks"),
            prompt="Choose checks.",
            required=False,
            choices=_choices(first_description=None),
            default_value=(
                ChoiceValue("stable-safe"),
                ChoiceValue("stable-fast"),
            ),
            constraints=SelectionValidationConstraints(
                minimum=0,
                maximum=2,
            ),
        )
        default_channel = _FakeChannel("")
        default = await CliInteractionRenderer(default_channel).render(
            _context(default_question)
        )
        default_answer = cast(
            MultipleSelectionAnswer,
            _answer(default, MultipleSelectionAnswer),
        )
        self.assertEqual(
            default_answer.values,
            (
                SelectedChoice(value=ChoiceValue("stable-safe")),
                SelectedChoice(value=ChoiceValue("stable-fast")),
            ),
        )
        self.assertEqual(default_channel.output.count("[default]"), 2)

        no_default = MultipleSelectionQuestion(
            question_id=QuestionId("checks"),
            prompt="Optional checks.",
            required=False,
            choices=_choices(),
            constraints=SelectionValidationConstraints(
                minimum=0,
                maximum=2,
            ),
        )
        skipped = await CliInteractionRenderer(_FakeChannel("")).render(
            _context(no_default)
        )
        assert isinstance(skipped, InputAnswerSubmission)
        self.assertEqual(skipped.answers, ())

        explicit_none = await CliInteractionRenderer(
            _FakeChannel("none")
        ).render(_context(no_default))
        none_answer = cast(
            MultipleSelectionAnswer,
            _answer(explicit_none, MultipleSelectionAnswer),
        )
        self.assertEqual(none_answer.values, ())

        exact_question = MultipleSelectionQuestion(
            question_id=QuestionId("checks"),
            prompt="Choose exactly one.",
            required=False,
            choices=_choices(),
            constraints=SelectionValidationConstraints(
                minimum=1,
                maximum=1,
            ),
        )
        exact_channel = _FakeChannel("1,2", "2")
        exact = await CliInteractionRenderer(exact_channel).render(
            _context(exact_question)
        )
        exact_answer = cast(
            MultipleSelectionAnswer,
            _answer(exact, MultipleSelectionAnswer),
        )
        self.assertEqual(len(exact_answer.values), 1)
        self.assertIn("Select exactly 1 options", exact_channel.output)

        required_empty = _FakeChannel("", "1")
        required_empty_result = await CliInteractionRenderer(
            required_empty
        ).render(_context(question))
        self.assertIsInstance(required_empty_result, InputAnswerSubmission)
        self.assertIn(
            "Invalid input: Select at least one option.",
            required_empty.output,
        )

        corrected_channel = _FakeChannel("1", "2")
        corrected_answer = MultipleSelectionAnswer(
            question_id=question.question_id,
            provenance=AnswerProvenance.HUMAN,
            values=(SelectedChoice(value=ChoiceValue("stable-safe")),),
        )
        validation_error = InputValidationError(
            InputErrorCode.INVALID_CARDINALITY,
            "answer.values",
            "selection needs correction",
        )
        with patch(
            "avalan.cli.interaction_renderer.MultipleSelectionAnswer",
            side_effect=(validation_error, corrected_answer),
        ):
            corrected = await CliInteractionRenderer(corrected_channel).render(
                _context(question)
            )
        self.assertEqual(
            _answer(corrected, MultipleSelectionAnswer),
            corrected_answer,
        )
        self.assertIn("selection needs correction", corrected_channel.output)

        cancelled = await CliInteractionRenderer(
            _FakeChannel(":cancel")
        ).render(_context(question))
        self.assertEqual(
            cancelled,
            AttachedInputDisconnected(
                reason=AttachedInputDisconnectReason.HANDLER_CANCELLED
            ),
        )

        other_cancelled = await CliInteractionRenderer(
            _FakeChannel("3", ":cancel")
        ).render(_context(question))
        self.assertEqual(
            other_cancelled,
            AttachedInputDisconnected(
                reason=AttachedInputDisconnectReason.HANDLER_CANCELLED
            ),
        )

    async def test_bundle_help_feedback_and_safe_control_output(
        self,
    ) -> None:
        injection = (
            "\x1b]8;;https://example.invalid\x07click\x1b]8;;\x07 "
            "[bold]\u202e\x00"
        )
        confirmation = _confirmation(
            prompt=f"Continue {injection}?",
            header="Header [bold]\u202e\x00",
            help_text=f"Question help {injection}",
        )
        text = TextQuestion(
            question_id=QuestionId("name"),
            prompt=f"Name {injection}",
            required=True,
        )
        selection = SingleSelectionQuestion(
            question_id=QuestionId("strategy"),
            prompt="Choose.",
            required=True,
            choices=_choices(first_label=f"Fast {injection}"),
        )
        feedback = InputValidationFeedback(
            code=InputErrorCode.INVALID_CARDINALITY,
            path=f"answer{injection}",
            message=f"Try again {injection}",
        )
        channel = _FakeChannel("?", "yes", "Ada", "1")
        stdout = StringIO()
        stderr = StringIO()

        with redirect_stdout(stdout), redirect_stderr(stderr):
            result = await CliInteractionRenderer(channel).render(
                _context(
                    confirmation,
                    text,
                    selection,
                    reason=f"Reason {injection}",
                    validation_error=feedback,
                )
            )

        self.assertIsInstance(result, InputAnswerSubmission)
        assert isinstance(result, InputAnswerSubmission)
        self.assertEqual(len(result.answers), 3)
        self.assertIn("Question 1/3", channel.output)
        self.assertIn("Question 2/3", channel.output)
        self.assertIn("Question 3/3", channel.output)
        self.assertIn("Previous input was not accepted", channel.output)
        self.assertGreaterEqual(channel.output.count("Question help"), 2)
        self.assertIn("click", channel.output)
        self.assertIn("[bold]", channel.output)
        for unsafe in ("\x1b", "\x07", "\x00", "\u202e"):
            self.assertNotIn(unsafe, channel.output)
        self.assertIn(r"\u202e", channel.output)
        self.assertEqual(stdout.getvalue(), "")
        self.assertEqual(stderr.getvalue(), "")

    async def test_every_presentation_hint_has_semantic_fallback(
        self,
    ) -> None:
        for hint in PresentationHint:
            with self.subTest(hint=hint):
                channel = _FakeChannel("yes")
                result = await CliInteractionRenderer(channel).render(
                    _context(_confirmation(hint=hint))
                )
                answer = cast(
                    ConfirmationAnswer,
                    _answer(result, ConfirmationAnswer),
                )
                self.assertTrue(answer.value)
                self.assertEqual(channel.read_count, 1)

    async def test_decline_input_cancel_and_channel_loss_are_distinct(
        self,
    ) -> None:
        declined = await CliInteractionRenderer(
            _FakeChannel(":decline")
        ).render(_context(_confirmation()))
        self.assertIsInstance(declined, InputDeclineSubmission)
        assert isinstance(declined, InputDeclineSubmission)
        self.assertIs(declined.provenance, AnswerProvenance.HUMAN)

        for command in (":cancel", ":cancel-input"):
            with self.subTest(command=command):
                cancelled = await CliInteractionRenderer(
                    _FakeChannel(command)
                ).render(_context(_confirmation()))
                self.assertEqual(
                    cancelled,
                    AttachedInputDisconnected(
                        reason=(
                            AttachedInputDisconnectReason.HANDLER_CANCELLED
                        )
                    ),
                )

        closed = await CliInteractionRenderer(_FakeChannel(None)).render(
            _context(_confirmation())
        )
        self.assertEqual(
            closed,
            AttachedInputDisconnected(
                reason=(AttachedInputDisconnectReason.CONTROL_CHANNEL_CLOSED)
            ),
        )

    async def test_run_cancel_and_steering_dispatch_separately(
        self,
    ) -> None:
        run_recorder = _CommandRecorder()
        run_cancelled = await CliInteractionRenderer(
            _FakeChannel(":cancel-run"),
            command_handler=run_recorder,
        ).render(_context(_confirmation()))

        self.assertEqual(
            run_cancelled,
            AttachedInputDisconnected(
                reason=AttachedInputDisconnectReason.HANDLER_CANCELLED
            ),
        )
        self.assertEqual(len(run_recorder.commands), 1)
        run_command = run_recorder.commands[0]
        self.assertIsInstance(run_command, CliRunCancellationCommand)
        self.assertIs(
            run_command.kind,
            CliInteractionCommandKind.RUN_CANCELLED,
        )

        steer_channel = _FakeChannel(":steer focus on safety", "yes")
        steer_recorder = _CommandRecorder()
        answered = await CliInteractionRenderer(
            steer_channel,
            command_handler=steer_recorder,
        ).render(_context(_confirmation()))
        answer = cast(
            ConfirmationAnswer,
            _answer(answered, ConfirmationAnswer),
        )
        self.assertTrue(answer.value)
        self.assertEqual(len(steer_recorder.commands), 1)
        steer_command = steer_recorder.commands[0]
        self.assertIsInstance(steer_command, CliSteeringCommand)
        assert isinstance(steer_command, CliSteeringCommand)
        self.assertEqual(steer_command.text, "focus on safety")
        self.assertIs(
            steer_command.kind,
            CliInteractionCommandKind.STEERING,
        )
        self.assertIn("input request is still pending", steer_channel.output)

        unavailable = _FakeChannel(
            ":steer no handler",
            ":cancel-run",
            "yes",
        )
        unavailable_result = await CliInteractionRenderer(unavailable).render(
            _context(_confirmation())
        )
        self.assertIsInstance(unavailable_result, InputAnswerSubmission)
        self.assertIn(
            "Steering is unavailable in this CLI session.",
            unavailable.output,
        )
        self.assertIn(
            "Containing-run cancellation is unavailable in this CLI session.",
            unavailable.output,
        )

        rejected_channel = _FakeChannel(
            ":steer unsupported",
            ":cancel-run",
            "yes",
        )
        rejected_recorder = _CommandRecorder(
            CliInteractionCommandDisposition.UNAVAILABLE
        )
        rejected_result = await CliInteractionRenderer(
            rejected_channel,
            command_handler=rejected_recorder,
        ).render(_context(_confirmation()))
        self.assertIsInstance(rejected_result, InputAnswerSubmission)
        self.assertEqual(len(rejected_recorder.commands), 2)
        self.assertNotIn("Steering sent", rejected_channel.output)

    async def test_controls_work_inside_multiline_and_other_prompts(
        self,
    ) -> None:
        multiline = MultilineTextQuestion(
            question_id=QuestionId("notes"),
            prompt="Notes.",
            required=True,
        )
        multiline_cancel = await CliInteractionRenderer(
            _FakeChannel(":cancel")
        ).render(_context(multiline))
        self.assertEqual(
            multiline_cancel,
            AttachedInputDisconnected(
                reason=AttachedInputDisconnectReason.HANDLER_CANCELLED
            ),
        )

        selection = SingleSelectionQuestion(
            question_id=QuestionId("strategy"),
            prompt="Choose.",
            required=True,
            choices=_choices(),
            allow_other=True,
        )
        other_decline = await CliInteractionRenderer(
            _FakeChannel("3", ":decline")
        ).render(_context(selection))
        self.assertIsInstance(other_decline, InputDeclineSubmission)

        text = TextQuestion(
            question_id=QuestionId("text"),
            prompt="Text.",
            required=True,
        )
        escaped = await CliInteractionRenderer(
            _FakeChannel("::steer literal")
        ).render(_context(text))
        escaped_answer = cast(TextAnswer, _answer(escaped, TextAnswer))
        self.assertEqual(escaped_answer.value, ":steer literal")

        help_channel = _FakeChannel(":steer", "?", "yes")
        help_result = await CliInteractionRenderer(help_channel).render(
            _context(_confirmation())
        )
        self.assertIsInstance(help_result, InputAnswerSubmission)
        self.assertGreaterEqual(
            help_channel.output.count(
                "No additional question help is available."
            ),
            2,
        )

    async def test_renderer_propagates_failures_and_is_cancellable(
        self,
    ) -> None:
        with self.assertRaises(AssertionError):
            await CliInteractionRenderer(_FakeChannel()).render(_context())

        failure = RuntimeError("reader failed")
        with self.assertRaisesRegex(RuntimeError, "reader failed"):
            await CliInteractionRenderer(_FakeChannel(failure)).render(
                _context(_confirmation())
            )

        blocking = _BlockingChannel()
        task = create_task(
            CliInteractionRenderer(blocking).render(_context(_confirmation()))
        )
        await blocking.reading.wait()
        task.cancel()
        with self.assertRaises(CancelledError):
            await task
        await sleep(0)
        self.assertTrue(task.done())

        renderer = CliInteractionRenderer(_FakeChannel())
        with self.assertRaisesRegex(
            AssertionError,
            "unsupported canonical question variant",
        ):
            await renderer._ask_question(
                cast(InputQuestion, _UnsupportedQuestion()),
                1,
                1,
            )

    async def test_channel_lifetime_remains_with_its_owner(self) -> None:
        channel = _FakeChannel("yes")

        await CliInteractionRenderer(channel).render(_context(_confirmation()))

        self.assertFalse(channel.closed)
        await channel.aclose()
        self.assertTrue(channel.closed)


class CliInteractionRendererHelperTestCase(TestCase):
    def test_commands_and_messages_are_typed(self) -> None:
        input_cancel: CliInteractionCommand = CliInputCancellationCommand()
        run_cancel: CliInteractionCommand = CliRunCancellationCommand()
        steering: CliInteractionCommand = CliSteeringCommand(text="redirect")

        self.assertIs(
            input_cancel.kind,
            CliInteractionCommandKind.INPUT_CANCELLED,
        )
        self.assertIs(
            run_cancel.kind,
            CliInteractionCommandKind.RUN_CANCELLED,
        )
        self.assertIs(
            steering.kind,
            CliInteractionCommandKind.STEERING,
        )
        with self.assertRaises(AssertionError):
            CliSteeringCommand(text="")
        with self.assertRaises(AssertionError):
            CliSteeringCommand(text="line\nbreak")

        self.assertEqual(_length_message(2, 2), "Enter exactly 2 characters.")
        self.assertEqual(
            _selection_count_message(0, 2),
            "Select between 0 and 2 options.",
        )
        self.assertEqual(
            _selection_count_message(2, 2),
            "Select exactly 2 options.",
        )

    def test_literal_text_neutralizes_all_terminal_control_forms(self) -> None:
        payload = (
            "[red]\x1b[31mCSI\x1b[0m"
            "\x1b]8;;https://example.invalid\x07OSC\x1b]8;;\x07"
            "\x1bPignored\x1b\\"
            "\x00\t\r\n\u202d\ud800"
        )

        rendered = _literal_text(payload)

        self.assertEqual(
            rendered,
            r"[red]CSIOSC\u0000\t\r\n\u202d\ud800",
        )
        self.assertNotIn("\x1b", rendered)
        self.assertEqual(
            _literal_text("[bold]x[/bold] [label](https://example.invalid)"),
            "[bold]x[/bold] [label](https://example.invalid)",
        )

    def test_source_has_no_stdout_or_sync_event_loop_entrypoint(self) -> None:
        source = Path("src/avalan/cli/interaction_renderer.py").read_text(
            encoding="utf-8"
        )

        self.assertNotIn("asyncio.run", source)
        self.assertNotIn("print(", source)
        self.assertNotIn("sys.stdout", source)

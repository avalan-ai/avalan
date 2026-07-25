"""Translate canonical task input to the frozen A2A extension."""

from .codec import decode_input_question, encode_input_question
from .entities import (
    AnsweredResolution,
    AnswerProvenance,
    CancellationScope,
    CancelledResolution,
    ChoiceValue,
    ConfirmationAnswer,
    ConfirmationQuestion,
    DeclinedResolution,
    FreeFormOther,
    InputAnswer,
    InputQuestion,
    InputRequest,
    InputRequestId,
    InputResolution,
    MultilineTextAnswer,
    MultilineTextQuestion,
    MultipleSelectionAnswer,
    MultipleSelectionQuestion,
    QuestionType,
    SelectedChoice,
    SelectionValue,
    SingleSelectionAnswer,
    SingleSelectionQuestion,
    TextAnswer,
    TextQuestion,
)
from .error import InputErrorCode, InputValidationError

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import cast, final

A2A_INPUT_EXTENSION_URI = "https://avalan.ai/extensions/task-input/v1"
A2A_INPUT_EXTENSION_DESCRIPTION = (
    "Avalan structured task-input request and resolution extension."
)
A2A_INPUT_EXTENSION_PARAMS: dict[str, object] = {
    "schema": "avalan.task-input.v1",
    "message_metadata_key": A2A_INPUT_EXTENSION_URI,
    "readable_text_fallback": True,
}


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class A2AInputRequestMetadata:
    """Carry one validated request decoded from A2A message metadata."""

    request_id: InputRequestId
    required: bool
    questions: tuple[InputQuestion, ...]


def encode_a2a_input_request_metadata(
    request: InputRequest | A2AInputRequestMetadata,
) -> dict[str, object]:
    """Encode one canonical request as A2A message metadata."""
    if not isinstance(request, InputRequest | A2AInputRequestMetadata):
        raise InputValidationError(
            InputErrorCode.INVALID_TYPE,
            "request",
            "value must be an input request or A2A request metadata",
        )
    return {
        "kind": "request",
        "request_id": str(request.request_id),
        "required": request.required,
        "questions": [
            _a2a_question_metadata(question) for question in request.questions
        ],
    }


def decode_a2a_input_request_metadata(
    value: object,
) -> A2AInputRequestMetadata:
    """Decode one A2A request metadata payload."""
    payload = _object(value, "metadata")
    _exact_keys(
        payload,
        {"kind", "request_id", "required", "questions"},
        "metadata",
    )
    if payload["kind"] != "request":
        raise _invalid("metadata.kind", "value must be request")
    required = payload["required"]
    if type(required) is not bool:
        raise _invalid("metadata.required", "value must be a boolean")
    questions_value = payload["questions"]
    if not _sequence(questions_value):
        raise _invalid("metadata.questions", "value must be an array")
    questions = tuple(
        _decode_a2a_question(question)
        for question in cast(
            list[object] | tuple[object, ...], questions_value
        )
    )
    if not 1 <= len(questions) <= 3:
        raise _invalid(
            "metadata.questions",
            "request must contain one to three questions",
        )
    return A2AInputRequestMetadata(
        request_id=InputRequestId(
            _string(payload["request_id"], "metadata.request_id")
        ),
        required=required,
        questions=questions,
    )


def encode_a2a_input_resolution_metadata(
    resolution: InputResolution,
) -> dict[str, object]:
    """Encode one canonical resolution as A2A message metadata."""
    if type(resolution) is AnsweredResolution:
        return {
            "kind": "resolution",
            "request_id": str(resolution.request_id),
            "action": "accept",
            "answers": {
                str(answer.question_id): _encode_a2a_answer(answer)
                for answer in resolution.answers
            },
        }
    if type(resolution) is DeclinedResolution:
        action = "decline"
    elif type(resolution) is CancelledResolution:
        action = "cancel"
    else:
        raise InputValidationError(
            InputErrorCode.INVALID_TYPE,
            "resolution",
            "A2A supports answered, declined, or cancelled resolutions",
        )
    return {
        "kind": "resolution",
        "request_id": str(resolution.request_id),
        "action": action,
    }


def decode_a2a_input_resolution_metadata(
    value: object,
    *,
    request: InputRequest | A2AInputRequestMetadata,
    resolved_at: datetime,
) -> InputResolution:
    """Decode and validate one A2A resolution metadata payload."""
    payload = _object(value, "metadata")
    if payload.get("kind") != "resolution":
        raise _invalid("metadata.kind", "value must be resolution")
    request_id = InputRequestId(
        _string(payload.get("request_id"), "metadata.request_id")
    )
    if request_id != request.request_id:
        raise InputValidationError(
            InputErrorCode.CORRELATION_MISMATCH,
            "metadata.request_id",
            "request identity does not match the pending interaction",
        )
    action = payload.get("action")
    if action == "accept":
        _exact_keys(
            payload,
            {"kind", "request_id", "action", "answers"},
            "metadata",
        )
        answers_value = _object(payload["answers"], "metadata.answers")
        question_ids = {
            str(question.question_id) for question in request.questions
        }
        answer_ids = set(answers_value)
        unknown_ids = answer_ids - question_ids
        if unknown_ids:
            raise _invalid(
                "metadata.answers",
                "answer keys must reference pending questions",
            )
        if question_ids - answer_ids:
            raise _invalid(
                "metadata.answers",
                "answer keys must include every pending question",
            )
        return AnsweredResolution(
            request_id=request_id,
            provenance=AnswerProvenance.EXTERNAL_CONTROLLER,
            resolved_at=resolved_at,
            answers=tuple(
                _decode_a2a_answer(
                    question,
                    answers_value[str(question.question_id)],
                )
                for question in request.questions
            ),
        )
    _exact_keys(payload, {"kind", "request_id", "action"}, "metadata")
    if action == "decline":
        return DeclinedResolution(
            request_id=request_id,
            provenance=AnswerProvenance.EXTERNAL_CONTROLLER,
            resolved_at=resolved_at,
        )
    if action == "cancel":
        return CancelledResolution(
            request_id=request_id,
            provenance=AnswerProvenance.EXTERNAL_CONTROLLER,
            resolved_at=resolved_at,
            scope=CancellationScope.REQUEST,
        )
    raise _invalid(
        "metadata.action",
        "value must be accept, decline, or cancel",
    )


def a2a_input_request_text(
    request: InputRequest | A2AInputRequestMetadata,
) -> str:
    """Render one readable non-authoritative A2A fallback."""
    lines = ["Additional input is required."]
    for index, question in enumerate(request.questions, start=1):
        lines.append(f"{index}. {question.prompt}")
        if isinstance(
            question,
            SingleSelectionQuestion | MultipleSelectionQuestion,
        ):
            lines.extend(f"   - {choice.label}" for choice in question.choices)
    return "\n".join(lines)


def _a2a_question_metadata(question: InputQuestion) -> dict[str, object]:
    encoded = dict(encode_input_question(question))
    encoded.pop("constraints", None)
    return cast(dict[str, object], encoded)


def _decode_a2a_question(value: object) -> InputQuestion:
    payload = _object(value, "metadata.question")
    kind_value = payload.get("kind")
    try:
        kind = QuestionType(_string(kind_value, "metadata.question.kind"))
    except ValueError:
        raise _invalid(
            "metadata.question.kind",
            "value must be a supported question kind",
        ) from None
    common_keys = {
        "question_id",
        "kind",
        "prompt",
        "required",
        "choices",
        "allow_other",
    }
    optional_keys = {
        "header",
        "help",
        "presentation_hint",
        "default_value",
        "recommended_choice",
    }
    if (
        not common_keys <= set(payload)
        or set(payload) - common_keys - optional_keys
    ):
        raise _invalid(
            "metadata.question",
            "question fields do not match the A2A schema",
        )
    canonical = dict(payload)
    if isinstance(canonical["choices"], tuple):
        canonical["choices"] = list(canonical["choices"])
    if kind in {QuestionType.TEXT, QuestionType.MULTILINE_TEXT}:
        canonical["constraints"] = (
            {"minimum_length": 0, "maximum_length": 4_096}
            if kind is QuestionType.TEXT
            else {"minimum_length": 0, "maximum_length": 65_536}
        )
    elif kind is QuestionType.MULTIPLE_SELECTION:
        choices = payload.get("choices")
        if not _sequence(choices):
            raise _invalid(
                "metadata.question.choices",
                "value must be an array",
            )
        allow_other = payload.get("allow_other")
        if type(allow_other) is not bool:
            raise _invalid(
                "metadata.question.allow_other",
                "value must be a boolean",
            )
        required = payload.get("required")
        if type(required) is not bool:
            raise _invalid(
                "metadata.question.required",
                "value must be a boolean",
            )
        choice_items = cast(list[object] | tuple[object, ...], choices)
        canonical["constraints"] = {
            "minimum": int(required),
            "maximum": min(20, len(choice_items) + int(allow_other)),
        }
    return decode_input_question(canonical)


def _encode_a2a_answer(answer: object) -> object:
    if isinstance(answer, ConfirmationAnswer):
        return answer.value
    if isinstance(answer, MultilineTextAnswer) and len(answer.value) > 10_000:
        raise _invalid(
            "answer.value",
            "multiline answer exceeds the A2A transport limit",
        )
    if isinstance(answer, TextAnswer | MultilineTextAnswer):
        return answer.value
    if isinstance(answer, SingleSelectionAnswer):
        return _encode_a2a_selection(answer.value)
    if isinstance(answer, MultipleSelectionAnswer):
        return [_encode_a2a_selection(value) for value in answer.values]
    raise InputValidationError(
        InputErrorCode.INVALID_TYPE,
        "answer",
        "value must be a supported answer",
    )


def _encode_a2a_selection(value: SelectionValue) -> dict[str, str]:
    if isinstance(value, SelectedChoice):
        return {"kind": "selected_choice", "value": str(value.value)}
    if isinstance(value, FreeFormOther):
        if len(value.text) > 1_000:
            raise _invalid(
                "answer.value.text",
                "free-form selection exceeds the A2A transport limit",
            )
        if "\r" in value.text or "\n" in value.text:
            raise _invalid(
                "answer.value.text",
                "free-form selection must be one line",
            )
        return {"kind": "free_form_other", "text": value.text}
    raise InputValidationError(
        InputErrorCode.INVALID_TYPE,
        "answer.value",
        "value must be a tagged selection",
    )


def _decode_a2a_answer(
    question: InputQuestion,
    value: object,
) -> InputAnswer:
    if isinstance(question, ConfirmationQuestion):
        if type(value) is not bool:
            raise _invalid("metadata.answers", "confirmation must be boolean")
        return ConfirmationAnswer(
            question_id=question.question_id,
            provenance=AnswerProvenance.EXTERNAL_CONTROLLER,
            value=value,
        )
    if isinstance(question, TextQuestion):
        return TextAnswer(
            question_id=question.question_id,
            provenance=AnswerProvenance.EXTERNAL_CONTROLLER,
            value=_answer_text(question, value),
        )
    if isinstance(question, MultilineTextQuestion):
        text = _answer_text(question, value)
        if len(text) > 10_000:
            raise _invalid(
                "metadata.answers",
                "multiline answer exceeds the A2A transport limit",
            )
        return MultilineTextAnswer(
            question_id=question.question_id,
            provenance=AnswerProvenance.EXTERNAL_CONTROLLER,
            value=text,
        )
    if isinstance(question, SingleSelectionQuestion):
        return SingleSelectionAnswer(
            question_id=question.question_id,
            provenance=AnswerProvenance.EXTERNAL_CONTROLLER,
            value=_decode_a2a_selection(question, value),
        )
    assert isinstance(question, MultipleSelectionQuestion)
    if not _sequence(value):
        raise _invalid("metadata.answers", "selection must be an array")
    return MultipleSelectionAnswer(
        question_id=question.question_id,
        provenance=AnswerProvenance.EXTERNAL_CONTROLLER,
        values=tuple(
            _decode_a2a_selection(question, item)
            for item in cast(list[object] | tuple[object, ...], value)
        ),
    )


def _answer_text(
    question: TextQuestion | MultilineTextQuestion,
    value: object,
) -> str:
    if not isinstance(value, str) or (question.required and not value):
        requirement = "non-empty " if question.required else ""
        raise _invalid(
            "metadata.answers",
            f"value must be a {requirement}string",
        )
    return value


def _decode_a2a_selection(
    question: SingleSelectionQuestion | MultipleSelectionQuestion,
    value: object,
) -> SelectionValue:
    payload = _object(value, "metadata.answer.selection")
    kind = payload.get("kind")
    if kind == "selected_choice":
        _exact_keys(
            payload,
            {"kind", "value"},
            "metadata.answer.selection",
        )
        selected = ChoiceValue(
            _string(payload["value"], "metadata.answer.selection.value")
        )
        if selected not in {choice.value for choice in question.choices}:
            raise _invalid(
                "metadata.answer.selection.value",
                "selected value is not an offered choice",
            )
        return SelectedChoice(value=selected)
    if kind == "free_form_other":
        _exact_keys(
            payload,
            {"kind", "text"},
            "metadata.answer.selection",
        )
        if not question.allow_other:
            raise _invalid(
                "metadata.answer.selection",
                "free-form selection is not allowed",
            )
        text = _string(payload["text"], "metadata.answer.selection.text")
        if len(text) > 1_000:
            raise _invalid(
                "metadata.answer.selection.text",
                "free-form selection exceeds the A2A transport limit",
            )
        if "\r" in text or "\n" in text:
            raise _invalid(
                "metadata.answer.selection.text",
                "free-form selection must be one line",
            )
        return FreeFormOther(text=text)
    raise _invalid(
        "metadata.answer.selection.kind",
        "value must be selected_choice or free_form_other",
    )


def _object(value: object, path: str) -> dict[str, object]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) for key in value
    ):
        raise _invalid(path, "value must be an object")
    return {cast(str, key): item for key, item in value.items()}


def _exact_keys(
    value: Mapping[str, object],
    expected: set[str],
    path: str,
) -> None:
    if set(value) != expected:
        raise _invalid(path, "object fields do not match the A2A schema")


def _string(value: object, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise _invalid(path, "value must be a non-empty string")
    return value


def _sequence(value: object) -> bool:
    return isinstance(value, list | tuple)


def _invalid(path: str, message: str) -> InputValidationError:
    return InputValidationError(InputErrorCode.INVALID_FORMAT, path, message)

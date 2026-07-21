"""Test primitive interaction-domain validation boundaries."""

from datetime import UTC, datetime, timedelta, timezone

import pytest

from avalan.interaction.error import InputErrorCode, InputValidationError
from avalan.interaction.validation import (
    MAX_STATE_REVISION,
    normalize_text,
    validate_aware_datetime,
    validate_bool,
    validate_choice_value,
    validate_finite_number,
    validate_int,
    validate_multiline_text,
    validate_opaque_id,
    validate_other_text,
    validate_presentation_text,
    validate_question_id,
    validate_single_line_text,
    validate_state_revision,
    validate_total_request_content,
)


@pytest.mark.parametrize(
    ("call", "code"),
    (
        (lambda: normalize_text(1, "value"), InputErrorCode.INVALID_TYPE),
        (
            lambda: normalize_text("\ud800", "value"),
            InputErrorCode.INVALID_FORMAT,
        ),
        (lambda: validate_opaque_id("", "id"), InputErrorCode.EMPTY),
        (
            lambda: validate_opaque_id("x" * 129, "id"),
            InputErrorCode.OUT_OF_BOUNDS,
        ),
        (
            lambda: validate_opaque_id("safe\u202eid", "id"),
            InputErrorCode.INVALID_FORMAT,
        ),
        (
            lambda: validate_question_id("1bad"),
            InputErrorCode.INVALID_FORMAT,
        ),
        (
            lambda: validate_choice_value(""),
            InputErrorCode.OUT_OF_BOUNDS,
        ),
        (
            lambda: validate_choice_value("bad\x00value"),
            InputErrorCode.INVALID_FORMAT,
        ),
        (
            lambda: validate_presentation_text(
                "   ",
                "text",
                minimum=1,
                maximum=10,
                maximum_bytes=40,
            ),
            InputErrorCode.EMPTY,
        ),
        (
            lambda: validate_presentation_text(
                "bad\nline",
                "text",
                minimum=1,
                maximum=20,
                maximum_bytes=80,
            ),
            InputErrorCode.INVALID_FORMAT,
        ),
        (
            lambda: validate_presentation_text(
                "é",
                "text",
                minimum=1,
                maximum=10,
                maximum_bytes=1,
            ),
            InputErrorCode.OUT_OF_BOUNDS,
        ),
        (
            lambda: validate_single_line_text("bad\rline", "answer"),
            InputErrorCode.INVALID_FORMAT,
        ),
        (
            lambda: validate_other_text("   "),
            InputErrorCode.EMPTY,
        ),
        (
            lambda: validate_other_text("bad\nother"),
            InputErrorCode.INVALID_FORMAT,
        ),
        (
            lambda: validate_aware_datetime("now", "timestamp"),
            InputErrorCode.INVALID_TYPE,
        ),
        (
            lambda: validate_aware_datetime(
                datetime(2026, 7, 20),
                "timestamp",
            ),
            InputErrorCode.NAIVE_TIMESTAMP,
        ),
        (lambda: validate_bool(1, "flag"), InputErrorCode.INVALID_TYPE),
        (
            lambda: validate_int(True, "number", minimum=0, maximum=1),
            InputErrorCode.INVALID_TYPE,
        ),
        (
            lambda: validate_int(2, "number", minimum=0, maximum=1),
            InputErrorCode.OUT_OF_BOUNDS,
        ),
        (
            lambda: validate_state_revision(
                MAX_STATE_REVISION + 1, "revision"
            ),
            InputErrorCode.OUT_OF_BOUNDS,
        ),
        (
            lambda: validate_total_request_content(("x" * 32_769,)),
            InputErrorCode.OUT_OF_BOUNDS,
        ),
        (
            lambda: validate_finite_number(True, "number"),
            InputErrorCode.INVALID_TYPE,
        ),
        (
            lambda: validate_finite_number(float("inf"), "number"),
            InputErrorCode.NON_JSON_VALUE,
        ),
    ),
)
def test_invalid_primitives(call: object, code: InputErrorCode) -> None:
    """Reject invalid primitives with stable content-safe codes."""
    assert callable(call)
    with pytest.raises(InputValidationError) as caught:
        call()
    assert caught.value.code is code
    assert "bad\nline" not in str(caught.value)


def test_normalization_and_valid_primitive_boundaries() -> None:
    """Normalize Unicode, newlines, timestamps, numbers, and bounds."""
    assert normalize_text("e\u0301", "value") == "é"
    assert validate_opaque_id("request-1", "id") == "request-1"
    assert validate_question_id("Question_1") == "Question_1"
    assert validate_choice_value("choice-1") == "choice-1"
    assert (
        validate_presentation_text(
            "Prompt",
            "prompt",
            minimum=1,
            maximum=10,
            maximum_bytes=40,
        )
        == "Prompt"
    )
    assert validate_single_line_text("", "answer") == ""
    assert validate_multiline_text("a\r\nb\rc", "answer") == "a\nb\nc"
    assert validate_other_text("custom") == "custom"
    offset = timezone(timedelta(hours=-3))
    timestamp = datetime(2026, 7, 20, 9, 0, tzinfo=offset)
    assert validate_aware_datetime(timestamp, "timestamp") == datetime(
        2026,
        7,
        20,
        12,
        0,
        tzinfo=UTC,
    )
    assert validate_bool(False, "flag") is False
    assert validate_int(1, "number", minimum=0, maximum=1) == 1
    assert (
        validate_state_revision(MAX_STATE_REVISION, "revision")
        == MAX_STATE_REVISION
    )
    validate_total_request_content(("safe",))
    assert validate_finite_number(1.5, "number") == 1.5

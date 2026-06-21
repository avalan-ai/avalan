from ...tool.display import (
    ToolDisplayDetail,
    ToolDisplayProjection,
    ToolDisplayScalar,
)

from collections.abc import Sequence

from rich.markup import escape

PROJECTION_PART_LIMIT = 120
PROJECTION_SUBJECT_LIMIT = 180
PROJECTION_SUMMARY_LIMIT = 120
PROJECTION_TERMINAL_LIMIT = 260
PROJECTION_DETAILS_LIMIT = 160
PROJECTION_DETAIL_VALUE_LIMIT = 64
PROJECTION_DETAIL_LABEL_LIMIT = 32
PROJECTION_DETAIL_COUNT = 3


def projection_subject_markup(
    projection: ToolDisplayProjection,
) -> str:
    """Return escaped action, target, and scope text."""
    assert isinstance(projection, ToolDisplayProjection)
    action = _bounded_text(projection.action, PROJECTION_PART_LIMIT)
    target = _bounded_optional_text(projection.target, PROJECTION_PART_LIMIT)
    scope = _bounded_optional_text(projection.scope, PROJECTION_PART_LIMIT)

    subject = action
    if target:
        subject = f"{subject} {target}"
    if scope and scope != target:
        subject = f"{subject} in {scope}"
    return escape(_bounded_text(subject, PROJECTION_SUBJECT_LIMIT))


def projection_status(
    projection: ToolDisplayProjection,
    fallback: str | None = None,
) -> str | None:
    """Return plain projected status or fallback text."""
    assert isinstance(projection, ToolDisplayProjection)
    if fallback is not None:
        assert isinstance(fallback, str)
    projected_status = _bounded_optional_text(
        projection.status,
        PROJECTION_PART_LIMIT,
    )
    fallback_status = (
        _bounded_optional_text(fallback, PROJECTION_PART_LIMIT)
        if fallback
        else None
    )
    return projected_status or fallback_status


def projection_outcome(
    projection: ToolDisplayProjection,
) -> str | None:
    """Return plain projected outcome text."""
    assert isinstance(projection, ToolDisplayProjection)
    return _bounded_optional_text(projection.outcome, PROJECTION_PART_LIMIT)


def projection_summary_markup(
    projection: ToolDisplayProjection,
) -> str | None:
    """Return escaped projected summary text."""
    assert isinstance(projection, ToolDisplayProjection)
    summary = _bounded_optional_text(
        projection.summary,
        PROJECTION_SUMMARY_LIMIT,
    )
    return escape(summary) if summary else None


def projection_details_markup(
    projection: ToolDisplayProjection,
) -> str | None:
    """Return escaped projected detail text."""
    assert isinstance(projection, ToolDisplayProjection)
    details = _details_text(projection.details)
    return escape(details) if details else None


def projection_terminal_markup(
    projection: ToolDisplayProjection,
    *,
    fallback_status: str | None = None,
) -> str:
    """Return escaped terminal status, outcome, summary, and details."""
    assert isinstance(projection, ToolDisplayProjection)
    if fallback_status is not None:
        assert isinstance(fallback_status, str)
    status = projection_status(projection, fallback_status)
    outcome = projection_outcome(projection)
    summary = _bounded_optional_text(
        projection.summary,
        PROJECTION_SUMMARY_LIMIT,
    )
    details = _details_text(projection.details)

    status_text = _status_outcome_text(status, outcome)
    parts = [
        part
        for part in (
            status_text,
            summary,
            f"details: {details}" if details else None,
        )
        if part
    ]
    return escape(
        _bounded_text(
            " - ".join(parts) or _bounded_text(projection.action),
            PROJECTION_TERMINAL_LIMIT,
        )
    )


def _status_outcome_text(
    status: str | None,
    outcome: str | None,
) -> str | None:
    if status and outcome and status != outcome:
        return f"{status} {outcome}"
    return outcome or status


def _details_text(details: Sequence[ToolDisplayDetail]) -> str | None:
    assert isinstance(details, Sequence)
    parts: list[str] = []
    for detail in details[:PROJECTION_DETAIL_COUNT]:
        detail_text = _detail_text(detail)
        if detail_text:
            parts.append(detail_text)
    if len(details) > PROJECTION_DETAIL_COUNT:
        parts.append("...")
    text = ", ".join(parts)
    return _bounded_optional_text(text, PROJECTION_DETAILS_LIMIT)


def _detail_text(detail: ToolDisplayDetail) -> str:
    assert isinstance(detail, ToolDisplayDetail)
    label = _bounded_text(detail.label, PROJECTION_DETAIL_LABEL_LIMIT)
    value = _scalar_text(detail.value, PROJECTION_DETAIL_VALUE_LIMIT)
    return f"{label}={value}"


def _scalar_text(value: ToolDisplayScalar, limit: int) -> str:
    if value is None:
        return "none"
    if isinstance(value, bool):
        return "true" if value else "false"
    return _bounded_text(str(value), limit)


def _bounded_optional_text(
    value: str | None,
    limit: int,
) -> str | None:
    if value is None:
        return None
    text = _bounded_text(value, limit)
    return text or None


def _bounded_text(
    value: str,
    limit: int = PROJECTION_PART_LIMIT,
) -> str:
    assert isinstance(value, str)
    assert isinstance(limit, int)
    assert limit > 3
    text = " ".join(value.split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."

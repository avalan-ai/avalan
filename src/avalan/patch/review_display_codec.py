"""Parse the detached non-authoritative approver-review view handle.

This lower-consumer module contains only immutable byte values, bounded public
headers, and content-free generic-log rendering.  It neither imports nor
references the trusted review host, its authority, its keys, or its additional
authenticated-data binding.
"""

from dataclasses import dataclass
from json import dumps
from typing import NewType, final

APPROVER_REVIEW_VIEW_SCHEMA_VERSION = 1
MAX_APPROVER_REVIEW_VIEW_BYTES = 1_048_576
MAX_APPROVER_REVIEW_PAGE_COUNT = 1_048_576
APPROVER_REVIEW_VIEW_NONCE_BYTES = 12
APPROVER_REVIEW_VIEW_TAG_BYTES = 16

_VIEW_MAGIC = b"ARV1"
_CORRELATION_LENGTH_BYTES = 1
_MIN_CORRELATION_BYTES = 23
_MAX_CORRELATION_BYTES = 55
_PAGE_COUNT_BYTES = 4
_MINIMUM_HEADER_BYTES = (
    len(_VIEW_MAGIC)
    + _CORRELATION_LENGTH_BYTES
    + _MIN_CORRELATION_BYTES
    + _PAGE_COUNT_BYTES
)
_CORRELATION_CHARACTERS = frozenset("abcdefghijklmnopqrstuvwxyz0123456789_")

ApproverReviewView = NewType("ApproverReviewView", bytes)


class ReviewDisplayCodecError(ValueError):
    """Report a malformed detached review handle without content."""


@final
@dataclass(frozen=True, slots=True)
class ReviewDisplayPublicHeader:
    """Store the bounded public metadata available to generic consumers."""

    correlation_id: str
    page_count: int

    def __post_init__(self) -> None:
        """Require one generic-safe correlation and finite page count."""
        _correlation_bytes(self.correlation_id)
        if (
            type(self.page_count) is not int
            or not 1 <= self.page_count <= MAX_APPROVER_REVIEW_PAGE_COUNT
        ):
            raise ReviewDisplayCodecError("review view handle is invalid")


@final
@dataclass(frozen=True, slots=True)
class ReviewDisplaySealedPayload:
    """Store opaque authenticated ciphertext parsed from a detached handle."""

    nonce: bytes
    ciphertext: bytes


def create_approver_review_view_handle(
    header: ReviewDisplayPublicHeader,
    nonce: bytes,
    ciphertext: bytes,
) -> ApproverReviewView:
    """Return one bounded detached handle without encrypting or decrypting."""
    correlation = _correlation_bytes(header.correlation_id)
    if (
        type(nonce) is not bytes
        or len(nonce) != APPROVER_REVIEW_VIEW_NONCE_BYTES
        or type(ciphertext) is not bytes
        or len(ciphertext) < APPROVER_REVIEW_VIEW_TAG_BYTES
    ):
        raise ReviewDisplayCodecError("review view handle is invalid")
    value = (
        _VIEW_MAGIC
        + bytes((len(correlation),))
        + correlation
        + header.page_count.to_bytes(_PAGE_COUNT_BYTES, "big")
        + nonce
        + ciphertext
    )
    if len(value) > MAX_APPROVER_REVIEW_VIEW_BYTES:
        raise ReviewDisplayCodecError("review view handle is invalid")
    return ApproverReviewView(value)


def review_display_public_header(
    view: ApproverReviewView,
) -> ReviewDisplayPublicHeader:
    """Return only bounded public correlation and pagination metadata."""
    value = _view_bytes(view)
    correlation_start = len(_VIEW_MAGIC) + _CORRELATION_LENGTH_BYTES
    correlation_end = correlation_start + value[len(_VIEW_MAGIC)]
    correlation = value[correlation_start:correlation_end]
    try:
        correlation_id = correlation.decode("ascii")
    except UnicodeDecodeError as error:
        raise ReviewDisplayCodecError(
            "review view handle is invalid"
        ) from error
    page_start = correlation_end
    page_count = int.from_bytes(
        value[page_start : page_start + _PAGE_COUNT_BYTES], "big"
    )
    return ReviewDisplayPublicHeader(correlation_id, page_count)


def review_display_sealed_payload(
    view: ApproverReviewView,
) -> ReviewDisplaySealedPayload:
    """Return opaque nonce and ciphertext after header validation."""
    value = _view_bytes(view)
    header = review_display_public_header(view)
    payload_start = (
        len(_VIEW_MAGIC)
        + _CORRELATION_LENGTH_BYTES
        + len(header.correlation_id)
        + _PAGE_COUNT_BYTES
    )
    return ReviewDisplaySealedPayload(
        value[
            payload_start : payload_start + APPROVER_REVIEW_VIEW_NONCE_BYTES
        ],
        value[payload_start + APPROVER_REVIEW_VIEW_NONCE_BYTES :],
    )


def render_review_log(view: ApproverReviewView) -> bytes:
    """Render content-free generic log metadata from a detached view handle."""
    header = review_display_public_header(view)
    return dumps(
        {
            "audience": "generic_log",
            "correlation_id": header.correlation_id,
            "event": "privileged_review_available",
            "review_page_count": header.page_count,
        },
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _view_bytes(view: ApproverReviewView) -> bytes:
    """Return one exact bounded handle byte sequence or a safe error."""
    if (
        type(view) is not bytes
        or len(view)
        < _MINIMUM_HEADER_BYTES
        + APPROVER_REVIEW_VIEW_NONCE_BYTES
        + APPROVER_REVIEW_VIEW_TAG_BYTES
        or len(view) > MAX_APPROVER_REVIEW_VIEW_BYTES
        or view[: len(_VIEW_MAGIC)] != _VIEW_MAGIC
        or not _MIN_CORRELATION_BYTES
        <= view[len(_VIEW_MAGIC)]
        <= _MAX_CORRELATION_BYTES
        or len(view)
        < len(_VIEW_MAGIC)
        + _CORRELATION_LENGTH_BYTES
        + view[len(_VIEW_MAGIC)]
        + _PAGE_COUNT_BYTES
        + APPROVER_REVIEW_VIEW_NONCE_BYTES
        + APPROVER_REVIEW_VIEW_TAG_BYTES
    ):
        raise ReviewDisplayCodecError("review view handle is invalid")
    return view


def _correlation_bytes(value: str) -> bytes:
    """Return one exact generic-safe public correlation encoding."""
    if (
        type(value) is not str
        or not _MIN_CORRELATION_BYTES <= len(value) <= _MAX_CORRELATION_BYTES
        or any(item not in _CORRELATION_CHARACTERS for item in value)
    ):
        raise ReviewDisplayCodecError("review view handle is invalid")
    return value.encode("ascii")

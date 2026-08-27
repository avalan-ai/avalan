"""Render complete approver review safely from detached trusted delivery.

This trusted-host module deliberately has no CLI, logger, shell-history,
broker, target, or mutation dependency. It can only build a privileged
display view from an exact approver projection boundary and its exact
authority witness. The delivered view itself is a detached bytes handle from
the data-only :mod:`avalan.patch.review_display_codec` module.

The lower-consumer boundary begins with that detached handle and the data-only
codec. Arbitrary same-process code that imports this trusted module or walks
global ``gc.get_objects()`` is a trusted-host compromise, outside this lower
consumer threat model, as it is for ``projection.py``.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from hmac import compare_digest
from json import JSONDecodeError, dumps, loads
from secrets import token_bytes
from typing import Never, NoReturn, TypeAlias, final

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from rich.text import Text

from avalan.patch.domain import PatchPublicCorrelationId
from avalan.patch.projection import (
    ApproverProjectionAuthority,
    ApproverProjectionBoundary,
    ProjectionError,
)
from avalan.patch.review_display_codec import (
    ApproverReviewView,
    ReviewDisplayCodecError,
    ReviewDisplayPublicHeader,
    ReviewDisplaySealedPayload,
    create_approver_review_view_handle,
    review_display_public_header,
    review_display_sealed_payload,
)

MAX_REVIEW_PAGE_CHARACTERS = 1024
_REVIEW_SCHEMA_VERSION = 1
_AES_GCM_KEY_BYTES = 32
_AES_GCM_NONCE_BYTES = 12
_SAFE_ASCII = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 +- =,;"
)

_DecodedJsonValue: TypeAlias = (
    str
    | int
    | float
    | bool
    | None
    | list["_DecodedJsonValue"]
    | dict[str, "_DecodedJsonValue"]
)
_DecodedJsonObject: TypeAlias = dict[str, _DecodedJsonValue]


class ReviewDisplayError(ValueError):
    """Report an invalid review display boundary without exposing content."""


@final
@dataclass(frozen=True, slots=True)
class ReviewPageIndex:
    """Identify one zero-based complete-review page."""

    value: int

    def __post_init__(self) -> None:
        """Require a finite nonnegative review page index."""
        if type(self.value) is not int or self.value < 0:
            raise ReviewDisplayError("review page index is invalid")


@final
@dataclass(frozen=True, slots=True)
class CompleteDiffPagination:
    """Describe complete safe-diff retrieval without rendering content."""

    page_count: int
    page_character_limit: int
    content_complete: bool

    def __post_init__(self) -> None:
        """Require an explicit complete finite page range."""
        if (
            type(self.page_count) is not int
            or self.page_count < 1
            or self.page_character_limit != MAX_REVIEW_PAGE_CHARACTERS
            or self.content_complete is not True
        ):
            raise ReviewDisplayError("review pagination is invalid")


@final
@dataclass(frozen=True, slots=True)
class ResolvedReviewPath:
    """Store one safely rendered resolved reviewer lineage path pair."""

    lineage_id: str
    source_path: str | None
    destination_path: str | None
    effects: tuple[str, ...]


@final
@dataclass(frozen=True, slots=True)
class ReviewRisk:
    """Store fixed-label atomicity and staging facts for one lineage."""

    lineage_id: str
    atomicity: str
    staging: str


@final
@dataclass(frozen=True, slots=True)
class ReviewRegion:
    """Store one complete numeric reviewer match region."""

    logical_start: int
    logical_end: int
    byte_start: int
    byte_end: int


@final
@dataclass(frozen=True, slots=True)
class TrustedReviewerActionPrompt:
    """Describe fixed reviewer controls without granting an approval."""

    correlation_id: PatchPublicCorrelationId
    actions: tuple[str, ...]


@final
@dataclass(frozen=True, slots=True, eq=False, init=False)
class ApproverReviewViewAuthority:
    """Bind one trusted-host authority to its exact sealed review handle."""

    _decryption_key: bytes
    _source_digest: str
    _terminal_digest: str
    _page_count: int
    _view_digest: bytes
    correlation_id: PatchPublicCorrelationId

    def __init__(self, issuer: Never) -> None:
        """Reject direct construction outside the trusted review factory."""
        del issuer
        raise ReviewDisplayError("review authority is factory-issued")

    def __repr__(self) -> str:
        """Render an opaque marker without review content."""
        return "ApproverReviewViewAuthority(<opaque>)"

    def __copy__(self) -> NoReturn:
        """Reject copying an exact review authority witness."""
        raise ReviewDisplayError("review authority cannot be copied")

    def __deepcopy__(self, memo: dict[int, object]) -> NoReturn:
        """Reject deep-copying an exact review authority witness."""
        del memo
        raise ReviewDisplayError("review authority cannot be copied")

    def __reduce__(self) -> NoReturn:
        """Reject serializing an authority witness."""
        raise ReviewDisplayError("review authority cannot be serialized")

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        """Reject protocol-specific authority serialization."""
        del protocol
        raise ReviewDisplayError("review authority cannot be serialized")


def safe_review_text(value: str) -> str:
    """Return a complete visible ASCII encoding for untrusted review text."""
    if type(value) is not str:
        raise ReviewDisplayError("review text is invalid")
    return "".join(_visible_character(item) for item in value)


def create_approver_review_view(
    boundary: ApproverProjectionBoundary,
    authority: ApproverProjectionAuthority,
) -> tuple[ApproverReviewView, ApproverReviewViewAuthority]:
    """Build one safe complete review from an exact approver projection."""
    if (
        type(boundary) is not ApproverProjectionBoundary
        or type(authority) is not ApproverProjectionAuthority
    ):
        raise ReviewDisplayError("approver projection boundary is invalid")
    try:
        delivery = boundary.project(authority)
    except ProjectionError as error:
        raise ReviewDisplayError("approver review is unavailable") from error
    decoded = _decode_approver_delivery(delivery)
    key = token_bytes(_AES_GCM_KEY_BYTES)
    review_authority = object.__new__(ApproverReviewViewAuthority)
    object.__setattr__(review_authority, "_decryption_key", key)
    object.__setattr__(
        review_authority,
        "_source_digest",
        boundary._terminal.source_digest,
    )
    object.__setattr__(
        review_authority,
        "_terminal_digest",
        boundary._terminal.terminal_digest,
    )
    object.__setattr__(review_authority, "_page_count", len(decoded.pages))
    object.__setattr__(
        review_authority,
        "correlation_id",
        decoded.correlation_id,
    )
    sealed = _seal_delivery(delivery, review_authority)
    view = create_approver_review_view_handle(
        ReviewDisplayPublicHeader(
            decoded.correlation_id.value,
            len(decoded.pages),
        ),
        sealed.nonce,
        sealed.ciphertext,
    )
    object.__setattr__(review_authority, "_view_digest", sha256(view).digest())
    return view, review_authority


def review_pagination(
    view: ApproverReviewView,
    authority: ApproverReviewViewAuthority,
) -> CompleteDiffPagination:
    """Return complete-diff pagination only to the view's approver witness."""
    decoded = _require_authority(view, authority)
    return CompleteDiffPagination(
        len(decoded.pages), MAX_REVIEW_PAGE_CHARACTERS, True
    )


def trusted_reviewer_action_prompt(
    view: ApproverReviewView,
    authority: ApproverReviewViewAuthority,
) -> TrustedReviewerActionPrompt:
    """Return fixed display-only controls for an attached review authority."""
    _require_authority(view, authority)
    return TrustedReviewerActionPrompt(
        authority.correlation_id,
        ("approve", "deny", "cancel"),
    )


def render_review_plain(
    view: ApproverReviewView,
    authority: ApproverReviewViewAuthority,
    page: ReviewPageIndex,
) -> str:
    """Render one complete-review page as safe fixed-label plain text."""
    decoded = _require_authority(view, authority)
    content = _page(decoded, page)
    lines = [
        "Privileged patch review",
        f"Audience correlation: {authority.correlation_id.value}",
        "Trusted runtime and target summary:",
    ]
    lines.extend(f"  {label}: {value}" for label, value in decoded.runtime)
    lines.append("Resolved paths:")
    if decoded.paths:
        for item, regions in zip(decoded.paths, decoded.regions, strict=True):
            lines.append(f"  Lineage: {item.lineage_id}")
            lines.append(
                f"    Resolved source: {_path_value(item.source_path)}"
            )
            lines.append(
                "    Resolved destination: "
                f"{_path_value(item.destination_path)}"
            )
            lines.append(f"    Effects: {', '.join(item.effects)}")
            lines.append(f"    Complete regions: {len(regions)}")
    else:
        lines.append("  None")
    lines.append("Policy warnings:")
    lines.extend(f"  {warning}" for warning in decoded.warnings)
    if not decoded.warnings:
        lines.append("  None")
    lines.append("Atomicity and staging risk:")
    lines.extend(
        "  Lineage "
        f"{risk.lineage_id}: atomicity={risk.atomicity}; "
        f"staging={risk.staging}"
        for risk in decoded.risks
    )
    lines.append("Untrusted model text (display-only; never an action):")
    lines.append(
        f"  Complete diff page {page.value + 1}/{len(decoded.pages)} "
        f"(safe page limit {MAX_REVIEW_PAGE_CHARACTERS} characters):"
    )
    lines.append(f"  {content}")
    lines.append("Trusted reviewer action:")
    lines.append(
        "  Use an attached policy-authorized control to approve, deny, or "
        "cancel. Display text cannot authorize a patch."
    )
    return "\n".join(lines)


def render_review_ansi(
    view: ApproverReviewView,
    authority: ApproverReviewViewAuthority,
    page: ReviewPageIndex,
) -> str:
    """Render one safe plain review string suitable for ANSI terminals."""
    return render_review_plain(view, authority, page)


def render_review_rich(
    view: ApproverReviewView,
    authority: ApproverReviewViewAuthority,
    page: ReviewPageIndex,
) -> Text:
    """Render one review page as literal Rich text without markup parsing."""
    return Text(render_review_plain(view, authority, page))


def render_review_json(
    view: ApproverReviewView,
    authority: ApproverReviewViewAuthority,
    page: ReviewPageIndex,
) -> bytes:
    """Render one authorized complete-review page as canonical JSON bytes."""
    decoded = _require_authority(view, authority)
    content = _page(decoded, page)
    payload = {
        "schema_version": _REVIEW_SCHEMA_VERSION,
        "audience": "approver_review",
        "correlation_id": authority.correlation_id.value,
        "runtime_target_summary": dict(decoded.runtime),
        "resolved_paths": tuple(
            {
                "lineage_id": item.lineage_id,
                "source_path": item.source_path,
                "destination_path": item.destination_path,
                "effects": item.effects,
                "regions": tuple(
                    {
                        "logical_start": region.logical_start,
                        "logical_end": region.logical_end,
                        "byte_start": region.byte_start,
                        "byte_end": region.byte_end,
                    }
                    for region in regions
                ),
            }
            for item, regions in zip(
                decoded.paths, decoded.regions, strict=True
            )
        ),
        "policy_warnings": decoded.warnings,
        "atomicity_staging_risk": tuple(
            {
                "lineage_id": risk.lineage_id,
                "atomicity": risk.atomicity,
                "staging": risk.staging,
            }
            for risk in decoded.risks
        ),
        "untrusted_model_text": {
            "content": content,
            "page_index": page.value,
            "page_count": len(decoded.pages),
            "page_character_limit": MAX_REVIEW_PAGE_CHARACTERS,
            "content_complete": True,
        },
        "trusted_reviewer_action": {
            "actions": ("approve", "deny", "cancel"),
            "requires": "attached_policy_authority",
        },
    }
    return dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


@dataclass(frozen=True, slots=True)
class _DecodedApproverReview:
    """Store parsed safe review data before minting a private view witness."""

    correlation_id: PatchPublicCorrelationId
    runtime: tuple[tuple[str, str], ...]
    paths: tuple[ResolvedReviewPath, ...]
    risks: tuple[ReviewRisk, ...]
    regions: tuple[tuple[ReviewRegion, ...], ...]
    warnings: tuple[str, ...]
    pages: tuple[str, ...]


def _decode_approver_delivery(value: bytes) -> _DecodedApproverReview:
    """Decode only a boundary-produced approver delivery into safe sections."""
    if type(value) is not bytes:
        raise ReviewDisplayError("approver review delivery is invalid")
    try:
        decoded: _DecodedJsonValue = loads(value.decode("utf-8"))
    except (JSONDecodeError, UnicodeDecodeError) as error:
        raise ReviewDisplayError(
            "approver review delivery is invalid"
        ) from error
    envelope = _mapping(decoded, "approver review delivery")
    _exact_keys(
        envelope,
        frozenset(
            (
                "schema_version",
                "audience",
                "correlation_id",
                "payload",
            )
        ),
        "approver review delivery",
    )
    if (
        _integer(envelope["schema_version"], "approver review delivery") != 1
        or _text(envelope["audience"], "approver review delivery")
        != "approver"
    ):
        raise ReviewDisplayError("approver review delivery is invalid")
    correlation_id = PatchPublicCorrelationId(
        _text(envelope["correlation_id"], "approver review delivery")
    )
    payload = _mapping(envelope["payload"], "approver review payload")
    _exact_keys(
        payload,
        frozenset(
            (
                "status",
                "mutation_state",
                "lineage_state",
                "requested_effect_occurred",
                "artifact_state",
                "workspace_change",
                "commit_set_exact",
                "postcondition",
                "diagnostic_code",
                "diff",
                "review",
            )
        ),
        "approver review payload",
    )
    runtime = _runtime(payload)
    review = _mapping(payload["review"], "complete review")
    _exact_keys(
        review,
        frozenset(
            (
                "lineages",
                "warnings",
                "diff",
                "expiry",
                "fingerprint",
                "runtime",
            )
        ),
        "complete review",
    )
    paths, risks, regions = _lineages(review["lineages"])
    warnings = _warnings(review["warnings"])
    diff = _complete_diff(review["diff"])
    runtime = runtime + _review_runtime(review["runtime"])
    return _DecodedApproverReview(
        correlation_id,
        runtime,
        paths,
        risks,
        regions,
        warnings,
        _pages(safe_review_text(diff)),
    )


def _runtime(
    payload: _DecodedJsonObject,
) -> tuple[tuple[str, str], ...]:
    """Return fixed-label terminal facts from trusted approver payload data."""
    return (
        (
            "Terminal status",
            safe_review_text(_text(payload["status"], "status")),
        ),
        (
            "Mutation state",
            safe_review_text(
                _text(payload["mutation_state"], "mutation state")
            ),
        ),
        (
            "Lineage state",
            safe_review_text(_text(payload["lineage_state"], "lineage state")),
        ),
        (
            "Artifact state",
            safe_review_text(
                _text(payload["artifact_state"], "artifact state")
            ),
        ),
        (
            "Workspace change",
            safe_review_text(
                _text(payload["workspace_change"], "workspace change")
            ),
        ),
    )


def _review_runtime(
    value: _DecodedJsonValue,
) -> tuple[tuple[str, str], ...]:
    """Return fixed-label runtime and target fields only for approvers."""
    runtime = _mapping(value, "review runtime")
    _exact_keys(
        runtime,
        frozenset(
            (
                "context_kind",
                "target_implementation",
                "target_platform",
                "approval_mode",
            )
        ),
        "review runtime",
    )
    return (
        (
            "Runtime context",
            safe_review_text(
                _text(runtime["context_kind"], "runtime context")
            ),
        ),
        (
            "Target implementation",
            safe_review_text(
                _text(
                    runtime["target_implementation"], "target implementation"
                )
            ),
        ),
        (
            "Target platform",
            safe_review_text(
                _text(runtime["target_platform"], "target platform")
            ),
        ),
        (
            "Approval mode",
            safe_review_text(_text(runtime["approval_mode"], "approval mode")),
        ),
    )


def _lineages(
    value: _DecodedJsonValue,
) -> tuple[
    tuple[ResolvedReviewPath, ...],
    tuple[ReviewRisk, ...],
    tuple[tuple[ReviewRegion, ...], ...],
]:
    """Return resolved path, risk, and complete region sections."""
    if type(value) is not list:
        raise ReviewDisplayError("review lineages are invalid")
    paths: list[ResolvedReviewPath] = []
    risks: list[ReviewRisk] = []
    all_regions: list[tuple[ReviewRegion, ...]] = []
    for item in value:
        lineage = _mapping(item, "review lineage")
        _exact_keys(
            lineage,
            frozenset(
                (
                    "lineage_id",
                    "source_path",
                    "destination_path",
                    "effects",
                    "regions",
                    "atomicity",
                    "staging",
                )
            ),
            "review lineage",
        )
        lineage_id = safe_review_text(
            _text(lineage["lineage_id"], "review lineage")
        )
        source_path = _optional_path(lineage["source_path"])
        destination_path = _optional_path(lineage["destination_path"])
        effects = _text_list(lineage["effects"], "review effects")
        paths.append(
            ResolvedReviewPath(
                lineage_id,
                source_path,
                destination_path,
                effects,
            )
        )
        risks.append(
            ReviewRisk(
                lineage_id,
                safe_review_text(_text(lineage["atomicity"], "atomicity")),
                safe_review_text(_text(lineage["staging"], "staging")),
            )
        )
        all_regions.append(_regions(lineage["regions"]))
    return tuple(paths), tuple(risks), tuple(all_regions)


def _optional_path(value: _DecodedJsonValue) -> str | None:
    """Return one safely rendered resolved path or an absent endpoint."""
    if value is None:
        return None
    path = _mapping(value, "review path")
    _exact_keys(path, frozenset(("value",)), "review path")
    return safe_review_text(_text(path["value"], "review path"))


def _regions(value: _DecodedJsonValue) -> tuple[ReviewRegion, ...]:
    """Return every exact numeric review region without content rendering."""
    if type(value) is not list:
        raise ReviewDisplayError("review regions are invalid")
    result: list[ReviewRegion] = []
    for item in value:
        region = _mapping(item, "review region")
        _exact_keys(
            region,
            frozenset(
                (
                    "logical_start",
                    "logical_end",
                    "byte_start",
                    "byte_end",
                )
            ),
            "review region",
        )
        result.append(
            ReviewRegion(
                _integer(region["logical_start"], "review region"),
                _integer(region["logical_end"], "review region"),
                _integer(region["byte_start"], "review region"),
                _integer(region["byte_end"], "review region"),
            )
        )
    return tuple(result)


def _warnings(value: _DecodedJsonValue) -> tuple[str, ...]:
    """Return fixed-section policy warning values as literal safe text."""
    if type(value) is not list:
        raise ReviewDisplayError("review warnings are invalid")
    result: list[str] = []
    for item in value:
        warning = _mapping(item, "review warning")
        _exact_keys(warning, frozenset(("value",)), "review warning")
        result.append(
            safe_review_text(_text(warning["value"], "review warning"))
        )
    return tuple(result)


def _complete_diff(value: _DecodedJsonValue) -> str:
    """Return the complete approver diff representation before safe paging."""
    review = _mapping(value, "review diff")
    _exact_keys(review, frozenset(("diff", "digest", "size")), "review diff")
    diff = _mapping(review["diff"], "review diff")
    _exact_keys(diff, frozenset(("value",)), "review diff")
    raw = diff["value"]
    if type(raw) is str:
        return raw
    encoded = _mapping(raw, "review diff")
    _exact_keys(encoded, frozenset(("encoding", "value")), "review diff")
    if _text(encoded["encoding"], "review diff") != "hex":
        raise ReviewDisplayError("review diff is invalid")
    return "hex " + _text(encoded["value"], "review diff")


def _pages(value: str) -> tuple[str, ...]:
    """Split complete safe review text into deterministic fixed-size pages."""
    if not value:
        return ("",)
    return tuple(
        value[index : index + MAX_REVIEW_PAGE_CHARACTERS]
        for index in range(0, len(value), MAX_REVIEW_PAGE_CHARACTERS)
    )


def _page(review: _DecodedApproverReview, page: ReviewPageIndex) -> str:
    """Return one bounded rendered review page after range validation."""
    if type(page) is not ReviewPageIndex or page.value >= len(review.pages):
        raise ReviewDisplayError("review page is unavailable")
    return review.pages[page.value]


def _require_authority(
    view: ApproverReviewView,
    authority: ApproverReviewViewAuthority,
) -> _DecodedApproverReview:
    """Authenticate, decrypt, and validate one exact approver review view."""
    if type(authority) is not ApproverReviewViewAuthority:
        raise ReviewDisplayError("review authority is not issued here")
    try:
        header = review_display_public_header(view)
        sealed = review_display_sealed_payload(view)
        is_bound = (
            compare_digest(sha256(view).digest(), authority._view_digest)
            and header.correlation_id == authority.correlation_id.value
            and header.page_count == authority._page_count
        )
    except (AttributeError, ReviewDisplayCodecError) as error:
        raise ReviewDisplayError(
            "review authority is not issued here"
        ) from error
    if not is_bound:
        raise ReviewDisplayError("review authority is not issued here")
    try:
        delivery = AESGCM(authority._decryption_key).decrypt(
            sealed.nonce,
            sealed.ciphertext,
            _review_aad(authority),
        )
    except (InvalidTag, ValueError) as error:
        raise ReviewDisplayError(
            "review authority is not issued here"
        ) from error
    decoded = _decode_approver_delivery(delivery)
    if (
        decoded.correlation_id != authority.correlation_id
        or len(decoded.pages) != authority._page_count
    ):
        raise ReviewDisplayError("review authority is not issued here")
    return decoded


def _seal_delivery(
    delivery: bytes,
    authority: ApproverReviewViewAuthority,
) -> ReviewDisplaySealedPayload:
    """Encrypt one detached delivery for the exact authorized review view."""
    nonce = token_bytes(_AES_GCM_NONCE_BYTES)
    return ReviewDisplaySealedPayload(
        nonce,
        AESGCM(authority._decryption_key).encrypt(
            nonce,
            delivery,
            _review_aad(authority),
        ),
    )


def _review_aad(authority: ApproverReviewViewAuthority) -> bytes:
    """Return fixed audience/source binding for sealed review bytes."""
    return (
        "approver_review\0"
        f"{authority.correlation_id.value}\0"
        f"{authority._source_digest}\0"
        f"{authority._terminal_digest}\0"
        f"{authority._page_count}"
    ).encode("ascii")


def _mapping(value: _DecodedJsonValue, label: str) -> _DecodedJsonObject:
    """Return an exact decoded JSON object or a content-free error."""
    if type(value) is not dict:
        raise ReviewDisplayError(f"{label} is invalid")
    return value


def _exact_keys(
    value: Mapping[str, _DecodedJsonValue],
    expected: frozenset[str],
    label: str,
) -> None:
    """Require one closed object schema before any review display derives."""
    if set(value) != expected:
        raise ReviewDisplayError(f"{label} schema is invalid")


def _text(value: _DecodedJsonValue, label: str) -> str:
    """Return one exact decoded string or a content-free error."""
    if type(value) is not str:
        raise ReviewDisplayError(f"{label} is invalid")
    return value


def _integer(value: _DecodedJsonValue, label: str) -> int:
    """Return one exact nonnegative decoded integer or a safe error."""
    if type(value) is not int or value < 0:
        raise ReviewDisplayError(f"{label} is invalid")
    return value


def _text_list(value: _DecodedJsonValue, label: str) -> tuple[str, ...]:
    """Return a bounded literal string list without model-authored labels."""
    if type(value) is not list:
        raise ReviewDisplayError(f"{label} is invalid")
    return tuple(safe_review_text(_text(item, label)) for item in value)


def _visible_character(value: str) -> str:
    """Return an inert ASCII representation preserving one code point."""
    codepoint = ord(value)
    if value in _SAFE_ASCII:
        return value
    return f"(U+{codepoint:04X})"


def _path_value(value: str | None) -> str:
    """Return a fixed absent marker or one already-safe resolved path."""
    return "absent" if value is None else value

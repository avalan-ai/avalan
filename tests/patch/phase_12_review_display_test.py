"""Exercise the Phase 12 detached privileged review display foundation."""

from asyncio import run
from copy import copy, deepcopy
from dataclasses import asdict
from gc import get_referents
from hashlib import sha256
from json import dumps, loads
from pathlib import Path
from pickle import dumps as pickle_dumps
from runpy import run_path

import pytest
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from rich.text import Text

import avalan.patch.review_display as display_module
import avalan.patch.review_display_codec as codec_module
from avalan.patch.projection import (
    PatchProjectionSource,
    create_approver_projection_boundary,
    create_model_projection_boundary,
)
from avalan.patch.review_display import (
    MAX_REVIEW_PAGE_CHARACTERS,
    ApproverReviewViewAuthority,
    CompleteDiffPagination,
    ReviewDisplayError,
    ReviewPageIndex,
    create_approver_review_view,
    render_review_ansi,
    render_review_json,
    render_review_plain,
    render_review_rich,
    review_pagination,
    safe_review_text,
    trusted_reviewer_action_prompt,
)
from avalan.patch.review_display_codec import (
    APPROVER_REVIEW_VIEW_NONCE_BYTES,
    ApproverReviewView,
    ReviewDisplayCodecError,
    ReviewDisplayPublicHeader,
    create_approver_review_view_handle,
    render_review_log,
    review_display_public_header,
)

_PHASE_TWELVE = run_path(
    str(Path("tests/patch/phase_12_contract_test.py").resolve())
)
_CANARY = (
    "\x1b[2J\x1b]8;;javascript:alert(1)\x07"
    "\x1bPhidden\x1b\\\x9b31m\x00\x1f\rFORGED APPROVED\n"
    "\u202ereversed\u2066hidden\u2069 <script>[bold]"
    " [label](file:/private/patch) data:text/plain,javascript:run() "
    "www.example.test"
)
_UNICODE_CANARY = "".join(
    (
        "\u00ad",
        "\u180e",
        "\u2061",
        "\u206f",
        "\u202e",
        "\ufeff",
        "\ufe00",
        "\ufe0f",
        chr(0xE0100),
        chr(0xE01EF),
        chr(0xE0000),
        chr(0xE007F),
        chr(0xFDD0),
        chr(0xFFFE),
        chr(0x10FFFF),
        "\u0430",
        "\ud800",
    )
)


def _source() -> PatchProjectionSource:
    """Return one complete-review-authorized Phase 12 source fixture."""
    source = run(
        _PHASE_TWELVE["_source"](_PHASE_TWELVE["_full_disclosures"]())
    )
    assert isinstance(source, PatchProjectionSource)
    return source


def _invalid_call(name: str, *values: object) -> None:
    """Call one private validation helper through a dynamic test seam."""
    function = getattr(display_module, name)
    function(*values)


def _invalid_asdict_call(value: object) -> None:
    """Call ``asdict`` dynamically to exercise a non-dataclass failure."""
    function: object = asdict
    getattr(function, "__call__")(value)


def _invalid_view_constructor_call() -> None:
    """Call the detached-view NewType dynamically without its byte input."""
    function: object = ApproverReviewView
    getattr(function, "__call__")()


def _contains_raw(
    value: object, raw: str, seen: set[int] | None = None
) -> bool:
    """Return whether direct object referents expose a raw protected canary."""
    if isinstance(value, type):
        return False
    active_seen = set() if seen is None else seen
    if id(value) in active_seen:
        return False
    active_seen.add(id(value))
    if type(value) is str:
        return raw in value
    if type(value) is bytes:
        return raw.encode() in value
    return any(
        _contains_raw(item, raw, active_seen) for item in get_referents(value)
    )


def _assert_ascii_strings(value: object) -> None:
    """Require recursively decoded renderer values to contain ASCII only."""
    if type(value) is str:
        assert all(item == "\n" or 0x20 <= ord(item) <= 0x7E for item in value)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _assert_ascii_strings(key)
            _assert_ascii_strings(item)
    if isinstance(value, list | tuple):
        for item in value:
            _assert_ascii_strings(item)


def _view() -> tuple[ApproverReviewView, ApproverReviewViewAuthority]:
    """Return one exact detached safe review view and its witness."""
    boundary = create_approver_projection_boundary(_source())
    return create_approver_review_view(boundary, boundary.authority())


def _malicious_view(
    *,
    diff: str = _CANARY,
) -> tuple[ApproverReviewView, ApproverReviewViewAuthority]:
    """Return a host-bound review fixture carrying hostile display canaries."""
    boundary = create_approver_projection_boundary(_source())
    delivery = loads(boundary.project(boundary.authority()))
    payload = delivery["payload"]
    review = payload["review"]
    assert isinstance(review, dict)
    review_diff = review["diff"]
    assert isinstance(review_diff, dict)
    complete = review_diff["diff"]
    assert isinstance(complete, dict)
    complete["value"] = diff
    lineages = review["lineages"]
    assert isinstance(lineages, list) and lineages
    lineage = lineages[0]
    assert isinstance(lineage, dict)
    for field in ("source_path", "destination_path"):
        path = lineage[field]
        assert isinstance(path, dict)
        path["value"] = _CANARY
    lineage["lineage_id"] = _CANARY
    lineage["atomicity"] = _CANARY
    lineage["staging"] = _CANARY
    lineage["effects"] = [_CANARY]
    review["warnings"] = [{"value": _CANARY}]
    runtime = review["runtime"]
    assert isinstance(runtime, dict)
    for key in runtime:
        runtime[key] = _CANARY
    object.__setattr__(boundary, "_review", review)
    return create_approver_review_view(boundary, boundary.authority())


def test_safe_review_text_neutralizes_terminal_markup_and_links() -> None:
    """Render hostile review text as a single literal non-forging string."""
    rendered = safe_review_text(_CANARY)

    _assert_ascii_strings(rendered)
    assert "\x1b" not in rendered
    assert "\x9b" not in rendered
    assert "\r" not in rendered
    assert "\n" not in rendered
    assert "\u202e" not in rendered
    assert "\u2066" not in rendered
    assert "<script>" not in rendered
    assert "[bold]" not in rendered
    assert "javascript:" not in rendered
    assert "file:" not in rendered
    assert "data:" not in rendered
    assert "www.example.test" not in rendered
    assert "(U+000D)FORGED APPROVED(U+000A)" in rendered
    assert "(U+003C)script(U+003E)" in rendered
    assert "(U+005B)bold(U+005D)" in rendered
    assert "javascript(U+003A)alert(U+0028)1(U+0029)" in rendered
    assert "www(U+002E)example(U+002E)test" in rendered


def test_safe_review_text_preserves_control_bodies_and_suffixes() -> None:
    """Encode every terminal-control character without suppressing content."""
    raw = (
        "\x1b]osc-body\x07osc-suffix"
        "\x1bPdcs-body\x1b\\dcs-suffix"
        "\x1b_apc-body\x1b\\apc-suffix"
        "\x1b^pm-body\x1b\\pm-suffix"
        "\x9dc1-osc-body\x9cc1-osc-suffix"
        "\x90c1-dcs-body\x9cc1-dcs-suffix"
        "\x9fc1-apc-body\x9cc1-apc-suffix"
        "\x9ec1-pm-body\x9cc1-pm-suffix"
        "\x1b]unterminated-body-final-suffix"
    )

    rendered = safe_review_text(raw)

    _assert_ascii_strings(rendered)
    for body_or_suffix in (
        "osc-body",
        "osc-suffix",
        "dcs-body",
        "dcs-suffix",
        "apc-body",
        "apc-suffix",
        "pm-body",
        "pm-suffix",
        "c1-osc-body",
        "c1-osc-suffix",
        "c1-dcs-body",
        "c1-dcs-suffix",
        "c1-apc-body",
        "c1-apc-suffix",
        "c1-pm-body",
        "c1-pm-suffix",
        "unterminated-body-final-suffix",
    ):
        assert body_or_suffix in rendered
    for codepoint in (
        0x001B,
        0x0007,
        0x005D,
        0x005F,
        0x005E,
        0x009D,
        0x0090,
        0x009F,
        0x009E,
        0x009C,
    ):
        assert f"(U+{codepoint:04X})" in rendered


def test_safe_review_text_encodes_all_non_ascii_unicode_as_visible_ascii() -> (
    None
):
    """Make confusable and invisible Unicode content inert in every form."""
    rendered = safe_review_text(_UNICODE_CANARY)

    _assert_ascii_strings(rendered)
    for item in _UNICODE_CANARY:
        assert f"(U+{ord(item):04X})" in rendered
        assert item not in rendered


def test_approver_view_has_structural_fixed_sections_and_complete_pages() -> (
    None
):
    """Keep complete review material limited to the approver witness."""
    view, authority = _malicious_view(
        diff="x" * (MAX_REVIEW_PAGE_CHARACTERS + 1)
    )
    pagination = review_pagination(view, authority)
    prompt = trusted_reviewer_action_prompt(view, authority)
    first_page = ReviewPageIndex(0)
    plain = render_review_plain(view, authority, first_page)
    ansi = render_review_ansi(view, authority, first_page)
    rich = render_review_rich(view, authority, first_page)
    json_value = render_review_json(view, authority, first_page)

    assert isinstance(pagination, CompleteDiffPagination)
    assert pagination.page_count == 2
    assert pagination.content_complete is True
    assert prompt.actions == ("approve", "deny", "cancel")
    assert prompt.correlation_id is authority.correlation_id
    assert plain == ansi
    assert isinstance(rich, Text)
    assert rich.plain == plain
    assert b"\x1b" not in json_value
    assert b"javascript:" not in json_value
    assert b"file:" not in json_value
    assert b"data:" not in json_value
    assert "Privileged patch review" in plain
    assert "Trusted runtime and target summary:" in plain
    assert "Resolved paths:" in plain
    assert "Policy warnings:" in plain
    assert "Atomicity and staging risk:" in plain
    assert "Untrusted model text (display-only; never an action):" in plain
    assert "Trusted reviewer action:" in plain
    assert "FORGED APPROVED" in plain
    assert "(U+000D)FORGED APPROVED(U+000A)" in plain
    decoded = loads(json_value)
    assert decoded["audience"] == "approver_review"
    assert decoded["correlation_id"] == authority.correlation_id.value
    assert decoded["untrusted_model_text"]["page_count"] == 2
    assert (
        decoded["trusted_reviewer_action"]["requires"]
        == "attached_policy_authority"
    )
    assert len(render_review_plain(view, authority, ReviewPageIndex(1))) > 0
    with pytest.raises(ReviewDisplayError, match="page is unavailable"):
        render_review_plain(view, authority, ReviewPageIndex(2))


def test_privileged_renderers_preserve_encoded_unicode_pages_completely() -> (
    None
):
    """Return every encoded untrusted code point through each review format."""
    raw_diff = "x" * (MAX_REVIEW_PAGE_CHARACTERS - 3) + _UNICODE_CANARY
    view, authority = _malicious_view(diff=raw_diff)
    pagination = review_pagination(view, authority)
    pages = tuple(
        loads(render_review_json(view, authority, ReviewPageIndex(index)))[
            "untrusted_model_text"
        ]["content"]
        for index in range(pagination.page_count)
    )
    plain = render_review_plain(view, authority, ReviewPageIndex(0))
    ansi = render_review_ansi(view, authority, ReviewPageIndex(0))
    rich = render_review_rich(view, authority, ReviewPageIndex(0))
    json_value = loads(render_review_json(view, authority, ReviewPageIndex(0)))

    assert pagination.page_count > 1
    assert "".join(pages) == safe_review_text(raw_diff)
    assert plain == ansi
    assert rich.plain == plain
    _assert_ascii_strings(plain)
    _assert_ascii_strings(ansi)
    _assert_ascii_strings(rich.plain)
    _assert_ascii_strings(json_value)
    for item in _UNICODE_CANARY:
        token = f"(U+{ord(item):04X})"
        assert token in "".join(pages)
        assert item not in plain
        assert item not in dumps(json_value)


def test_generic_log_excludes_complete_review_and_all_canaries() -> None:
    """Keep generic logs and history data content-free without authority."""
    view, authority = _malicious_view()
    privileged = render_review_json(view, authority, ReviewPageIndex(0))
    generic_log = render_review_log(view)

    assert b"privileged_review_available" in generic_log
    assert authority.correlation_id.value.encode() in generic_log
    for forbidden in (
        b"FORGED APPROVED",
        b"javascript",
        b"file",
        b"data",
        b"script",
        b"private/patch",
        b"reversed",
    ):
        assert forbidden not in generic_log
    assert b"FORGED APPROVED" in privileged
    assert b"\x1b" not in generic_log
    assert b"\r" not in generic_log
    assert b"\n" not in generic_log


def test_view_graph_never_exposes_privileged_content() -> None:
    """Keep raw content out of the detached byte handle and its C graph."""
    raw = "view-raw-diff-path-runtime-canary"
    view, authority = _malicious_view(diff=raw)
    view_shapes = (
        view,
        repr(view),
        copy(view),
        deepcopy(view),
        pickle_dumps(view),
        tuple(get_referents(view)),
    )

    assert review_pagination(view, authority).content_complete is True
    assert type(view) is bytes
    assert view.__class__ is bytes
    assert not hasattr(bytes.__repr__, "__globals__")
    assert not hasattr(bytes.__reduce_ex__, "__globals__")
    for shape in view_shapes:
        assert not _contains_raw(shape, raw)
        assert not _contains_raw(shape, _CANARY)
    with pytest.raises(TypeError):
        vars(view)
    with pytest.raises(TypeError):
        _invalid_asdict_call(view)
    with pytest.raises(AttributeError):
        getattr(view, "_sealed_delivery")


def test_data_only_view_codec_cannot_reach_trusted_review_host() -> None:
    """Keep detached parsing and generic logs outside trusted globals."""
    view, authority = _view()
    source = Path("src/avalan/patch/review_display_codec.py").read_text(
        encoding="utf-8"
    )

    assert render_review_log.__module__ == "avalan.patch.review_display_codec"
    assert display_module not in codec_module.__dict__.values()
    assert "review_display import" not in source
    assert "AESGCM" not in source
    assert "_review_aad" not in source
    assert review_display_public_header(view).correlation_id == (
        authority.correlation_id.value
    )


def test_data_only_handle_cannot_reseal_or_cross_authorize_review() -> None:
    """Reject arbitrary wrapped bytes and a different authority key."""
    boundary = create_approver_projection_boundary(_source())
    projection_authority = boundary.authority()
    view, authority = create_approver_review_view(
        boundary, projection_authority
    )
    other_view, other_authority = create_approver_review_view(
        boundary, projection_authority
    )
    header = review_display_public_header(view)
    other_sealed = codec_module.review_display_sealed_payload(other_view)
    untrusted_handle = create_approver_review_view_handle(
        header,
        b"x" * APPROVER_REVIEW_VIEW_NONCE_BYTES,
        b"x" * 16,
    )

    assert (
        header.correlation_id
        == review_display_public_header(other_view).correlation_id
    )
    with pytest.raises(InvalidTag):
        AESGCM(authority._decryption_key).decrypt(
            other_sealed.nonce,
            other_sealed.ciphertext,
            display_module._review_aad(authority),
        )
    with pytest.raises(ReviewDisplayError, match="authority is not issued"):
        render_review_plain(other_view, authority, ReviewPageIndex(0))
    with pytest.raises(ReviewDisplayError, match="authority is not issued"):
        render_review_plain(untrusted_handle, authority, ReviewPageIndex(0))
    assert review_pagination(other_view, other_authority).content_complete


def test_review_authority_rejects_tamper_wrong_view_and_reconstruction() -> (
    None
):
    """Reject tampered bytes and malformed authorities without a registry."""
    view, authority = _view()
    other_view, other_authority = _view()
    forged_authority = object.__new__(ApproverReviewViewAuthority)
    tampered = ApproverReviewView(view[:-1] + bytes((view[-1] ^ 1,)))

    with pytest.raises(ReviewDisplayError, match="authority is not issued"):
        render_review_plain(other_view, authority, ReviewPageIndex(0))
    with pytest.raises(ReviewDisplayError, match="authority is not issued"):
        render_review_plain(other_view, forged_authority, ReviewPageIndex(0))
    with pytest.raises(ReviewDisplayError, match="authority is not issued"):
        render_review_plain(tampered, authority, ReviewPageIndex(0))
    object.__setattr__(authority, "_view_digest", sha256(tampered).digest())
    with pytest.raises(ReviewDisplayError, match="authority is not issued"):
        render_review_plain(tampered, authority, ReviewPageIndex(0))
    with pytest.raises(ReviewDisplayError):
        copy(authority)
    with pytest.raises(ReviewDisplayError):
        deepcopy(authority)
    with pytest.raises(ReviewDisplayError):
        pickle_dumps(authority)
    assert review_pagination(other_view, other_authority).content_complete


def test_review_authority_keeps_source_binding_out_of_delivery() -> None:
    """Keep source/terminal digests in trusted AAD rather than delivery."""
    view, authority = _view()
    sealed = codec_module.review_display_sealed_payload(view)
    delivery = AESGCM(authority._decryption_key).decrypt(
        sealed.nonce,
        sealed.ciphertext,
        display_module._review_aad(authority),
    )
    envelope = loads(delivery)
    assert isinstance(envelope, dict)
    assert "source_digest" not in envelope
    assert "terminal_digest" not in envelope
    assert "issuer_id" not in envelope
    envelope["correlation_id"] = "public_" + "a" * 16
    altered = dumps(envelope, sort_keys=True, separators=(",", ":")).encode()
    rebound = display_module._seal_delivery(altered, authority)
    altered_view = create_approver_review_view_handle(
        review_display_public_header(view),
        rebound.nonce,
        rebound.ciphertext,
    )
    object.__setattr__(
        authority,
        "_view_digest",
        sha256(altered_view).digest(),
    )

    with pytest.raises(ReviewDisplayError, match="authority is not issued"):
        render_review_plain(altered_view, authority, ReviewPageIndex(0))


def test_approver_review_boundaries_reject_substitution() -> None:
    """Reject forged, copied, wrong-audience, and unavailable review values."""
    view, authority = _view()
    model_boundary = create_model_projection_boundary(_source())
    model_authority = model_boundary.authority()

    with pytest.raises(ReviewDisplayError, match="authority is not issued"):
        getattr(render_review_plain, "__call__")(
            view,
            model_authority,
            ReviewPageIndex(0),
        )
    forged_authority = object.__new__(ApproverReviewViewAuthority)
    with pytest.raises(ReviewDisplayError, match="authority is not issued"):
        render_review_json(view, forged_authority, ReviewPageIndex(0))
    with pytest.raises(ReviewDisplayCodecError):
        render_review_log(ApproverReviewView(b"forged"))
    for value in (authority,):
        with pytest.raises(ReviewDisplayError):
            copy(value)
        with pytest.raises(ReviewDisplayError):
            deepcopy(value)
        with pytest.raises(ReviewDisplayError):
            pickle_dumps(value)
    with pytest.raises(TypeError):
        _invalid_view_constructor_call()
    with pytest.raises(TypeError):
        getattr(ApproverReviewViewAuthority, "__call__")()
    hidden_source = run(_PHASE_TWELVE["_source"](frozenset()))
    assert isinstance(hidden_source, PatchProjectionSource)
    hidden_boundary = create_approver_projection_boundary(hidden_source)
    with pytest.raises(ReviewDisplayError, match="unavailable"):
        create_approver_review_view(
            hidden_boundary,
            hidden_boundary.authority(),
        )
    with pytest.raises(ReviewDisplayError, match="boundary is invalid"):
        getattr(create_approver_review_view, "__call__")(
            model_boundary,
            model_authority,
        )


def test_review_decoder_rejects_malformed_delivery() -> None:
    """Fail closed before malformed review becomes a privileged display."""
    boundary = create_approver_projection_boundary(_source())
    review = boundary._review
    assert review is not None
    malformed = dict(review)
    malformed["unexpected"] = "value"
    object.__setattr__(boundary, "_review", malformed)

    with pytest.raises(ReviewDisplayError, match="schema"):
        create_approver_review_view(boundary, boundary.authority())


def test_review_display_keeps_cli_history_and_logger_paths_unwired() -> None:
    """Keep the foundation detached from real CLI and logging integration."""
    source = Path("src/avalan/patch/review_display.py").read_text(
        encoding="utf-8"
    )

    assert "avalan.cli" not in source
    assert "logging import" not in source
    assert "PlanApprovalBroker" not in source
    assert "from avalan.cli" not in source
    assert "subprocess" not in source


def test_review_display_closed_helpers_and_empty_sections_fail_closed() -> (
    None
):
    """Cover malformed display inputs without creating review truth."""
    boundary = create_approver_projection_boundary(_source())
    review = dict(boundary._review or {})
    review["lineages"] = ()
    review["warnings"] = ()
    object.__setattr__(boundary, "_review", review)
    view, authority = create_approver_review_view(
        boundary, boundary.authority()
    )

    plain = render_review_plain(view, authority, ReviewPageIndex(0))
    assert "Resolved paths:\n  None" in plain
    assert "Policy warnings:\n  None" in plain
    assert display_module._pages("") == ("",)
    assert display_module._optional_path(None) is None
    assert display_module._visible_character("\t") == "(U+0009)"
    assert (
        display_module._complete_diff(
            {
                "diff": {"value": {"encoding": "hex", "value": "0a"}},
                "digest": {},
                "size": {},
            }
        )
        == "hex 0a"
    )
    with pytest.raises(ReviewDisplayError):
        display_module._complete_diff(
            {
                "diff": {"value": {"encoding": "base64", "value": "Cg"}},
                "digest": {},
                "size": {},
            }
        )
    with pytest.raises(ReviewDisplayError):
        ReviewPageIndex(-1)
    with pytest.raises(ReviewDisplayError):
        CompleteDiffPagination(0, MAX_REVIEW_PAGE_CHARACTERS, True)
    with pytest.raises(ReviewDisplayError):
        CompleteDiffPagination(1, 1, True)
    with pytest.raises(ReviewDisplayError):
        CompleteDiffPagination(1, MAX_REVIEW_PAGE_CHARACTERS, False)
    with pytest.raises(ReviewDisplayError):
        getattr(ApproverReviewViewAuthority, "__init__")(
            object.__new__(ApproverReviewViewAuthority),
            None,
        )
    assert repr(authority) == "ApproverReviewViewAuthority(<opaque>)"
    with pytest.raises(ReviewDisplayError):
        authority.__reduce__()
    with pytest.raises(ReviewDisplayError):
        getattr(display_module, "safe_review_text")(None)
    with pytest.raises(ReviewDisplayError):
        _invalid_call("_decode_approver_delivery", None)
    with pytest.raises(ReviewDisplayError):
        display_module._decode_approver_delivery(b"not json")
    delivery = loads(boundary.project(boundary.authority()))
    delivery["audience"] = "model"
    with pytest.raises(ReviewDisplayError):
        display_module._decode_approver_delivery(
            dumps(delivery, sort_keys=True, separators=(",", ":")).encode()
        )
    for name, arguments in (
        ("_mapping", ("value", "label")),
        ("_text", (None, "label")),
        ("_integer", (True, "label")),
        ("_text_list", ({}, "label")),
        ("_lineages", ({},)),
        ("_regions", ({},)),
        ("_warnings", ({},)),
        ("_complete_diff", ({},)),
    ):
        with pytest.raises(ReviewDisplayError):
            _invalid_call(name, *arguments)
    with pytest.raises(ReviewDisplayCodecError):
        ReviewDisplayPublicHeader("invalid", 1)
    with pytest.raises(ReviewDisplayCodecError):
        ReviewDisplayPublicHeader(authority.correlation_id.value, 0)
    with pytest.raises(ReviewDisplayCodecError):
        create_approver_review_view_handle(
            ReviewDisplayPublicHeader(authority.correlation_id.value, 1),
            b"x",
            b"x" * 16,
        )
    with pytest.raises(ReviewDisplayCodecError):
        create_approver_review_view_handle(
            ReviewDisplayPublicHeader(authority.correlation_id.value, 1),
            b"x" * APPROVER_REVIEW_VIEW_NONCE_BYTES,
            b"x",
        )
    with pytest.raises(ReviewDisplayCodecError):
        create_approver_review_view_handle(
            ReviewDisplayPublicHeader(authority.correlation_id.value, 1),
            b"x" * APPROVER_REVIEW_VIEW_NONCE_BYTES,
            b"x" * codec_module.MAX_APPROVER_REVIEW_VIEW_BYTES,
        )
    malformed_header = ApproverReviewView(
        b"ARV1"
        + bytes((39,))
        + b"\xff" * 39
        + (1).to_bytes(4, "big")
        + b"x" * APPROVER_REVIEW_VIEW_NONCE_BYTES
        + b"x" * 16
    )
    with pytest.raises(ReviewDisplayCodecError):
        review_display_public_header(malformed_header)
    malformed_length = ApproverReviewView(b"ARV1" + b"\x00" + b"x" * 64)
    with pytest.raises(ReviewDisplayCodecError):
        review_display_public_header(malformed_length)
    maximum_correlation = b"public_" + b"a" * 48
    maximum_boundary = (
        b"ARV1"
        + bytes((len(maximum_correlation),))
        + maximum_correlation
        + (1).to_bytes(4, "big")
    )
    truncated_payload = ApproverReviewView(maximum_boundary + b"x" * 27)
    empty_payload = ApproverReviewView(maximum_boundary)
    for malformed_view in (truncated_payload, empty_payload):
        for parser in (
            review_display_public_header,
            codec_module.review_display_sealed_payload,
            render_review_log,
        ):
            with pytest.raises(ReviewDisplayCodecError):
                parser(malformed_view)
    with pytest.raises(ReviewDisplayError):
        _invalid_call(
            "_page",
            display_module._require_authority(view, authority),
            object(),
        )

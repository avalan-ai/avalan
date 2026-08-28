"""Run the dormant local patch-review test profile without log exposure.

The module is intentionally not registered with the normal CLI parser, agent
commands, server, or external protocols.  A caller must already hold the
authenticated ``PatchSdkHost`` and the detached approver-review capability.
It renders that capability directly to an attached terminal, accepts only one
fixed action, and never stores review text in history or generic logging.
"""

import sys
from asyncio import (
    CancelledError,
    Future,
    Lock,
    create_task,
    get_running_loop,
    shield,
    wait_for,
)
from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
from hmac import compare_digest
from json import JSONDecodeError, loads
from os import (
    close,
    dup,
    fdopen,
    fstat,
    get_blocking,
    getpgrp,
    isatty,
    set_blocking,
    tcgetpgrp,
    ttyname,
)
from os import (
    write as os_write,
)
from secrets import token_bytes
from termios import TCSANOW, tcgetattr, tcsetattr
from typing import Callable, Literal, Never, NoReturn, TextIO, TypeAlias, final

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from avalan.patch.domain import (
    PatchObserverCorrelationId,
    PatchPending,
    PatchRequestId,
    PatchResult,
)
from avalan.patch.review_display import ReviewDisplayError, safe_review_text
from avalan.patch.toolset import (
    PatchSdkHost,
    PatchSdkInvocationReview,
    PatchToolError,
)

LOCAL_PATCH_REVIEW_TEST_PROFILE = "local-patch-review-test"
_TerminalAttributes: TypeAlias = list[int | list[bytes | int]]
_PREAPPROVAL_VIEW_KEY_BYTES = 32
_PREAPPROVAL_VIEW_NONCE_BYTES = 12
_PREAPPROVAL_PAGE_CHARACTERS = 1024
_RECONCILIATION_TIMEOUT_SECONDS = 1.0


class PatchCliReviewError(RuntimeError):
    """Report a bounded local patch-review terminal failure."""


class PatchCliReviewState(StrEnum):
    """Name the closed local patch-review outcomes."""

    TERMINAL = "terminal"
    PENDING = "pending"
    DENIED = "denied"
    CANCELLED = "cancelled"


class _ContinuationState(StrEnum):
    """Name the exact one-owner detached settlement states."""

    READY = "ready"
    ACTIVE = "active"
    RECONCILING = "reconciling"
    CLOSED = "closed"


@final
@dataclass(frozen=True, slots=True)
class PatchCliReviewResult:
    """Store a bounded terminal or nonterminal local CLI observation."""

    state: PatchCliReviewState
    result: PatchResult | None = None
    continuation: "PatchCliReviewContinuation | None" = None

    def __post_init__(self) -> None:
        """Require exactly one terminal result or bound continuation."""
        if self.state is PatchCliReviewState.TERMINAL:
            if (
                type(self.result) is not PatchResult
                or self.continuation is not None
            ):
                raise PatchCliReviewError(
                    "patch CLI terminal result is invalid"
                )
            return
        if self.result is not None:
            raise PatchCliReviewError(
                "patch CLI nonterminal result is invalid"
            )
        if self.state is PatchCliReviewState.PENDING:
            if type(self.continuation) is not PatchCliReviewContinuation:
                raise PatchCliReviewError("patch CLI continuation is invalid")
            return
        if self.continuation is not None:
            raise PatchCliReviewError("patch CLI cancellation is invalid")


@final
@dataclass(frozen=True, slots=True, repr=False, init=False)
class PatchCliInvocationReviewBinding:
    """Bind one host-issued preapproval review to one pending invocation."""

    _host: PatchSdkHost
    _host_review: PatchSdkInvocationReview
    _request_id: PatchRequestId
    _correlation_id: PatchObserverCorrelationId
    _review: bytes
    _review_digest: bytes
    _profile_claimed: bool

    def __init__(self, issuer: Never) -> None:
        """Reject construction outside the trusted host preparation path."""
        del issuer
        raise PatchCliReviewError("patch CLI review binding is host-issued")

    def __repr__(self) -> str:
        """Render an opaque host-review binding marker."""
        return "PatchCliInvocationReviewBinding(<opaque>)"

    def __copy__(self) -> NoReturn:
        """Reject copies that could detach host review identity."""
        raise PatchCliReviewError("patch CLI review binding cannot be copied")

    def __deepcopy__(self, memo: dict[int, object]) -> NoReturn:
        """Reject deep copies that could detach host review identity."""
        del memo
        raise PatchCliReviewError("patch CLI review binding cannot be copied")

    def __reduce__(self) -> NoReturn:
        """Reject serializing a host-issued review binding."""
        raise PatchCliReviewError(
            "patch CLI review binding cannot be serialized"
        )

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        """Reject protocol-specific review-binding serialization."""
        del protocol
        raise PatchCliReviewError(
            "patch CLI review binding cannot be serialized"
        )


@final
@dataclass(frozen=True, slots=True, repr=False, init=False)
class PatchCliReviewContinuation:
    """Bind one single-owner detached terminal wait to one pending envelope."""

    _profile_marker: object
    _binding_digest: bytes
    _pending: PatchPending
    _state: _ContinuationState
    _reconciliation_generation: int

    def __init__(self, issuer: Never) -> None:
        """Reject direct construction outside the local profile factory."""
        del issuer
        raise PatchCliReviewError("patch CLI continuation is factory-issued")

    def __repr__(self) -> str:
        """Render an opaque continuation marker without pending identifiers."""
        return "PatchCliReviewContinuation(<opaque>)"

    def __copy__(self) -> NoReturn:
        """Reject copies that could duplicate a pending settlement owner."""
        raise PatchCliReviewError("patch CLI continuation cannot be copied")

    def __deepcopy__(self, memo: dict[int, object]) -> NoReturn:
        """Reject deep copies that could duplicate a pending owner."""
        del memo
        raise PatchCliReviewError("patch CLI continuation cannot be copied")

    def __reduce__(self) -> NoReturn:
        """Reject serializing a live pending invocation binding."""
        raise PatchCliReviewError(
            "patch CLI continuation cannot be serialized"
        )

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        """Reject protocol-specific continuation serialization."""
        del protocol
        raise PatchCliReviewError(
            "patch CLI continuation cannot be serialized"
        )


@final
@dataclass(frozen=True, slots=True, repr=False, init=False)
class ExactPatchCliPreauthorization:
    """Bind one one-shot headless approval to an exact local review binding."""

    _profile_marker: object
    _binding_digest: bytes
    _consumed: bool

    def __init__(self, issuer: Never) -> None:
        """Reject direct construction outside the test-profile factory."""
        del issuer
        raise PatchCliReviewError(
            "patch CLI preauthorization is factory-issued"
        )

    def __repr__(self) -> str:
        """Render an opaque preauthorization marker without review content."""
        return "ExactPatchCliPreauthorization(<opaque>)"

    def __copy__(self) -> NoReturn:
        """Reject copies that could replay a headless approval."""
        raise PatchCliReviewError(
            "patch CLI preauthorization cannot be copied"
        )

    def __deepcopy__(self, memo: dict[int, object]) -> NoReturn:
        """Reject deep copies that could replay a headless approval."""
        del memo
        raise PatchCliReviewError(
            "patch CLI preauthorization cannot be copied"
        )

    def __reduce__(self) -> NoReturn:
        """Reject serializing a headless approval authority."""
        raise PatchCliReviewError(
            "patch CLI preauthorization cannot be serialized"
        )

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        """Reject protocol-specific preauthorization serialization."""
        del protocol
        raise PatchCliReviewError(
            "patch CLI preauthorization cannot be serialized"
        )


@final
@dataclass(frozen=True, slots=True, repr=False, init=False)
class DetachedPatchCliApproval:
    """Bind one one-shot detached approval to an exact review binding."""

    _profile_marker: object
    _binding_digest: bytes
    _consumed: bool

    def __init__(self, issuer: Never) -> None:
        """Reject direct construction outside the test-profile factory."""
        del issuer
        raise PatchCliReviewError("detached patch approval is factory-issued")

    def __repr__(self) -> str:
        """Render an opaque detached approval marker without review content."""
        return "DetachedPatchCliApproval(<opaque>)"

    def __copy__(self) -> NoReturn:
        """Reject copies that could replay detached approval."""
        raise PatchCliReviewError("detached patch approval cannot be copied")

    def __deepcopy__(self, memo: dict[int, object]) -> NoReturn:
        """Reject deep copies that could replay detached approval."""
        del memo
        raise PatchCliReviewError("detached patch approval cannot be copied")

    def __reduce__(self) -> NoReturn:
        """Reject serializing detached approval authority."""
        raise PatchCliReviewError(
            "detached patch approval cannot be serialized"
        )

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        """Reject protocol-specific detached-approval serialization."""
        del protocol
        raise PatchCliReviewError(
            "detached patch approval cannot be serialized"
        )


@final
@dataclass(frozen=True, slots=True, repr=False, init=False)
class LocalPatchReviewTestProfile:
    """Keep the closed unregistered local interactive review test profile."""

    _marker: object
    _binding: PatchCliInvocationReviewBinding
    _view_key: bytes
    _view_nonce: bytes
    _view_ciphertext: bytes
    _run_lock: Lock
    _owner_lock: Lock
    _approval_claimed: bool
    _continuation: PatchCliReviewContinuation | None
    _consumed: bool
    name: str

    def __init__(self, issuer: Never) -> None:
        """Reject direct construction outside the local profile factory."""
        del issuer
        raise PatchCliReviewError("patch CLI profile is factory-issued")

    def __repr__(self) -> str:
        """Render a profile marker without host or complete-review content."""
        return "LocalPatchReviewTestProfile(<opaque>)"

    def __copy__(self) -> NoReturn:
        """Reject copies that could detach review authority from its host."""
        raise PatchCliReviewError("patch CLI profile cannot be copied")

    def __deepcopy__(self, memo: dict[int, object]) -> NoReturn:
        """Reject deep copies that detach review authority from its host."""
        del memo
        raise PatchCliReviewError("patch CLI profile cannot be copied")

    def __reduce__(self) -> NoReturn:
        """Reject serializing the trusted review display authority."""
        raise PatchCliReviewError("patch CLI profile cannot be serialized")

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        """Reject protocol-specific profile serialization."""
        del protocol
        raise PatchCliReviewError("patch CLI profile cannot be serialized")


async def prepare_local_patch_review_binding(
    host: PatchSdkHost,
) -> PatchCliInvocationReviewBinding:
    """Prepare one exact host-issued local preapproval review binding.

    Args:
        host: Exact authenticated SDK host holding one pending invocation.

    Returns:
        Opaque binding for that host, handle, request, correlation, and review.
    """
    if type(host) is not PatchSdkHost:
        raise PatchCliReviewError("patch CLI host is invalid")
    try:
        host_review = await host.prepare_approval_review()
    except PatchToolError as error:
        raise PatchCliReviewError("patch CLI review is unavailable") from error
    if type(host_review) is not PatchSdkInvocationReview:
        raise PatchCliReviewError("patch CLI review is unavailable")
    binding = object.__new__(PatchCliInvocationReviewBinding)
    object.__setattr__(binding, "_host", host)
    object.__setattr__(binding, "_host_review", host_review)
    object.__setattr__(binding, "_request_id", host_review._pending.request_id)
    object.__setattr__(
        binding, "_correlation_id", host_review._pending.correlation_id
    )
    object.__setattr__(binding, "_review", host_review._review)
    object.__setattr__(binding, "_review_digest", host_review._review_digest)
    object.__setattr__(binding, "_profile_claimed", False)
    _require_binding(binding)
    return binding


def create_local_patch_review_test_profile(
    binding: PatchCliInvocationReviewBinding,
) -> LocalPatchReviewTestProfile:
    """Create the only dormant local profile from one host review binding.

    Args:
        binding: Exact opaque host-issued pending invocation review binding.

    Returns:
        Unregistered local profile sealed to that exact host review binding.
    """
    _require_binding(binding)
    if binding._profile_claimed:
        raise PatchCliReviewError(
            "patch CLI review binding is already claimed"
        )
    key = token_bytes(_PREAPPROVAL_VIEW_KEY_BYTES)
    nonce = token_bytes(_PREAPPROVAL_VIEW_NONCE_BYTES)
    try:
        ciphertext = AESGCM(key).encrypt(
            nonce,
            binding._host_review._review,
            _view_aad(binding),
        )
    except ValueError as error:
        raise PatchCliReviewError("patch CLI review is unavailable") from error
    object.__setattr__(binding, "_profile_claimed", True)
    profile = object.__new__(LocalPatchReviewTestProfile)
    object.__setattr__(profile, "_marker", object())
    object.__setattr__(profile, "_binding", binding)
    object.__setattr__(profile, "_view_key", key)
    object.__setattr__(profile, "_view_nonce", nonce)
    object.__setattr__(profile, "_view_ciphertext", ciphertext)
    object.__setattr__(profile, "_run_lock", Lock())
    object.__setattr__(profile, "_owner_lock", Lock())
    object.__setattr__(profile, "_approval_claimed", False)
    object.__setattr__(profile, "_continuation", None)
    object.__setattr__(profile, "_consumed", False)
    object.__setattr__(profile, "name", LOCAL_PATCH_REVIEW_TEST_PROFILE)
    return profile


def create_exact_patch_cli_preauthorization(
    profile: LocalPatchReviewTestProfile,
) -> ExactPatchCliPreauthorization:
    """Issue one exact headless preauthorization for a local test profile.

    Args:
        profile: The exact authenticated local review profile to bind.

    Returns:
        An opaque exact preauthorization for that one profile and review.
    """
    _require_profile(profile)
    authorization = object.__new__(ExactPatchCliPreauthorization)
    object.__setattr__(authorization, "_profile_marker", profile._marker)
    object.__setattr__(
        authorization, "_binding_digest", profile._binding._review_digest
    )
    object.__setattr__(authorization, "_consumed", False)
    return authorization


def create_detached_patch_cli_approval(
    profile: LocalPatchReviewTestProfile,
) -> DetachedPatchCliApproval:
    """Issue one exact detached approval for a local test profile.

    Args:
        profile: The exact authenticated local review profile to bind.

    Returns:
        An opaque detached approval for that one profile and review.
    """
    _require_profile(profile)
    approval = object.__new__(DetachedPatchCliApproval)
    object.__setattr__(approval, "_profile_marker", profile._marker)
    object.__setattr__(
        approval, "_binding_digest", profile._binding._review_digest
    )
    object.__setattr__(approval, "_consumed", False)
    return approval


async def run_local_patch_review(
    profile: LocalPatchReviewTestProfile,
    *,
    preauthorization: ExactPatchCliPreauthorization | None = None,
    detached_approval: DetachedPatchCliApproval | None = None,
    input_stream: TextIO | None = None,
    output_stream: TextIO | None = None,
    error_stream: TextIO | None = None,
) -> PatchCliReviewResult:
    """Render and resolve the current authenticated local patch review.

    Args:
        profile: Exact local profile holding an invoked authenticated SDK host.
        preauthorization: Optional exact headless approval for this review.
        detached_approval: Optional exact detached approval for this review.
        input_stream: Attached terminal input for fixed action tokens only.
        output_stream: Attached terminal output for structured review only.
        error_stream: Attached terminal error stream for session validation.

    Returns:
        A terminal result, a durable pending continuation, or cancellation.
    """
    _require_profile(profile)
    active_input = sys.stdin if input_stream is None else input_stream
    active_output = sys.stdout if output_stream is None else output_stream
    active_error = sys.stderr if error_stream is None else error_stream
    if profile._run_lock.locked() or profile._consumed:
        raise PatchCliReviewError("patch CLI review is already consumed")
    async with profile._run_lock:
        session = _attached_terminal_session(
            active_input, active_output, active_error
        )
        if session is None:
            if not _has_exact_headless_authority(
                profile, preauthorization, detached_approval
            ):
                raise PatchCliReviewError(
                    "patch CLI review requires an attached terminal"
                )
            _consume_headless_authority(
                profile, preauthorization, detached_approval
            )
            return await _approve(profile, active_output, attached=False)
        try:
            with _TerminalStateGuard(
                session, active_input, active_output, active_error
            ) as terminal:
                terminal.require_current()
                _render_complete_review(
                    profile,
                    terminal.output_stream,
                    require_current=terminal.require_current,
                )
                action = await _read_action(
                    terminal.input_stream,
                    terminal.output_stream,
                    require_current=terminal.require_current,
                )
                terminal.require_current()
                if action is None:
                    object.__setattr__(profile, "_consumed", True)
                    await _cancel(profile)
                    return PatchCliReviewResult(PatchCliReviewState.CANCELLED)
                if action == "approve":
                    return await _approve(
                        profile,
                        terminal.output_stream,
                        attached=True,
                        require_current=terminal.require_current,
                    )
                if action == "deny":
                    object.__setattr__(profile, "_consumed", True)
                    await _cancel(profile)
                    _write(terminal.output_stream, "Patch review denied.\n")
                    return PatchCliReviewResult(PatchCliReviewState.DENIED)
                if action == "cancel":
                    object.__setattr__(profile, "_consumed", True)
                    await _cancel(profile)
                    _write(terminal.output_stream, "Patch review cancelled.\n")
                    return PatchCliReviewResult(PatchCliReviewState.CANCELLED)
                object.__setattr__(profile, "_consumed", True)
                _write(
                    terminal.output_stream, "Patch review action rejected.\n"
                )
                return PatchCliReviewResult(PatchCliReviewState.CANCELLED)
        except (CancelledError, KeyboardInterrupt):
            object.__setattr__(profile, "_consumed", True)
            await _cancel(profile)
            return PatchCliReviewResult(PatchCliReviewState.CANCELLED)


async def resume_local_patch_review(
    profile: LocalPatchReviewTestProfile,
    continuation: PatchCliReviewContinuation,
    *,
    output_stream: TextIO | None = None,
    error_stream: TextIO | None = None,
) -> PatchCliReviewResult:
    """Await one detached local pending request without issuing another effect.

    Args:
        profile: Exact profile that minted the detached pending continuation.
        continuation: The exact durable pending envelope to await once.
        output_stream: Attached terminal output for bounded state messages.
        error_stream: Attached terminal error stream for session validation.

    Returns:
        The exact terminal result for the original durable invocation.
    """
    _require_profile(profile)
    await _claim_continuation(profile, continuation)
    try:
        active_output = sys.stdout if output_stream is None else output_stream
        active_error = sys.stderr if error_stream is None else error_stream
        session = _output_terminal_session(active_output, active_error)
        if session is not None:
            settlement_attempted = False
            try:
                with _TerminalStateGuard(
                    session, None, active_output, active_error
                ) as terminal:
                    terminal.require_current()
                    _write(
                        terminal.output_stream,
                        "Patch settlement remains pending.\n",
                    )
                    settlement_attempted = True
                    result = await _await_and_close(profile, continuation)
                    terminal.require_current()
                    _write_terminal_result(terminal.output_stream, result)
                    return PatchCliReviewResult(
                        PatchCliReviewState.TERMINAL, result
                    )
            except PatchCliReviewError:
                if settlement_attempted:
                    raise
                await _await_and_close(profile, continuation)
                raise PatchCliReviewError(
                    "patch CLI terminal was lost after settlement"
                ) from None
        result = await _await_and_close(profile, continuation)
        return PatchCliReviewResult(PatchCliReviewState.TERMINAL, result)
    except BaseException:
        reconciliation = _schedule_failed_resume_reconciliation(
            profile, continuation
        )
        if reconciliation is not None:
            try:
                await shield(reconciliation)
            except CancelledError:
                # The detached finalizer remains independently scheduled.
                # It will close or rearm before any later owner can claim.
                pass
        raise


async def read_local_patch_review_result(
    profile: LocalPatchReviewTestProfile,
    *,
    output_stream: TextIO | None = None,
    error_stream: TextIO | None = None,
) -> PatchCliReviewResult:
    """Read one existing local invocation without initiating another effect.

    Args:
        profile: Exact profile holding the original authenticated invocation.
        output_stream: Optional terminal for bounded current-state output.
        error_stream: Attached terminal error stream for session validation.

    Returns:
        The existing terminal result or a bound pending continuation.
    """
    _require_profile(profile)
    active_output = sys.stdout if output_stream is None else output_stream
    active_error = sys.stderr if error_stream is None else error_stream
    session = _output_terminal_session(active_output, active_error)
    if session is None:
        return await _read_local_patch_review_outcome(profile)
    with _TerminalStateGuard(
        session, None, active_output, active_error
    ) as terminal:
        result = await _read_local_patch_review_outcome(
            profile, terminal=terminal
        )
        terminal.require_current()
        return result


async def _read_local_patch_review_outcome(
    profile: LocalPatchReviewTestProfile,
    *,
    terminal: "_TerminalStateGuard | None" = None,
) -> PatchCliReviewResult:
    """Validate and project one current exact invocation observation."""
    try:
        outcome = await profile._binding._host.inspect()
    except PatchToolError as error:
        raise PatchCliReviewError("patch CLI result is unavailable") from error
    _require_profile(profile)
    if type(outcome) is PatchResult:
        if terminal is not None:
            terminal.require_current()
            await terminal.write(_terminal_result_text(outcome))
        return PatchCliReviewResult(PatchCliReviewState.TERMINAL, outcome)
    if type(outcome) is PatchPending:
        if terminal is not None:
            terminal.require_current()
            await terminal.write("Patch settlement remains pending.\n")
        return PatchCliReviewResult(
            PatchCliReviewState.PENDING,
            continuation=await _continuation_for(profile, outcome),
        )
    raise PatchCliReviewError("patch CLI result is invalid")


async def _await_terminal(
    profile: LocalPatchReviewTestProfile,
    continuation: PatchCliReviewContinuation,
) -> PatchResult:
    """Await the original pending envelope without changing its invocation."""
    try:
        return await profile._binding._host.await_terminal(
            continuation._pending
        )
    except PatchToolError as error:
        raise PatchCliReviewError(
            "patch CLI terminal result is unavailable"
        ) from error


async def _await_and_close(
    profile: LocalPatchReviewTestProfile,
    continuation: PatchCliReviewContinuation,
) -> PatchResult:
    """Await exactly one active continuation and close terminal truth."""
    result = await _await_terminal(profile, continuation)
    await _close_continuation(profile, continuation)
    return result


async def _approve(
    profile: LocalPatchReviewTestProfile,
    output_stream: TextIO,
    *,
    attached: bool,
    require_current: Callable[[], None] | None = None,
) -> PatchCliReviewResult:
    """Approve only the current exact host-bound plan and observe its state."""
    async with profile._owner_lock:
        _require_profile(profile)
        if profile._approval_claimed or profile._consumed:
            raise PatchCliReviewError("patch CLI review is already consumed")
        _require_binding(profile._binding, require_pending=True)
        object.__setattr__(profile, "_approval_claimed", True)
        object.__setattr__(profile, "_consumed", True)
    if require_current is not None:
        require_current()
    try:
        outcome = await profile._binding._host.approve_review(
            profile._binding._host_review
        )
    except PatchToolError as error:
        raise PatchCliReviewError(
            "patch CLI approval is unavailable"
        ) from error
    if type(outcome) is PatchResult:
        if attached:
            _write_terminal_result(output_stream, outcome)
        return PatchCliReviewResult(PatchCliReviewState.TERMINAL, outcome)
    if type(outcome) is not PatchPending:
        raise PatchCliReviewError("patch CLI approval outcome is invalid")
    continuation = await _continuation_for(profile, outcome)
    if attached:
        _write(output_stream, "Patch settlement remains pending.\n")
    return PatchCliReviewResult(
        PatchCliReviewState.PENDING, continuation=continuation
    )


async def _cancel(profile: LocalPatchReviewTestProfile) -> None:
    """Preserve the existing pending request without initiating an effect."""
    try:
        outcome = await profile._binding._host.cancel()
    except PatchToolError as error:
        raise PatchCliReviewError(
            "patch CLI cancellation is unavailable"
        ) from error
    if type(outcome) is not PatchPending:
        raise PatchCliReviewError("patch CLI cancellation outcome is invalid")


async def _continuation_for(
    profile: LocalPatchReviewTestProfile,
    pending: PatchPending,
) -> PatchCliReviewContinuation:
    """Return the one ready continuation for the current pending call."""
    async with profile._owner_lock:
        _require_profile(profile)
        if (
            type(pending) is not PatchPending
            or pending != profile._binding._host_review._pending
        ):
            raise PatchCliReviewError("patch CLI continuation is invalid")
        continuation = profile._continuation
        if continuation is not None:
            _require_continuation(profile, continuation)
            return continuation
        continuation = object.__new__(PatchCliReviewContinuation)
        object.__setattr__(continuation, "_profile_marker", profile._marker)
        object.__setattr__(
            continuation, "_binding_digest", profile._binding._review_digest
        )
        object.__setattr__(continuation, "_pending", pending)
        object.__setattr__(continuation, "_state", _ContinuationState.READY)
        object.__setattr__(continuation, "_reconciliation_generation", 0)
        object.__setattr__(profile, "_continuation", continuation)
        return continuation


def _require_profile(profile: LocalPatchReviewTestProfile) -> None:
    """Reject a forged local review profile before host operation."""
    if (
        type(profile) is not LocalPatchReviewTestProfile
        or type(profile._binding) is not PatchCliInvocationReviewBinding
        or type(profile._view_key) is not bytes
        or len(profile._view_key) != _PREAPPROVAL_VIEW_KEY_BYTES
        or type(profile._view_nonce) is not bytes
        or len(profile._view_nonce) != _PREAPPROVAL_VIEW_NONCE_BYTES
        or type(profile._view_ciphertext) is not bytes
        or type(profile._run_lock) is not Lock
        or type(profile._owner_lock) is not Lock
        or type(profile._approval_claimed) is not bool
        or (
            profile._continuation is not None
            and type(profile._continuation) is not PatchCliReviewContinuation
        )
        or type(profile._consumed) is not bool
        or profile.name != LOCAL_PATCH_REVIEW_TEST_PROFILE
    ):
        raise PatchCliReviewError("patch CLI profile is invalid")
    try:
        _require_binding(
            profile._binding,
            require_pending=not profile._consumed,
        )
        _decrypt_bound_view(profile)
    except (InvalidTag, ReviewDisplayError, ValueError) as error:
        raise PatchCliReviewError("patch CLI review is unavailable") from error


def _require_binding(
    binding: PatchCliInvocationReviewBinding,
    *,
    require_pending: bool = True,
) -> None:
    """Reject a binding not issued for the exact live host invocation."""
    if (
        type(binding) is not PatchCliInvocationReviewBinding
        or type(binding._host) is not PatchSdkHost
        or type(binding._host_review) is not PatchSdkInvocationReview
        or type(binding._request_id) is not PatchRequestId
        or type(binding._correlation_id) is not PatchObserverCorrelationId
        or type(binding._review) is not bytes
        or type(binding._review_digest) is not bytes
        or not compare_digest(binding._review, binding._host_review._review)
        or binding._review_digest != binding._host_review._review_digest
        or not compare_digest(
            binding._review_digest, sha256(binding._review).digest()
        )
        or binding._request_id != binding._host_review._pending.request_id
        or binding._correlation_id
        != binding._host_review._pending.correlation_id
        or type(binding._profile_claimed) is not bool
    ):
        raise PatchCliReviewError("patch CLI review binding is invalid")
    if require_pending:
        try:
            binding._host.validate_approval_review(binding._host_review)
        except PatchToolError as error:
            raise PatchCliReviewError(
                "patch CLI review binding is invalid"
            ) from error
        return
    try:
        binding._host.validate_invocation_review(binding._host_review)
    except PatchToolError as error:
        raise PatchCliReviewError(
            "patch CLI review binding is invalid"
        ) from error


def _require_continuation(
    profile: LocalPatchReviewTestProfile,
    continuation: PatchCliReviewContinuation,
    *,
    require_ready: bool = True,
) -> None:
    """Reject a mismatched continuation before any terminal settlement wait."""
    if (
        type(continuation) is not PatchCliReviewContinuation
        or continuation._profile_marker is not profile._marker
        or continuation._binding_digest != profile._binding._review_digest
        or type(continuation._pending) is not PatchPending
        or continuation._pending != profile._binding._host_review._pending
        or continuation is not profile._continuation
        or type(continuation._state) is not _ContinuationState
        or type(continuation._reconciliation_generation) is not int
        or continuation._reconciliation_generation < 0
        or (
            require_ready
            and continuation._state is not _ContinuationState.READY
        )
    ):
        raise PatchCliReviewError("patch CLI continuation is invalid")


async def _claim_continuation(
    profile: LocalPatchReviewTestProfile,
    continuation: PatchCliReviewContinuation,
) -> None:
    """Claim the one current continuation before any terminal await."""
    async with profile._owner_lock:
        _require_profile(profile)
        _require_continuation(profile, continuation)
        object.__setattr__(continuation, "_state", _ContinuationState.ACTIVE)
        object.__setattr__(profile, "_consumed", True)


async def _close_continuation(
    profile: LocalPatchReviewTestProfile,
    continuation: PatchCliReviewContinuation,
) -> None:
    """Close one active continuation after verified terminal settlement."""
    async with profile._owner_lock:
        _require_profile(profile)
        _require_continuation(
            profile,
            continuation,
            require_ready=False,
        )
        if continuation._state is not _ContinuationState.ACTIVE:
            raise PatchCliReviewError("patch CLI continuation is invalid")
        object.__setattr__(continuation, "_state", _ContinuationState.CLOSED)


async def _reconcile_failed_resume(
    profile: LocalPatchReviewTestProfile,
    continuation: PatchCliReviewContinuation,
    generation: int | None = None,
) -> None:
    """Finish one scheduled failed-wait reconciliation without propagating."""
    active_generation = (
        _mark_continuation_reconciling(profile, continuation)
        if generation is None
        else generation
    )
    if active_generation is None:
        return
    host: PatchSdkHost | None = None
    pending: PatchPending | None = None
    try:
        host = profile._binding._host
        pending = continuation._pending
        outcome = await wait_for(
            host.inspect(), timeout=_RECONCILIATION_TIMEOUT_SECONDS
        )
    except BaseException:
        outcome = None
    try:
        async with profile._owner_lock:
            if (
                type(continuation) is not PatchCliReviewContinuation
                or continuation is not profile._continuation
                or continuation._state is not _ContinuationState.RECONCILING
                or continuation._reconciliation_generation != active_generation
            ):
                return
            try:
                _require_profile(profile)
                _require_continuation(
                    profile,
                    continuation,
                    require_ready=False,
                )
                exact_pending = (
                    host is not None
                    and pending is not None
                    and profile._binding._host is host
                    and continuation._pending == pending
                    and type(outcome) is PatchPending
                    and outcome == pending
                )
            except BaseException:
                exact_pending = False
            object.__setattr__(
                continuation,
                "_state",
                (
                    _ContinuationState.READY
                    if exact_pending
                    else _ContinuationState.CLOSED
                ),
            )
    except BaseException:
        _force_close_reconciling_continuation(
            profile, continuation, active_generation
        )


def _schedule_failed_resume_reconciliation(
    profile: LocalPatchReviewTestProfile,
    continuation: PatchCliReviewContinuation,
) -> Future[None] | None:
    """Atomically schedule the one finalizer for an abnormal active wait."""
    generation = _mark_continuation_reconciling(profile, continuation)
    if generation is None:
        return None
    reconciliation = _reconcile_failed_resume(
        profile, continuation, generation
    )
    try:
        task = create_task(reconciliation)
    except BaseException:
        try:
            reconciliation.close()
        except BaseException:
            pass
        _force_close_reconciling_continuation(
            profile, continuation, generation
        )
        return None
    try:
        if task.done():
            _reconciliation_done_callback(
                task, profile, continuation, generation
            )
        else:
            task.add_done_callback(
                lambda completed: _reconciliation_done_callback(
                    completed, profile, continuation, generation
                )
            )
    except BaseException:
        try:
            task.cancel()
        except BaseException:
            pass
        _force_close_reconciling_continuation(
            profile, continuation, generation
        )
        return None
    return task


def _mark_continuation_reconciling(
    profile: LocalPatchReviewTestProfile,
    continuation: PatchCliReviewContinuation,
) -> int | None:
    """Move one active owner into the nonclaimable finalization state."""
    if (
        type(continuation) is not PatchCliReviewContinuation
        or continuation is not profile._continuation
        or continuation._state is not _ContinuationState.ACTIVE
    ):
        return None
    generation = continuation._reconciliation_generation + 1
    object.__setattr__(continuation, "_reconciliation_generation", generation)
    object.__setattr__(continuation, "_state", _ContinuationState.RECONCILING)
    return generation


def _force_close_reconciling_continuation(
    profile: LocalPatchReviewTestProfile,
    continuation: PatchCliReviewContinuation,
    generation: int,
) -> None:
    """Close only the exact still-reconciling finalization generation."""
    if (
        type(continuation) is PatchCliReviewContinuation
        and continuation is profile._continuation
        and continuation._state is _ContinuationState.RECONCILING
        and continuation._reconciliation_generation == generation
    ):
        object.__setattr__(continuation, "_state", _ContinuationState.CLOSED)


def _reconciliation_done_callback(
    task: Future[None],
    profile: LocalPatchReviewTestProfile,
    continuation: PatchCliReviewContinuation,
    generation: int,
) -> None:
    """Retrieve finalizer completion and close only its stale active state."""
    try:
        task.result()
    except BaseException:
        pass
    try:
        _force_close_reconciling_continuation(
            profile, continuation, generation
        )
    except BaseException:
        pass


def _has_exact_headless_authority(
    profile: LocalPatchReviewTestProfile,
    preauthorization: ExactPatchCliPreauthorization | None,
    detached_approval: DetachedPatchCliApproval | None,
) -> bool:
    """Return whether exact opaque headless authority exists."""
    return _matches_preauthorization(
        profile, preauthorization
    ) or _matches_detached(profile, detached_approval)


def _matches_preauthorization(
    profile: LocalPatchReviewTestProfile,
    preauthorization: ExactPatchCliPreauthorization | None,
) -> bool:
    """Return whether one exact factory-issued preauthorization matches."""
    return (
        type(preauthorization) is ExactPatchCliPreauthorization
        and preauthorization._profile_marker is profile._marker
        and preauthorization._binding_digest == profile._binding._review_digest
        and not preauthorization._consumed
    )


def _matches_detached(
    profile: LocalPatchReviewTestProfile,
    detached_approval: DetachedPatchCliApproval | None,
) -> bool:
    """Return whether one exact factory-issued detached approval matches."""
    return (
        type(detached_approval) is DetachedPatchCliApproval
        and detached_approval._profile_marker is profile._marker
        and detached_approval._binding_digest
        == profile._binding._review_digest
        and not detached_approval._consumed
    )


def _consume_headless_authority(
    profile: LocalPatchReviewTestProfile,
    preauthorization: ExactPatchCliPreauthorization | None,
    detached_approval: DetachedPatchCliApproval | None,
) -> None:
    """Consume the one exact headless authority before the approval effect."""
    if _matches_preauthorization(profile, preauthorization):
        assert preauthorization is not None
        object.__setattr__(preauthorization, "_consumed", True)
        return
    if _matches_detached(profile, detached_approval):
        assert detached_approval is not None
        object.__setattr__(detached_approval, "_consumed", True)
        return
    raise PatchCliReviewError("patch CLI headless authority is invalid")


def _render_complete_review(
    profile: LocalPatchReviewTestProfile,
    output_stream: TextIO,
    *,
    require_current: Callable[[], None] | None = None,
) -> None:
    """Render every exact authorized review page directly to the terminal."""
    try:
        review = _decrypt_bound_view(profile)
        canonical = review.decode("utf-8")
        loads(canonical)
        safe = safe_review_text(canonical)
        pages = _review_pages(safe)
        pending = profile._binding._host_review._pending
        for index, page in enumerate(pages, start=1):
            if require_current is not None:
                require_current()
            _write(
                output_stream,
                "Privileged patch preapproval review\n"
                "Trusted host invocation binding:\n"
                f"  Request: {pending.request_id.value}\n"
                f"  Observer correlation: {pending.correlation_id.value}\n"
                "  Plan review digest: "
                f"{profile._binding._review_digest.hex()}\n"
                "Trusted host complete plan review:\n"
                f"  Complete review page {index}/{len(pages)} "
                "(safe page limit "
                f"{_PREAPPROVAL_PAGE_CHARACTERS} characters):\n"
                f"  {page}\n"
                "Trusted reviewer action:\n"
                "  Use exact approve, deny, or cancel.\n",
            )
    except (
        InvalidTag,
        JSONDecodeError,
        UnicodeDecodeError,
        ReviewDisplayError,
        ValueError,
    ) as error:
        raise PatchCliReviewError(
            "patch CLI review renderer failed"
        ) from error


def _decrypt_bound_view(profile: LocalPatchReviewTestProfile) -> bytes:
    """Decrypt only the exact canonical host review sealed in the binding."""
    review = AESGCM(profile._view_key).decrypt(
        profile._view_nonce,
        profile._view_ciphertext,
        _view_aad(profile._binding),
    )
    if (
        not compare_digest(review, profile._binding._review)
        or not compare_digest(review, profile._binding._host_review._review)
        or not compare_digest(
            sha256(review).digest(), profile._binding._review_digest
        )
        or not compare_digest(
            sha256(review).digest(),
            profile._binding._host_review._review_digest,
        )
    ):
        raise ValueError("preapproval view is not host-bound")
    return review


def _review_pages(value: str) -> tuple[str, ...]:
    """Split one complete host-owned review into fixed direct-only pages."""
    if not value:
        return ("",)
    return tuple(
        value[index : index + _PREAPPROVAL_PAGE_CHARACTERS]
        for index in range(0, len(value), _PREAPPROVAL_PAGE_CHARACTERS)
    )


def _view_aad(binding: PatchCliInvocationReviewBinding) -> bytes:
    """Return exact host request/correlation/review binding for the view."""
    return (
        "patch_cli_preapproval\0"
        f"{binding._request_id.value}\0"
        f"{binding._correlation_id.value}\0"
        f"{binding._review_digest.hex()}"
    ).encode("ascii")


async def _read_action(
    input_stream: TextIO,
    output_stream: TextIO,
    *,
    require_current: Callable[[], None] | None = None,
) -> str | None:
    """Read one fixed reviewer action without retaining terminal input."""
    if require_current is not None:
        require_current()
    _write(output_stream, "Review action [approve|deny|cancel]: ")
    try:
        descriptor = input_stream.fileno()
    except (OSError, ValueError) as error:
        raise PatchCliReviewError("patch CLI terminal input failed") from error
    loop = get_running_loop()
    received: Future[str] = loop.create_future()

    def receive() -> None:
        """Read exactly one terminal line when the descriptor becomes ready."""
        loop.remove_reader(descriptor)
        try:
            if require_current is not None:
                require_current()
            value = input_stream.readline()
        except (OSError, PatchCliReviewError) as error:
            received.set_exception(error)
            return
        received.set_result(value)

    loop.add_reader(descriptor, receive)
    try:
        value = await received
    except (OSError, PatchCliReviewError) as error:
        raise PatchCliReviewError("patch CLI terminal input failed") from error
    finally:
        loop.remove_reader(descriptor)
    if value == "":
        return None
    return value.rstrip("\r\n")


def _write_terminal_result(output_stream: TextIO, result: PatchResult) -> None:
    """Write only bounded terminal truth after the SDK host settles."""
    _write(output_stream, _terminal_result_text(result))


def _terminal_result_text(result: PatchResult) -> str:
    """Return one bounded terminal result for direct terminal output."""
    return (
        "Patch terminal result: "
        f"status={result.status.value}; "
        f"mutation_state={result.truth.mutation_state.value}.\n"
    )


def _write(output_stream: TextIO, value: str) -> None:
    """Write one directly rendered review value without generic logging."""
    try:
        written = output_stream.write(value)
        if written != len(value):
            raise PatchCliReviewError("patch CLI terminal output failed")
        output_stream.flush()
    except (OSError, ValueError) as error:
        raise PatchCliReviewError(
            "patch CLI terminal output failed"
        ) from error


def _is_output_terminal(output_stream: TextIO) -> bool:
    """Return whether a bounded post-settlement result may be displayed."""
    try:
        return output_stream.isatty()
    except OSError:
        return False


def _is_attached_terminal(input_stream: TextIO, output_stream: TextIO) -> bool:
    """Return whether two legacy test controls are both terminal attached."""
    try:
        return input_stream.isatty() and output_stream.isatty()
    except OSError:
        return False


@final
@dataclass(frozen=True, slots=True)
class _TerminalIdentity:
    """Store one descriptor's exact terminal identity at session capture."""

    descriptor: int
    device: int
    inode: int
    raw_device: int
    mode: int
    terminal_name: str


@final
@dataclass(frozen=True, slots=True)
class _TerminalSession:
    """Store the exact foreground terminal identity and input state."""

    device: int
    inode: int
    foreground_process_group: int
    attributes: _TerminalAttributes
    identities: tuple[_TerminalIdentity, ...]


def _attached_terminal_session(
    input_stream: TextIO,
    output_stream: TextIO,
    error_stream: TextIO,
) -> _TerminalSession | None:
    """Return one same-device foreground terminal session or reject a swap."""
    streams = (input_stream, output_stream, error_stream)
    try:
        attached = tuple(stream.isatty() for stream in streams)
    except OSError as error:
        raise PatchCliReviewError(
            "patch CLI terminal is unavailable"
        ) from error
    if not any(attached):
        return None
    if not all(attached):
        raise PatchCliReviewError("patch CLI terminals do not match")
    return _terminal_session(streams)


def _output_terminal_session(
    output_stream: TextIO,
    error_stream: TextIO,
) -> _TerminalSession | None:
    """Return one same-device foreground output session for resumption."""
    try:
        attached = (output_stream.isatty(), error_stream.isatty())
    except OSError as error:
        raise PatchCliReviewError(
            "patch CLI terminal is unavailable"
        ) from error
    if not any(attached):
        return None
    if not all(attached):
        raise PatchCliReviewError("patch CLI terminals do not match")
    return _terminal_session((output_stream, error_stream))


def _terminal_session(streams: tuple[TextIO, ...]) -> _TerminalSession:
    """Snapshot one exact foreground terminal after same-device validation."""
    try:
        descriptors = tuple(stream.fileno() for stream in streams)
        identities = tuple(
            _terminal_identity(descriptor) for descriptor in descriptors
        )
        if any(
            identity.device != identities[0].device
            or identity.inode != identities[0].inode
            or identity.raw_device != identities[0].raw_device
            or identity.mode != identities[0].mode
            or identity.terminal_name != identities[0].terminal_name
            for identity in identities[1:]
        ):
            raise PatchCliReviewError("patch CLI terminals do not match")
        foreground = tcgetpgrp(descriptors[0])
        if foreground != getpgrp():
            raise PatchCliReviewError("patch CLI is not foreground")
        attributes = tcgetattr(descriptors[0])
    except (OSError, ValueError) as error:
        raise PatchCliReviewError(
            "patch CLI terminal is unavailable"
        ) from error
    return _TerminalSession(
        identities[0].device,
        identities[0].inode,
        foreground,
        attributes,
        identities,
    )


@final
@dataclass(frozen=True, slots=True)
class _TerminalHandle:
    """Keep one original terminal descriptor and its stable duplicate."""

    original_descriptor: int
    duplicate_descriptor: int
    identity: _TerminalIdentity


@final
class _TerminalStateGuard:
    """Pin, validate, and restore one same-device terminal session."""

    def __init__(
        self,
        session: _TerminalSession,
        input_stream: TextIO | None,
        output_stream: TextIO,
        error_stream: TextIO | None = None,
    ) -> None:
        """Capture all terminal streams before direct privileged review."""
        self._session = session
        self._input_stream = input_stream
        self._output_stream = output_stream
        self._error_stream = (
            output_stream if error_stream is None else error_stream
        )
        self._input_handle: _TerminalHandle | None = None
        self._output_handle: _TerminalHandle | None = None
        self._error_handle: _TerminalHandle | None = None
        self._input_duplicate_stream: TextIO | None = None
        self._output_duplicate_stream: TextIO | None = None
        self._output_blocking: bool | None = None
        self._entered = False
        self._closed = False

    @property
    def input_stream(self) -> TextIO:
        """Return the stable duplicated input stream after validation."""
        if self._input_duplicate_stream is None:
            raise PatchCliReviewError(
                "patch CLI terminal input is unavailable"
            )
        return self._input_duplicate_stream

    @property
    def output_stream(self) -> TextIO:
        """Return the stable duplicated output stream after validation."""
        if self._output_duplicate_stream is None:
            raise PatchCliReviewError(
                "patch CLI terminal output is unavailable"
            )
        return self._output_duplicate_stream

    def __enter__(self) -> "_TerminalStateGuard":
        """Duplicate attached handles and enter the alternate review screen."""
        try:
            identities = self._session_identities()
            self._require_original_streams_current(identities)
            input_identity = (
                None if self._input_stream is None else identities[0]
            )
            output_identity = identities[-2]
            error_identity = identities[-1]
            if self._input_stream is None:
                self._input_handle = None
            else:
                assert input_identity is not None
                self._input_handle = _duplicate_terminal_handle(
                    self._input_stream, input_identity
                )
            self._output_handle = _duplicate_terminal_handle(
                self._output_stream, output_identity
            )
            self._error_handle = _duplicate_terminal_handle(
                self._error_stream, error_identity
            )
            self._input_duplicate_stream = (
                None
                if self._input_handle is None
                else fdopen(
                    self._input_handle.duplicate_descriptor,
                    "r",
                    encoding="utf-8",
                    closefd=False,
                )
            )
            self._output_duplicate_stream = fdopen(
                self._output_handle.duplicate_descriptor,
                "w",
                encoding="utf-8",
                closefd=False,
            )
            self._output_blocking = get_blocking(
                self._output_handle.duplicate_descriptor
            )
            set_blocking(self._output_handle.duplicate_descriptor, False)
            self.require_current()
            self._entered = True
            self._write_escape("\x1b[?1049h\x1b[?25l")
            return self
        except (OSError, PatchCliReviewError, ValueError) as error:
            try:
                self._restore()
            except PatchCliReviewError:
                pass
            if type(error) is PatchCliReviewError:
                raise error
            raise PatchCliReviewError(
                "patch CLI terminal is unavailable"
            ) from error

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: object | None,
    ) -> Literal[False]:
        """Restore terminal state and close duplicates for every outcome."""
        del exception_type, exception, traceback
        self._restore()
        return False

    def require_current(self) -> None:
        """Require every captured terminal descriptor to remain exact."""
        self._require_original_streams_current(self._session_identities())
        handles = tuple(
            handle
            for handle in (
                self._input_handle,
                self._output_handle,
                self._error_handle,
            )
            if handle is not None
        )
        if not handles:
            raise PatchCliReviewError("patch CLI terminal is unavailable")
        try:
            for handle in handles:
                _require_terminal_handle(handle, handle.original_descriptor)
                _require_terminal_handle(handle, handle.duplicate_descriptor)
            control = self._control_handle()
            foreground = tcgetpgrp(control.original_descriptor)
            if (
                foreground != self._session.foreground_process_group
                or foreground != getpgrp()
                or tcgetpgrp(control.duplicate_descriptor) != foreground
            ):
                raise PatchCliReviewError(
                    "patch CLI terminal is not foreground"
                )
        except (OSError, ValueError) as error:
            raise PatchCliReviewError(
                "patch CLI terminal is unavailable"
            ) from error

    async def write(self, value: str) -> None:
        """Write bounded UTF-8 bytes without blocking the event loop."""
        if self._output_handle is None:
            raise PatchCliReviewError(
                "patch CLI terminal output is unavailable"
            )
        data = value.encode("utf-8")
        offset = 0
        while offset < len(data):
            self._require_stable_output()
            try:
                written = os_write(
                    self._output_handle.duplicate_descriptor, data[offset:]
                )
            except BlockingIOError:
                await self._wait_until_output_writable()
                continue
            except OSError as error:
                raise PatchCliReviewError(
                    "patch CLI terminal output failed"
                ) from error
            if written <= 0:
                raise PatchCliReviewError("patch CLI terminal output failed")
            offset += written
        self._require_stable_output()

    async def _wait_until_output_writable(self) -> None:
        """Yield until the stable nonblocking output descriptor is writable."""
        if self._output_handle is None:
            raise PatchCliReviewError(
                "patch CLI terminal output is unavailable"
            )
        descriptor = self._output_handle.duplicate_descriptor
        loop = get_running_loop()
        writable: Future[None] = loop.create_future()

        def ready() -> None:
            """Resolve one registered nonblocking output readiness wait."""
            loop.remove_writer(descriptor)
            if not writable.done():
                writable.set_result(None)

        registered = False
        try:
            loop.add_writer(descriptor, ready)
            registered = True
            await writable
        except (OSError, RuntimeError, ValueError) as error:
            raise PatchCliReviewError(
                "patch CLI terminal output failed"
            ) from error
        finally:
            if registered:
                loop.remove_writer(descriptor)

    def _restore(self) -> None:
        """Restore through stable duplicates, never swapped streams."""
        failure: PatchCliReviewError | None = None
        try:
            self.require_current()
        except PatchCliReviewError as error:
            failure = error
        control = self._control_handle_or_none()
        if failure is None and control is not None:
            try:
                self._require_stable_control(control)
                tcsetattr(
                    control.duplicate_descriptor,
                    TCSANOW,
                    self._session.attributes,
                )
                restored = tcgetattr(control.duplicate_descriptor)
                _require_terminal_handle(control, control.duplicate_descriptor)
                if restored != self._session.attributes:
                    raise PatchCliReviewError(
                        "patch CLI terminal restore failed"
                    )
            except (OSError, ValueError) as error:
                failure = PatchCliReviewError(
                    "patch CLI terminal restore failed"
                )
                failure.__cause__ = error
            except PatchCliReviewError as error:
                failure = error
        if failure is None and self._entered:
            try:
                self._write_escape("\x1b[0m\x1b[?25h\x1b[?1049l")
            except PatchCliReviewError as error:
                failure = error
            finally:
                self._entered = False
        elif self._entered:
            self._entered = False
        self._close_duplicates(restore_blocking=failure is None)
        if failure is not None:
            raise failure

    def _control_handle(self) -> _TerminalHandle:
        """Return input control or the stable output control."""
        control = self._control_handle_or_none()
        if control is None:
            raise PatchCliReviewError("patch CLI terminal is unavailable")
        return control

    def _control_handle_or_none(self) -> _TerminalHandle | None:
        """Return the active input or output terminal control handle."""
        return (
            self._output_handle
            if self._input_handle is None
            else self._input_handle
        )

    def _session_identities(self) -> tuple[_TerminalIdentity, ...]:
        """Return the exact role-ordered identities captured before entry."""
        expected_count = 2 if self._input_stream is None else 3
        identities = self._session.identities
        if (
            len(identities) != expected_count
            or any(
                not _is_terminal_identity(identity) for identity in identities
            )
            or any(
                identity.device != self._session.device
                or identity.inode != self._session.inode
                for identity in identities
            )
        ):
            raise PatchCliReviewError("patch CLI terminal changed")
        return identities

    def _require_original_streams_current(
        self,
        identities: tuple[_TerminalIdentity, ...],
    ) -> None:
        """Require every original role descriptor before any guard action."""
        streams: tuple[TextIO, ...] = (
            (self._output_stream, self._error_stream)
            if self._input_stream is None
            else (
                self._input_stream,
                self._output_stream,
                self._error_stream,
            )
        )
        try:
            for stream, identity in zip(streams, identities, strict=True):
                _require_terminal_identity(
                    identity,
                    stream.fileno(),
                    require_descriptor=True,
                )
        except (OSError, ValueError) as error:
            raise PatchCliReviewError(
                "patch CLI terminal is unavailable"
            ) from error

    def _close_duplicates(self, *, restore_blocking: bool = True) -> None:
        """Close duplicates without restoring flags to an invalid handle."""
        if self._closed:
            return
        self._closed = True
        if self._output_duplicate_stream is not None:
            try:
                self._output_duplicate_stream.close()
            except OSError:
                pass
            self._output_duplicate_stream = None
        if self._input_duplicate_stream is not None:
            try:
                self._input_duplicate_stream.close()
            except OSError:
                pass
            self._input_duplicate_stream = None
        if (
            restore_blocking
            and self._output_handle is not None
            and self._output_blocking is not None
        ):
            try:
                set_blocking(
                    self._output_handle.duplicate_descriptor,
                    self._output_blocking,
                )
            except OSError:
                pass
        for handle in (
            self._input_handle,
            self._output_handle,
            self._error_handle,
        ):
            if handle is None:
                continue
            try:
                close(handle.duplicate_descriptor)
            except OSError:
                pass

    def _write_escape(self, value: str) -> None:
        """Write bounded terminal control bytes through stable output only."""
        if self._output_handle is None:
            raise PatchCliReviewError(
                "patch CLI terminal output is unavailable"
            )
        data = value.encode("ascii")
        try:
            self._require_stable_output()
            written = os_write(self._output_handle.duplicate_descriptor, data)
            if written != len(data):
                raise PatchCliReviewError("patch CLI terminal is unavailable")
        except (BlockingIOError, OSError) as error:
            raise PatchCliReviewError(
                "patch CLI terminal is unavailable"
            ) from error

    def _require_stable_control(self, control: _TerminalHandle) -> None:
        """Require all captured handles before retained-control restore."""
        self.require_current()
        _require_terminal_handle(control, control.duplicate_descriptor)
        foreground = tcgetpgrp(control.duplicate_descriptor)
        if (
            foreground != self._session.foreground_process_group
            or foreground != getpgrp()
        ):
            raise PatchCliReviewError("patch CLI terminal is not foreground")

    def _require_stable_output(self) -> None:
        """Require all captured handles before terminal escape output."""
        if self._output_handle is None:
            raise PatchCliReviewError(
                "patch CLI terminal output is unavailable"
            )
        try:
            self.require_current()
        except PatchCliReviewError as error:
            if not isinstance(error.__cause__, (OSError, ValueError)):
                raise
            raise PatchCliReviewError(
                "patch CLI terminal output failed"
            ) from error
        _require_terminal_handle(
            self._output_handle,
            self._output_handle.duplicate_descriptor,
        )
        self._require_stable_control(self._control_handle())


def _duplicate_terminal_handle(
    stream: TextIO,
    identity: _TerminalIdentity,
) -> _TerminalHandle:
    """Duplicate one exact snapshot-matching terminal descriptor."""
    descriptor = stream.fileno()
    _require_terminal_identity(identity, descriptor, require_descriptor=True)
    duplicate_descriptor = dup(descriptor)
    try:
        _require_terminal_identity(
            identity, duplicate_descriptor, require_descriptor=False
        )
    except (OSError, PatchCliReviewError, ValueError):
        close(duplicate_descriptor)
        raise
    return _TerminalHandle(
        descriptor,
        duplicate_descriptor,
        identity,
    )


def _require_terminal_handle(
    handle: _TerminalHandle,
    descriptor: int,
) -> None:
    """Require one original or duplicate descriptor to retain its identity."""
    _require_terminal_identity(
        handle.identity,
        descriptor,
        require_descriptor=descriptor == handle.original_descriptor,
    )


def _terminal_identity(descriptor: int) -> _TerminalIdentity:
    """Capture every stable identity component for one terminal descriptor."""
    current = fstat(descriptor)
    name = ttyname(descriptor)
    if not isatty(descriptor):
        raise PatchCliReviewError("patch CLI terminal is unavailable")
    return _TerminalIdentity(
        descriptor,
        current.st_dev,
        current.st_ino,
        current.st_rdev,
        current.st_mode,
        name,
    )


def _require_terminal_identity(
    identity: _TerminalIdentity,
    descriptor: int,
    *,
    require_descriptor: bool,
) -> None:
    """Require a descriptor to retain the exact pre-guard session identity."""
    current = _terminal_identity(descriptor)
    if (
        (require_descriptor and descriptor != identity.descriptor)
        or current.device != identity.device
        or current.inode != identity.inode
        or current.raw_device != identity.raw_device
        or current.mode != identity.mode
        or current.terminal_name != identity.terminal_name
    ):
        raise PatchCliReviewError("patch CLI terminal changed")


def _is_terminal_identity(value: object) -> bool:
    """Return whether one session identity retains strict primitive fields."""
    return (
        type(value) is _TerminalIdentity
        and type(value.descriptor) is int
        and type(value.device) is int
        and type(value.inode) is int
        and type(value.raw_device) is int
        and type(value.mode) is int
        and type(value.terminal_name) is str
    )

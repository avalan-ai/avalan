"""Cover exact local patch-review binding and one-shot control boundaries."""

from asyncio import (
    CancelledError,
    Event,
    Future,
    create_task,
    gather,
    get_running_loop,
    run,
    sleep,
    wait_for,
)
from contextlib import contextmanager
from copy import copy, deepcopy
from io import StringIO
from os import (
    close,
    dup,
    dup2,
    environ,
    fdopen,
    fstat,
    get_blocking,
    pipe,
    read,
    write,
)
from pathlib import Path
from pickle import dumps
from pty import openpty
from runpy import run_path
from select import select
from sys import argv
from termios import tcgetattr
from typing import Any, Callable, Iterator, Literal, Never, TextIO, cast
from unittest.mock import AsyncMock, patch

import pytest
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from avalan.cli import patch_review
from avalan.cli.patch_review import (
    DetachedPatchCliApproval,
    ExactPatchCliPreauthorization,
    LocalPatchReviewTestProfile,
    PatchCliInvocationReviewBinding,
    PatchCliReviewContinuation,
    PatchCliReviewError,
    PatchCliReviewResult,
    PatchCliReviewState,
    _attached_terminal_session,
    _is_attached_terminal,
    _is_output_terminal,
    _output_terminal_session,
    _read_action,
    _require_binding,
    _require_continuation,
    _require_profile,
    _review_pages,
    _TerminalStateGuard,
    _write,
    create_detached_patch_cli_approval,
    create_exact_patch_cli_preauthorization,
    create_local_patch_review_test_profile,
    prepare_local_patch_review_binding,
    read_local_patch_review_result,
    resume_local_patch_review,
    run_local_patch_review,
)
from avalan.patch.domain import (
    LifecyclePhase,
    PatchPending,
    PatchPendingOperationId,
)
from avalan.patch.toolset import (
    PatchSdkHost,
    PatchSdkInvocationReview,
    PatchToolError,
)


def _profile(
    *, pending_after_approval: bool = False
) -> tuple[Any, LocalPatchReviewTestProfile]:
    """Return one invoked exact test profile from the PTY fixture."""
    fixture = run_path("tests/cli/patch_review_pty_e2e_test.py")
    service, profile = run(
        fixture["_profile"](pending_after_approval=pending_after_approval)
    )
    assert type(profile) is LocalPatchReviewTestProfile
    return service, profile


class _TestTerminalGuard:
    """Provide stable direct streams for a narrowly mocked terminal guard."""

    def __init__(self, exit_error: PatchCliReviewError | None = None) -> None:
        """Create isolated stable input and output streams."""
        self.input_stream = StringIO()
        self.output_stream = StringIO()
        self._exit_error = exit_error

    def __enter__(self) -> "_TestTerminalGuard":
        """Enter the inert test-only terminal guard."""
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: object | None,
    ) -> Literal[False]:
        """Exit without suppressing the tested outcome."""
        del exception_type, exception, traceback
        if self._exit_error is not None:
            raise self._exit_error
        return False

    def require_current(self) -> None:
        """Keep the mocked terminal identity current for action tests."""


def test_patch_cli_binding_is_exact_opaque_and_single_profile_owner() -> None:
    """Reject unrelated, copied, replayed, and forged local review bindings."""
    _service, profile = _profile()
    assert type(profile) is LocalPatchReviewTestProfile
    binding = profile._binding
    assert type(binding) is PatchCliInvocationReviewBinding
    assert repr(binding) == "PatchCliInvocationReviewBinding(<opaque>)"
    for value in (binding, profile):
        with pytest.raises(PatchCliReviewError):
            copy(value)
        with pytest.raises(PatchCliReviewError):
            deepcopy(value)
        with pytest.raises(PatchCliReviewError):
            dumps(value)
        with pytest.raises(PatchCliReviewError):
            value.__reduce__()
    with pytest.raises(PatchCliReviewError, match="already claimed"):
        create_local_patch_review_test_profile(binding)
    invalid_binding = object.__new__(PatchCliInvocationReviewBinding)
    object.__setattr__(invalid_binding, "_host", object())
    with pytest.raises(PatchCliReviewError):
        create_local_patch_review_test_profile(invalid_binding)
    with pytest.raises(PatchCliReviewError):
        _require_binding(invalid_binding)
    object.__setattr__(profile, "_view_key", b"")
    with pytest.raises(PatchCliReviewError):
        _require_profile(profile)
    with pytest.raises(PatchCliReviewError):
        patch_review.PatchCliReviewResult(PatchCliReviewState.TERMINAL)
    with pytest.raises(PatchCliReviewError):
        patch_review.PatchCliReviewResult(
            PatchCliReviewState.PENDING,
            continuation=None,
        )
    with pytest.raises(PatchCliReviewError):
        patch_review.PatchCliReviewResult(
            PatchCliReviewState.DENIED,
            continuation=object.__new__(
                patch_review.PatchCliReviewContinuation
            ),
        )


def test_patch_cli_opaque_controls_reject_every_copy_and_constructor() -> None:
    """Keep all factory-only lifecycle controls nonconstructible and opaque."""
    _service, profile = _profile(pending_after_approval=True)
    pending = run(read_local_patch_review_result(profile))
    assert pending.continuation is not None
    values = (
        pending.continuation,
        create_exact_patch_cli_preauthorization(profile),
        create_detached_patch_cli_approval(profile),
    )
    for value in values:
        assert repr(value).endswith("(<opaque>)")
        with pytest.raises(PatchCliReviewError):
            copy(value)
        with pytest.raises(PatchCliReviewError):
            deepcopy(value)
        with pytest.raises(PatchCliReviewError):
            value.__reduce__()
        with pytest.raises(PatchCliReviewError):
            dumps(value)
    for factory_type in (
        ExactPatchCliPreauthorization,
        DetachedPatchCliApproval,
    ):
        with pytest.raises(PatchCliReviewError):
            factory_type(cast(Never, None))
    with pytest.raises(PatchCliReviewError):
        patch_review.PatchCliReviewContinuation(cast(Never, None))
    with pytest.raises(PatchCliReviewError):
        LocalPatchReviewTestProfile(cast(Never, None))
    with pytest.raises(PatchCliReviewError):
        PatchCliInvocationReviewBinding(cast(Never, None))


def test_patch_sdk_review_binding_stays_exact_across_settlement() -> None:
    """Keep exact host, handle, and digest identity across settlement."""
    service, profile = _profile()
    host = profile._binding._host
    review = profile._binding._host_review
    with pytest.raises(PatchToolError, match="host-issued"):
        PatchSdkInvocationReview(cast(Never, None))
    assert type(review) is PatchSdkInvocationReview
    assert repr(review) == "PatchSdkInvocationReview(<opaque>)"
    host.validate_approval_review(review)
    host.validate_invocation_review(review)
    for operation in (copy, deepcopy, dumps):
        with pytest.raises(PatchToolError):
            operation(review)
    with pytest.raises(PatchToolError, match="cannot be serialized"):
        review.__reduce__()
    with patch.object(
        service, "review", new=AsyncMock(return_value={"bad": set()})
    ):
        with pytest.raises(PatchToolError, match="review is invalid"):
            run(host.prepare_approval_review())
    run(
        run_local_patch_review(
            profile,
            preauthorization=create_exact_patch_cli_preauthorization(profile),
        )
    )
    with pytest.raises(PatchToolError):
        host.validate_approval_review(review)
    host.validate_invocation_review(review)
    with pytest.raises(PatchToolError, match="review is not pending"):
        run(host.prepare_approval_review())


def test_patch_cli_preauthorization_and_detached_approval_are_one_shot() -> (
    None
):
    """Consume exact headless approval before the host approval transition."""
    _service, profile = _profile()
    authorization = create_exact_patch_cli_preauthorization(profile)
    detached = create_detached_patch_cli_approval(profile)
    for value in (authorization, detached):
        assert repr(value).endswith("(<opaque>)")
        with pytest.raises(PatchCliReviewError):
            copy(value)
        with pytest.raises(PatchCliReviewError):
            deepcopy(value)
        with pytest.raises(PatchCliReviewError):
            dumps(value)
    result = run(
        run_local_patch_review(profile, preauthorization=authorization)
    )
    assert result.state is PatchCliReviewState.TERMINAL
    with pytest.raises(PatchCliReviewError, match="already consumed"):
        run(run_local_patch_review(profile, preauthorization=authorization))

    _service, profile = _profile()
    detached = create_detached_patch_cli_approval(profile)
    result = run(run_local_patch_review(profile, detached_approval=detached))
    assert result.state is PatchCliReviewState.TERMINAL
    with pytest.raises(PatchCliReviewError, match="already consumed"):
        run(run_local_patch_review(profile, detached_approval=detached))


def test_patch_cli_rejects_cross_profile_authority_and_host_substitution() -> (
    None
):
    """Reject a valid-but-different profile, host, and review authority."""
    first_service, first_profile = _profile()
    second_service, second_profile = _profile()
    authority = create_exact_patch_cli_preauthorization(first_profile)
    with pytest.raises(PatchCliReviewError, match="attached terminal"):
        run(run_local_patch_review(second_profile, preauthorization=authority))
    assert first_service.approvals == 0
    assert second_service.approvals == 0
    object.__setattr__(
        first_profile._binding, "_host", second_profile._binding._host
    )
    with pytest.raises(PatchCliReviewError, match="binding is invalid"):
        _require_binding(first_profile._binding)


def test_patch_cli_concurrent_headless_replay_approves_at_most_once() -> None:
    """Fence simultaneous replay before a second host approval call."""
    service, profile = _profile()
    authorization = create_exact_patch_cli_preauthorization(profile)

    async def exercise() -> tuple[object, object]:
        """Run the same one-shot authority twice without a second effect."""
        return await gather(
            run_local_patch_review(profile, preauthorization=authorization),
            run_local_patch_review(profile, preauthorization=authorization),
            return_exceptions=True,
        )

    first, second = run(exercise())
    assert {type(first), type(second)} == {
        PatchCliReviewError,
        patch_review.PatchCliReviewResult,
    }
    assert getattr(service, "approvals", 0) == 1


def test_patch_cli_direct_approval_has_one_atomic_owner() -> None:
    """Claim direct approval before concurrent or stale replay reaches host."""
    service, profile = _profile()

    async def exercise() -> tuple[object, object]:
        """Hold one host approval open while a direct replay races it."""
        original = profile._binding._host.approve_review
        started = Event()
        release = Event()
        calls = 0

        async def delayed(review: PatchSdkInvocationReview) -> object:
            """Expose the otherwise direct approval ownership race."""
            nonlocal calls
            calls += 1
            started.set()
            await release.wait()
            return await original(review)

        with patch.object(
            profile._binding._host,
            "approve_review",
            new=delayed,
        ):
            owner = create_task(
                patch_review._approve(profile, StringIO(), attached=False)
            )
            await started.wait()
            duplicate = await gather(
                patch_review._approve(profile, StringIO(), attached=False),
                return_exceptions=True,
            )
            release.set()
            result = await owner
        assert calls == 1
        return result, duplicate[0]

    owner, duplicate = run(exercise())
    assert type(owner) is patch_review.PatchCliReviewResult
    assert type(duplicate) is PatchCliReviewError
    assert getattr(service, "approvals", 0) == 1
    with pytest.raises(PatchCliReviewError, match="already consumed"):
        run(patch_review._approve(profile, StringIO(), attached=False))
    assert getattr(service, "approvals", 0) == 1


def test_patch_cli_pending_continuation_has_one_safe_resume_owner() -> None:
    """Claim a detached pending continuation before the only terminal await."""
    _service, profile = _profile(pending_after_approval=True)
    pending = run(
        run_local_patch_review(
            profile,
            detached_approval=create_detached_patch_cli_approval(profile),
        )
    )
    assert pending.state is PatchCliReviewState.PENDING
    assert pending.continuation is not None
    with pytest.raises(PatchCliReviewError):
        copy(pending.continuation)
    terminal = run(resume_local_patch_review(profile, pending.continuation))
    assert terminal.state is PatchCliReviewState.TERMINAL
    with pytest.raises(PatchCliReviewError, match="continuation is invalid"):
        _require_continuation(profile, pending.continuation)
    with pytest.raises(PatchCliReviewError, match="continuation is invalid"):
        run(resume_local_patch_review(profile, pending.continuation))


def test_patch_cli_continuation_has_one_read_and_settlement_owner() -> None:
    """Reuse one continuation and await its pending call at most once."""
    service, profile = _profile(pending_after_approval=True)

    async def exercise() -> tuple[
        PatchCliReviewResult,
        object,
        object,
        PatchCliReviewContinuation,
    ]:
        """Race repeated reads and resumes against one held settlement wait."""
        first, second = await gather(
            read_local_patch_review_result(profile),
            read_local_patch_review_result(profile),
        )
        assert first.continuation is not None
        assert first.continuation is second.continuation
        continuation = first.continuation
        original = profile._binding._host.await_terminal
        started = Event()
        release = Event()
        calls = 0

        async def delayed(pending: PatchPending) -> object:
            """Hold one durable wait long enough to reproduce replay."""
            nonlocal calls
            calls += 1
            started.set()
            await release.wait()
            return await original(pending)

        with patch.object(
            profile._binding._host,
            "await_terminal",
            new=delayed,
        ):
            owner = create_task(
                resume_local_patch_review(profile, continuation)
            )
            await started.wait()
            duplicate_read = await gather(
                read_local_patch_review_result(profile),
                return_exceptions=True,
            )
            duplicate_resume = await gather(
                resume_local_patch_review(profile, continuation),
                return_exceptions=True,
            )
            release.set()
            result = await owner
        assert calls == 1
        return result, duplicate_read[0], duplicate_resume[0], continuation

    owner, duplicate_read, duplicate_resume, continuation = run(exercise())
    assert type(owner) is patch_review.PatchCliReviewResult
    assert type(duplicate_read) is PatchCliReviewError
    assert type(duplicate_resume) is PatchCliReviewError
    assert getattr(service, "waits", 0) == 1
    with pytest.raises(PatchCliReviewError, match="continuation is invalid"):
        run(read_local_patch_review_result(profile))
    with pytest.raises(PatchCliReviewError, match="continuation is invalid"):
        run(resume_local_patch_review(profile, continuation))
    assert getattr(service, "waits", 0) == 1


def test_patch_cli_cancelled_wait_rearms_one_concurrent_retry_owner() -> None:
    """Rearm only a same-pending cancelled wait before one later retry."""
    service, profile = _profile(pending_after_approval=True)
    pending = run(
        run_local_patch_review(
            profile,
            detached_approval=create_detached_patch_cli_approval(profile),
        )
    )
    assert pending.continuation is not None
    continuation = pending.continuation

    async def exercise() -> tuple[object, object]:
        """Cancel once, then hold the exact retry against a duplicate owner."""
        cancelled_calls = 0

        async def cancelled_wait(value: PatchPending) -> object:
            """Cancel the first host wait while preserving its pending call."""
            nonlocal cancelled_calls
            assert value == continuation._pending
            cancelled_calls += 1
            raise CancelledError()

        with patch.object(
            profile._binding._host,
            "await_terminal",
            new=cancelled_wait,
        ):
            with pytest.raises(CancelledError):
                await resume_local_patch_review(profile, continuation)
        recovered = await read_local_patch_review_result(profile)
        assert recovered.continuation is continuation

        original = profile._binding._host.await_terminal
        started = Event()
        release = Event()
        retry_calls = 0

        async def delayed_retry(value: PatchPending) -> object:
            """Hold the rearmed wait to expose a concurrent retry attempt."""
            nonlocal retry_calls
            assert value == continuation._pending
            retry_calls += 1
            started.set()
            await release.wait()
            return await original(value)

        with patch.object(
            profile._binding._host,
            "await_terminal",
            new=delayed_retry,
        ):
            owner = create_task(
                resume_local_patch_review(profile, continuation)
            )
            await started.wait()
            duplicate = await gather(
                resume_local_patch_review(profile, continuation),
                return_exceptions=True,
            )
            release.set()
            result = await owner
        assert cancelled_calls == 1
        assert retry_calls == 1
        return result, duplicate[0]

    result, duplicate = run(exercise())
    assert type(result) is PatchCliReviewResult
    assert type(duplicate) is PatchCliReviewError
    assert getattr(service, "waits", 0) == 1
    with pytest.raises(PatchCliReviewError, match="continuation is invalid"):
        run(resume_local_patch_review(profile, continuation))
    assert getattr(service, "waits", 0) == 1


def test_patch_cli_double_cancellation_reconciles_before_later_read() -> None:
    """Finish the detached finalizer after repeated cancellation interrupts."""
    _service, profile = _profile(pending_after_approval=True)
    pending = run(
        run_local_patch_review(
            profile,
            detached_approval=create_detached_patch_cli_approval(profile),
        )
    )
    assert pending.continuation is not None
    continuation = pending.continuation

    async def exercise() -> None:
        """Interrupt the owner again while its exact pending inspect waits."""
        inspect_started = Event()
        release_inspect = Event()

        async def cancelled_wait(value: PatchPending) -> object:
            """Abort the original active terminal wait."""
            assert value == continuation._pending
            raise CancelledError()

        async def held_inspect() -> object:
            """Hold the independent finalizer until after a second cancel."""
            inspect_started.set()
            await release_inspect.wait()
            return continuation._pending

        with (
            patch.object(
                profile._binding._host,
                "await_terminal",
                new=cancelled_wait,
            ),
            patch.object(
                profile._binding._host,
                "inspect",
                new=held_inspect,
            ),
        ):
            owner = create_task(
                resume_local_patch_review(profile, continuation)
            )
            await inspect_started.wait()
            assert str(continuation._state) == "reconciling"
            duplicate = await gather(
                resume_local_patch_review(profile, continuation),
                return_exceptions=True,
            )
            assert type(duplicate[0]) is PatchCliReviewError
            owner.cancel()
            with pytest.raises(CancelledError):
                await owner
            assert str(continuation._state) == "reconciling"
            release_inspect.set()
            await sleep(0)
            await sleep(0)
        recovered = await read_local_patch_review_result(profile)
        assert recovered.continuation is continuation
        assert str(continuation._state) == "ready"

    run(exercise())


def test_patch_cli_arbitrary_finalizer_failures_close_fail_closed() -> None:
    """Close a continuation for RuntimeError and nonstandard base failures."""

    class _HostAbort(BaseException):
        """Model one host boundary failure outside ordinary exceptions."""

    async def exercise(
        profile: LocalPatchReviewTestProfile,
        error: BaseException,
    ) -> None:
        """Fail wait and inspect, then prove a later read cannot rearm."""
        pending = await run_local_patch_review(
            profile,
            detached_approval=create_detached_patch_cli_approval(profile),
        )
        assert pending.continuation is not None
        continuation = pending.continuation

        async def broken_wait(value: PatchPending) -> object:
            """Raise the exact abnormal host wait outcome."""
            assert value == continuation._pending
            raise error

        async def broken_inspect() -> object:
            """Raise an unrelated finalizer inspection failure."""
            raise error

        with (
            patch.object(
                profile._binding._host,
                "await_terminal",
                new=broken_wait,
            ),
            patch.object(
                profile._binding._host,
                "inspect",
                new=broken_inspect,
            ),
        ):
            with pytest.raises(type(error)):
                await resume_local_patch_review(profile, continuation)
        assert continuation._state.value == "closed"
        with pytest.raises(
            PatchCliReviewError, match="continuation is invalid"
        ):
            await read_local_patch_review_result(profile)
        with pytest.raises(
            PatchCliReviewError, match="continuation is invalid"
        ):
            await resume_local_patch_review(profile, continuation)

    for error in (RuntimeError("inspect failed"), _HostAbort("host abort")):
        _service, profile = _profile(pending_after_approval=True)
        run(exercise(profile, error))


def test_patch_cli_finalizer_closes_when_scheduling_or_locking_fails() -> None:
    """Close every abnormal owner when the bounded finalizer cannot proceed."""

    def active_continuation() -> (
        tuple[LocalPatchReviewTestProfile, PatchCliReviewContinuation]
    ):
        """Return one direct pending continuation in its active owner state."""
        _service, profile = _profile(pending_after_approval=True)
        pending = run(
            run_local_patch_review(
                profile,
                detached_approval=create_detached_patch_cli_approval(profile),
            )
        )
        assert pending.continuation is not None
        object.__setattr__(
            pending.continuation,
            "_state",
            patch_review._ContinuationState.ACTIVE,
        )
        return profile, pending.continuation

    def unavailable_task(coroutine: Any) -> Never:
        """Dispose the supplied finalizer before reproducing scheduler loss."""
        coroutine.close()
        raise RuntimeError("scheduler lost")

    scheduled_profile, scheduled = active_continuation()
    with patch.object(
        patch_review,
        "create_task",
        new=unavailable_task,
    ):
        assert (
            patch_review._schedule_failed_resume_reconciliation(
                scheduled_profile, scheduled
            )
            is None
        )
    assert scheduled._state.value == "closed"
    patch_review._force_close_reconciling_continuation(
        scheduled_profile,
        scheduled,
        scheduled._reconciliation_generation,
    )

    class _BrokenReconciliation:
        """Refuse disposal after a scheduler failure."""

        def close(self) -> None:
            """Raise while closing the unstarted finalizer."""
            raise RuntimeError("finalizer close lost")

    def broken_reconciliation(*arguments: Any) -> Any:
        """Return the close-failing reconciliation stand-in."""
        del arguments
        return _BrokenReconciliation()

    close_profile, close_continuation = active_continuation()
    with (
        patch.object(
            patch_review,
            "_reconcile_failed_resume",
            new=broken_reconciliation,
        ),
        patch.object(
            patch_review,
            "create_task",
            new=unavailable_task,
        ),
    ):
        assert (
            patch_review._schedule_failed_resume_reconciliation(
                close_profile, close_continuation
            )
            is None
        )
    assert close_continuation._state.value == "closed"

    class _CallbackFailure:
        """Refuse callback attachment after accepting a finalizer task."""

        def done(self) -> bool:
            """Remain pending so callback attachment is required."""
            return False

        def add_done_callback(self, callback: Any) -> None:
            """Fail before retaining the completion observer."""
            del callback
            raise RuntimeError("callback lost")

        def cancel(self) -> bool:
            """Fail after callback attachment has already been lost."""
            raise RuntimeError("cancel lost")

    def callback_failure_task(coroutine: Any) -> Any:
        """Close the unstarted finalizer before returning the fake task."""
        coroutine.close()
        return _CallbackFailure()

    callback_profile, callback_continuation = active_continuation()
    with patch.object(
        patch_review,
        "create_task",
        new=callback_failure_task,
    ):
        assert (
            patch_review._schedule_failed_resume_reconciliation(
                callback_profile, callback_continuation
            )
            is None
        )
    assert callback_continuation._state.value == "closed"

    closed_profile, closed = active_continuation()
    object.__setattr__(
        closed, "_state", patch_review._ContinuationState.CLOSED
    )
    run(patch_review._reconcile_failed_resume(closed_profile, closed))
    assert closed._state.value == "closed"

    class _BrokenLock:
        """Raise while the reconciler attempts its one locked transition."""

        async def __aenter__(self) -> None:
            """Fail before final state validation."""
            raise RuntimeError("lock lost")

        async def __aexit__(
            self,
            exception_type: type[BaseException] | None,
            exception: BaseException | None,
            traceback: object | None,
        ) -> Literal[False]:
            """Never suppress the simulated lock failure."""
            del exception_type, exception, traceback
            return False

    locked_profile, locked = active_continuation()
    object.__setattr__(locked_profile, "_owner_lock", _BrokenLock())
    run(patch_review._reconcile_failed_resume(locked_profile, locked))
    assert locked._state.value == "closed"


def test_patch_cli_reconciliation_backstop_fences_generations() -> None:
    """Close prestart cancellation without closing newer generation work."""
    prestart_service, prestart_profile = _profile(pending_after_approval=True)
    prestart_pending = run(
        run_local_patch_review(
            prestart_profile,
            detached_approval=create_detached_patch_cli_approval(
                prestart_profile
            ),
        )
    )
    assert prestart_pending.continuation is not None
    prestart = prestart_pending.continuation

    async def exercise_prestart() -> None:
        """Return an already-cancelled finalizer before its first step."""

        async def cancelled_wait(value: PatchPending) -> object:
            """Abort the active host wait that owns the continuation."""
            assert value == prestart._pending
            raise CancelledError()

        def cancelled_task(coroutine: Any) -> Future[None]:
            """Close the unstarted coroutine and return its cancelled task."""
            coroutine.close()
            task: Future[None] = Future()
            task.cancel()
            return task

        with (
            patch.object(
                prestart_profile._binding._host,
                "await_terminal",
                new=cancelled_wait,
            ),
            patch.object(
                patch_review,
                "create_task",
                new=cancelled_task,
            ),
        ):
            with pytest.raises(CancelledError):
                await resume_local_patch_review(prestart_profile, prestart)
        assert prestart._state.value == "closed"
        with pytest.raises(
            PatchCliReviewError, match="continuation is invalid"
        ):
            await read_local_patch_review_result(prestart_profile)
        assert getattr(prestart_service, "approvals", 0) == 1
        assert getattr(prestart_service, "waits", 0) == 0

    run(exercise_prestart())

    stale_service, stale_profile = _profile(pending_after_approval=True)
    stale_pending = run(
        run_local_patch_review(
            stale_profile,
            detached_approval=create_detached_patch_cli_approval(
                stale_profile
            ),
        )
    )
    assert stale_pending.continuation is not None
    stale = stale_pending.continuation

    async def exercise_stale_callback() -> None:
        """Race one callback with rearm, next generation, and close."""
        await patch_review._claim_continuation(stale_profile, stale)
        first_generation = patch_review._mark_continuation_reconciling(
            stale_profile, stale
        )
        assert first_generation is not None
        completed: Future[None] = Future()
        completed.set_result(None)
        await patch_review._reconcile_failed_resume(
            stale_profile, stale, first_generation
        )
        assert str(stale._state) == "ready"
        patch_review._reconciliation_done_callback(
            completed, stale_profile, stale, first_generation
        )
        assert str(stale._state) == "ready"

        await patch_review._claim_continuation(stale_profile, stale)
        second_generation = patch_review._mark_continuation_reconciling(
            stale_profile, stale
        )
        assert second_generation is not None
        assert second_generation > first_generation
        await patch_review._reconcile_failed_resume(
            stale_profile, stale, first_generation
        )
        assert str(stale._state) == "reconciling"
        patch_review._reconciliation_done_callback(
            completed, stale_profile, stale, first_generation
        )
        assert str(stale._state) == "reconciling"
        failed: Future[None] = Future()
        failed.set_exception(RuntimeError("finalizer destroyed"))
        patch_review._reconciliation_done_callback(
            failed, stale_profile, stale, second_generation
        )
        assert str(stale._state) == "closed"
        patch_review._reconciliation_done_callback(
            completed, stale_profile, stale, first_generation
        )
        assert str(stale._state) == "closed"
        with patch.object(
            patch_review,
            "_force_close_reconciling_continuation",
            side_effect=RuntimeError("callback close lost"),
        ):
            patch_review._reconciliation_done_callback(
                completed, stale_profile, stale, first_generation
            )
        with pytest.raises(
            PatchCliReviewError, match="continuation is invalid"
        ):
            await read_local_patch_review_result(stale_profile)

    run(exercise_stale_callback())
    assert getattr(stale_service, "approvals", 0) == 1
    assert getattr(stale_service, "waits", 0) == 0


def test_patch_cli_transient_wait_rearms_and_terminal_race_closes() -> None:
    """Rearm exact pending errors and close terminal or unknown races."""

    async def transient_wait(value: PatchPending) -> object:
        """Fail one host wait without creating terminal truth."""
        del value
        raise PatchToolError("transient")

    service, profile = _profile(pending_after_approval=True)
    pending = run(
        run_local_patch_review(
            profile,
            detached_approval=create_detached_patch_cli_approval(profile),
        )
    )
    assert pending.continuation is not None
    continuation = pending.continuation
    with patch.object(
        profile._binding._host,
        "await_terminal",
        new=transient_wait,
    ):
        with pytest.raises(PatchCliReviewError, match="terminal result"):
            run(resume_local_patch_review(profile, continuation))
    recovered = run(read_local_patch_review_result(profile))
    assert recovered.continuation is continuation
    result = run(resume_local_patch_review(profile, continuation))
    assert result.state is PatchCliReviewState.TERMINAL
    assert getattr(service, "waits", 0) == 1

    terminal_service, terminal_profile = _profile(pending_after_approval=True)
    terminal_pending = run(
        run_local_patch_review(
            terminal_profile,
            detached_approval=create_detached_patch_cli_approval(
                terminal_profile
            ),
        )
    )
    assert terminal_pending.continuation is not None
    terminal_continuation = terminal_pending.continuation
    terminal = terminal_service._base["_result"](terminal_service.request_id)
    with (
        patch.object(
            terminal_profile._binding._host,
            "await_terminal",
            new=transient_wait,
        ),
        patch.object(
            terminal_profile._binding._host,
            "inspect",
            new=AsyncMock(return_value=terminal),
        ),
    ):
        with pytest.raises(PatchCliReviewError, match="terminal result"):
            run(
                resume_local_patch_review(
                    terminal_profile,
                    terminal_continuation,
                )
            )
        observed = run(read_local_patch_review_result(terminal_profile))
    assert observed.state is PatchCliReviewState.TERMINAL
    with pytest.raises(PatchCliReviewError, match="continuation is invalid"):
        run(
            resume_local_patch_review(
                terminal_profile,
                terminal_continuation,
            )
        )
    assert getattr(terminal_service, "waits", 0) == 0

    unknown_service, unknown_profile = _profile(pending_after_approval=True)
    unknown_pending = run(
        run_local_patch_review(
            unknown_profile,
            detached_approval=create_detached_patch_cli_approval(
                unknown_profile
            ),
        )
    )
    assert unknown_pending.continuation is not None
    unknown_continuation = unknown_pending.continuation
    with (
        patch.object(
            unknown_profile._binding._host,
            "await_terminal",
            new=transient_wait,
        ),
        patch.object(
            unknown_profile._binding._host,
            "inspect",
            new=AsyncMock(side_effect=PatchToolError("unknown")),
        ),
    ):
        with pytest.raises(PatchCliReviewError, match="terminal result"):
            run(
                resume_local_patch_review(
                    unknown_profile,
                    unknown_continuation,
                )
            )
    with pytest.raises(PatchCliReviewError, match="continuation is invalid"):
        run(
            resume_local_patch_review(
                unknown_profile,
                unknown_continuation,
            )
        )
    assert getattr(unknown_service, "waits", 0) == 0

    different_service, different_profile = _profile(
        pending_after_approval=True
    )
    different_pending = run(
        run_local_patch_review(
            different_profile,
            detached_approval=create_detached_patch_cli_approval(
                different_profile
            ),
        )
    )
    assert different_pending.continuation is not None
    different_continuation = different_pending.continuation
    different_outcome = PatchPending(
        1,
        PatchPendingOperationId("pending_" + "c" * 16),
        different_continuation._pending.request_id,
        different_continuation._pending.correlation_id,
        LifecyclePhase.SETTLEMENT_PENDING,
    )
    with (
        patch.object(
            different_profile._binding._host,
            "await_terminal",
            new=transient_wait,
        ),
        patch.object(
            different_profile._binding._host,
            "inspect",
            new=AsyncMock(return_value=different_outcome),
        ),
    ):
        with pytest.raises(PatchCliReviewError, match="terminal result"):
            run(
                resume_local_patch_review(
                    different_profile,
                    different_continuation,
                )
            )
    with pytest.raises(PatchCliReviewError, match="continuation is invalid"):
        run(
            resume_local_patch_review(
                different_profile,
                different_continuation,
            )
        )
    assert getattr(different_service, "waits", 0) == 0

    close_service, close_profile = _profile(pending_after_approval=True)
    close_pending = run(
        run_local_patch_review(
            close_profile,
            detached_approval=create_detached_patch_cli_approval(
                close_profile
            ),
        )
    )
    assert close_pending.continuation is not None
    with pytest.raises(PatchCliReviewError, match="continuation is invalid"):
        run(
            patch_review._close_continuation(
                close_profile,
                close_pending.continuation,
            )
        )
    assert getattr(close_service, "waits", 0) == 0

    invalid_service, invalid_profile = _profile(pending_after_approval=True)
    invalid_pending = run(
        run_local_patch_review(
            invalid_profile,
            detached_approval=create_detached_patch_cli_approval(
                invalid_profile
            ),
        )
    )
    assert invalid_pending.continuation is not None
    object.__setattr__(
        invalid_pending.continuation,
        "_state",
        patch_review._ContinuationState.ACTIVE,
    )
    object.__setattr__(invalid_profile, "_view_ciphertext", b"invalid")
    run(
        patch_review._reconcile_failed_resume(
            invalid_profile,
            invalid_pending.continuation,
        )
    )
    assert invalid_pending.continuation._state.value == "closed"
    assert getattr(invalid_service, "waits", 0) == 0


def test_patch_cli_interactive_actions_and_detached_terminal_resume() -> None:
    """Accept only fixed interactive actions and await the original pending."""
    for action, expected in (
        (None, PatchCliReviewState.CANCELLED),
        ("deny", PatchCliReviewState.DENIED),
        ("cancel", PatchCliReviewState.CANCELLED),
        ("approve", PatchCliReviewState.TERMINAL),
        ("a", PatchCliReviewState.CANCELLED),
    ):
        service, profile = _profile()
        with (
            patch.object(
                patch_review,
                "_attached_terminal_session",
                return_value=object(),
            ),
            patch.object(
                patch_review,
                "_TerminalStateGuard",
                return_value=_TestTerminalGuard(),
            ),
            patch.object(patch_review, "_render_complete_review"),
            patch.object(
                patch_review,
                "_read_action",
                new=AsyncMock(return_value=action),
            ),
        ):
            result = run(
                run_local_patch_review(profile, output_stream=StringIO())
            )
        assert result.state is expected
        assert service.approvals == (action == "approve")

    service, profile = _profile(pending_after_approval=True)
    pending = run(
        run_local_patch_review(
            profile,
            detached_approval=create_detached_patch_cli_approval(profile),
        )
    )
    assert pending.continuation is not None
    with (
        patch.object(
            patch_review, "_output_terminal_session", return_value=object()
        ),
        patch.object(
            patch_review,
            "_TerminalStateGuard",
            return_value=_TestTerminalGuard(),
        ),
    ):
        result = run(
            resume_local_patch_review(
                profile,
                pending.continuation,
                output_stream=StringIO(),
            )
        )
    assert result.state is PatchCliReviewState.TERMINAL
    assert service.approvals == 1
    assert service.waits == 1

    service, profile = _profile(pending_after_approval=True)
    pending = run(
        run_local_patch_review(
            profile,
            detached_approval=create_detached_patch_cli_approval(profile),
        )
    )
    assert pending.continuation is not None

    with (
        patch.object(
            patch_review, "_output_terminal_session", return_value=object()
        ),
        patch.object(
            patch_review,
            "_TerminalStateGuard",
            return_value=_TestTerminalGuard(
                PatchCliReviewError("terminal lost")
            ),
        ),
    ):
        with pytest.raises(PatchCliReviewError, match="terminal lost"):
            run(
                resume_local_patch_review(
                    profile,
                    pending.continuation,
                    output_stream=StringIO(),
                )
            )
    assert service.approvals == 1
    assert service.waits == 1


def test_patch_cli_cancel_and_terminal_loss_preserve_original_pending() -> (
    None
):
    """Cancel or lose a terminal without issuing a second host effect."""
    service, profile = _profile()
    with (
        patch.object(
            patch_review, "_attached_terminal_session", return_value=object()
        ),
        patch.object(
            patch_review,
            "_TerminalStateGuard",
            return_value=_TestTerminalGuard(),
        ),
        patch.object(patch_review, "_render_complete_review"),
        patch.object(
            patch_review,
            "_read_action",
            new=AsyncMock(side_effect=CancelledError()),
        ),
    ):
        result = run(run_local_patch_review(profile, output_stream=StringIO()))
    assert result.state is PatchCliReviewState.CANCELLED
    assert service.approvals == 0

    service, profile = _profile(pending_after_approval=True)
    pending = run(
        run_local_patch_review(
            profile,
            detached_approval=create_detached_patch_cli_approval(profile),
        )
    )
    assert pending.continuation is not None

    @contextmanager
    def lost_terminal() -> Iterator[None]:
        """Fail before the resume terminal can render settlement state."""
        raise PatchCliReviewError("terminal lost")
        yield None

    with (
        patch.object(
            patch_review, "_output_terminal_session", return_value=object()
        ),
        patch.object(
            patch_review, "_TerminalStateGuard", return_value=lost_terminal()
        ),
    ):
        with pytest.raises(PatchCliReviewError, match="lost after settlement"):
            run(
                resume_local_patch_review(
                    profile,
                    pending.continuation,
                    output_stream=StringIO(),
                )
            )
    assert service.approvals == 1
    assert service.waits == 1


def test_patch_cli_preapproval_pages_are_host_bound_and_direct_only() -> None:
    """Render sealed host review pages without raw input disclosure."""
    _service, profile = _profile()
    output = StringIO()
    patch_review._render_complete_review(profile, output)
    rendered = output.getvalue()
    assert "Privileged patch preapproval review" in rendered
    assert "PATCH-CLI-PRIVILEGED-REVIEW-CANARY-7F3C2B1D" in rendered
    assert "PATCH-CLI-RAW-HOST-INPUT-CANARY-6A5A4D8E" not in rendered
    assert len(_review_pages("x" * 1025)) == 2
    object.__setattr__(profile, "_view_ciphertext", b"bad")
    with pytest.raises(PatchCliReviewError, match="renderer failed"):
        patch_review._render_complete_review(profile, StringIO())


def test_patch_cli_canaries_stay_outside_history_logs_and_errors(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Keep host input absent outside direct privileged review rendering."""
    raw_canary = "PATCH-CLI-RAW-HOST-INPUT-CANARY-6A5A4D8E"
    review_canary = "PATCH-CLI-PRIVILEGED-REVIEW-CANARY-7F3C2B1D"
    history_path = tmp_path / "shell-history"
    monkeypatch.setenv("HISTFILE", str(history_path))
    _service, profile = _profile()
    output = StringIO()
    with caplog.at_level("DEBUG"):
        patch_review._render_complete_review(profile, output)
    assert review_canary in output.getvalue()
    assert raw_canary not in output.getvalue()
    assert raw_canary not in "\0".join(argv)
    assert raw_canary not in "\0".join(environ.values())
    assert raw_canary not in caplog.text
    assert not history_path.exists()
    object.__setattr__(profile, "_view_ciphertext", b"bad")
    with pytest.raises(PatchCliReviewError) as error:
        _require_profile(profile)
    assert raw_canary not in str(error.value)


def test_patch_cli_rejects_forged_preapproval_view_before_approval() -> None:
    """Reject a valid encrypted but different canonical preapproval view."""
    service, profile = _profile()
    authorization = create_exact_patch_cli_preauthorization(profile)
    replacement = b'{"candidate":"forged","plan":"other"}'
    object.__setattr__(
        profile,
        "_view_ciphertext",
        AESGCM(profile._view_key).encrypt(
            profile._view_nonce,
            replacement,
            patch_review._view_aad(profile._binding),
        ),
    )
    with pytest.raises(PatchCliReviewError, match="review is unavailable"):
        _require_profile(profile)
    with pytest.raises(PatchCliReviewError, match="renderer failed"):
        patch_review._render_complete_review(profile, StringIO())
    with pytest.raises(PatchCliReviewError, match="review is unavailable"):
        run(run_local_patch_review(profile, preauthorization=authorization))
    with pytest.raises(PatchCliReviewError, match="review is unavailable"):
        run(patch_review._approve(profile, StringIO(), attached=False))
    assert service.approvals == 0


def test_patch_cli_rejects_every_post_prepare_view_substitution() -> None:
    """Reject key, nonce, digest, AAD, and other-review substitutions."""

    def rejected(
        mutator: Callable[[LocalPatchReviewTestProfile], None],
    ) -> None:
        """Apply one post-prepare mutation and prove it reaches no approval."""
        service, profile = _profile()
        authorization = create_exact_patch_cli_preauthorization(profile)
        mutator(profile)
        with pytest.raises(PatchCliReviewError):
            _require_profile(profile)
        with pytest.raises(PatchCliReviewError):
            patch_review._render_complete_review(profile, StringIO())
        with pytest.raises(PatchCliReviewError):
            run(
                run_local_patch_review(profile, preauthorization=authorization)
            )
        with pytest.raises(PatchCliReviewError):
            run(patch_review._approve(profile, StringIO(), attached=False))
        assert service.approvals == 0

    rejected(
        lambda profile: object.__setattr__(profile, "_view_key", b"x" * 32)
    )
    rejected(
        lambda profile: object.__setattr__(profile, "_view_nonce", b"x" * 12)
    )
    rejected(
        lambda profile: object.__setattr__(
            profile._binding, "_review_digest", b"x" * 32
        )
    )
    rejected(
        lambda profile: object.__setattr__(
            profile._binding, "_review", b'{"candidate":"forged"}'
        )
    )
    rejected(
        lambda profile: object.__setattr__(
            profile._binding._host_review,
            "_review",
            b'{"candidate":"forged"}',
        )
    )
    second_service, second_profile = _profile()
    assert second_service.approvals == 0

    def replace_with_other(profile: LocalPatchReviewTestProfile) -> None:
        """Replace all encrypted view material with another valid review."""
        object.__setattr__(profile, "_view_key", second_profile._view_key)
        object.__setattr__(profile, "_view_nonce", second_profile._view_nonce)
        object.__setattr__(
            profile, "_view_ciphertext", second_profile._view_ciphertext
        )

    rejected(replace_with_other)
    rejected(
        lambda profile: object.__setattr__(
            profile._binding,
            "_correlation_id",
            second_profile._binding._correlation_id,
        )
    )


def test_patch_cli_revalidates_before_each_page_prompt_and_read() -> None:
    """Invoke the terminal-current fence for display, prompt, and read."""
    _service, profile = _profile()
    calls = 0

    def current() -> None:
        """Record one terminal-current check without changing identity."""
        nonlocal calls
        calls += 1

    patch_review._render_complete_review(
        profile,
        StringIO(),
        require_current=current,
    )
    input_fd, output_fd = pipe()
    input_stream = fdopen(input_fd, "r", encoding="utf-8")
    try:
        write(output_fd, b"approve\n")
        assert (
            run(
                _read_action(
                    input_stream,
                    StringIO(),
                    require_current=current,
                )
            )
            == "approve"
        )
    finally:
        input_stream.close()
        close(output_fd)
    assert calls == 3

    _service, profile = _profile()
    object.__setattr__(profile, "_view_ciphertext", b"bad")
    with pytest.raises(PatchCliReviewError, match="review is unavailable"):
        _require_profile(profile)

    _service, profile = _profile()
    noncanonical = b'{"plan": "noncanonical"}'
    object.__setattr__(
        profile,
        "_view_ciphertext",
        AESGCM(profile._view_key).encrypt(
            profile._view_nonce,
            noncanonical,
            patch_review._view_aad(profile._binding),
        ),
    )
    with pytest.raises(PatchCliReviewError, match="renderer failed"):
        patch_review._render_complete_review(profile, StringIO())


def test_patch_cli_read_and_terminal_helpers_handle_bounded_output() -> None:
    """Read existing truth without an extra approval effect."""
    service, profile = _profile(pending_after_approval=True)
    output = StringIO()
    pending = run(
        read_local_patch_review_result(profile, output_stream=output)
    )
    assert pending.state is PatchCliReviewState.PENDING
    assert pending.continuation is not None
    assert service.approvals == 0
    assert _review_pages("") == ("",)

    service, profile = _profile()
    run(
        run_local_patch_review(
            profile,
            preauthorization=create_exact_patch_cli_preauthorization(profile),
        )
    )
    terminal = run(
        read_local_patch_review_result(profile, output_stream=output)
    )
    assert terminal.state is PatchCliReviewState.TERMINAL
    assert service.approvals == 1

    class _BrokenTerminal:
        """Raise when the terminal predicate is queried."""

        def isatty(self) -> bool:
            """Reject terminal introspection for the negative helper path."""
            raise OSError("lost")

    broken = cast(TextIO, _BrokenTerminal())
    assert not _is_output_terminal(broken)
    assert not _is_attached_terminal(broken, broken)
    with pytest.raises(PatchCliReviewError, match="terminal is unavailable"):
        _attached_terminal_session(broken, broken, broken)
    with pytest.raises(PatchCliReviewError, match="terminal is unavailable"):
        _output_terminal_session(broken, broken)


@pytest.mark.parametrize(
    ("pending_after_approval", "expected_state", "message"),
    (
        (
            True,
            PatchCliReviewState.PENDING,
            b"Patch settlement remains pending.",
        ),
        (
            False,
            PatchCliReviewState.TERMINAL,
            b"Patch terminal result: status=committed; ",
        ),
    ),
)
def test_patch_cli_detached_read_projects_through_real_pinned_pty(
    pending_after_approval: bool,
    expected_state: PatchCliReviewState,
    message: bytes,
) -> None:
    """Project normal pending and terminal reads through exact PTY handles."""
    _service, profile = _profile(pending_after_approval=pending_after_approval)
    if not pending_after_approval:
        run(
            run_local_patch_review(
                profile,
                preauthorization=create_exact_patch_cli_preauthorization(
                    profile
                ),
            )
        )
    master, slave = openpty()
    output_tty = fdopen(dup(slave), "w", encoding="utf-8")
    error_tty = fdopen(dup(slave), "w", encoding="utf-8")
    try:
        with (
            patch.object(patch_review, "tcgetpgrp", return_value=1),
            patch.object(patch_review, "getpgrp", return_value=1),
        ):
            observed = run(
                read_local_patch_review_result(
                    profile,
                    output_stream=output_tty,
                    error_stream=error_tty,
                )
            )
        assert observed.state is expected_state
        if expected_state is PatchCliReviewState.PENDING:
            assert observed.continuation is not None
        else:
            assert observed.result is not None
        readable, _, _ = select((master,), (), (), 1)
        assert readable == [master]
        assert message in read(master, 65536)
    finally:
        output_tty.close()
        error_tty.close()
        close(master)
        close(slave)


@pytest.mark.parametrize("swap", ("descriptor", "foreground"))
def test_patch_cli_detached_read_fences_terminal_swap_before_truth(
    swap: str,
) -> None:
    """Disclose no current truth after a real descriptor or foreground swap."""
    _service, profile = _profile(pending_after_approval=True)
    master, slave = openpty()
    other_master, other_slave = openpty()
    output_tty = fdopen(dup(slave), "w", encoding="utf-8")
    error_tty = fdopen(dup(slave), "w", encoding="utf-8")
    foreground = 1
    original_inspect = profile._binding._host.inspect

    async def swapped_inspect() -> object:
        """Swap the actual output boundary after terminal guard entry."""
        nonlocal foreground
        outcome = await original_inspect()
        if swap == "descriptor":
            dup2(other_slave, output_tty.fileno())
        else:
            foreground = 2
        return outcome

    def current_foreground(descriptor: int) -> int:
        """Return the controlled foreground state for the real PTY."""
        del descriptor
        return foreground

    try:
        with (
            patch.object(
                profile._binding._host,
                "inspect",
                new=swapped_inspect,
            ),
            patch.object(patch_review, "tcgetpgrp", new=current_foreground),
            patch.object(patch_review, "getpgrp", return_value=1),
        ):
            with pytest.raises(
                PatchCliReviewError,
                match=(
                    "terminal changed"
                    if swap == "descriptor"
                    else "not foreground"
                ),
            ):
                run(
                    read_local_patch_review_result(
                        profile,
                        output_stream=output_tty,
                        error_stream=error_tty,
                    )
                )
        terminal_output = b""
        readable, _, _ = select((master,), (), (), 0)
        if readable:
            terminal_output = read(master, 65536)
        replacement_output = b""
        readable, _, _ = select((other_master,), (), (), 0)
        if readable:
            replacement_output = read(other_master, 65536)
        for output in (terminal_output, replacement_output):
            assert b"Patch settlement remains pending." not in output
            assert b"Patch terminal result:" not in output
    finally:
        output_tty.close()
        error_tty.close()
        close(master)
        close(other_master)
        close(slave)
        close(other_slave)


def test_patch_cli_detached_read_schedules_nonblocking_pty_backpressure() -> (
    None
):
    """Yield under real PTY backpressure, then finish the bounded write."""
    _service, profile = _profile(pending_after_approval=True)
    master, slave = openpty()
    output_tty = fdopen(dup(slave), "w", encoding="utf-8")
    error_tty = fdopen(dup(slave), "w", encoding="utf-8")
    original_inspect = profile._binding._host.inspect

    async def filled_inspect() -> object:
        """Fill the guarded nonblocking PTY before returning host truth."""
        outcome = await original_inspect()
        payload = b"x" * 4096
        while True:
            try:
                write(output_tty.fileno(), payload)
            except BlockingIOError:
                return outcome

    async def exercise() -> tuple[PatchCliReviewResult, bytes]:
        """Prove another task runs while terminal output awaits readiness."""
        with (
            patch.object(
                profile._binding._host,
                "inspect",
                new=filled_inspect,
            ),
            patch.object(patch_review, "tcgetpgrp", return_value=1),
            patch.object(patch_review, "getpgrp", return_value=1),
        ):
            reader = create_task(
                read_local_patch_review_result(
                    profile,
                    output_stream=output_tty,
                    error_stream=error_tty,
                )
            )
            ticked = Event()
            get_running_loop().call_soon(ticked.set)
            await ticked.wait()
            await sleep(0)
            assert not reader.done()

            async def drain_output() -> bytes:
                """Drain actual PTY bytes while preserving event-loop turns."""
                observed = b""
                while not reader.done():
                    if select((master,), (), (), 0)[0]:
                        observed += read(master, 65536)
                    await sleep(0)
                while select((master,), (), (), 0)[0]:
                    observed += read(master, 65536)
                return observed

            drainer = create_task(drain_output())
            observed = await wait_for(reader, 2)
            return observed, await drainer

    try:
        observed, output = run(exercise())
        assert observed.state is PatchCliReviewState.PENDING
        assert observed.continuation is not None
        assert b"Patch settlement remains pending." in output
    finally:
        output_tty.close()
        error_tty.close()
        close(master)
        close(slave)


def test_patch_cli_detached_read_cancellation_ignores_queued_pty_writer() -> (
    None
):
    """Cancel before a queued real-PTY writer without disclosure or errors."""
    _service, profile = _profile(pending_after_approval=True)
    master, slave = openpty()
    output_tty = fdopen(dup(slave), "w", encoding="utf-8")
    error_tty = fdopen(dup(slave), "w", encoding="utf-8")
    original_blocking = get_blocking(output_tty.fileno())
    original_inspect = profile._binding._host.inspect
    original_wait = _TerminalStateGuard._wait_until_output_writable
    writer_registered = Event()

    async def filled_inspect() -> object:
        """Fill the guarded nonblocking PTY before returning host truth."""
        outcome = await original_inspect()
        payload = b"x" * 4096
        while True:
            try:
                write(output_tty.fileno(), payload)
            except BlockingIOError:
                try:
                    write(output_tty.fileno(), b"x")
                except BlockingIOError:
                    return outcome

    async def observe_writer_registration(
        terminal: _TerminalStateGuard,
    ) -> None:
        """Signal only after the real output readiness writer is registered."""
        loop = get_running_loop()
        original_add_writer = loop.add_writer

        def register_writer(
            descriptor: int,
            callback: Callable[[], None],
        ) -> None:
            """Register the original writer before exposing the wait point."""
            original_add_writer(descriptor, callback)
            writer_registered.set()

        with patch.object(loop, "add_writer", new=register_writer):
            await original_wait(terminal)

    async def exercise() -> tuple[bytes, list[dict[str, Any]]]:
        """Queue cancellation before writable delivery and audit the loop."""
        loop = get_running_loop()
        loop_failures: list[dict[str, Any]] = []
        previous_handler = loop.get_exception_handler()

        def record_loop_failure(
            active_loop: Any, context: dict[str, Any]
        ) -> None:
            """Retain unexpected callback failures without generic logging."""
            del active_loop
            loop_failures.append(context)

        loop.set_exception_handler(record_loop_failure)
        output = b""
        try:
            with (
                patch.object(
                    profile._binding._host,
                    "inspect",
                    new=filled_inspect,
                ),
                patch.object(
                    _TerminalStateGuard,
                    "_wait_until_output_writable",
                    new=observe_writer_registration,
                ),
                patch.object(patch_review, "tcgetpgrp", return_value=1),
                patch.object(patch_review, "getpgrp", return_value=1),
            ):
                reader = create_task(
                    read_local_patch_review_result(
                        profile,
                        output_stream=output_tty,
                        error_stream=error_tty,
                    )
                )
                await wait_for(writer_registered.wait(), 2)
                assert not reader.done()
                loop.call_soon(reader.cancel)
                while select((master,), (), (), 0)[0]:
                    output += read(master, 65536)
                with pytest.raises(CancelledError):
                    await reader
                await sleep(0)
                await sleep(0)
                while select((master,), (), (), 0)[0]:
                    output += read(master, 65536)
                assert reader.cancelled()
                assert get_blocking(output_tty.fileno()) is original_blocking
        finally:
            loop.set_exception_handler(previous_handler)
        return output, loop_failures

    try:
        output, loop_failures = run(exercise())
        assert loop_failures == []
        assert b"Patch settlement remains pending." not in output
        assert b"Patch terminal result:" not in output
    finally:
        output_tty.close()
        error_tty.close()
        close(master)
        close(slave)


def test_patch_cli_host_failures_and_bounded_existing_result_paths() -> None:
    """Translate invalid host observations without bypassing exact bindings."""
    _service, profile = _profile(pending_after_approval=True)
    with patch.object(patch_review, "_is_output_terminal", return_value=True):
        pending = run(
            read_local_patch_review_result(profile, output_stream=StringIO())
        )
    assert pending.state is PatchCliReviewState.PENDING

    _service, profile = _profile()
    terminal = run(
        run_local_patch_review(
            profile,
            preauthorization=create_exact_patch_cli_preauthorization(profile),
        )
    )
    assert terminal.result is not None
    with pytest.raises(PatchCliReviewError, match="nonterminal result"):
        patch_review.PatchCliReviewResult(
            PatchCliReviewState.DENIED,
            result=terminal.result,
        )
    with patch.object(patch_review, "_is_output_terminal", return_value=True):
        observed = run(
            read_local_patch_review_result(profile, output_stream=StringIO())
        )
    assert observed.state is PatchCliReviewState.TERMINAL

    _service, profile = _profile()
    with patch.object(
        profile._binding._host,
        "inspect",
        new=AsyncMock(side_effect=PatchToolError("unavailable")),
    ):
        with pytest.raises(PatchCliReviewError, match="result is unavailable"):
            run(read_local_patch_review_result(profile))
    with patch.object(
        profile._binding._host,
        "inspect",
        new=AsyncMock(return_value=object()),
    ):
        with pytest.raises(PatchCliReviewError, match="result is invalid"):
            run(read_local_patch_review_result(profile))
    with pytest.raises(PatchCliReviewError, match="continuation is invalid"):
        run(
            patch_review._continuation_for(
                profile,
                cast(PatchPending, object()),
            )
        )
    with patch.object(
        profile._binding._host,
        "approve_review",
        new=AsyncMock(side_effect=PatchToolError("unavailable")),
    ):
        with pytest.raises(
            PatchCliReviewError, match="approval is unavailable"
        ):
            run(patch_review._approve(profile, StringIO(), attached=False))
    _service, profile = _profile()
    with patch.object(
        profile._binding._host,
        "approve_review",
        new=AsyncMock(return_value=object()),
    ):
        with pytest.raises(PatchCliReviewError, match="approval outcome"):
            run(patch_review._approve(profile, StringIO(), attached=False))
    with patch.object(
        profile._binding._host,
        "cancel",
        new=AsyncMock(side_effect=PatchToolError("unavailable")),
    ):
        with pytest.raises(
            PatchCliReviewError, match="cancellation is unavailable"
        ):
            run(patch_review._cancel(profile))
    with patch.object(
        profile._binding._host,
        "cancel",
        new=AsyncMock(return_value=object()),
    ):
        with pytest.raises(PatchCliReviewError, match="cancellation outcome"):
            run(patch_review._cancel(profile))

    with patch.object(
        profile._binding._host,
        "prepare_approval_review",
        new=AsyncMock(side_effect=PatchToolError("unavailable")),
    ):
        with pytest.raises(PatchCliReviewError, match="review is unavailable"):
            run(prepare_local_patch_review_binding(profile._binding._host))
    with patch.object(
        profile._binding._host,
        "prepare_approval_review",
        new=AsyncMock(return_value=object()),
    ):
        with pytest.raises(PatchCliReviewError, match="review is unavailable"):
            run(prepare_local_patch_review_binding(profile._binding._host))

    _service, profile = _profile()
    object.__setattr__(profile._binding, "_profile_claimed", False)
    with patch("avalan.cli.patch_review.AESGCM") as cipher:
        cipher.return_value.encrypt.side_effect = ValueError("invalid")
        with pytest.raises(PatchCliReviewError, match="review is unavailable"):
            create_local_patch_review_test_profile(profile._binding)
    assert repr(profile) == "LocalPatchReviewTestProfile(<opaque>)"

    _service, profile = _profile()

    async def locked_run() -> None:
        """Reject a reentrant review before it spends authority."""
        await profile._run_lock.acquire()
        try:
            with pytest.raises(PatchCliReviewError, match="already consumed"):
                await run_local_patch_review(profile)
        finally:
            profile._run_lock.release()

    run(locked_run())

    _service, profile = _profile(pending_after_approval=True)
    pending = run(
        run_local_patch_review(
            profile,
            detached_approval=create_detached_patch_cli_approval(profile),
        )
    )
    assert pending.continuation is not None
    with patch.object(
        profile._binding._host,
        "await_terminal",
        new=AsyncMock(side_effect=PatchToolError("unavailable")),
    ):
        with pytest.raises(
            PatchCliReviewError, match="terminal result is unavailable"
        ):
            run(patch_review._await_terminal(profile, pending.continuation))

    _service, profile = _profile(pending_after_approval=True)
    pending = run(read_local_patch_review_result(profile))
    assert pending.continuation is not None
    attached = run(patch_review._approve(profile, StringIO(), attached=True))
    assert attached.state is PatchCliReviewState.PENDING
    with pytest.raises(PatchCliReviewError, match="headless authority"):
        patch_review._consume_headless_authority(profile, None, None)

    _service, profile = _profile()
    run(
        run_local_patch_review(
            profile,
            preauthorization=create_exact_patch_cli_preauthorization(profile),
        )
    )
    with patch.object(
        profile._binding._host,
        "validate_invocation_review",
        side_effect=PatchToolError("unavailable"),
    ):
        with pytest.raises(PatchCliReviewError, match="binding is invalid"):
            run(read_local_patch_review_result(profile))


def test_patch_cli_terminal_io_helpers_and_state_guard_fail_closed() -> None:
    """Reject broken terminal I/O and restore a real same-device PTY state."""
    input_fd, output_fd = pipe()
    input_stream = fdopen(input_fd, "r", encoding="utf-8")
    output_stream = StringIO()
    try:
        write(output_fd, b"approve\r\n")
        assert run(_read_action(input_stream, output_stream)) == "approve"
    finally:
        input_stream.close()
        close(output_fd)
    eof_fd, eof_writer = pipe()
    eof_stream = fdopen(eof_fd, "r", encoding="utf-8")
    close(eof_writer)
    try:
        assert run(_read_action(eof_stream, StringIO())) is None
    finally:
        eof_stream.close()
    with pytest.raises(PatchCliReviewError, match="terminal input failed"):
        run(_read_action(StringIO(), StringIO()))

    class _ReadFailure:
        """Expose one readable descriptor whose terminal read fails."""

        def __init__(self, descriptor: int) -> None:
            """Store the descriptor watched by the event loop."""
            self._descriptor = descriptor

        def fileno(self) -> int:
            """Return the readable descriptor."""
            return self._descriptor

        def readline(self) -> str:
            """Fail after readiness without retaining input text."""
            raise OSError("lost")

    read_fd, write_fd = pipe()
    try:
        write(write_fd, b"x")
        with pytest.raises(PatchCliReviewError, match="terminal input failed"):
            run(_read_action(cast(TextIO, _ReadFailure(read_fd)), StringIO()))
    finally:
        close(read_fd)
        close(write_fd)

    class _BrokenOutput(StringIO):
        """Raise on direct terminal output."""

        def write(self, value: str) -> int:
            """Reject every write after accepting the typed value."""
            del value
            raise OSError("lost")

    with pytest.raises(PatchCliReviewError, match="terminal output failed"):
        _write(_BrokenOutput(), "x")

    class _ShortOutput(StringIO):
        """Report a short direct write without claiming full disclosure."""

        def write(self, value: str) -> int:
            """Return fewer characters than the supplied bounded value."""
            return len(value) - 1

    with pytest.raises(PatchCliReviewError, match="terminal output failed"):
        _write(_ShortOutput(), "short")

    broken_read, broken_write = pipe()
    broken_output = fdopen(broken_write, "w", encoding="utf-8")
    close(broken_read)
    try:
        with pytest.raises(
            PatchCliReviewError, match="terminal output failed"
        ):
            _write(broken_output, "actual pipe flush failure")
    finally:
        try:
            broken_output.close()
        except OSError:
            pass

    closed_read, closed_write = pipe()
    closed_output = fdopen(closed_write, "w", encoding="utf-8")
    closed_output.close()
    close(closed_read)
    with pytest.raises(PatchCliReviewError, match="terminal output failed"):
        _write(closed_output, "actual closed descriptor write failure")

    class _Attached(StringIO):
        """Choose terminal attachment independently for a stream."""

        def __init__(self, attached: bool) -> None:
            """Keep one fixed terminal-attachment answer."""
            super().__init__()
            self._attached = attached

        def isatty(self) -> bool:
            """Return the fixed attachment answer."""
            return self._attached

    assert (
        _attached_terminal_session(
            _Attached(False), _Attached(False), _Attached(False)
        )
        is None
    )
    with pytest.raises(PatchCliReviewError, match="do not match"):
        _attached_terminal_session(
            _Attached(True), _Attached(False), _Attached(True)
        )
    with pytest.raises(PatchCliReviewError, match="do not match"):
        _output_terminal_session(_Attached(True), _Attached(False))
    with pytest.raises(PatchCliReviewError, match="terminal is unavailable"):
        patch_review._terminal_session((StringIO(),))

    master, slave = openpty()
    input_tty = open(slave, "r", encoding="utf-8", closefd=False)
    output_tty = fdopen(dup(slave), "w", encoding="utf-8")
    try:
        with (
            patch.object(patch_review, "tcgetpgrp", return_value=1),
            patch.object(patch_review, "getpgrp", return_value=1),
        ):
            session = _attached_terminal_session(
                input_tty, output_tty, output_tty
            )
            assert session is not None
            with _TerminalStateGuard(session, input_tty, output_tty):
                pass
            with pytest.raises(
                PatchCliReviewError, match="terminal is unavailable"
            ):
                with _TerminalStateGuard(session, StringIO(), output_tty):
                    pass
            guard = _TerminalStateGuard(session, input_tty, output_tty)
            guard.__enter__()
            with patch.object(
                patch_review, "ttyname", return_value="/swapped"
            ):
                with pytest.raises(
                    PatchCliReviewError, match="terminal changed"
                ):
                    guard.require_current()
            guard.__exit__(None, None, None)
            with pytest.raises(
                PatchCliReviewError, match="terminal is unavailable"
            ):
                with _TerminalStateGuard(session, input_tty, _BrokenOutput()):
                    pass
            guard = _TerminalStateGuard(session, input_tty, output_tty)
            guard.__enter__()
            with patch.object(patch_review, "tcgetpgrp", return_value=2):
                with pytest.raises(
                    PatchCliReviewError, match="not foreground"
                ):
                    guard.__exit__(None, None, None)
            with patch.object(
                patch_review, "tcsetattr", side_effect=OSError("lost")
            ):
                with pytest.raises(
                    PatchCliReviewError, match="restore failed"
                ):
                    with _TerminalStateGuard(session, input_tty, output_tty):
                        pass
                with pytest.raises(
                    PatchCliReviewError, match="terminal is unavailable"
                ):
                    with _TerminalStateGuard(
                        session, input_tty, _BrokenOutput()
                    ):
                        pass
    finally:
        input_tty.close()
        output_tty.close()
        close(master)
        close(slave)


def test_patch_cli_terminal_identity_rejects_mismatch_and_background() -> None:
    """Require same foreground terminal and restore exact saved attributes."""
    master_one, slave_one = openpty()
    master_two, slave_two = openpty()
    first = open(slave_one, "r", encoding="utf-8", closefd=False)
    first_output = fdopen(dup(slave_one), "w", encoding="utf-8")
    second = open(slave_two, "w", encoding="utf-8", closefd=False)
    try:
        with pytest.raises(PatchCliReviewError, match="do not match"):
            _attached_terminal_session(first, second, second)
        with (
            patch.object(patch_review, "tcgetpgrp", return_value=2),
            patch.object(patch_review, "getpgrp", return_value=1),
        ):
            with pytest.raises(PatchCliReviewError, match="not foreground"):
                _attached_terminal_session(first, first, first)
        with (
            patch.object(patch_review, "tcgetpgrp", return_value=1),
            patch.object(patch_review, "getpgrp", return_value=1),
        ):
            session = _attached_terminal_session(
                first, first_output, first_output
            )
            assert session is not None
            assert session.attributes == tcgetattr(slave_one)
            assert (
                _output_terminal_session(first_output, first_output)
                is not None
            )
    finally:
        first.close()
        first_output.close()
        second.close()
        close(master_one)
        close(master_two)
        close(slave_one)
        close(slave_two)


def test_patch_cli_guard_rejects_swapped_or_lost_terminal_descriptors() -> (
    None
):
    """Fence descriptor swaps, loss, and foreground changes before effect."""
    for swapped in ("input", "output", "error"):
        master, slave = openpty()
        other_master, other_slave = openpty()
        input_tty = open(slave, "r", encoding="utf-8", closefd=False)
        output_tty = fdopen(dup(slave), "w", encoding="utf-8")
        error_tty = fdopen(dup(slave), "w", encoding="utf-8")
        try:
            with (
                patch.object(patch_review, "tcgetpgrp", return_value=1),
                patch.object(patch_review, "getpgrp", return_value=1),
            ):
                session = _attached_terminal_session(
                    input_tty, output_tty, error_tty
                )
                assert session is not None
                guard = _TerminalStateGuard(
                    session, input_tty, output_tty, error_tty
                )
                guard.__enter__()
                target = {
                    "input": input_tty,
                    "output": output_tty,
                    "error": error_tty,
                }[swapped]
                dup2(other_slave, target.fileno())
                with pytest.raises(
                    PatchCliReviewError, match="terminal changed"
                ):
                    guard.require_current()
                handles = (
                    guard._input_handle,
                    guard._output_handle,
                    guard._error_handle,
                )
                with pytest.raises(
                    PatchCliReviewError, match="terminal changed"
                ):
                    guard.__exit__(None, None, None)
                if swapped == "output":
                    readable, _, _ = select((other_master,), (), (), 0)
                    assert readable == []
                for handle in handles:
                    assert handle is not None
                    with pytest.raises(OSError):
                        fstat(handle.duplicate_descriptor)
        finally:
            input_tty.close()
            output_tty.close()
            error_tty.close()
            close(master)
            close(other_master)
            close(slave)
        close(other_slave)

    master, slave = openpty()
    other_master, other_slave = openpty()
    input_tty = open(slave, "r", encoding="utf-8", closefd=False)
    output_tty = fdopen(dup(slave), "w", encoding="utf-8")
    try:
        with (
            patch.object(patch_review, "tcgetpgrp", return_value=1),
            patch.object(patch_review, "getpgrp", return_value=1),
        ):
            session = _attached_terminal_session(
                input_tty, output_tty, output_tty
            )
            assert session is not None
            guard = _TerminalStateGuard(session, input_tty, output_tty)
            guard.__enter__()
            assert guard._output_handle is not None
            dup2(other_slave, guard._output_handle.duplicate_descriptor)
            with pytest.raises(PatchCliReviewError, match="terminal changed"):
                guard.require_current()
            with pytest.raises(PatchCliReviewError, match="terminal changed"):
                guard.__exit__(None, None, None)
            readable, _, _ = select((other_master,), (), (), 0)
            assert readable == []
    finally:
        input_tty.close()
        output_tty.close()
        close(master)
        close(other_master)
        close(slave)
        close(other_slave)

    master, slave = openpty()
    input_tty = open(slave, "r", encoding="utf-8", closefd=False)
    output_tty = fdopen(dup(slave), "w", encoding="utf-8")
    try:
        with (
            patch.object(patch_review, "tcgetpgrp", return_value=1),
            patch.object(patch_review, "getpgrp", return_value=1),
        ):
            session = _attached_terminal_session(
                input_tty, output_tty, output_tty
            )
            assert session is not None
            guard = _TerminalStateGuard(session, input_tty, output_tty)
            guard.__enter__()
            with patch.object(
                patch_review, "fstat", side_effect=OSError("lost")
            ):
                with pytest.raises(
                    PatchCliReviewError, match="terminal is unavailable"
                ):
                    guard.require_current()
                with pytest.raises(
                    PatchCliReviewError, match="terminal is unavailable"
                ):
                    guard.__exit__(None, None, None)
    finally:
        input_tty.close()
        output_tty.close()
        close(master)
        close(slave)

    master, slave = openpty()
    input_tty = open(slave, "r", encoding="utf-8", closefd=False)
    output_tty = fdopen(dup(slave), "w", encoding="utf-8")
    try:
        with (
            patch.object(patch_review, "tcgetpgrp", return_value=1),
            patch.object(patch_review, "getpgrp", return_value=1),
        ):
            session = _attached_terminal_session(
                input_tty, output_tty, output_tty
            )
            assert session is not None
            guard = _TerminalStateGuard(session, input_tty, output_tty)
            guard.__enter__()
            with patch.object(patch_review, "tcgetpgrp", return_value=2):
                with pytest.raises(
                    PatchCliReviewError, match="not foreground"
                ):
                    guard.require_current()
                with pytest.raises(
                    PatchCliReviewError, match="not foreground"
                ):
                    guard.__exit__(None, None, None)
    finally:
        input_tty.close()
        output_tty.close()
        close(master)
        close(slave)


def test_patch_cli_guard_rejects_snapshot_substitution_before_entry() -> None:
    """Reject partial, all-stream, and close/reopen swaps before output."""
    for swapped in ("input", "output", "error", "all", "reopened"):
        master, slave = openpty()
        other_master, other_slave = openpty()
        input_tty = open(slave, "r", encoding="utf-8", closefd=False)
        output_tty = fdopen(dup(slave), "w", encoding="utf-8")
        error_tty = fdopen(dup(slave), "w", encoding="utf-8")
        try:
            with (
                patch.object(patch_review, "tcgetpgrp", return_value=1),
                patch.object(patch_review, "getpgrp", return_value=1),
            ):
                session = _attached_terminal_session(
                    input_tty, output_tty, error_tty
                )
                assert session is not None
                streams = {
                    "input": input_tty,
                    "output": output_tty,
                    "error": error_tty,
                }
                targets = (
                    tuple(streams.values())
                    if swapped == "all"
                    else (
                        streams["error" if swapped == "reopened" else swapped],
                    )
                )
                for target in targets:
                    if swapped == "reopened":
                        close(target.fileno())
                    dup2(other_slave, target.fileno())
                guard = _TerminalStateGuard(
                    session, input_tty, output_tty, error_tty
                )
                with pytest.raises(
                    PatchCliReviewError, match="terminal changed"
                ):
                    guard.__enter__()
                readable, _, _ = select((other_master,), (), (), 0)
                assert readable == []
                assert guard._closed is True
        finally:
            input_tty.close()
            output_tty.close()
            error_tty.close()
            close(master)
            close(other_master)
            close(slave)
            close(other_slave)


def test_patch_cli_guard_bound_session_edge_failures_are_fenced() -> None:
    """Cover output-only sessions and bounded identity-failure cleanup."""
    master, slave = openpty()
    input_tty = open(slave, "r", encoding="utf-8", closefd=False)
    output_tty = fdopen(dup(slave), "w", encoding="utf-8")
    try:
        with (
            patch.object(patch_review, "tcgetpgrp", return_value=1),
            patch.object(patch_review, "getpgrp", return_value=1),
        ):
            output_session = _output_terminal_session(output_tty, output_tty)
            assert output_session is not None
            with _TerminalStateGuard(output_session, None, output_tty):
                pass

            session = _attached_terminal_session(
                input_tty, output_tty, output_tty
            )
            assert session is not None
            guard = _TerminalStateGuard(session, input_tty, output_tty)
            guard.__enter__()
            with patch.object(
                patch_review, "tcgetpgrp", side_effect=OSError("lost")
            ):
                with pytest.raises(
                    PatchCliReviewError, match="terminal is unavailable"
                ):
                    guard.require_current()
            guard.__exit__(None, None, None)

            guard = _TerminalStateGuard(session, input_tty, output_tty)
            with patch.object(
                patch_review, "get_blocking", side_effect=OSError("lost")
            ):
                with pytest.raises(
                    PatchCliReviewError, match="terminal is unavailable"
                ):
                    guard.__enter__()

            corrupted = _attached_terminal_session(
                input_tty, output_tty, output_tty
            )
            assert corrupted is not None
            object.__setattr__(corrupted, "identities", ())
            with pytest.raises(PatchCliReviewError, match="terminal changed"):
                _TerminalStateGuard(
                    corrupted, input_tty, output_tty
                ).__enter__()

            guard = _TerminalStateGuard(session, input_tty, output_tty)
            guard.__enter__()
            control = guard._control_handle()
            with patch.object(
                patch_review,
                "tcgetpgrp",
                side_effect=(1, 1, 2),
            ):
                with pytest.raises(
                    PatchCliReviewError, match="not foreground"
                ):
                    guard._require_stable_control(control)
            guard.__exit__(None, None, None)
    finally:
        input_tty.close()
        output_tty.close()
        close(master)
        close(slave)


def test_patch_cli_guard_async_output_failures_close_real_descriptors() -> (
    None
):
    """Fence unavailable, zero, closed, and kernel-failed async output."""
    master, slave = openpty()
    output_tty = fdopen(dup(slave), "w", encoding="utf-8")
    try:
        with (
            patch.object(patch_review, "tcgetpgrp", return_value=1),
            patch.object(patch_review, "getpgrp", return_value=1),
        ):
            session = _output_terminal_session(output_tty, output_tty)
            assert session is not None
            unavailable = _TerminalStateGuard(session, None, output_tty)
            with pytest.raises(
                PatchCliReviewError, match="output is unavailable"
            ):
                run(unavailable.write("x"))
            with pytest.raises(
                PatchCliReviewError, match="output is unavailable"
            ):
                run(unavailable._wait_until_output_writable())

            zero_write = _TerminalStateGuard(session, None, output_tty)
            zero_write.__enter__()
            with patch.object(patch_review, "os_write", return_value=0):
                with pytest.raises(
                    PatchCliReviewError, match="terminal output failed"
                ):
                    run(zero_write.write("x"))
            zero_write.__exit__(None, None, None)

            closed_writer = _TerminalStateGuard(session, None, output_tty)
            closed_writer.__enter__()
            assert closed_writer._output_handle is not None
            close(closed_writer._output_handle.duplicate_descriptor)
            with pytest.raises(
                PatchCliReviewError, match="terminal output failed"
            ):
                run(closed_writer._wait_until_output_writable())
            closed_writer._close_duplicates(restore_blocking=False)
    finally:
        output_tty.close()
        close(master)
        close(slave)

    master, slave = openpty()
    output_tty = fdopen(dup(slave), "w", encoding="utf-8")
    try:
        with (
            patch.object(patch_review, "tcgetpgrp", return_value=1),
            patch.object(patch_review, "getpgrp", return_value=1),
        ):
            session = _output_terminal_session(output_tty, output_tty)
            assert session is not None
            lost_identity = _TerminalStateGuard(session, None, output_tty)
            lost_identity.__enter__()
            with patch.object(
                patch_review, "ttyname", side_effect=OSError("lost")
            ):
                with pytest.raises(
                    PatchCliReviewError, match="terminal output failed"
                ):
                    run(lost_identity.write("bounded truth"))
            lost_identity._close_duplicates(restore_blocking=False)

            foreground_failure = _TerminalStateGuard(session, None, output_tty)
            foreground_failure.__enter__()
            while select((master,), (), (), 0)[0]:
                read(master, 65536)
            terminal_error = PatchCliReviewError(
                "patch CLI terminal is not foreground"
            )
            with patch.object(
                foreground_failure,
                "require_current",
                side_effect=terminal_error,
            ):
                with pytest.raises(PatchCliReviewError) as raised:
                    run(foreground_failure.write("bounded truth"))
            assert raised.value is terminal_error
            assert select((master,), (), (), 0)[0] == []
            foreground_failure._close_duplicates(restore_blocking=False)

            failed_os_write = _TerminalStateGuard(session, None, output_tty)
            failed_os_write.__enter__()
            while select((master,), (), (), 0)[0]:
                read(master, 65536)
            with patch.object(
                patch_review, "os_write", side_effect=OSError("lost")
            ):
                with pytest.raises(
                    PatchCliReviewError, match="terminal output failed"
                ):
                    run(failed_os_write.write("bounded truth"))
            assert select((master,), (), (), 0)[0] == []
            failed_os_write._close_duplicates(restore_blocking=False)

            failed_write = _TerminalStateGuard(session, None, output_tty)
            failed_write.__enter__()
            close(master)
            master = -1
            with pytest.raises(
                PatchCliReviewError, match="terminal output failed"
            ):
                run(failed_write.write("bounded truth"))
            failed_write._close_duplicates(restore_blocking=False)
    finally:
        output_tty.close()
        if master >= 0:
            close(master)
        close(slave)


def test_patch_cli_guard_closes_duplicates_and_fences_restore_failures() -> (
    None
):
    """Cover unavailable handles and safe duplicated-descriptor cleanup."""

    class _BrokenClose:
        """Fail one duplicate wrapper close without retaining terminal data."""

        def close(self) -> None:
            """Raise the expected bounded close failure."""
            raise OSError("close failed")

    master, slave = openpty()
    input_tty = open(slave, "r", encoding="utf-8", closefd=False)
    output_tty = fdopen(dup(slave), "w", encoding="utf-8")
    try:
        with (
            patch.object(patch_review, "tcgetpgrp", return_value=1),
            patch.object(patch_review, "getpgrp", return_value=1),
        ):
            session = _attached_terminal_session(
                input_tty, output_tty, output_tty
            )
            assert session is not None

            guard = _TerminalStateGuard(session, input_tty, output_tty)
            with pytest.raises(
                PatchCliReviewError, match="input is unavailable"
            ):
                _ = guard.input_stream
            with pytest.raises(
                PatchCliReviewError, match="output is unavailable"
            ):
                _ = guard.output_stream
            with pytest.raises(
                PatchCliReviewError, match="terminal is unavailable"
            ):
                guard._control_handle()
            with patch.object(
                patch_review,
                "_duplicate_terminal_handle",
                side_effect=PatchCliReviewError("duplicate failed"),
            ):
                with pytest.raises(
                    PatchCliReviewError, match="duplicate failed"
                ):
                    guard.__enter__()

            guard = _TerminalStateGuard(session, input_tty, output_tty)
            guard.__enter__()
            with patch.object(patch_review, "tcgetattr", return_value=[]):
                with pytest.raises(
                    PatchCliReviewError, match="restore failed"
                ):
                    guard.__exit__(None, None, None)

            guard = _TerminalStateGuard(session, input_tty, output_tty)
            guard.__enter__()
            with patch.object(
                patch_review,
                "_require_terminal_handle",
                side_effect=PatchCliReviewError("terminal changed"),
            ):
                with pytest.raises(
                    PatchCliReviewError, match="terminal changed"
                ):
                    guard.__exit__(None, None, None)

            guard = _TerminalStateGuard(session, input_tty, output_tty)
            with pytest.raises(
                PatchCliReviewError, match="output is unavailable"
            ):
                guard._write_escape("x")
            with pytest.raises(
                PatchCliReviewError, match="output is unavailable"
            ):
                guard._require_stable_output()

            guard = _TerminalStateGuard(session, input_tty, output_tty)
            guard.__enter__()
            assert guard.input_stream.fileno() != input_tty.fileno()
            assert guard.output_stream.fileno() != output_tty.fileno()
            with patch.object(patch_review, "os_write", return_value=1):
                with pytest.raises(
                    PatchCliReviewError, match="terminal is unavailable"
                ):
                    guard._write_escape("xx")
            guard.__exit__(None, None, None)

            guard = _TerminalStateGuard(session, input_tty, output_tty)
            guard.__enter__()
            with patch.object(
                patch_review, "os_write", side_effect=OSError("lost")
            ):
                with pytest.raises(
                    PatchCliReviewError, match="terminal is unavailable"
                ):
                    guard.__exit__(None, None, None)

            guard = _TerminalStateGuard(session, input_tty, output_tty)
            guard.__enter__()
            handles = (
                guard._input_handle,
                guard._output_handle,
                guard._error_handle,
            )
            guard._input_duplicate_stream = cast(TextIO, _BrokenClose())
            guard._output_duplicate_stream = cast(TextIO, _BrokenClose())
            with (
                patch.object(
                    patch_review, "set_blocking", side_effect=OSError("lost")
                ),
                patch.object(
                    patch_review, "close", side_effect=OSError("lost")
                ),
            ):
                guard._close_duplicates()
            guard._close_duplicates()
            for handle in handles:
                assert handle is not None
                close(handle.duplicate_descriptor)

            with patch.object(patch_review, "isatty", return_value=False):
                with pytest.raises(
                    PatchCliReviewError, match="terminal is unavailable"
                ):
                    patch_review._duplicate_terminal_handle(
                        input_tty, session.identities[0]
                    )
            with patch.object(
                patch_review,
                "ttyname",
                side_effect=(
                    session.identities[0].terminal_name,
                    "/dev/replaced",
                ),
            ):
                with pytest.raises(
                    PatchCliReviewError, match="terminal changed"
                ):
                    patch_review._duplicate_terminal_handle(
                        input_tty, session.identities[0]
                    )
    finally:
        input_tty.close()
        output_tty.close()
        close(master)
        close(slave)


def test_patch_cli_prepare_rejects_unrelated_host_review() -> None:
    """Require host preparation to originate from one pending invocation."""
    with pytest.raises(PatchCliReviewError):
        run(prepare_local_patch_review_binding(cast(PatchSdkHost, object())))
    with pytest.raises(PatchCliReviewError):
        ExactPatchCliPreauthorization(cast(Never, None))
    with pytest.raises(PatchCliReviewError):
        DetachedPatchCliApproval(cast(Never, None))

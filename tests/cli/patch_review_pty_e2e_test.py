"""Exercise the unregistered local patch-review profile through real PTYs."""

from asyncio import Future, get_running_loop, run
from errno import EIO
from fcntl import ioctl
from json import dumps, loads
from os import (
    WNOHANG,
    _exit,
    close,
    fork,
    kill,
    pipe,
    read,
    setsid,
    waitpid,
    waitstatus_to_exitcode,
    write,
)
from pathlib import Path
from pty import openpty
from runpy import run_path
from select import select
from signal import SIGINT
from termios import ECHO, ICANON, TIOCSCTTY, tcgetattr
from time import monotonic
from typing import Any, cast
from unittest.mock import patch

import pytest

from avalan.cli import patch_review
from avalan.cli.patch_review import (
    PatchCliReviewError,
    PatchCliReviewState,
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
    OperationType,
    PatchPending,
    PatchPendingOperationId,
)
from avalan.patch.review_display import ReviewDisplayError
from avalan.patch.toolset import (
    PatchCapabilitySnapshot,
    PatchInvocationHandle,
    PatchSdkHost,
)

_RAW_CANARY = "PATCH-CLI-RAW-HOST-INPUT-CANARY-6A5A4D8E"
_REVIEW_CANARY = "PATCH-CLI-PRIVILEGED-REVIEW-CANARY-7F3C2B1D"


def _phase_module(name: str) -> dict[str, Any]:
    """Load one existing phase fixture in an isolated test namespace."""
    path = Path("tests/patch") / name
    return run_path(str(path))


def _settled(value: Any) -> Future[Any]:
    """Return one resolved future in the current loop."""
    future: Future[Any] = get_running_loop().create_future()
    future.set_result(value)
    return future


class _PendingAfterApprovalService:
    """Expose one pending approval that settles only through a later read."""

    def __init__(self, base: Any) -> None:
        """Initialize the phase-nine service shape without target effects."""
        self._base = base
        self.invocations: list[bytes] = []
        self.approvals = 0
        self.waits = 0
        self.request_id: Any = None
        self.correlation_id: Any = None
        self.settlement = self

    def inspect(self, handle: PatchInvocationHandle) -> Future[Any]:
        """Return the current pending observation for the issued handle."""
        assert type(handle) is PatchInvocationHandle
        return _settled(self._pending())

    def await_terminal(
        self, handle: PatchInvocationHandle, pending: PatchPending
    ) -> Future[Any]:
        """Settle only the original pending request after one restart read."""
        assert type(handle) is PatchInvocationHandle
        assert pending == self._pending()
        self.waits += 1
        return _settled(self._base["_result"](self.request_id))

    async def invoke(
        self,
        operation: Any,
        raw_arguments: bytes,
        capability: Any,
        request_id: Any,
        correlation_id: Any,
    ) -> PatchPending:
        """Record one request and suspend it before reviewer approval."""
        del operation, capability
        self.invocations.append(raw_arguments)
        self.request_id = request_id
        self.correlation_id = correlation_id
        return self._pending()

    async def review(self, handle: PatchInvocationHandle) -> dict[str, str]:
        """Return the exact host-owned complete preapproval review only."""
        assert type(handle) is PatchInvocationHandle
        return {
            "candidate": _REVIEW_CANARY,
            "plan": "exact-local-pending-plan",
        }

    async def approve(self, handle: PatchInvocationHandle) -> PatchPending:
        """Start the same fenced settlement without a second invocation."""
        assert type(handle) is PatchInvocationHandle
        self.approvals += 1
        return self._pending()

    def subscribe(self, handle: PatchInvocationHandle) -> Any:
        """Reject unused lifecycle subscription in this PTY-only service."""
        del handle
        raise AssertionError("PTY review does not subscribe")

    async def cancel(self, handle: PatchInvocationHandle) -> PatchPending:
        """Preserve the known pending request without an effect."""
        assert type(handle) is PatchInvocationHandle
        return self._pending()

    def _pending(self) -> PatchPending:
        """Return the one exact durable pending envelope for this service."""
        assert self.request_id is not None
        assert self.correlation_id is not None
        return PatchPending(
            1,
            PatchPendingOperationId("pending_" + "b" * 16),
            self.request_id,
            self.correlation_id,
            LifecyclePhase.SETTLEMENT_PENDING,
        )


class _TerminalAfterApprovalService(_PendingAfterApprovalService):
    """Settle the reviewed request directly when exact approval is received."""

    def inspect(self, handle: PatchInvocationHandle) -> Future[Any]:
        """Report pending before approval and terminal truth only afterward."""
        assert type(handle) is PatchInvocationHandle
        if self.approvals:
            return _settled(self._base["_result"](self.request_id))
        return _settled(self._pending())

    async def approve(self, handle: PatchInvocationHandle) -> Any:
        """Return the terminal result for the same prepared request only."""
        assert type(handle) is PatchInvocationHandle
        self.approvals += 1
        return self._base["_result"](self.request_id)


async def _profile(*, pending_after_approval: bool) -> tuple[Any, Any]:
    """Return a real authenticated SDK host and detached review profile."""
    phase_nine = _phase_module("phase_9_contract_test.py")
    if pending_after_approval:
        service = _PendingAfterApprovalService(phase_nine)
    else:
        service = _TerminalAfterApprovalService(phase_nine)
    toolset = await phase_nine["_toolset_async"](
        service,
        PatchCapabilitySnapshot(edit_available=True, apply_available=False),
    )
    host = PatchSdkHost(service, toolset.capability)
    await _invoke(host)
    binding = await prepare_local_patch_review_binding(host)
    profile = create_local_patch_review_test_profile(binding)
    return service, profile


async def _invoke(host: PatchSdkHost) -> None:
    """Start one pending edit with a raw-content canary."""
    outcome = await host.invoke_json(
        operation=OperationType.EDIT,
        arguments={
            "path": "note.txt",
            "edits": [{"old_text": "old", "new_text": _RAW_CANARY}],
        },
    )
    assert type(outcome) is PatchPending


def _child(
    slave: int,
    result_fd: int,
    case: str,
) -> None:
    """Run one real test-profile CLI interaction inside the child PTY."""
    from os import dup2, fdopen, fstat, getpgrp, tcgetpgrp

    dup2(slave, 0)
    dup2(slave, 1)
    dup2(slave, 2)
    close(slave)
    setsid()
    ioctl(0, TIOCSCTTY, 0)
    input_stream = fdopen(0, "r", encoding="utf-8", closefd=False)
    output_stream = fdopen(1, "w", encoding="utf-8", closefd=False)
    error_stream = fdopen(2, "w", encoding="utf-8", closefd=False)
    service, profile = run(_profile(pending_after_approval=case == "pending"))
    before = tcgetattr(0)
    identity_before = tuple(
        (fstat(descriptor).st_dev, fstat(descriptor).st_ino)
        for descriptor in (0, 1, 2)
    )
    foreground_before = tcgetpgrp(0) == getpgrp()
    payload: dict[str, object]
    try:
        if case == "renderer_failure":
            with patch.object(
                patch_review,
                "_review_pages",
                side_effect=ReviewDisplayError("blocked"),
            ):
                run(
                    run_local_patch_review(
                        profile,
                        input_stream=input_stream,
                        output_stream=output_stream,
                        error_stream=error_stream,
                    )
                )
            raise AssertionError("renderer failure must reject")
        result = run(
            run_local_patch_review(
                profile,
                input_stream=input_stream,
                output_stream=output_stream,
                error_stream=error_stream,
            )
        )
        later_read_terminal = False
        if case == "approve":
            later_read = run(
                read_local_patch_review_result(
                    profile,
                    output_stream=output_stream,
                    error_stream=error_stream,
                )
            )
            later_read_terminal = (
                later_read.state is PatchCliReviewState.TERMINAL
            )
        if case == "pending":
            assert result.state is PatchCliReviewState.PENDING
            assert result.continuation is not None
            result = run(
                resume_local_patch_review(
                    profile,
                    result.continuation,
                    output_stream=output_stream,
                    error_stream=error_stream,
                )
            )
        payload = {
            "state": result.state.value,
            "terminal": result.result is not None,
            "later_read_terminal": later_read_terminal,
            "invocations": len(service.invocations),
            "approvals": getattr(service, "approvals", 0),
            "waits": getattr(service, "waits", 0),
            "restored": before == tcgetattr(0),
        }
    except PatchCliReviewError as error:
        payload = {
            "error": str(error),
            "invocations": len(service.invocations),
            "approvals": getattr(service, "approvals", 0),
            "waits": getattr(service, "waits", 0),
            "restored": before == tcgetattr(0),
        }
    after = tcgetattr(0)
    payload["same_terminal"] = (
        fstat(0).st_dev == fstat(1).st_dev == fstat(2).st_dev
        and fstat(0).st_ino == fstat(1).st_ino == fstat(2).st_ino
    )
    payload["terminal_identity_restored"] = identity_before == tuple(
        (fstat(descriptor).st_dev, fstat(descriptor).st_ino)
        for descriptor in (0, 1, 2)
    )
    payload["foreground_before"] = foreground_before
    payload["foreground"] = tcgetpgrp(0) == getpgrp()
    payload["echo_restored"] = bool(after[3] & ECHO)
    payload["canonical_restored"] = bool(after[3] & ICANON)
    write(result_fd, dumps(payload, sort_keys=True).encode("utf-8"))
    close(result_fd)


def _spawn(case: str) -> tuple[int, int, int]:
    """Spawn one PTY child and return process, terminal, and result pipe."""
    master, slave = openpty()
    result_read, result_write = pipe()
    pid = fork()
    if pid == 0:
        close(master)
        close(result_read)
        _child(slave, result_write, case)
        _exit(0)
    close(slave)
    close(result_write)
    return pid, master, result_read


def _read_until(fd: int, marker: bytes) -> bytes:
    """Read terminal bytes until a fixed prompt appears or the child exits."""
    output = b""
    deadline = monotonic() + 15
    while marker not in output and monotonic() < deadline:
        readable, _, _ = select((fd,), (), (), 0.2)
        if readable:
            chunk = _read_pty_master(fd)
            if chunk is None:
                break
            output += chunk
    return output


def _read_pty_master(fd: int) -> bytes | None:
    """Read a PTY master chunk, mapping Linux closed-slave EIO to EOF."""
    try:
        return read(fd, 65536)
    except OSError as error:
        if error.errno == EIO:
            return None
        raise


def _finish(
    pid: int, master: int, result_fd: int, output: bytes
) -> tuple[dict[str, object], bytes]:
    """Collect child output, status, and bounded result receipt."""
    deadline = monotonic() + 15
    status = None
    while status is None and monotonic() < deadline:
        readable, _, _ = select((master,), (), (), 0.05)
        if readable:
            chunk = _read_pty_master(master)
            if chunk:
                output += chunk
        observed, value = waitpid(pid, WNOHANG)
        if observed:
            status = value
            break
    assert status is not None
    while True:
        readable, _, _ = select((master,), (), (), 0)
        if not readable:
            break
        chunk = _read_pty_master(master)
        if not chunk:
            break
        output += chunk
    payload = loads(read(result_fd, 65536).decode("utf-8"))
    close(master)
    close(result_fd)
    assert waitstatus_to_exitcode(status) == 0
    assert isinstance(payload, dict)
    return cast(dict[str, object], payload), output


def test_patch_cli_pty_eio_is_terminal_eof_and_other_errors_propagate() -> (
    None
):
    """Treat only Linux PTY EIO as EOF while retaining other read failures."""
    terminal_eof = OSError(EIO, "PTY slave closed")
    with patch(f"{__name__}.read", side_effect=terminal_eof):
        assert _read_pty_master(17) is None

    unexpected = OSError(999, "unexpected PTY read failure")
    with (
        patch(f"{__name__}.read", side_effect=unexpected),
        pytest.raises(OSError) as raised,
    ):
        _read_pty_master(17)
    assert raised.value is unexpected


def test_patch_e2e_023_real_pty_review_approve_and_later_read() -> None:
    """Complete privileged review and approval through a real isolated PTY."""
    pid, master, result_fd = _spawn("approve")
    output = _read_until(master, b"Review action [approve|deny|cancel]: ")
    if b"Review action [approve|deny|cancel]: " not in output:
        payload, output = _finish(pid, master, result_fd, output)
        pytest.fail(f"PTY review did not reach action prompt: {payload}")
    write(master, b"approve\n")
    payload, output = _finish(pid, master, result_fd, output)

    assert payload == {
        "approvals": 1,
        "invocations": 1,
        "later_read_terminal": True,
        "foreground_before": True,
        "foreground": True,
        "same_terminal": True,
        "terminal_identity_restored": True,
        "echo_restored": True,
        "canonical_restored": True,
        "restored": True,
        "state": "terminal",
        "terminal": True,
        "waits": 0,
    }
    assert b"Privileged patch preapproval review" in output
    assert b"Trusted reviewer action" in output
    assert b"Patch terminal result: status=committed" in output
    assert _REVIEW_CANARY.encode("utf-8") in output
    assert _RAW_CANARY.encode("utf-8") not in output
    assert b"\x1b[?1049h\x1b[?25l" in output
    assert b"\x1b[0m\x1b[?25h\x1b[?1049l" in output


def test_patch_e2e_024_real_pty_pending_restart_awaits_same_invocation() -> (
    None
):
    """Detach and resume one pending invocation without a second effect."""
    pid, master, result_fd = _spawn("pending")
    output = _read_until(master, b"Review action [approve|deny|cancel]: ")
    write(master, b"approve\n")
    payload, output = _finish(pid, master, result_fd, output)

    assert payload == {
        "approvals": 1,
        "invocations": 1,
        "later_read_terminal": False,
        "foreground_before": True,
        "foreground": True,
        "same_terminal": True,
        "terminal_identity_restored": True,
        "echo_restored": True,
        "canonical_restored": True,
        "restored": True,
        "state": "terminal",
        "terminal": True,
        "waits": 1,
    }
    assert output.count(b"Patch settlement remains pending.") == 2
    assert b"Patch terminal result: status=committed" in output
    assert _REVIEW_CANARY.encode("utf-8") in output
    assert _RAW_CANARY.encode("utf-8") not in output


@pytest.mark.parametrize(
    ("case", "input_bytes", "expected_state"),
    (
        ("approve", b"deny\n", "denied"),
        ("approve", b"a\n", "cancelled"),
        ("approve", b"approve-all\n", "cancelled"),
        ("approve", b"\x04", "cancelled"),
    ),
)
def test_patch_cli_pty_rejects_nonexact_or_cancelled_actions(
    case: str,
    input_bytes: bytes,
    expected_state: str,
) -> None:
    """Reject deny, raw shorthand, future approval, and EOF without writes."""
    pid, master, result_fd = _spawn(case)
    output = _read_until(master, b"Review action [approve|deny|cancel]: ")
    write(master, input_bytes)
    payload, output = _finish(pid, master, result_fd, output)

    assert payload["state"] == expected_state
    assert payload["invocations"] == 1
    assert payload["approvals"] == 0
    assert payload["waits"] == 0
    assert payload["restored"] is True
    assert payload["same_terminal"] is True
    assert payload["terminal_identity_restored"] is True
    assert payload["foreground_before"] is True
    assert payload["foreground"] is True
    assert payload["echo_restored"] is True
    assert payload["canonical_restored"] is True
    assert _RAW_CANARY.encode("utf-8") not in output


def test_patch_cli_pty_sigint_restores_terminal_without_approval() -> None:
    """Cancel a real PTY review with SIGINT before any approval mutation."""
    pid, master, result_fd = _spawn("approve")
    output = _read_until(master, b"Review action [approve|deny|cancel]: ")
    kill(pid, SIGINT)
    payload, output = _finish(pid, master, result_fd, output)

    assert payload["state"] == "cancelled"
    assert payload["approvals"] == 0
    assert payload["restored"] is True
    assert payload["same_terminal"] is True
    assert payload["terminal_identity_restored"] is True
    assert payload["foreground_before"] is True
    assert payload["foreground"] is True
    assert payload["echo_restored"] is True
    assert payload["canonical_restored"] is True
    assert b"\x1b[0m\x1b[?25h\x1b[?1049l" in output
    assert _RAW_CANARY.encode("utf-8") not in output


def test_patch_cli_pty_renderer_failure_restores_without_approval() -> None:
    """Fail a renderer before controls without approving content."""
    pid, master, result_fd = _spawn("renderer_failure")
    payload, output = _finish(pid, master, result_fd, b"")

    assert payload["error"] == "patch CLI review renderer failed"
    assert payload["approvals"] == 0
    assert payload["restored"] is True
    assert payload["same_terminal"] is True
    assert payload["terminal_identity_restored"] is True
    assert payload["foreground_before"] is True
    assert payload["foreground"] is True
    assert _RAW_CANARY.encode("utf-8") not in output


def test_patch_cli_headless_requires_exact_authority() -> None:
    """Fail closed headless except for exact profile-bound authority."""

    async def exercise() -> None:
        """Start three isolated invocations under the same trusted profile."""
        service, profile = await _profile(pending_after_approval=False)
        with pytest.raises(PatchCliReviewError, match="attached terminal"):
            await run_local_patch_review(profile)
        assert len(service.invocations) == 1
        assert getattr(service, "approvals", 0) == 0
        first = await run_local_patch_review(
            profile,
            preauthorization=create_exact_patch_cli_preauthorization(profile),
        )
        assert first.state is PatchCliReviewState.TERMINAL

    run(exercise())

    async def detached() -> None:
        """Allow headless approval only with exact detached review."""
        service, profile = await _profile(pending_after_approval=False)
        result = await run_local_patch_review(
            profile,
            detached_approval=create_detached_patch_cli_approval(profile),
        )
        assert result.state is PatchCliReviewState.TERMINAL
        assert len(service.invocations) == 1

    run(detached())

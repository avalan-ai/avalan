"""Provide deterministic typed fixtures for structured-input tests."""

from asyncio import Event, Queue
from collections.abc import Iterable
from dataclasses import dataclass
from typing import ClassVar, NewType

TestRunId = NewType("TestRunId", str)
TestTurnId = NewType("TestTurnId", str)
TestTaskId = NewType("TestTaskId", str)
TestAgentId = NewType("TestAgentId", str)
TestBranchId = NewType("TestBranchId", str)
TestModelCallId = NewType("TestModelCallId", str)
TestRequestId = NewType("TestRequestId", str)
TestContinuationId = NewType("TestContinuationId", str)
TestStreamSessionId = NewType("TestStreamSessionId", str)
TestUserId = NewType("TestUserId", str)
TestTenantId = NewType("TestTenantId", str)
TestParticipantId = NewType("TestParticipantId", str)
TestSessionId = NewType("TestSessionId", str)
TestStateRevision = NewType("TestStateRevision", int)


@dataclass(frozen=True, kw_only=True, slots=True)
class TestPrincipal:
    """Identify one trusted test principal."""

    __test__: ClassVar[bool] = False

    user_id: TestUserId | None = None
    tenant_id: TestTenantId | None = None
    participant_id: TestParticipantId | None = None
    session_id: TestSessionId | None = None


@dataclass(frozen=True, kw_only=True, slots=True)
class TestCorrelation:
    """Identify one deterministic logical execution."""

    __test__: ClassVar[bool] = False

    run_id: TestRunId
    turn_id: TestTurnId
    task_id: TestTaskId | None
    agent_id: TestAgentId
    branch_id: TestBranchId
    parent_branch_id: TestBranchId | None
    model_call_id: TestModelCallId
    request_id: TestRequestId
    continuation_id: TestContinuationId
    stream_session_id: TestStreamSessionId
    state_revision: TestStateRevision


@dataclass(frozen=True, kw_only=True, slots=True)
class ScriptedProviderCall:
    """Describe one expected provider call and its response."""

    request: str
    response: str


class ManualClock:
    """Provide manually advanced monotonic time."""

    def __init__(self, initial: float = 0.0) -> None:
        assert initial >= 0.0
        self._now = initial

    def now(self) -> float:
        """Return the current deterministic instant."""
        return self._now

    def advance(self, seconds: float) -> None:
        """Advance time without sleeping."""
        assert seconds >= 0.0
        self._now += seconds


class OpaqueIdFactory:
    """Create deterministic opaque identifiers with distinct static types."""

    def __init__(self, prefix: str = "fixture") -> None:
        assert prefix
        self._prefix = prefix
        self._sequence = 0

    def _next(self, kind: str) -> str:
        self._sequence += 1
        return f"{self._prefix}-{kind}-{self._sequence:04d}"

    def run_id(self) -> TestRunId:
        """Return the next run identifier."""
        return TestRunId(self._next("run"))

    def turn_id(self) -> TestTurnId:
        """Return the next turn identifier."""
        return TestTurnId(self._next("turn"))

    def task_id(self) -> TestTaskId:
        """Return the next task identifier."""
        return TestTaskId(self._next("task"))

    def agent_id(self) -> TestAgentId:
        """Return the next agent identifier."""
        return TestAgentId(self._next("agent"))

    def branch_id(self) -> TestBranchId:
        """Return the next branch identifier."""
        return TestBranchId(self._next("branch"))

    def model_call_id(self) -> TestModelCallId:
        """Return the next model-call identifier."""
        return TestModelCallId(self._next("model-call"))

    def request_id(self) -> TestRequestId:
        """Return the next request identifier."""
        return TestRequestId(self._next("request"))

    def continuation_id(self) -> TestContinuationId:
        """Return the next continuation identifier."""
        return TestContinuationId(self._next("continuation"))

    def stream_session_id(self) -> TestStreamSessionId:
        """Return the next stream-session identifier."""
        return TestStreamSessionId(self._next("stream-session"))

    def user_id(self) -> TestUserId:
        """Return the next user identifier."""
        return TestUserId(self._next("user"))

    def tenant_id(self) -> TestTenantId:
        """Return the next tenant identifier."""
        return TestTenantId(self._next("tenant"))

    def participant_id(self) -> TestParticipantId:
        """Return the next participant identifier."""
        return TestParticipantId(self._next("participant"))

    def session_id(self) -> TestSessionId:
        """Return the next session identifier."""
        return TestSessionId(self._next("session"))

    def state_revision(self) -> TestStateRevision:
        """Return the next state revision."""
        self._sequence += 1
        return TestStateRevision(self._sequence)


class ScriptedProvider:
    """Return a finite deterministic sequence of provider responses."""

    def __init__(self, calls: Iterable[ScriptedProviderCall]) -> None:
        self._calls = tuple(calls)
        self._index = 0

    @property
    def call_count(self) -> int:
        """Return the number of completed calls."""
        return self._index

    async def generate(self, request: str) -> str:
        """Validate and resolve the next scripted provider call."""
        if self._index >= len(self._calls):
            raise RuntimeError("scripted provider exhausted")
        call = self._calls[self._index]
        if request != call.request:
            raise ValueError("scripted provider request mismatch")
        self._index += 1
        return call.response


class AsyncBarrier:
    """Coordinate deterministic concurrency without timing sleeps."""

    def __init__(self, parties: int) -> None:
        assert parties > 0
        self._parties = parties
        self._arrivals = 0
        self._released = Event()

    async def arrive_and_wait(self) -> None:
        """Wait until every expected participant has arrived."""
        self._arrivals += 1
        if self._arrivals > self._parties:
            raise RuntimeError("async barrier received too many arrivals")
        if self._arrivals == self._parties:
            self._released.set()
        await self._released.wait()


class LocalProtocolPeer:
    """Exchange typed local messages without network access."""

    def __init__(self) -> None:
        self._incoming: Queue[str] = Queue()
        self._outgoing: Queue[str] = Queue()

    async def send_to_runtime(self, message: str) -> None:
        """Send one message to the runtime-facing queue."""
        await self._incoming.put(message)

    async def receive_from_client(self) -> str:
        """Receive one message sent to the runtime."""
        return await self._incoming.get()

    async def send_to_client(self, message: str) -> None:
        """Send one message to the client-facing queue."""
        await self._outgoing.put(message)

    async def receive_from_runtime(self) -> str:
        """Receive one message sent to the client."""
        return await self._outgoing.get()

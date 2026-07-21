"""Lock the static types of deterministic interaction fixtures."""

from typing import assert_type

from input_contract_fixtures import (
    AsyncBarrier,
    LocalProtocolPeer,
    ManualClock,
    OpaqueIdFactory,
    ScriptedProvider,
    ScriptedProviderCall,
    TestAgentId,
    TestBranchId,
    TestContinuationId,
    TestCorrelation,
    TestModelCallId,
    TestParticipantId,
    TestPrincipal,
    TestRequestId,
    TestRunId,
    TestSessionId,
    TestStateRevision,
    TestStreamSessionId,
    TestTaskId,
    TestTenantId,
    TestTurnId,
    TestUserId,
)

factory = OpaqueIdFactory()
assert_type(factory.run_id(), TestRunId)
assert_type(factory.turn_id(), TestTurnId)
assert_type(factory.task_id(), TestTaskId)
assert_type(factory.agent_id(), TestAgentId)
assert_type(factory.branch_id(), TestBranchId)
assert_type(factory.model_call_id(), TestModelCallId)
assert_type(factory.request_id(), TestRequestId)
assert_type(factory.continuation_id(), TestContinuationId)
assert_type(factory.stream_session_id(), TestStreamSessionId)
assert_type(factory.user_id(), TestUserId)
assert_type(factory.tenant_id(), TestTenantId)
assert_type(factory.participant_id(), TestParticipantId)
assert_type(factory.session_id(), TestSessionId)
assert_type(factory.state_revision(), TestStateRevision)

principal = TestPrincipal(
    user_id=factory.user_id(),
    tenant_id=factory.tenant_id(),
    participant_id=factory.participant_id(),
    session_id=factory.session_id(),
)
assert_type(principal, TestPrincipal)
assert_type(principal.user_id, TestUserId | None)

correlation = TestCorrelation(
    run_id=factory.run_id(),
    turn_id=factory.turn_id(),
    task_id=factory.task_id(),
    agent_id=factory.agent_id(),
    branch_id=factory.branch_id(),
    parent_branch_id=None,
    model_call_id=factory.model_call_id(),
    request_id=factory.request_id(),
    continuation_id=factory.continuation_id(),
    stream_session_id=factory.stream_session_id(),
    state_revision=factory.state_revision(),
)
assert_type(correlation, TestCorrelation)
assert_type(correlation.request_id, TestRequestId)
assert_type(correlation.state_revision, TestStateRevision)

clock = ManualClock()
assert_type(clock.now(), float)
assert_type(clock.advance(1.0), None)

provider = ScriptedProvider(
    [ScriptedProviderCall(request="in", response="out")]
)
assert_type(provider, ScriptedProvider)
assert_type(provider.call_count, int)


async def exercise_async_contracts() -> None:
    """Lock async fixture result types."""
    barrier = AsyncBarrier(parties=1)
    assert_type(await barrier.arrive_and_wait(), None)
    peer = LocalProtocolPeer()
    assert_type(await peer.send_to_runtime("request"), None)
    assert_type(await peer.receive_from_client(), str)
    assert_type(await peer.send_to_client("resolution"), None)
    assert_type(await peer.receive_from_runtime(), str)
    assert_type(await provider.generate("in"), str)

"""Regress sequential response ownership and automatic synchronization."""

from asyncio import Event, create_task, gather, wait_for
from typing import Any, cast
from unittest import IsolatedAsyncioTestCase

from execution_response_ownership_adversarial_test import (
    _Agent,
    _FailingSyncAgent,
    _Harness,
)

from avalan.agent.execution import AgentExecutionStatus
from avalan.entities import (
    EngineMessageIdempotencyKey,
    Message,
    MessageRole,
)


def _call_transcript(
    harness: _Harness,
    call_index: int,
) -> tuple[tuple[MessageRole, str], ...]:
    """Return exact role and text pairs sent in one model call."""
    input_value = harness.manager.calls[call_index].context.input
    messages = (
        [input_value]
        if isinstance(input_value, Message)
        else cast(list[Message], input_value)
    )
    transcript: list[tuple[MessageRole, str]] = []
    for message in messages:
        assert isinstance(message.content, str)
        transcript.append((message.role, message.content))
    return tuple(transcript)


class _BlockingSyncAgent(_Agent):
    """Hold one user-memory append and count exact sink attempts."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.sync_started = Event()
        self.release_sync = Event()
        self.sync_attempts: list[tuple[MessageRole, str]] = []
        self._block_once = True
        super().__init__(*args, **kwargs)

    async def sync_message(
        self,
        message: Message,
        *,
        idempotency_key: EngineMessageIdempotencyKey | None = None,
    ) -> None:
        """Block one append before delegating to the real memory sink."""
        assert isinstance(message.content, str)
        self.sync_attempts.append((message.role, message.content))
        if self._block_once:
            self._block_once = False
            self.sync_started.set()
            await self.release_sync.wait()
        await super().sync_message(
            message,
            idempotency_key=idempotency_key,
        )


class SequentialResponseSynchronizationTest(IsolatedAsyncioTestCase):
    """Require deterministic history without merging active invocations."""

    async def test_next_root_call_syncs_completed_exchange(self) -> None:
        harness = _Harness()
        try:
            first = await harness.orchestrator("first")
            self.assertEqual(await first.to_str(), "answer:first")
            self.assertEqual(harness.recent(), ())

            second = await harness.orchestrator("second")

            self.assertEqual(
                _call_transcript(harness, 1),
                (
                    (MessageRole.USER, "first"),
                    (MessageRole.ASSISTANT, "answer:first"),
                    (MessageRole.USER, "second"),
                ),
            )
            self.assertEqual(
                harness.recent(),
                (
                    (MessageRole.USER, "first"),
                    (MessageRole.ASSISTANT, "answer:first"),
                ),
            )
            self.assertEqual(
                tuple(harness.orchestrator._pending_responses.values()),
                (second,),
            )
            self.assertEqual(await second.to_str(), "answer:second")

            await harness.exit()

            self.assertEqual(
                harness.recent(),
                (
                    (MessageRole.USER, "first"),
                    (MessageRole.ASSISTANT, "answer:first"),
                    (MessageRole.USER, "second"),
                    (MessageRole.ASSISTANT, "answer:second"),
                ),
            )
            self.assertEqual(harness.orchestrator._pending_responses, {})
        finally:
            await harness.close()

    async def test_explicit_running_sync_retains_response_until_exit(
        self,
    ) -> None:
        harness = _Harness()
        try:
            response = await harness.orchestrator("running")
            execution = response.execution
            assert execution is not None
            self.assertIs(execution.status, AgentExecutionStatus.RUNNING)

            await harness.orchestrator.sync_messages(response)

            self.assertEqual(
                harness.recent(),
                ((MessageRole.USER, "running"),),
            )
            self.assertEqual(
                tuple(harness.orchestrator._pending_responses.values()),
                (response,),
            )

            self.assertEqual(await response.to_str(), "answer:running")
            self.assertIs(execution.status, AgentExecutionStatus.COMPLETED)
            self.assertEqual(
                harness.recent(),
                ((MessageRole.USER, "running"),),
            )

            await harness.exit()

            self.assertEqual(
                harness.recent(),
                (
                    (MessageRole.USER, "running"),
                    (MessageRole.ASSISTANT, "answer:running"),
                ),
            )
            self.assertEqual(harness.orchestrator._pending_responses, {})
        finally:
            await harness.close()

    async def test_failed_automatic_sync_retries_before_model_dispatch(
        self,
    ) -> None:
        harness = _Harness(agent_type=_FailingSyncAgent)
        try:
            first = await harness.orchestrator("retry-first")
            self.assertEqual(await first.to_str(), "answer:retry-first")

            with self.assertRaisesRegex(
                RuntimeError,
                "injected assistant memory failure",
            ):
                await harness.orchestrator("retry-second")

            self.assertEqual(len(harness.manager.calls), 1)
            self.assertEqual(
                harness.recent(),
                ((MessageRole.USER, "retry-first"),),
            )
            self.assertEqual(
                tuple(harness.orchestrator._pending_responses.values()),
                (first,),
            )

            second = await harness.orchestrator("retry-second")

            self.assertEqual(len(harness.manager.calls), 2)
            self.assertEqual(
                _call_transcript(harness, 1),
                (
                    (MessageRole.USER, "retry-first"),
                    (MessageRole.ASSISTANT, "answer:retry-first"),
                    (MessageRole.USER, "retry-second"),
                ),
            )
            self.assertEqual(await second.to_str(), "answer:retry-second")

            await harness.exit()

            self.assertEqual(
                harness.recent(),
                (
                    (MessageRole.USER, "retry-first"),
                    (MessageRole.ASSISTANT, "answer:retry-first"),
                    (MessageRole.USER, "retry-second"),
                    (MessageRole.ASSISTANT, "answer:retry-second"),
                ),
            )
            self.assertEqual(harness.orchestrator._pending_responses, {})
        finally:
            await harness.close()

    async def test_active_responses_remain_isolated(self) -> None:
        harness = _Harness()
        try:
            alpha = await harness.orchestrator("alpha")
            beta = await harness.orchestrator("beta")

            self.assertEqual(
                _call_transcript(harness, 0),
                ((MessageRole.USER, "alpha"),),
            )
            self.assertEqual(
                _call_transcript(harness, 1),
                ((MessageRole.USER, "beta"),),
            )
            self.assertEqual(harness.recent(), ())
            self.assertEqual(
                tuple(harness.orchestrator._pending_responses.values()),
                (alpha, beta),
            )

            self.assertEqual(
                await gather(alpha.to_str(), beta.to_str()),
                ["answer:alpha", "answer:beta"],
            )
            await harness.exit()

            self.assertEqual(
                harness.recent(),
                (
                    (MessageRole.USER, "alpha"),
                    (MessageRole.ASSISTANT, "answer:alpha"),
                    (MessageRole.USER, "beta"),
                    (MessageRole.ASSISTANT, "answer:beta"),
                ),
            )
            self.assertEqual(harness.orchestrator._pending_responses, {})
        finally:
            await harness.close()

    async def test_blocked_active_sync_does_not_block_new_root(
        self,
    ) -> None:
        harness = _Harness(agent_type=_BlockingSyncAgent)
        agent = cast(_BlockingSyncAgent, harness.agent)
        first_sync = None
        duplicate_sync = None
        try:
            active = await harness.orchestrator("active")
            first_sync = create_task(
                harness.orchestrator.sync_messages(active)
            )
            await wait_for(agent.sync_started.wait(), timeout=1)
            duplicate_sync = create_task(
                harness.orchestrator.sync_messages(active)
            )

            unrelated = await wait_for(
                harness.orchestrator("unrelated"),
                timeout=1,
            )

            self.assertFalse(first_sync.done())
            self.assertFalse(duplicate_sync.done())
            self.assertEqual(
                _call_transcript(harness, 1),
                ((MessageRole.USER, "unrelated"),),
            )
            self.assertEqual(harness.recent(), ())

            agent.release_sync.set()
            await gather(first_sync, duplicate_sync)

            self.assertEqual(
                agent.sync_attempts,
                [(MessageRole.USER, "active")],
            )
            self.assertEqual(
                tuple(harness.orchestrator._pending_responses.values()),
                (active, unrelated),
            )
            self.assertEqual(
                await gather(active.to_str(), unrelated.to_str()),
                ["answer:active", "answer:unrelated"],
            )

            await harness.exit()

            self.assertEqual(
                harness.recent(),
                (
                    (MessageRole.USER, "active"),
                    (MessageRole.ASSISTANT, "answer:active"),
                    (MessageRole.USER, "unrelated"),
                    (MessageRole.ASSISTANT, "answer:unrelated"),
                ),
            )
            self.assertEqual(harness.orchestrator._pending_responses, {})
        finally:
            agent.release_sync.set()
            pending = tuple(
                task
                for task in (first_sync, duplicate_sync)
                if task is not None and not task.done()
            )
            if pending:
                await gather(*pending, return_exceptions=True)
            await harness.close()


if __name__ == "__main__":
    from unittest import main

    main()

"""Cover orchestration convergence boundaries with behavioral state probes."""

from asyncio import CancelledError, Lock
from types import SimpleNamespace
from typing import Any, cast
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import AsyncMock, patch

from avalan.agent.execution import AgentExecution, AgentExecutionStatus
from avalan.agent.orchestrator import Orchestrator
from avalan.entities import Message, MessageRole
from avalan.model.response.text import TextGenerationResponse


class _SyncResponse:
    """Expose terminal transitions observed by orchestration ownership."""

    def __init__(self, *, terminalize_on_sync: bool = False) -> None:
        self.execution = SimpleNamespace(status=AgentExecutionStatus.RUNNING)
        self.ownership_cleanup_complete = True
        self.sync_calls = 0
        self.terminalize_on_sync = terminalize_on_sync

    async def sync_messages(self) -> None:
        """Record synchronization and optionally become terminal."""
        self.sync_calls += 1
        if self.terminalize_on_sync:
            self.execution.status = AgentExecutionStatus.COMPLETED


def _ownership_orchestrator() -> Orchestrator:
    """Return a minimally initialized owner for private state transitions."""
    orchestrator = object.__new__(Orchestrator)
    orchestrator._pending_responses_lock = Lock()
    orchestrator._pending_responses = {}
    orchestrator._pending_provider_cleanups = {}
    return orchestrator


class OrchestratorExecutionMessageCoverageTest(TestCase):
    """Exercise each supported invocation input representation."""

    def test_execution_messages_normalize_text_and_homogeneous_lists(
        self,
    ) -> None:
        first = Message(role=MessageRole.USER, content="first")
        second = Message(role=MessageRole.ASSISTANT, content="second")

        text_messages = Orchestrator._execution_messages("hello")
        message_list = Orchestrator._execution_messages([first, second])
        text_list = Orchestrator._execution_messages(["one", "two"])

        self.assertEqual(len(text_messages), 1)
        self.assertEqual(text_messages[0].role, MessageRole.USER)
        self.assertEqual(text_messages[0].content, "hello")
        self.assertEqual(message_list, (first, second))
        self.assertEqual(
            tuple(message.content for message in text_list),
            ("one", "two"),
        )
        self.assertTrue(
            all(message.role is MessageRole.USER for message in text_list)
        )


class OrchestratorOwnershipCoverageTest(IsolatedAsyncioTestCase):
    """Exercise interrupted settlement and ownership synchronization."""

    async def test_cancelled_cleanup_task_is_attached_to_primary_failure(
        self,
    ) -> None:
        orchestrator = _ownership_orchestrator()
        primary = RuntimeError("wrapper construction failed")
        cleanup = AsyncMock(side_effect=CancelledError())

        with patch.object(
            orchestrator,
            "_cleanup_unowned_provider_response",
            cleanup,
        ):
            await orchestrator._settle_unowned_provider_response(
                cast(TextGenerationResponse, object()),
                cast(AgentExecution, object()),
                primary,
                cancelled=False,
            )

        self.assertEqual(cleanup.await_count, 1)
        self.assertEqual(
            getattr(primary, "__notes__", ()),
            ["post-provider cleanup failure: CancelledError: "],
        )

    async def test_sync_owned_only_skips_unowned_response(self) -> None:
        orchestrator = _ownership_orchestrator()
        response = _SyncResponse()

        await orchestrator._sync_owned_response(
            cast(Any, response),
            owned_only=True,
        )
        await orchestrator._sync_owned_response(
            cast(Any, response),
            owned_only=False,
        )

        self.assertEqual(response.sync_calls, 1)
        self.assertEqual(orchestrator._pending_responses, {})

    async def test_sync_rechecks_new_terminal_before_release(self) -> None:
        orchestrator = _ownership_orchestrator()
        response = _SyncResponse(terminalize_on_sync=True)
        orchestrator._pending_responses[id(response)] = cast(Any, response)

        await orchestrator._sync_owned_response(
            cast(Any, response),
            owned_only=True,
        )

        self.assertEqual(response.sync_calls, 2)
        self.assertEqual(
            response.execution.status,
            AgentExecutionStatus.COMPLETED,
        )
        self.assertNotIn(id(response), orchestrator._pending_responses)

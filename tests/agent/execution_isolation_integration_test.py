"""Exercise invocation isolation through a real agent and orchestrator."""

from asyncio import Event, gather
from dataclasses import asdict, dataclass
from json import dumps
from logging import getLogger
from types import MappingProxyType, SimpleNamespace
from typing import Any, cast
from unittest import IsolatedAsyncioTestCase
from uuid import uuid4

from avalan.agent import (
    AgentOperation,
    EngineEnvironment,
    Specification,
)
from avalan.agent.engine import EngineAgent
from avalan.agent.execution import AgentExecution, ExecutionIdFactory
from avalan.agent.orchestrator import Orchestrator
from avalan.entities import (
    EngineMessage,
    EngineUri,
    Message,
    MessageRole,
    MessageToolCall,
    TransformerEngineSettings,
)
from avalan.event.manager import EventManager, EventManagerMode
from avalan.interaction.entities import (
    BranchId,
    ModelCallId,
    RunId,
    StreamSessionId,
    TaskId,
    TurnId,
)
from avalan.memory import RecentMessageMemory
from avalan.memory.manager import MemoryManager
from avalan.model.call import ModelCall, ModelCallContext
from avalan.model.manager import ModelManager
from avalan.model.response.text import TextGenerationResponse
from avalan.tool.manager import ToolManager


class _ExecutionIds:
    """Mint deterministic, distinct identifiers for concurrent invocations."""

    def __init__(self) -> None:
        self._run = 0
        self._turn = 0
        self._model_call = 0
        self._task = 0
        self._branch = 0
        self._stream = 0

    async def new_run_id(self) -> RunId:
        """Return a distinct run identifier."""
        self._run += 1
        return RunId(f"run-{self._run}")

    async def new_turn_id(self) -> TurnId:
        """Return a distinct turn identifier."""
        self._turn += 1
        return TurnId(f"turn-{self._turn}")

    async def new_model_call_id(self) -> ModelCallId:
        """Return a distinct model-call identifier."""
        self._model_call += 1
        return ModelCallId(f"model-call-{self._model_call}")

    async def new_task_id(self) -> TaskId:
        """Return a distinct task identifier."""
        self._task += 1
        return TaskId(f"task-{self._task}")

    async def new_branch_id(self) -> BranchId:
        """Return a distinct branch identifier."""
        self._branch += 1
        return BranchId(f"branch-{self._branch}")

    async def new_stream_session_id(self) -> StreamSessionId:
        """Return a distinct stream identifier."""
        self._stream += 1
        return StreamSessionId(f"stream-{self._stream}")


class _Engine:
    """Expose the model surface consumed by the real engine agent."""

    model_id = "isolation-model"
    model_type = "fake"

    def __init__(self) -> None:
        self.tokenizer = SimpleNamespace(eos_token="<isolated-eos>")


class _Agent(EngineAgent):
    """Prepare prompts while retaining the production execution path."""

    def _prepare_call(self, context: ModelCallContext) -> dict[str, object]:
        return {"instructions": context.specification.instructions}


def _message_text(input_value: object) -> str:
    """Return the final plain-text message in one model call."""
    messages = (
        [input_value]
        if isinstance(input_value, Message)
        else cast(list[Message], input_value)
    )
    content = messages[-1].content
    assert isinstance(content, str)
    return content


def _text_response(text: str) -> TextGenerationResponse:
    """Return one repo-native non-stream model response."""
    return TextGenerationResponse(
        lambda: text,
        logger=getLogger(),
        use_async_generator=False,
    )


class _ConcurrentModelManager:
    """Overlap the first two real model calls before returning results."""

    def __init__(self) -> None:
        self.calls: list[ModelCall] = []
        self._both_started = Event()
        self.overlap = False
        self.outputs: dict[str, str] = {}

    async def __call__(self, call: ModelCall) -> TextGenerationResponse:
        self.calls.append(call)
        if self.overlap and len(self.calls) <= 2:
            if len(self.calls) == 2:
                self._both_started.set()
            await self._both_started.wait()
        prompt = _message_text(call.context.input)
        return _text_response(self.outputs.get(prompt, f"answer:{prompt}"))


@dataclass(frozen=True, slots=True)
class _Invocation:
    """Capture public results and task-local execution observations."""

    execution: AgentExecution
    text: str
    prompt_text: str
    output: str | None


class ExecutionIsolationIntegrationTest(IsolatedAsyncioTestCase):
    """Exercise shared-instance isolation at real invocation boundaries."""

    async def asyncSetUp(self) -> None:
        self.logger = getLogger()
        self.event_manager = EventManager(mode=EventManagerMode.TEST)
        self.memory = MemoryManager(
            agent_id=uuid4(),
            participant_id=uuid4(),
            permanent_message_memory=None,
            recent_message_memory=RecentMessageMemory(),
            text_partitioner=None,
            logger=self.logger,
            event_manager=self.event_manager,
        )
        self.tool = ToolManager.create_instance()
        self.engine = _Engine()
        self.model_manager = _ConcurrentModelManager()
        engine_uri = EngineUri(
            host=None,
            port=None,
            user=None,
            password=None,
            vendor=None,
            model_id=self.engine.model_id,
            params={},
        )
        environment = EngineEnvironment(
            engine_uri=engine_uri,
            settings=TransformerEngineSettings(),
        )
        self.operations = [
            AgentOperation(
                specification=Specification(instructions="first operation"),
                environment=environment,
            ),
            AgentOperation(
                specification=Specification(instructions="second operation"),
                environment=environment,
            ),
        ]
        self.agent = _Agent(
            cast(Any, self.engine),
            self.memory,
            self.tool,
            self.event_manager,
            cast(ModelManager, self.model_manager),
            engine_uri,
        )
        self.orchestrator = Orchestrator(
            self.logger,
            cast(ModelManager, self.model_manager),
            self.memory,
            self.tool,
            self.event_manager,
            self.operations,
        )
        environment_hash = dumps(asdict(environment))
        self.orchestrator._engine_agents[environment_hash] = self.agent
        self.ids = _ExecutionIds()

    async def asyncTearDown(self) -> None:
        await self.event_manager.aclose()

    async def _invoke(self, prompt: str, operation_index: int) -> _Invocation:
        response = await self.orchestrator(
            prompt,
            operation_index=operation_index,
            execution_id_factory=cast(ExecutionIdFactory, self.ids),
        )
        execution = response.execution
        assert execution is not None
        text = await response.to_str()
        last_prompt = execution.last_prompt
        assert last_prompt is not None
        prompt_text = _message_text(last_prompt.input)
        return _Invocation(
            execution=execution,
            text=text,
            prompt_text=prompt_text,
            output=execution.last_response,
        )

    async def test_concurrent_invocations_are_fully_isolated(self) -> None:
        self.model_manager.overlap = True
        alpha, beta = await gather(
            self._invoke("alpha", 1),
            self._invoke("beta", 1),
        )

        self.assertEqual(alpha.text, "answer:alpha")
        self.assertEqual(beta.text, "answer:beta")
        self.assertEqual(alpha.prompt_text, "alpha")
        self.assertEqual(beta.prompt_text, "beta")
        self.assertEqual(alpha.output, alpha.execution.last_response)
        self.assertEqual(beta.output, beta.execution.last_response)
        self.assertIsNot(alpha.execution, beta.execution)

        origins = (
            alpha.execution.initial_origin,
            beta.execution.initial_origin,
        )
        for field_name in (
            "run_id",
            "turn_id",
            "model_call_id",
            "task_id",
            "branch_id",
            "stream_session_id",
        ):
            self.assertEqual(
                len({getattr(origin, field_name) for origin in origins}),
                2,
                field_name,
            )
        self.assertEqual(
            tuple(message.content for message in alpha.execution.messages),
            ("alpha", "answer:alpha"),
        )
        self.assertEqual(
            tuple(message.content for message in beta.execution.messages),
            ("beta", "answer:beta"),
        )
        self.assertEqual(alpha.execution.definition, beta.execution.definition)
        self.assertEqual(alpha.execution.operation_index, 1)
        self.assertEqual(beta.execution.operation_index, 1)

    async def test_definition_sync_and_tool_configuration_are_stable(
        self,
    ) -> None:
        seed_before = self.tool.export_model_capability_seed()
        tokenizer_before = vars(self.engine.tokenizer).copy()
        invocation = await self._invoke("synchronize", 0)

        self.assertEqual(invocation.execution.operation_index, 0)
        same_operation = await self._invoke("repeat", 0)
        self.assertEqual(
            invocation.execution.definition,
            same_operation.execution.definition,
        )
        other_operation = await self._invoke("other", 1)
        self.assertNotEqual(
            invocation.execution.operation_id,
            other_operation.execution.operation_id,
        )
        self.assertEqual(self.tool.export_model_capability_seed(), seed_before)
        self.assertEqual(vars(self.engine.tokenizer), tokenizer_before)

        await self.agent.sync_messages(invocation.execution)
        synced = tuple(self.memory.recent_messages or ())
        await self.agent.sync_messages(invocation.execution)
        self.assertEqual(tuple(self.memory.recent_messages or ()), synced)
        self.assertEqual(
            tuple(item.message.content for item in synced),
            (
                "synchronize",
                "answer:synchronize",
                "repeat",
                "answer:repeat",
            ),
        )

    async def test_prior_history_hydrates_prompt_without_resyncing_prefix(
        self,
    ) -> None:
        prior = (
            Message(role=MessageRole.USER, content="prior question"),
            Message(role=MessageRole.ASSISTANT, content="prior answer"),
        )
        for message in prior:
            await self.memory.append_message(
                EngineMessage(
                    agent_id=uuid4(),
                    model_id=self.engine.model_id,
                    message=message,
                )
            )

        invocation = await self._invoke("follow up", 0)
        prompt = cast(
            list[Message], self.model_manager.calls[-1].operation.input
        )
        self.assertEqual(
            tuple(message.content for message in prompt),
            ("prior question", "prior answer", "follow up"),
        )
        self.assertEqual(
            tuple(
                message.content for message in invocation.execution.messages
            ),
            (
                "prior question",
                "prior answer",
                "follow up",
                "answer:follow up",
            ),
        )

        await self.agent.sync_messages(invocation.execution)
        synchronized = tuple(self.memory.recent_messages or ())
        await self.agent.sync_messages(invocation.execution)

        self.assertEqual(
            tuple(self.memory.recent_messages or ()), synchronized
        )
        self.assertEqual(
            tuple(item.message.content for item in synchronized),
            (
                "prior question",
                "prior answer",
                "follow up",
                "answer:follow up",
            ),
        )

    async def test_prior_history_recursively_snapshots_mapping_proxies(
        self,
    ) -> None:
        prior = Message(
            role=MessageRole.ASSISTANT,
            content="prior tool request",
            tool_calls=[
                MessageToolCall(
                    name="request_user_input",
                    arguments=cast(
                        Any,
                        MappingProxyType(
                            {
                                "question": MappingProxyType(
                                    {"prompt": "Continue?"}
                                )
                            }
                        ),
                    ),
                )
            ],
        )
        await self.memory.append_message(
            EngineMessage(
                agent_id=uuid4(),
                model_id=self.engine.model_id,
                message=prior,
            )
        )

        invocation = await self._invoke("follow mapping", 0)
        call_messages = cast(
            list[Message], self.model_manager.calls[-1].operation.input
        )
        snapshotted_calls = call_messages[0].tool_calls
        assert snapshotted_calls is not None

        self.assertEqual(
            snapshotted_calls[0].arguments,
            {"question": {"prompt": "Continue?"}},
        )
        self.assertIsNot(call_messages[0], prior)
        self.assertEqual(invocation.text, "answer:follow mapping")

    async def test_concurrent_prompts_share_only_the_prior_snapshot(
        self,
    ) -> None:
        prior = Message(role=MessageRole.USER, content="shared prior")
        await self.memory.append_message(
            EngineMessage(
                agent_id=uuid4(),
                model_id=self.engine.model_id,
                message=prior,
            )
        )
        self.model_manager.overlap = True

        alpha, beta = await gather(
            self._invoke("alpha with history", 0),
            self._invoke("beta with history", 0),
        )

        prompts = {
            cast(
                str, cast(list[Message], call.operation.input)[-1].content
            ): tuple(
                message.content
                for message in cast(list[Message], call.operation.input)
            )
            for call in self.model_manager.calls
        }
        self.assertEqual(
            prompts,
            {
                "alpha with history": ("shared prior", "alpha with history"),
                "beta with history": ("shared prior", "beta with history"),
            },
        )
        await self.agent.sync_messages(alpha.execution)
        await self.agent.sync_messages(beta.execution)
        synchronized = tuple(
            item.message.content for item in self.memory.recent_messages or ()
        )
        self.assertEqual(synchronized.count("shared prior"), 1)
        self.assertEqual(synchronized.count("alpha with history"), 1)
        self.assertEqual(synchronized.count("beta with history"), 1)

    async def test_empty_model_output_is_not_added_to_memory(self) -> None:
        self.model_manager.outputs["empty"] = ""
        invocation = await self._invoke("empty", 0)

        await self.agent.sync_messages(invocation.execution)

        self.assertEqual(
            tuple(
                item.message.content
                for item in self.memory.recent_messages or ()
            ),
            ("empty",),
        )


if __name__ == "__main__":
    from unittest import main

    main()

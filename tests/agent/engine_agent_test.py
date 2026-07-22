from collections.abc import AsyncIterator
from dataclasses import replace
from logging import getLogger
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, MagicMock

from avalan.agent import Specification
from avalan.agent.engine import EngineAgent
from avalan.entities import (
    ChatSettings,
    EngineMessage,
    EngineUri,
    GenerationSettings,
    Message,
    MessageRole,
    ReasoningEffort,
    ReasoningSettings,
    ReasoningTag,
)
from avalan.event import EventType
from avalan.event.manager import EventManager
from avalan.model import TextGenerationResponse
from avalan.model.call import ModelCallContext
from avalan.model.capability import ModelCapabilityCatalog
from avalan.model.manager import ModelManager
from avalan.model.stream import (
    CanonicalStreamItem,
    StreamChannel,
    StreamItemCorrelation,
    StreamItemKind,
    StreamProviderEvent,
    StreamReasoningRepresentation,
    StreamTerminalOutcome,
    StreamVisibility,
    TextGenerationNonStreamResult,
)
from avalan.tool.manager import ToolManager


class DummyEngine:
    def __init__(self) -> None:
        self.model_id = "m"
        self.model_type = "t"
        self.called_with = None
        self.input_token_count = MagicMock(return_value=3)

    async def __call__(self, input, **kwargs):
        self.called_with = (input, kwargs)
        return "out"


class DummyAgent(EngineAgent):
    def _prepare_call(self, context: ModelCallContext):
        return {}


class FakeMemory:
    def __init__(self) -> None:
        self.has_permanent_message = False
        self.has_recent_message = True
        self.recent_message = object()
        self.recent_messages: list[EngineMessage] = []

    async def append_message(self, message: EngineMessage) -> None:
        self.recent_messages.append(message)


class EngineAgentPropertyTestCase(IsolatedAsyncioTestCase):
    def setUp(self):
        self.memory = MagicMock()
        self.engine = DummyEngine()
        self.tool = MagicMock(spec=ToolManager)
        self.tool.export_model_capability_seed.return_value = (
            ToolManager.create_instance().export_model_capability_seed()
        )
        self.event_manager = MagicMock(spec=EventManager)
        self.event_manager.trigger = AsyncMock()
        self.model_manager = AsyncMock(spec=ModelManager)
        self.model_manager.return_value = "out"
        self.engine_uri = EngineUri(
            host=None,
            port=None,
            user=None,
            password=None,
            vendor=None,
            model_id="m",
            params={},
        )
        self.agent = DummyAgent(
            self.engine,
            self.memory,
            self.tool,
            self.event_manager,
            self.model_manager,
            self.engine_uri,
        )

    async def test_memory_and_engine_property(self):
        self.assertIs(self.agent.memory, self.memory)
        self.assertIs(self.agent.engine, self.engine)

    async def test_input_token_count_no_prompt(self):
        result = await self.agent.input_token_count()
        self.assertIsNone(result)
        self.engine.input_token_count.assert_not_called()
        self.event_manager.trigger.assert_not_called()

    async def test_input_token_count_with_prompt(self):
        self.agent._last_prompt = ("hi", "inst", "sys", None)
        result = await self.agent.input_token_count()
        self.assertEqual(result, 3)
        self.engine.input_token_count.assert_called_once_with(
            "hi",
            instructions="inst",
            system_prompt="sys",
            developer_prompt=None,
        )
        called_types = [
            c.args[0].type for c in self.event_manager.trigger.await_args_list
        ]
        self.assertIn(EventType.INPUT_TOKEN_COUNT_BEFORE, called_types)
        self.assertIn(EventType.INPUT_TOKEN_COUNT_AFTER, called_types)


class EngineAgentRunTestCase(IsolatedAsyncioTestCase):
    def _make_agent(self, last_output=None, params=None):
        memory = FakeMemory()
        engine = DummyEngine()
        tool = MagicMock(spec=ToolManager)
        tool.export_model_capability_seed.return_value = (
            ToolManager.create_instance().export_model_capability_seed()
        )
        event_manager = MagicMock(spec=EventManager)
        event_manager.trigger = AsyncMock()
        model_manager = AsyncMock(spec=ModelManager)
        model_manager.return_value = "out"
        engine_uri = EngineUri(
            host=None,
            port=None,
            user=None,
            password=None,
            vendor=None,
            model_id="m",
            params=params or {},
        )
        agent = DummyAgent(
            engine,
            memory,
            tool,
            event_manager,
            model_manager,
            engine_uri,
        )
        agent._last_output = last_output
        return agent, engine, memory, model_manager

    def _make_context(self, input_value: Message | str):
        return ModelCallContext(
            specification=Specification(role=None, goal=None),
            input=input_value,
        )

    async def test_run_with_settings_and_previous_response(self):
        last_response = TextGenerationResponse(
            lambda: "prev", logger=getLogger(), use_async_generator=False
        )
        agent, engine, memory, manager = self._make_agent(last_response)

        settings = GenerationSettings(max_new_tokens=1)
        message = Message(role=MessageRole.USER, content="hi")
        context = self._make_context(message)
        await agent._run(
            context,
            message,
            settings=settings,
            top_p=0.7,
        )

        self.assertEqual(len(memory.recent_messages), 2)
        self.assertEqual(
            memory.recent_messages[0].message.role, MessageRole.ASSISTANT
        )
        self.assertEqual(
            memory.recent_messages[1].message.role, MessageRole.USER
        )
        prompt = agent.last_prompt
        self.assertIsNotNone(prompt)
        self.assertEqual(
            [message.role for message in prompt[0]],
            [MessageRole.ASSISTANT, MessageRole.USER],
        )

        manager.assert_awaited_once()
        task = manager.await_args.args[0]
        self.assertEqual(task.engine_uri, agent.engine_uri)
        self.assertIs(task.model, engine)
        self.assertEqual(
            task.operation.generation_settings,
            replace(settings, top_p=0.7),
        )
        self.assertEqual(replace(task.context, capability=None), context)
        self.assertIs(task.context.capability, task.capability)
        self.assertEqual(agent._last_output, "out")

    async def test_run_with_settings_no_previous_response(self):
        agent, engine, memory, manager = self._make_agent()
        settings = GenerationSettings(max_new_tokens=1)
        message = Message(role=MessageRole.USER, content="hi")
        context = self._make_context(message)
        await agent._run(
            context,
            message,
            settings=settings,
            top_p=0.7,
        )

        self.assertEqual(len(memory.recent_messages), 1)
        self.assertEqual(
            memory.recent_messages[0].message.role, MessageRole.USER
        )
        manager.assert_awaited_once()
        task = manager.await_args.args[0]
        self.assertEqual(
            task.operation.generation_settings, replace(settings, top_p=0.7)
        )
        self.assertEqual(replace(task.context, capability=None), context)
        self.assertIs(task.context.capability, task.capability)

    async def test_run_preserves_context_capability_identity(self) -> None:
        agent, _engine, _memory, manager = self._make_agent()
        capability = ModelCapabilityCatalog.create()
        message = Message(role=MessageRole.USER, content="hi")
        context = ModelCallContext(
            specification=Specification(role=None, goal=None),
            input=message,
            capability=capability,
        )
        agent._tool.export_model_capability_seed.reset_mock()

        await agent._run(context, message)

        task = manager.await_args.args[0]
        self.assertIs(task.context, context)
        self.assertIs(task.capability, capability)
        agent._tool.export_model_capability_seed.assert_not_called()

    async def test_run_keeps_instructions_distinct_from_messages(self):
        agent, _engine, _memory, manager = self._make_agent()
        message = Message(role=MessageRole.USER, content="hi")
        context = self._make_context(message)

        await agent._run(
            context,
            message,
            instructions="provider instructions",
            system_prompt="system prompt",
            developer_prompt="developer prompt",
        )

        task = manager.await_args.args[0]
        text = task.operation.parameters["text"]
        self.assertEqual(text.instructions, "provider instructions")
        self.assertEqual(text.system_prompt, "system prompt")
        self.assertEqual(text.developer_prompt, "developer prompt")
        self.assertEqual(task.operation.input, [message])
        self.assertNotIn("provider instructions", str(task.operation.input))

    async def test_run_kwargs_only_with_previous_response(self):
        last_response = TextGenerationResponse(
            lambda: "prev", logger=getLogger(), use_async_generator=False
        )
        agent, engine, memory, manager = self._make_agent(last_response)

        message = Message(role=MessageRole.USER, content="hi")
        context = self._make_context(message)
        await agent._run(context, message, temperature=0.4)

        self.assertEqual(len(memory.recent_messages), 2)
        self.assertEqual(
            memory.recent_messages[0].message.role, MessageRole.ASSISTANT
        )
        self.assertEqual(
            memory.recent_messages[1].message.role, MessageRole.USER
        )
        manager.assert_awaited_once()
        task = manager.await_args.args[0]
        self.assertEqual(task.operation.generation_settings.temperature, 0.4)
        self.assertFalse(task.operation.generation_settings.do_sample)
        self.assertEqual(replace(task.context, capability=None), context)
        self.assertIs(task.context.capability, task.capability)

    async def test_run_kwargs_only_no_previous_response(self):
        agent, engine, memory, manager = self._make_agent()
        message = Message(role=MessageRole.USER, content="hi")
        context = self._make_context(message)
        await agent._run(context, message, temperature=0.4)

        self.assertEqual(len(memory.recent_messages), 1)
        self.assertEqual(
            memory.recent_messages[0].message.role, MessageRole.USER
        )
        manager.assert_awaited_once()
        task = manager.await_args.args[0]
        self.assertEqual(task.operation.generation_settings.temperature, 0.4)
        self.assertFalse(task.operation.generation_settings.do_sample)
        self.assertEqual(replace(task.context, capability=None), context)
        self.assertIs(task.context.capability, task.capability)

    async def test_run_defaults_from_uri(self):
        agent, engine, _, manager = self._make_agent(
            params={"temperature": 0.6, "max_new_tokens": 5}
        )
        message = Message(role=MessageRole.USER, content="hi")
        context = self._make_context(message)
        await agent._run(context, message)
        manager.assert_awaited_once()
        settings = manager.await_args.args[0].operation.generation_settings
        self.assertEqual(settings.temperature, 0.6)
        self.assertEqual(settings.max_new_tokens, 5)

    async def test_run_settings_overridden_by_uri(self):
        agent, engine, _, manager = self._make_agent(params={"top_p": 0.5})
        message = Message(role=MessageRole.USER, content="hi")
        context = self._make_context(message)
        await agent._run(
            context,
            message,
            settings=GenerationSettings(),
        )
        manager.assert_awaited_once()
        used_settings = manager.await_args.args[
            0
        ].operation.generation_settings
        self.assertEqual(used_settings.top_p, 0.5)

    async def test_run_normalizes_nested_settings_from_dicts(self):
        agent, _, _, manager = self._make_agent()
        message = Message(role=MessageRole.USER, content="hi")
        context = self._make_context(message)

        await agent._run(
            context,
            message,
            chat_settings={"enable_thinking": False},
            reasoning={"effort": "xhigh", "tag": "think"},
        )

        manager.assert_awaited_once()
        used_settings = manager.await_args.args[
            0
        ].operation.generation_settings
        self.assertEqual(
            used_settings.chat_settings,
            ChatSettings(enable_thinking=False),
        )
        self.assertEqual(
            used_settings.reasoning,
            ReasoningSettings(
                effort=ReasoningEffort.XHIGH,
                tag=ReasoningTag.THINK,
            ),
        )

    async def test_run_with_list_of_strings_converts_to_user_messages(self):
        agent, _, memory, manager = self._make_agent()
        context = self._make_context("unused")

        await agent._run(context, ["hello", "world"])

        self.assertEqual(len(memory.recent_messages), 2)
        first = memory.recent_messages[0].message
        second = memory.recent_messages[1].message
        self.assertEqual(first.role, MessageRole.USER)
        self.assertEqual(first.content, "hello")
        self.assertEqual(second.role, MessageRole.USER)
        self.assertEqual(second.content, "world")
        manager.assert_awaited_once()

    async def test_run_child_context_uses_explicit_messages(self) -> None:
        last_response = TextGenerationResponse(
            lambda: "tool call",
            logger=getLogger(),
            use_async_generator=False,
        )
        agent, _, memory, manager = self._make_agent(last_response)
        message = Message(role=MessageRole.USER, content="hi")
        capability = ModelCapabilityCatalog.create()
        parent_context = replace(
            self._make_context(message), capability=capability
        )
        child_context = ModelCallContext(
            specification=parent_context.specification,
            input=[message],
            capability=capability,
            parent=parent_context,
        )
        agent._tool.export_model_capability_seed.reset_mock()

        await agent._run(child_context, [message])

        self.assertEqual(memory.recent_messages, [])
        task = manager.await_args.args[0]
        self.assertEqual(task.operation.input, [message])
        self.assertIs(task.context, child_context)
        self.assertIs(task.capability, capability)
        agent._tool.export_model_capability_seed.assert_not_called()
        self.assertEqual(agent.last_prompt, ([message], None, None, None))
        self.assertEqual(agent._last_output, "out")

    async def test_sync_messages_appends_last_output_when_memory_enabled(
        self,
    ) -> None:
        agent, _, memory, _ = self._make_agent()
        agent._last_output = TextGenerationResponse(
            lambda: "assistant",
            logger=getLogger(),
            use_async_generator=False,
        )

        await agent.sync_messages()

        self.assertEqual(len(memory.recent_messages), 1)
        synced = memory.recent_messages[0].message
        self.assertEqual(synced.role, MessageRole.ASSISTANT)
        self.assertEqual(synced.content, "assistant")

    async def test_sync_messages_keeps_rich_reasoning_out_of_memory(
        self,
    ) -> None:
        summary = "private summary sentinel"
        answer = '{"answer":true}'
        terminals = (
            StreamItemKind.STREAM_COMPLETED,
            StreamItemKind.STREAM_ERRORED,
            StreamItemKind.STREAM_CANCELLED,
        )
        for terminal in terminals:
            events = (
                StreamProviderEvent(
                    kind=StreamItemKind.REASONING_DELTA,
                    text_delta=summary,
                    correlation=StreamItemCorrelation(
                        protocol_item_id="reasoning-memory",
                        provider_output_index=0,
                        provider_summary_index=0,
                    ),
                    visibility=StreamVisibility.PRIVATE,
                    reasoning_representation=(
                        StreamReasoningRepresentation.SUMMARY
                    ),
                    segment_instance_ordinal=0,
                ),
                StreamProviderEvent(
                    kind=StreamItemKind.ANSWER_DELTA,
                    text_delta=answer,
                ),
                StreamProviderEvent(kind=StreamItemKind.ANSWER_DONE),
                StreamProviderEvent(
                    kind=terminal,
                    data=(
                        {"message": "provider failed"}
                        if terminal is StreamItemKind.STREAM_ERRORED
                        else None
                    ),
                ),
            )
            result = TextGenerationNonStreamResult(
                events,
                answer_text=answer,
                provider_family="openai",
            )
            response = TextGenerationResponse(
                result,
                logger=getLogger(),
                use_async_generator=False,
            )
            agent, _, memory, _ = self._make_agent(response)

            await agent.sync_messages()

            self.assertEqual(len(memory.recent_messages), 1)
            synced = memory.recent_messages[0].message
            self.assertEqual(synced.role, MessageRole.ASSISTANT)
            self.assertEqual(synced.content, answer)
            self.assertNotIn(summary, str(synced.content))

    async def test_sync_messages_appends_partial_output_for_errored_stream(
        self,
    ) -> None:
        async def output_gen() -> AsyncIterator[CanonicalStreamItem]:
            yield CanonicalStreamItem(
                stream_session_id="engine-agent-stream",
                run_id="engine-agent-run",
                turn_id="engine-agent-turn",
                sequence=0,
                kind=StreamItemKind.STREAM_STARTED,
                channel=StreamChannel.CONTROL,
            )
            yield CanonicalStreamItem(
                stream_session_id="engine-agent-stream",
                run_id="engine-agent-run",
                turn_id="engine-agent-turn",
                sequence=1,
                kind=StreamItemKind.ANSWER_DELTA,
                channel=StreamChannel.ANSWER,
                text_delta="partial",
            )
            yield CanonicalStreamItem(
                stream_session_id="engine-agent-stream",
                run_id="engine-agent-run",
                turn_id="engine-agent-turn",
                sequence=2,
                kind=StreamItemKind.ANSWER_DONE,
                channel=StreamChannel.ANSWER,
            )
            yield CanonicalStreamItem(
                stream_session_id="engine-agent-stream",
                run_id="engine-agent-run",
                turn_id="engine-agent-turn",
                sequence=3,
                kind=StreamItemKind.STREAM_ERRORED,
                channel=StreamChannel.CONTROL,
                data={"message": "provider failed"},
                terminal_outcome=StreamTerminalOutcome.ERRORED,
            )
            yield CanonicalStreamItem(
                stream_session_id="engine-agent-stream",
                run_id="engine-agent-run",
                turn_id="engine-agent-turn",
                sequence=4,
                kind=StreamItemKind.STREAM_CLOSED,
                channel=StreamChannel.CONTROL,
            )

        agent, _, memory, _ = self._make_agent()
        response = TextGenerationResponse(
            lambda: output_gen(),
            logger=getLogger(),
            use_async_generator=True,
        )
        _ = [item async for item in response]
        agent._last_output = response

        with self.assertRaisesRegex(RuntimeError, "provider failed"):
            await response.to_str()

        await agent.sync_messages()

        self.assertEqual(len(memory.recent_messages), 1)
        synced = memory.recent_messages[0].message
        self.assertEqual(synced.role, MessageRole.ASSISTANT)
        self.assertEqual(synced.content, "partial")


class EngineAgentCallTestCase(IsolatedAsyncioTestCase):
    def setUp(self):
        self.memory = MagicMock()
        self.engine = DummyEngine()
        self.tool = MagicMock(spec=ToolManager)
        self.tool.export_model_capability_seed.return_value = (
            ToolManager.create_instance().export_model_capability_seed()
        )
        self.event_manager = MagicMock(spec=EventManager)
        self.event_manager.trigger = AsyncMock()
        self.model_manager = AsyncMock(spec=ModelManager)
        self.engine_uri = EngineUri(
            host=None,
            port=None,
            user=None,
            password=None,
            vendor=None,
            model_id="m",
            params={},
        )
        self.agent = DummyAgent(
            self.engine,
            self.memory,
            self.tool,
            self.event_manager,
            self.model_manager,
            self.engine_uri,
        )

    async def test_call_sets_root_parent_when_missing(self):
        specification = Specification(role=None, goal=None)
        root_parent = ModelCallContext(
            specification=specification,
            input="root",
        )
        parent_context = ModelCallContext(
            specification=specification,
            input="parent",
            root_parent=root_parent,
        )
        child_context = ModelCallContext(
            specification=specification,
            input="child",
            parent=parent_context,
        )

        run_args = {"temperature": 0.25}
        self.agent._prepare_call = MagicMock(return_value=run_args)
        self.agent._run = AsyncMock(return_value="call-result")

        result = await self.agent(child_context)

        self.assertEqual(result, "call-result")
        self.agent._prepare_call.assert_called_once()
        prepared_context = self.agent._prepare_call.call_args.args[0]
        self.assertIs(prepared_context.root_parent, root_parent)

        self.agent._run.assert_awaited_once()
        run_context = self.agent._run.await_args.args[0]
        self.assertIs(run_context.root_parent, root_parent)
        self.assertIs(run_context.parent, parent_context)
        self.assertIsNotNone(run_context.capability)
        self.assertIs(prepared_context.capability, run_context.capability)
        self.assertEqual(self.agent._run.await_args.kwargs, run_args)
        self.assertIsNone(child_context.root_parent)
        self.assertEqual(self.event_manager.trigger.await_count, 4)

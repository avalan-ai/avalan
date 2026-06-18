from logging import getLogger
from unittest import IsolatedAsyncioTestCase
from unittest.mock import MagicMock

from avalan.agent import AgentOperation, EngineEnvironment, Specification
from avalan.agent.engine import EngineAgent
from avalan.agent.orchestrator.response.orchestrator_response import (
    OrchestratorResponse,
)
from avalan.entities import (
    EngineUri,
    GenerationSettings,
    Input,
    Message,
    MessageRole,
    TransformerEngineSettings,
)
from avalan.model import TextGenerationResponse
from avalan.model.call import ModelCallContext
from avalan.model.stream import (
    CanonicalStreamItem,
    StreamChannel,
    StreamItemKind,
    StreamTerminalOutcome,
)


class _DummyEngine:
    def __init__(self) -> None:
        self.model_id = "m"
        self.tokenizer = MagicMock()


def _dummy_operation() -> AgentOperation:
    env = EngineEnvironment(
        engine_uri=EngineUri(
            host=None,
            port=None,
            user=None,
            password=None,
            vendor=None,
            model_id="m",
            params={},
        ),
        settings=TransformerEngineSettings(),
    )
    spec = Specification(role="assistant", goal=None)
    return AgentOperation(specification=spec, environment=env)


def _dummy_response() -> TextGenerationResponse:
    async def output_gen():
        for sequence, text in enumerate(("a", "b"), start=1):
            yield CanonicalStreamItem(
                stream_session_id="step-stream",
                run_id="step-run",
                turn_id="step-turn",
                sequence=sequence,
                kind=StreamItemKind.ANSWER_DELTA,
                channel=StreamChannel.ANSWER,
                text_delta=text,
            )

    async def canonical_output_gen():
        yield CanonicalStreamItem(
            stream_session_id="step-stream",
            run_id="step-run",
            turn_id="step-turn",
            sequence=0,
            kind=StreamItemKind.STREAM_STARTED,
            channel=StreamChannel.CONTROL,
        )
        async for item in output_gen():
            yield item
        yield CanonicalStreamItem(
            stream_session_id="step-stream",
            run_id="step-run",
            turn_id="step-turn",
            sequence=3,
            kind=StreamItemKind.ANSWER_DONE,
            channel=StreamChannel.ANSWER,
        )
        yield CanonicalStreamItem(
            stream_session_id="step-stream",
            run_id="step-run",
            turn_id="step-turn",
            sequence=4,
            kind=StreamItemKind.STREAM_COMPLETED,
            channel=StreamChannel.CONTROL,
            usage={},
            terminal_outcome=StreamTerminalOutcome.COMPLETED,
        )

    settings = GenerationSettings()
    return TextGenerationResponse(
        lambda **_: canonical_output_gen(),
        logger=getLogger(),
        use_async_generator=True,
        generation_settings=settings,
        settings=settings,
    )


def _make_response(
    input_value: Input,
    response: TextGenerationResponse,
    agent: EngineAgent,
    operation: AgentOperation,
    engine_args: dict,
    **kwargs,
) -> OrchestratorResponse:
    context = ModelCallContext(
        specification=operation.specification,
        input=input_value,
        engine_args=dict(engine_args),
    )
    return OrchestratorResponse(
        input_value,
        response,
        agent,
        operation,
        engine_args,
        context,
        **kwargs,
    )


class OrchestratorResponseStepTestCase(IsolatedAsyncioTestCase):
    async def test_step_counter(self):
        engine = _DummyEngine()
        agent = MagicMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()
        resp = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            _dummy_response(),
            agent,
            operation,
            {},
        )
        resp.__aiter__()
        first = await resp.__anext__()
        self.assertIs(first.kind, StreamItemKind.STREAM_STARTED)
        self.assertEqual(resp._step, 0)
        second = await resp.__anext__()
        self.assertIs(second.kind, StreamItemKind.ANSWER_DELTA)
        self.assertEqual(resp._step, 1)
        third = await resp.__anext__()
        self.assertIs(third.kind, StreamItemKind.ANSWER_DELTA)
        self.assertEqual(resp._step, 2)
        while True:
            try:
                await resp.__anext__()
            except StopAsyncIteration:
                break
        resp.__aiter__()
        self.assertEqual(resp._step, 0)

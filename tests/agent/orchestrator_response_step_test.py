from avalan.agent.engine import EngineAgent
from avalan.agent.orchestrator.response.orchestrator_response import (
    OrchestratorResponse,
)
from avalan.agent import EngineEnvironment, AgentOperation, Specification
from avalan.entities import (
    EngineUri,
    GenerationSettings,
    Input,
    Message,
    MessageRole,
    Token,
    TransformerEngineSettings,
)
from avalan.model import TextGenerationResponse
from avalan.model.call import ModelCallContext
from logging import getLogger
from unittest import IsolatedAsyncioTestCase
from unittest.mock import MagicMock


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
        yield "a"
        yield Token(id=1, token="b")

    settings = GenerationSettings()
    return TextGenerationResponse(
        lambda **_: output_gen(),
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
        await resp.__anext__()
        self.assertEqual(resp._step, 1)
        await resp.__anext__()
        self.assertEqual(resp._step, 2)
        with self.assertRaises(StopAsyncIteration):
            await resp.__anext__()
        resp.__aiter__()
        self.assertEqual(resp._step, 0)

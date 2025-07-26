from avalan.agent.engine import EngineAgent
from avalan.agent.orchestrator.response.orchestrator_response import (
    OrchestratorResponse,
)
from avalan.agent import EngineEnvironment, AgentOperation, Specification
from avalan.entities import (
    EngineUri,
    Message,
    MessageRole,
    Token,
    TransformerEngineSettings,
    GenerationSettings,
)
from avalan.model import TextGenerationResponse
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
        use_async_generator=True,
        generation_settings=settings,
        settings=settings,
    )


class OrchestratorResponseStepTestCase(IsolatedAsyncioTestCase):
    async def test_step_counter(self):
        engine = _DummyEngine()
        agent = MagicMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()
        resp = OrchestratorResponse(
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

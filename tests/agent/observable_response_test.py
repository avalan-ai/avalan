from avalan.agent.orchestrator import ObservableTextGenerationResponse
from avalan.event.manager import EventManager
from avalan.event import EventType
from avalan.model.entities import Token
from avalan.model.nlp.text import TextGenerationResponse
from unittest import IsolatedAsyncioTestCase, main
from unittest.mock import AsyncMock, MagicMock


class ObservableTextGenerationResponseIterationTestCase(IsolatedAsyncioTestCase):
    async def test_iteration_emits_events_and_stream_end(self):
        async def output_gen():
            yield "a"
            yield Token(id=5, token="b")

        def output_fn():
            return output_gen()

        response = TextGenerationResponse(output_fn, use_async_generator=True)

        event_manager = MagicMock(spec=EventManager)
        event_manager.trigger = AsyncMock()
        tokenizer = MagicMock()
        tokenizer.encode.return_value = [42]

        obs = ObservableTextGenerationResponse(response, event_manager, "m", tokenizer)

        self.assertIs(obs.__aiter__(), obs)
        tokens = []
        async for t in obs:
            tokens.append(t)

        self.assertEqual(tokens, ["a", Token(id=5, token="b")])

        calls = event_manager.trigger.await_args_list
        token_events = [c.args[0] for c in calls if c.args[0].type == EventType.TOKEN_GENERATED]
        self.assertEqual(len(token_events), 2)
        self.assertEqual(
            token_events[0].payload,
            {"token_id": 42, "model_id": "m", "token": "a", "step": 0},
        )
        self.assertEqual(
            token_events[1].payload,
            {"token_id": 5, "model_id": "m", "token": "b", "step": 1},
        )
        self.assertTrue(any(c.args[0].type == EventType.STREAM_END for c in calls))
        self.assertEqual(obs.input_token_count, response.input_token_count)


class ObservableTextGenerationResponseDelegationTestCase(IsolatedAsyncioTestCase):
    async def test_delegated_methods(self):
        response = MagicMock(spec=TextGenerationResponse)
        response.input_token_count = 10
        response.__aiter__.return_value = []
        response.to_str = AsyncMock(return_value="str")
        response.to_json = AsyncMock(return_value="json")
        response.to = AsyncMock(return_value={"x": 1})
        response.add_done_callback = MagicMock()

        event_manager = MagicMock(spec=EventManager)
        event_manager.trigger = AsyncMock()

        obs = ObservableTextGenerationResponse(response, event_manager, "m", None)

        self.assertEqual(obs.input_token_count, 10)
        self.assertIs(obs.__aiter__(), obs)
        response.__aiter__.assert_called_once()

        self.assertEqual(await obs.to_str(), "str")
        self.assertEqual(await obs.to_json(), "json")
        self.assertEqual(await obs.to(dict), {"x": 1})
        response.to_str.assert_awaited_once()
        response.to_json.assert_awaited_once()
        response.to.assert_awaited_once_with(dict)


if __name__ == "__main__":
    main()

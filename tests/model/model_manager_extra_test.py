import asyncio
from logging import Logger
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import AsyncMock, MagicMock, patch

from avalan.agent import Specification
from avalan.entities import (
    Backend,
    EngineUri,
    GenerationSettings,
    Modality,
    Operation,
    OperationParameters,
    OperationTextParameters,
    TransformerEngineSettings,
)
from avalan.event import EventType
from avalan.event.manager import EventManager
from avalan.model.call import ModelCall, ModelCallContext
from avalan.model.hubs.huggingface import HuggingfaceHub
from avalan.model.manager import ModelManager


class ModelManagerExtraTestCase(TestCase):
    def setUp(self):
        self.hub = MagicMock(spec=HuggingfaceHub)
        self.hub.cache_dir = "cache"
        self.logger = MagicMock(spec=Logger)

    def test_parse_uri_invalid_scheme(self):
        manager = ModelManager(self.hub, self.logger)
        with self.assertRaises(ValueError):
            manager.parse_uri("http://openai/gpt-4o")

    def test_parse_uri_params(self):
        manager = ModelManager(self.hub, self.logger)
        uri = manager.parse_uri(
            "ai://openai/gpt-4o?temperature=0.6&max_new_tokens=8192&backend=mlx"
        )
        self.assertEqual(uri.params["temperature"], 0.6)
        self.assertEqual(uri.params["max_new_tokens"], 8192)
        self.assertEqual(uri.params["backend"], "mlx")

    def test_get_engine_settings_user_password_no_secret(self):
        manager = ModelManager(self.hub, self.logger)
        uri = EngineUri(
            host="openai",
            port=None,
            user="token",
            password="pass",
            vendor="openai",
            model_id="gpt",
            params={},
        )
        settings = manager.get_engine_settings(uri)
        self.assertIsNone(settings.access_token)

    def test_load_passes_arguments(self):
        manager = ModelManager(self.hub, self.logger)
        uri = EngineUri(
            host=None,
            port=None,
            user=None,
            password=None,
            vendor=None,
            model_id="model",
            params={},
        )
        with (
            patch.object(manager, "get_engine_settings") as get_mock,
            patch.object(manager, "load_engine") as load_mock,
        ):
            get_mock.return_value = TransformerEngineSettings()
            load_mock.return_value = "model"
            result = manager.load(
                uri,
                base_url="url",
                quiet=True,
                attention="sd",
                trust_remote_code=True,
                backend="mlxlm",
            )
        args = get_mock.call_args.args[1]
        self.assertTrue(args["disable_loading_progress_bar"])
        self.assertEqual(args["base_url"], "url")
        self.assertEqual(args["attention"], "sd")
        self.assertEqual(args["backend"], "mlxlm")
        self.assertTrue(args["trust_remote_code"])
        load_mock.assert_called_once_with(
            uri, get_mock.return_value, Modality.TEXT_GENERATION
        )
        self.assertEqual(result, "model")

    def test_load_backend_from_uri(self):
        manager = ModelManager(self.hub, self.logger)
        uri = EngineUri(
            host=None,
            port=None,
            user=None,
            password=None,
            vendor=None,
            model_id="model",
            params={"backend": "mlx"},
        )
        with (
            patch.object(manager, "get_engine_settings") as get_mock,
            patch.object(manager, "load_engine") as load_mock,
        ):
            get_mock.return_value = TransformerEngineSettings()
            load_mock.return_value = "model"
            manager.load(uri)
        args = get_mock.call_args.args[1]
        self.assertEqual(args["backend"], Backend.MLXLM)

    def test_load_engine_invalid_modality(self):
        manager = ModelManager(self.hub, self.logger)
        uri = manager.parse_uri("ai://tok@openai/gpt-4o")
        settings = TransformerEngineSettings()
        with self.assertRaises(NotImplementedError):
            manager.load_engine(uri, settings, "invalid")  # type: ignore[arg-type]

    def test_load_invalid_modality(self):
        manager = ModelManager(self.hub, self.logger)
        uri = manager.parse_uri("ai://tok@openai/gpt-4o")
        with self.assertRaises(NotImplementedError):
            manager.load(uri, modality="invalid")  # type: ignore[arg-type]

    def test_load_output_hidden_states(self):
        uri = EngineUri(
            host=None,
            port=None,
            user=None,
            password=None,
            vendor=None,
            model_id="m",
            params={},
        )
        manager = ModelManager(self.hub, self.logger)
        for modality in (Modality.TEXT_GENERATION, Modality.EMBEDDING):
            for value in (True, False, None):
                with self.subTest(modality=modality, value=value):
                    with (
                        patch.object(manager, "get_engine_settings") as ges,
                        patch.object(manager, "load_engine") as le,
                    ):
                        manager._stack.enter_context = MagicMock()
                        ges.return_value = TransformerEngineSettings()
                        le.return_value = "model"
                        result = manager.load(
                            uri, modality=modality, output_hidden_states=value
                        )
                    args = ges.call_args.args[1]
                    if value is None:
                        self.assertNotIn("output_hidden_states", args)
                    else:
                        self.assertEqual(args["output_hidden_states"], value)
                    le.assert_called_once_with(uri, ges.return_value, modality)
                    self.assertEqual(result, "model")

    def test_parse_uri_boolean_params(self):
        manager = ModelManager(self.hub, self.logger)
        uri = manager.parse_uri("ai://openai/gpt-4o?stream=true&debug=false")
        self.assertTrue(uri.params["stream"])
        self.assertFalse(uri.params["debug"])

    def test_async_context_manager(self):
        manager = ModelManager(self.hub, self.logger)

        async def run():
            async with manager as mm:
                self.assertIs(mm, manager)

        asyncio.run(run())


class ModelManagerEventDispatchTestCase(IsolatedAsyncioTestCase):
    async def test_triggers_events_and_closes_stack(self) -> None:
        hub = MagicMock(spec=HuggingfaceHub)
        hub.cache_dir = "cache"
        logger = MagicMock(spec=Logger)
        event_manager = MagicMock(spec=EventManager)
        event_manager.trigger = AsyncMock()
        manager = ModelManager(hub, logger, event_manager=event_manager)
        manager._stack.aclose = AsyncMock()  # type: ignore[assignment]

        engine_uri = EngineUri(
            host=None,
            port=None,
            user=None,
            password=None,
            vendor=None,
            model_id="model",
            params={},
        )
        operation = Operation(
            generation_settings=GenerationSettings(),
            input="prompt",
            modality=Modality.TEXT_GENERATION,
            parameters=OperationParameters(
                text=OperationTextParameters(
                    system_prompt="system",
                    developer_prompt="dev",
                )
            ),
        )
        expected = object()

        async def handler(
            engine_uri_arg: EngineUri,
            model_arg: object,
            operation_arg: Operation,
            tool_arg: object,
        ) -> object:
            self.assertIs(engine_uri_arg, engine_uri)
            self.assertIs(operation_arg, operation)
            self.assertIsNone(tool_arg)
            return expected

        model = object()
        context = ModelCallContext(
            specification=Specification(role=None, goal=None),
            input=operation.input,
        )
        task = ModelCall(
            engine_uri=engine_uri,
            model=model,
            operation=operation,
            tool=None,
            context=context,
        )
        with patch(
            "avalan.model.manager.ModalityRegistry.get",
            return_value=handler,
        ):
            result = await manager(task)

        self.assertIs(result, expected)
        self.assertEqual(event_manager.trigger.await_count, 2)
        before_event = event_manager.trigger.await_args_list[0].args[0]
        self.assertEqual(
            before_event.type, EventType.MODEL_MANAGER_CALL_BEFORE
        )
        self.assertEqual(
            before_event.payload["modality"], Modality.TEXT_GENERATION
        )
        self.assertIs(before_event.payload["task"], task)
        self.assertIs(before_event.payload["context"], context)
        after_event = event_manager.trigger.await_args_list[1].args[0]
        self.assertEqual(after_event.type, EventType.MODEL_MANAGER_CALL_AFTER)
        self.assertIs(after_event.payload["result"], expected)
        self.assertIs(after_event.payload["task"], task)
        self.assertIs(after_event.payload["context"], context)
        self.assertIsNotNone(after_event.started)
        self.assertIsNotNone(after_event.finished)
        self.assertIsNotNone(after_event.elapsed)

        manager.__exit__(None, None, None)
        await asyncio.sleep(0)
        manager._stack.aclose.assert_awaited_once()

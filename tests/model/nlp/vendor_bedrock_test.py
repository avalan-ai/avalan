from base64 import b64encode
from contextlib import AsyncExitStack
from dataclasses import dataclass
from importlib import import_module, reload
from importlib.machinery import ModuleSpec
from json import loads
from sys import modules
from types import ModuleType, SimpleNamespace
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import AsyncMock, MagicMock, patch

from avalan.entities import (
    GenerationSettings,
    Message,
    MessageContentFile,
    MessageContentImage,
    MessageContentText,
    MessageRole,
    ToolCall,
    ToolCallDiagnostic,
    ToolCallDiagnosticCode,
    ToolCallDiagnosticStage,
    ToolCallError,
    ToolCallResult,
    TransformerEngineSettings,
)
from avalan.model.stream import (
    CanonicalStreamItem,
    StreamItemKind,
    StreamTerminalOutcome,
    accumulate_canonical_stream_items,
)
from avalan.task.usage import (
    usage_observation_from_response,
    usage_totals_from_response,
)


class AsyncIter:
    def __init__(self, items):
        self._iter = iter(items)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._iter)
        except StopIteration as exc:
            raise StopAsyncIteration from exc


class FakeBedrockError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.response = {"Error": {"Code": code, "Message": message}}


def patch_bedrock_imports():
    aioboto3_stub = ModuleType("aioboto3")
    aioboto3_stub.__spec__ = ModuleSpec("aioboto3", loader=None)
    session_mock = MagicMock()
    aioboto3_stub.Session = MagicMock(return_value=session_mock)

    transformers_stub = ModuleType("transformers")
    transformers_stub.__spec__ = ModuleSpec("transformers", loader=None)
    transformers_stub.PreTrainedModel = type("PreTrainedModel", (), {})
    transformers_stub.PreTrainedTokenizer = type("PreTrainedTokenizer", (), {})
    transformers_stub.PreTrainedTokenizerFast = type(
        "PreTrainedTokenizerFast", (), {}
    )
    transformers_stub.__getattr__ = lambda name: MagicMock()

    transformers_utils_stub = ModuleType("transformers.utils")
    transformers_utils_stub.__spec__ = ModuleSpec(
        "transformers.utils", loader=None
    )
    transformers_utils_stub.get_json_schema = MagicMock()
    transformers_logging_stub = ModuleType("transformers.utils.logging")
    transformers_logging_stub.__spec__ = ModuleSpec(
        "transformers.utils.logging", loader=None
    )
    transformers_logging_stub.disable_progress_bar = MagicMock()
    transformers_logging_stub.enable_progress_bar = MagicMock()
    transformers_utils_stub.logging = transformers_logging_stub

    tokenization_stub = ModuleType("transformers.tokenization_utils_base")
    tokenization_stub.__spec__ = ModuleSpec(
        "transformers.tokenization_utils_base", loader=None
    )
    tokenization_stub.BatchEncoding = MagicMock()

    generation_stub = ModuleType("transformers.generation")
    generation_stub.__spec__ = ModuleSpec(
        "transformers.generation", loader=None
    )
    generation_stub.__path__ = []
    generation_stub.StoppingCriteria = MagicMock()
    generation_stub.StoppingCriteriaList = MagicMock()
    stopping_criteria_stub = ModuleType(
        "transformers.generation.stopping_criteria"
    )
    stopping_criteria_stub.__spec__ = ModuleSpec(
        "transformers.generation.stopping_criteria", loader=None
    )
    stopping_criteria_stub.StoppingCriteria = MagicMock()

    diffusers_stub = ModuleType("diffusers")
    diffusers_stub.__spec__ = ModuleSpec("diffusers", loader=None)
    diffusers_stub.DiffusionPipeline = MagicMock()

    patcher = patch.dict(
        modules,
        {
            "aioboto3": aioboto3_stub,
            "transformers": transformers_stub,
            "transformers.utils": transformers_utils_stub,
            "transformers.utils.logging": transformers_logging_stub,
            "transformers.tokenization_utils_base": tokenization_stub,
            "transformers.generation": generation_stub,
            "transformers.generation.stopping_criteria": (
                stopping_criteria_stub
            ),
            "diffusers": diffusers_stub,
        },
    )
    patcher.start()
    return aioboto3_stub, session_mock, patcher


class ClientContext:
    def __init__(self, client):
        self._client = client

    async def __aenter__(self):
        return self._client

    async def __aexit__(self, exc_type, exc, tb):
        return False


class BedrockTestCase(IsolatedAsyncioTestCase):
    def setUp(self):
        self.aioboto3_stub, self.session_mock, self.patch = (
            patch_bedrock_imports()
        )
        reload(import_module("avalan.model.nlp.text.vendor.bedrock"))
        self.mod = import_module("avalan.model.nlp.text.vendor.bedrock")
        self.client = SimpleNamespace(
            converse_stream=AsyncMock(), converse=AsyncMock()
        )
        self.session_mock.client.return_value = ClientContext(self.client)

    def tearDown(self):
        self.patch.stop()

    async def test_stream_processing(self):
        events = [
            {
                "contentBlockDelta": {
                    "contentBlockIndex": 0,
                    "delta": {"reasoning": {"text": "think"}},
                }
            },
            {
                "contentBlockStart": {
                    "contentBlockIndex": 1,
                    "contentBlock": {
                        "toolUse": {
                            "toolUseId": "id1",
                            "name": "pkg__tool",
                        }
                    },
                }
            },
            {
                "contentBlockDelta": {
                    "contentBlockIndex": 1,
                    "delta": {"toolUse": {"input": "{"}},
                }
            },
            {
                "contentBlockDelta": {
                    "contentBlockIndex": 1,
                    "delta": {"toolUse": {"input": '"a":1}'}},
                }
            },
            {
                "contentBlockDelta": {
                    "contentBlockIndex": 0,
                    "delta": {"text": {"text": "hi"}},
                }
            },
            {"contentBlockStop": {"contentBlockIndex": 1}},
            {"messageStop": {"reason": "finished"}},
        ]
        stream = self.mod.BedrockStream(AsyncIter(events))

        items = [item async for item in stream]
        accumulator = accumulate_canonical_stream_items(items)

        self.assertEqual(
            [item.kind for item in items],
            [
                StreamItemKind.STREAM_STARTED,
                StreamItemKind.REASONING_DELTA,
                StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                StreamItemKind.ANSWER_DELTA,
                StreamItemKind.TOOL_CALL_READY,
                StreamItemKind.TOOL_CALL_DONE,
                StreamItemKind.ANSWER_DONE,
                StreamItemKind.REASONING_DONE,
                StreamItemKind.STREAM_COMPLETED,
                StreamItemKind.STREAM_CLOSED,
            ],
        )
        self.assertEqual(accumulator.reasoning_text, "think")
        self.assertEqual(accumulator.answer_text, "hi")
        self.assertEqual(
            accumulator.tool_call_arguments,
            {"id1": '{"a":1}'},
        )
        ready = next(
            item
            for item in items
            if item.kind is StreamItemKind.TOOL_CALL_READY
        )
        self.assertEqual(ready.data, {"name": "pkg__tool"})

    async def test_stream_public_iterator_yields_canonical_items(self):
        stream = self.mod.BedrockStream(
            AsyncIter(
                [
                    {
                        "contentBlockDelta": {
                            "contentBlockIndex": 0,
                            "delta": {"text": {"text": "hi"}},
                        }
                    }
                ]
            )
        )

        items = [item async for item in stream]

        self.assertEqual(
            [item.kind for item in items],
            [
                StreamItemKind.STREAM_STARTED,
                StreamItemKind.ANSWER_DELTA,
                StreamItemKind.ANSWER_DONE,
                StreamItemKind.STREAM_COMPLETED,
                StreamItemKind.STREAM_CLOSED,
            ],
        )
        self.assertEqual(items[1].text_delta, "hi")
        self.assertEqual({item.provider_family for item in items}, {"bedrock"})

    async def test_stream_direct_anext_yields_canonical_items(self):
        stream = self.mod.BedrockStream(
            AsyncIter(
                [
                    {
                        "contentBlockDelta": {
                            "contentBlockIndex": 0,
                            "delta": {"text": {"text": "hi"}},
                        }
                    }
                ]
            )
        )

        started = await stream.__anext__()
        delta = await stream.__anext__()

        self.assertIs(started.kind, StreamItemKind.STREAM_STARTED)
        self.assertIs(delta.kind, StreamItemKind.ANSWER_DELTA)
        self.assertEqual(delta.text_delta, "hi")
        self.assertEqual(delta.provider_family, "bedrock")

    async def test_stream_initial_input_and_stop_event(self):
        events = [
            {
                "contentBlockStart": {
                    "contentBlockIndex": 0,
                    "contentBlock": {
                        "toolUse": {
                            "toolUseId": "id2",
                            "name": "pkg.tool",
                            "input": {"foo": 1},
                        }
                    },
                }
            },
            {
                "contentBlockDelta": {
                    "contentBlockIndex": 0,
                    "delta": {"toolUse": {"input": {"bar": 2}}},
                }
            },
            {
                "contentBlockStop": {
                    "contentBlockIndex": 0,
                    "contentBlock": {
                        "toolUse": {
                            "toolUseId": "id2",
                            "name": "pkg.tool",
                            "input": "done",
                        }
                    },
                }
            },
            {"messageStop": {}},
        ]
        stream = self.mod.BedrockStream(AsyncIter(events))

        items = [item async for item in stream]
        accumulator = accumulate_canonical_stream_items(items)

        self.assertEqual(
            [item.kind for item in items],
            [
                StreamItemKind.STREAM_STARTED,
                StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                StreamItemKind.TOOL_CALL_READY,
                StreamItemKind.TOOL_CALL_DONE,
                StreamItemKind.STREAM_COMPLETED,
                StreamItemKind.STREAM_CLOSED,
            ],
        )
        self.assertEqual(
            accumulator.tool_call_arguments,
            {"id2": '{"foo": 1}{"bar": 2}done'},
        )
        ready = next(
            item
            for item in items
            if item.kind is StreamItemKind.TOOL_CALL_READY
        )
        self.assertEqual(ready.data, {"name": "pkg.tool"})

    async def test_stream_stops_on_message_event(self):
        stop_event = {"messageStop": {}}
        stream = self.mod.BedrockStream(
            AsyncIter(
                [
                    stop_event,
                    {
                        "contentBlockDelta": {
                            "contentBlockIndex": 0,
                            "delta": {"text": {"text": "late"}},
                        }
                    },
                ]
            )
        )

        items = [item async for item in stream]

        self.assertEqual(
            [item.kind for item in items],
            [
                StreamItemKind.STREAM_STARTED,
                StreamItemKind.STREAM_COMPLETED,
                StreamItemKind.STREAM_CLOSED,
            ],
        )
        self.assertEqual(items[1].provider_payload, stop_event)
        self.assertEqual(items[1].provider_event_type, "messageStop")
        self.assertIsNone(stream.usage)

    async def test_stream_records_usage_from_terminal_metadata(self):
        events = [
            {
                "contentBlockDelta": {
                    "contentBlockIndex": 0,
                    "delta": {"text": {"text": "hi"}},
                }
            },
            {"messageStop": {"reason": "done"}},
            {
                "contentBlockDelta": {
                    "contentBlockIndex": 0,
                    "delta": {"text": {"text": "ignored"}},
                }
            },
            {
                "metadata": {
                    "usage": {
                        "inputTokens": 5,
                        "cacheReadInputTokens": 1,
                        "cacheWriteInputTokens": 2,
                        "outputTokens": 7,
                        "totalTokens": 12,
                        "reasoning": {"text": "private thinking text"},
                        "cacheDetails": {
                            "cacheRead": {
                                "ephemeral5mInputTokens": 3,
                                "ephemeral1hInputTokens": 4,
                            },
                            "cacheWrite": {
                                "ephemeral5mInputTokens": 5,
                                "ephemeral1hInputTokens": 6,
                            },
                        },
                    }
                }
            },
        ]
        stream = self.mod.BedrockStream(AsyncIter(events))

        items = [item async for item in stream]
        accumulator = accumulate_canonical_stream_items(items)
        observation = usage_observation_from_response(stream)
        totals = usage_totals_from_response(stream)

        self.assertEqual(accumulator.answer_text, "hi")
        self.assertEqual(
            accumulator.final_usage,
            {
                "inputTokens": 5,
                "cacheReadInputTokens": 1,
                "cacheWriteInputTokens": 2,
                "outputTokens": 7,
                "totalTokens": 12,
                "reasoning": {"text": "private thinking text"},
                "cacheDetails": {
                    "cacheRead": {
                        "ephemeral5mInputTokens": 3,
                        "ephemeral1hInputTokens": 4,
                    },
                    "cacheWrite": {
                        "ephemeral5mInputTokens": 5,
                        "ephemeral1hInputTokens": 6,
                    },
                },
            },
        )
        self.assertEqual(stream.provider_family, "bedrock")
        self.assertIsNotNone(observation)
        assert observation is not None
        self.assertEqual(
            observation.metadata,
            {
                "provider_family": "bedrock",
                "cache_creation_ephemeral_5m_input_tokens": 5,
                "cache_creation_ephemeral_1h_input_tokens": 6,
                "cache_read_ephemeral_5m_input_tokens": 3,
                "cache_read_ephemeral_1h_input_tokens": 4,
            },
        )
        self.assertIsNotNone(totals)
        assert totals is not None
        self.assertEqual(totals.input_tokens, 5)
        self.assertEqual(totals.cached_input_tokens, 1)
        self.assertEqual(totals.cache_creation_input_tokens, 2)
        self.assertEqual(totals.output_tokens, 7)
        self.assertIsNone(totals.reasoning_tokens)
        self.assertEqual(totals.total_tokens, 12)
        self.assertNotIn("private thinking text", str(observation))

    async def test_stream_defers_usage_until_metadata_stream_exhausts(self):
        events = [
            {
                "metadata": {
                    "usage": {
                        "inputTokens": 0,
                        "cacheReadInputTokens": 0,
                        "cacheWriteInputTokens": 0,
                        "outputTokens": 0,
                        "totalTokens": 0,
                    }
                }
            },
            {
                "contentBlockDelta": {
                    "contentBlockIndex": 0,
                    "delta": {"text": {"text": "late"}},
                }
            },
            {"messageStop": {"reason": "done"}},
        ]
        stream = self.mod.BedrockStream(AsyncIter(events))

        items = [item async for item in stream]
        accumulator = accumulate_canonical_stream_items(items)
        totals = usage_totals_from_response(stream)

        self.assertEqual(accumulator.answer_text, "late")
        self.assertEqual(
            accumulator.final_usage,
            {
                "inputTokens": 0,
                "cacheReadInputTokens": 0,
                "cacheWriteInputTokens": 0,
                "outputTokens": 0,
                "totalTokens": 0,
            },
        )
        self.assertIsNotNone(totals)
        assert totals is not None
        self.assertEqual(totals.input_tokens, 0)
        self.assertEqual(totals.cached_input_tokens, 0)
        self.assertEqual(totals.cache_creation_input_tokens, 0)
        self.assertEqual(totals.output_tokens, 0)
        self.assertIsNone(totals.reasoning_tokens)
        self.assertEqual(totals.total_tokens, 0)

    async def test_canonical_stream_maps_bedrock_events(self):
        events = [
            {
                "contentBlockDelta": {
                    "contentBlockIndex": 0,
                    "delta": {"reasoning": {"text": "think"}},
                }
            },
            {
                "contentBlockStart": {
                    "contentBlockIndex": 1,
                    "contentBlock": {
                        "toolUse": {
                            "toolUseId": "call-1",
                            "name": "lookup",
                            "input": {"q": 1},
                        }
                    },
                }
            },
            {
                "contentBlockDelta": {
                    "contentBlockIndex": 1,
                    "delta": {"toolUse": {"input": '{"more":'}},
                }
            },
            {
                "contentBlockDelta": {
                    "contentBlockIndex": 0,
                    "delta": {"text": {"text": "answer"}},
                }
            },
            {
                "contentBlockStop": {
                    "contentBlockIndex": 1,
                    "contentBlock": {
                        "toolUse": {
                            "toolUseId": "call-1",
                            "name": "lookup",
                            "input": '"yes"}',
                        }
                    },
                }
            },
            {"messageStop": {"reason": "finished"}},
            {
                "metadata": {
                    "usage": {
                        "inputTokens": 2,
                        "outputTokens": 3,
                    }
                }
            },
        ]
        stream = self.mod.BedrockStream(AsyncIter(events))

        items = [
            item
            async for item in stream.canonical_stream(
                stream_session_id="session",
                run_id="run",
                turn_id="turn",
            )
        ]

        self.assertEqual(
            [item.kind for item in items],
            [
                StreamItemKind.STREAM_STARTED,
                StreamItemKind.REASONING_DELTA,
                StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                StreamItemKind.ANSWER_DELTA,
                StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                StreamItemKind.TOOL_CALL_READY,
                StreamItemKind.TOOL_CALL_DONE,
                StreamItemKind.ANSWER_DONE,
                StreamItemKind.REASONING_DONE,
                StreamItemKind.USAGE_COMPLETED,
                StreamItemKind.STREAM_COMPLETED,
                StreamItemKind.STREAM_CLOSED,
            ],
        )
        accumulator = accumulate_canonical_stream_items(items)
        self.assertEqual(accumulator.answer_text, "answer")
        self.assertEqual(accumulator.reasoning_text, "think")
        self.assertEqual(
            accumulator.tool_call_arguments,
            {"call-1": '{"q": 1}{"more":"yes"}'},
        )
        self.assertEqual(
            accumulator.final_usage,
            {"inputTokens": 2, "outputTokens": 3},
        )
        argument_items = [
            item
            for item in items
            if item.kind is StreamItemKind.TOOL_CALL_ARGUMENT_DELTA
        ]
        self.assertEqual(
            [item.provider_event_type for item in argument_items],
            [
                "contentBlockStart",
                "contentBlockDelta",
                "contentBlockStop",
            ],
        )
        self.assertEqual(argument_items[0].provider_payload, events[1])
        self.assertEqual(argument_items[1].provider_payload, events[2])
        self.assertEqual(argument_items[2].provider_payload, events[4])
        self.assertEqual(items[10].provider_payload, events[-1])
        self.assertEqual(items[10].provider_event_type, "metadata.usage")
        self.assertEqual(items[11].provider_payload, events[-2])
        self.assertEqual(items[11].provider_event_type, "messageStop")
        ready = next(
            item
            for item in items
            if item.kind is StreamItemKind.TOOL_CALL_READY
        )
        self.assertEqual(ready.data, {"name": "lookup"})

    async def test_canonical_stream_maps_bedrock_text_delta_without_legacy(
        self,
    ):
        event = {
            "contentBlockDelta": {
                "contentBlockIndex": 0,
                "delta": {"text": {"text": "hello"}},
            }
        }
        stream = self.mod.BedrockStream(
            AsyncIter([event, {"messageStop": {"reason": "finished"}}])
        )

        items = [
            item
            async for item in stream.canonical_stream(
                stream_session_id="session",
                run_id="run",
                turn_id="turn",
            )
        ]

        self.assertTrue(
            all(isinstance(item, CanonicalStreamItem) for item in items)
        )
        self.assertEqual(
            [item.kind for item in items],
            [
                StreamItemKind.STREAM_STARTED,
                StreamItemKind.ANSWER_DELTA,
                StreamItemKind.ANSWER_DONE,
                StreamItemKind.STREAM_COMPLETED,
                StreamItemKind.STREAM_CLOSED,
            ],
        )
        self.assertEqual(
            accumulate_canonical_stream_items(items).answer_text,
            "hello",
        )
        self.assertEqual(items[1].provider_payload, event)
        self.assertEqual(items[1].provider_event_type, "contentBlockDelta")

    async def test_canonical_stream_maps_malformed_bedrock_tool_delta_to_error(
        self,
    ):
        events = [
            {
                "contentBlockDelta": {
                    "contentBlockIndex": 1,
                    "delta": {"toolUse": {"input": '{"q":1}'}},
                }
            }
        ]
        stream = self.mod.BedrockStream(AsyncIter(events))

        items = [
            item
            async for item in stream.canonical_stream(
                stream_session_id="session",
                run_id="run",
                turn_id="turn",
            )
        ]

        self.assertEqual(
            [item.kind for item in items],
            [
                StreamItemKind.STREAM_STARTED,
                StreamItemKind.STREAM_ERRORED,
                StreamItemKind.STREAM_CLOSED,
            ],
        )
        self.assertIn("missing start event", str(items[1].data))

    async def test_canonical_stream_preserves_malformed_bedrock_payload(
        self,
    ):
        event = {
            "contentBlockDelta": {
                "contentBlockIndex": "bad",
                "delta": {"text": {"text": "legacy-token"}},
            }
        }
        stream = self.mod.BedrockStream(AsyncIter([event]))

        items = [
            item
            async for item in stream.canonical_stream(
                stream_session_id="session",
                run_id="run",
                turn_id="turn",
            )
        ]

        self.assertTrue(
            all(isinstance(item, CanonicalStreamItem) for item in items)
        )
        self.assertEqual(
            [item.kind for item in items],
            [
                StreamItemKind.STREAM_STARTED,
                StreamItemKind.STREAM_ERRORED,
                StreamItemKind.STREAM_CLOSED,
            ],
        )
        error_item = items[1]
        error_data = error_item.data
        assert isinstance(error_data, dict)
        self.assertEqual(error_data["error_type"], "ValueError")
        self.assertIn("index must be an integer", error_data["message"])
        self.assertEqual(error_item.provider_payload, event)
        self.assertEqual(error_item.provider_event_type, "contentBlockDelta")
        self.assertIs(
            error_item.terminal_outcome,
            StreamTerminalOutcome.ERRORED,
        )
        self.assertNotIn(
            StreamItemKind.ANSWER_DELTA,
            [item.kind for item in items],
        )

    async def test_canonical_stream_maps_duplicate_bedrock_tool_stop_to_error(
        self,
    ):
        tool = {"toolUseId": "call-1", "name": "lookup"}
        events = [
            {
                "contentBlockStart": {
                    "contentBlockIndex": 1,
                    "contentBlock": {"toolUse": tool},
                }
            },
            {
                "contentBlockStop": {
                    "contentBlockIndex": 1,
                    "contentBlock": {"toolUse": tool},
                }
            },
            {
                "contentBlockStop": {
                    "contentBlockIndex": 1,
                    "contentBlock": {"toolUse": tool},
                }
            },
        ]
        stream = self.mod.BedrockStream(AsyncIter(events))

        items = [
            item
            async for item in stream.canonical_stream(
                stream_session_id="session",
                run_id="run",
                turn_id="turn",
            )
        ]

        self.assertEqual(
            [item.kind for item in items],
            [
                StreamItemKind.STREAM_STARTED,
                StreamItemKind.TOOL_CALL_READY,
                StreamItemKind.TOOL_CALL_DONE,
                StreamItemKind.STREAM_ERRORED,
                StreamItemKind.STREAM_CLOSED,
            ],
        )

    async def test_canonical_stream_skips_bedrock_content_after_message_stop(
        self,
    ):
        events = [
            {"messageStop": {"reason": "done"}},
            {
                "contentBlockDelta": {
                    "contentBlockIndex": 0,
                    "delta": {"text": {"text": "late"}},
                }
            },
            {
                "metadata": {
                    "usage": {
                        "inputTokens": 1,
                        "outputTokens": 2,
                    }
                }
            },
        ]
        stream = self.mod.BedrockStream(AsyncIter(events))

        items = [
            item
            async for item in stream.canonical_stream(
                stream_session_id="session",
                run_id="run",
                turn_id="turn",
            )
        ]

        self.assertNotIn(
            StreamItemKind.ANSWER_DELTA,
            [item.kind for item in items],
        )
        accumulator = accumulate_canonical_stream_items(items)
        self.assertEqual(
            accumulator.final_usage,
            {"inputTokens": 1, "outputTokens": 2},
        )

    async def test_canonical_stream_bedrock_mapping_edge_cases(self):
        stream = self.mod.BedrockStream(AsyncIter([]))

        self.assertEqual(stream._provider_events_from_event({}), ())
        self.assertIsNone(stream._provider_event_type(SimpleNamespace()))
        self.assertIsNone(stream._provider_event_type({"unknown": {}}))
        self.assertEqual(
            stream._provider_events_from_event(
                {
                    "contentBlockStart": {
                        "contentBlockIndex": 0,
                        "contentBlock": {"text": "ignored"},
                    }
                }
            ),
            (),
        )
        self.assertEqual(
            stream._provider_events_from_event(
                {
                    "contentBlockDelta": {
                        "contentBlockIndex": 0,
                        "delta": {"toolUse": {"input": ""}},
                    }
                }
            ),
            (),
        )
        self.assertEqual(
            stream._provider_events_from_event(
                {
                    "contentBlockDelta": {
                        "contentBlockIndex": 0,
                        "delta": {},
                    }
                }
            ),
            (),
        )
        self.assertEqual(
            stream._provider_events_from_event(
                {"contentBlockStop": {"contentBlockIndex": 0}}
            ),
            (),
        )

        stream._canonical_tool_blocks[1] = {
            "id": "call-1",
            "name": "lookup",
            "arguments_seen": False,
        }
        sparse_stop = stream._content_block_stop_events(
            {"contentBlockIndex": 1},
            None,
        )
        self.assertEqual(
            [event.kind for event in sparse_stop],
            [StreamItemKind.TOOL_CALL_READY, StreamItemKind.TOOL_CALL_DONE],
        )
        self.assertEqual(
            stream._mark_tool_ready("call-2", "lookup", None)[0].kind,
            StreamItemKind.TOOL_CALL_READY,
        )
        self.assertEqual(stream._mark_tool_ready("call-2", "lookup", None), ())

        invalid_events = [
            {"contentBlockStart": {"contentBlockIndex": "bad"}},
            {
                "contentBlockStart": {
                    "contentBlockIndex": 0,
                    "contentBlock": {
                        "toolUse": {
                            "toolUseId": "call-1",
                            "name": 1,
                        }
                    },
                }
            },
            {
                "contentBlockStart": {
                    "contentBlockIndex": 0,
                    "contentBlock": {
                        "toolUse": {
                            "toolUseId": None,
                            "name": "lookup",
                        }
                    },
                }
            },
            {"contentBlockDelta": {"contentBlockIndex": "bad"}},
            {"contentBlockStop": {"contentBlockIndex": "bad"}},
            {
                "contentBlockStop": {
                    "contentBlockIndex": 0,
                    "contentBlock": {
                        "toolUse": {
                            "toolUseId": "call-1",
                            "name": 1,
                        }
                    },
                }
            },
        ]
        for event in invalid_events:
            with self.assertRaises(ValueError):
                stream._provider_events_from_event(event)

    async def test_stream_failure_after_metadata_keeps_usage_unavailable(
        self,
    ):
        class FailingIter:
            def __init__(self):
                self._count = 0

            def __aiter__(self):
                return self

            async def __anext__(self):
                self._count += 1
                if self._count == 1:
                    return {"metadata": {"usage": {"inputTokens": 1}}}
                raise RuntimeError("provider failure")

        stream = self.mod.BedrockStream(FailingIter())

        items = [item async for item in stream]

        self.assertEqual(
            [item.kind for item in items],
            [
                StreamItemKind.STREAM_STARTED,
                StreamItemKind.STREAM_ERRORED,
                StreamItemKind.STREAM_CLOSED,
            ],
        )
        self.assertIn("provider failure", str(items[1].data))
        self.assertIsNone(stream.usage)
        self.assertIsNone(usage_totals_from_response(stream))

    async def test_client_stream_invocation(self):
        self.client.converse_stream.return_value = {"stream": AsyncIter([])}
        exit_stack = AsyncExitStack()
        client = self.mod.BedrockClient(
            exit_stack=exit_stack,
            region_name="us-east-1",
            endpoint_url="https://example.com",
        )
        client._system_prompt = MagicMock(return_value=None)
        client._template_messages = MagicMock(
            return_value=[{"role": "user", "content": []}]
        )
        client._inference_config = MagicMock(return_value=None)
        client._tool_config = MagicMock(return_value=None)

        with patch.object(self.mod, "BedrockStream") as StreamMock:
            result = await client("model", [], GenerationSettings())

        self.session_mock.client.assert_called_once_with(
            "bedrock-runtime",
            region_name="us-east-1",
            endpoint_url="https://example.com",
        )
        self.client.converse_stream.assert_awaited_once_with(
            modelId="model", messages=[{"role": "user", "content": []}]
        )
        StreamMock.assert_called_once_with(
            events=self.client.converse_stream.return_value["stream"]
        )
        self.assertIs(result, StreamMock.return_value)
        await exit_stack.aclose()

    async def test_provider_instructions_are_rejected_before_api_call(self):
        exit_stack = AsyncExitStack()
        client = self.mod.BedrockClient(exit_stack=exit_stack)

        with self.assertRaisesRegex(AssertionError, "provider instructions"):
            await client("model", [], instructions="private policy")

        self.session_mock.client.assert_not_called()
        self.client.converse_stream.assert_not_awaited()
        self.client.converse.assert_not_awaited()
        await exit_stack.aclose()

    async def test_client_stream_payload_includes_configs(self):
        self.client.converse_stream.return_value = {"stream": AsyncIter([])}
        exit_stack = AsyncExitStack()
        client = self.mod.BedrockClient(exit_stack=exit_stack)
        client._system_prompt = MagicMock(return_value="sys")
        client._template_messages = MagicMock(
            return_value=[{"role": "user", "content": []}]
        )
        client._inference_config = MagicMock(return_value={"maxTokens": 3})
        client._tool_config = MagicMock(return_value={"tools": []})

        await client("model", [], GenerationSettings())

        kwargs = self.client.converse_stream.await_args.kwargs
        self.assertEqual(kwargs["system"], [{"text": "sys"}])
        self.assertEqual(kwargs["inferenceConfig"], {"maxTokens": 3})
        self.assertEqual(kwargs["toolConfig"], {"tools": []})
        await exit_stack.aclose()

    async def test_client_stream_invalid_model_identifier_hint(self):
        self.client.converse_stream.side_effect = FakeBedrockError(
            "ValidationException",
            "The provided model identifier is invalid.",
        )
        exit_stack = AsyncExitStack()
        client = self.mod.BedrockClient(
            exit_stack=exit_stack, region_name="us-east-1"
        )
        client._system_prompt = MagicMock(return_value=None)
        client._template_messages = MagicMock(
            return_value=[{"role": "user", "content": []}]
        )
        client._inference_config = MagicMock(return_value=None)
        client._tool_config = MagicMock(return_value=None)

        with self.assertRaises(ValueError) as exc_info:
            await client(
                "anthropic.claude-3-5-sonnet-20241022-v1:0",
                [],
                GenerationSettings(),
            )

        self.assertIn(
            "Invalid Amazon Bedrock model identifier "
            "'anthropic.claude-3-5-sonnet-20241022-v1:0'.",
            str(exc_info.exception),
        )
        self.assertIn(
            "Requested region: 'us-east-1'.", str(exc_info.exception)
        )
        self.assertIn(
            "Try 'us.' as the model ID prefix.", str(exc_info.exception)
        )
        await exit_stack.aclose()

    async def test_client_stream_invalid_model_identifier_hint_without_prefix(
        self,
    ):
        self.client.converse_stream.side_effect = FakeBedrockError(
            "ValidationException",
            "The provided model identifier is invalid.",
        )
        exit_stack = AsyncExitStack()
        client = self.mod.BedrockClient(
            exit_stack=exit_stack, region_name="ap-southeast-1"
        )
        client._system_prompt = MagicMock(return_value=None)
        client._template_messages = MagicMock(
            return_value=[{"role": "user", "content": []}]
        )
        client._inference_config = MagicMock(return_value=None)
        client._tool_config = MagicMock(return_value=None)

        with self.assertRaises(ValueError) as exc_info:
            await client(
                "anthropic.claude-sonnet-4-6",
                [],
                GenerationSettings(),
            )

        self.assertIn(
            "inference profile such as 'us.anthropic...'",
            str(exc_info.exception),
        )
        await exit_stack.aclose()

    async def test_client_stream_other_validation_error_is_not_rewritten(self):
        error = FakeBedrockError(
            "ValidationException", "Malformed request payload."
        )
        self.client.converse_stream.side_effect = error
        exit_stack = AsyncExitStack()
        client = self.mod.BedrockClient(exit_stack=exit_stack)
        client._system_prompt = MagicMock(return_value=None)
        client._template_messages = MagicMock(
            return_value=[{"role": "user", "content": []}]
        )
        client._inference_config = MagicMock(return_value=None)
        client._tool_config = MagicMock(return_value=None)

        with self.assertRaises(FakeBedrockError) as exc_info:
            await client("model", [], GenerationSettings())

        self.assertIs(exc_info.exception, error)
        await exit_stack.aclose()

    async def test_client_stream_end_of_life_model_hint(self):
        self.client.converse_stream.side_effect = FakeBedrockError(
            "ResourceNotFoundException",
            "This model version has reached the end of its life.",
        )
        exit_stack = AsyncExitStack()
        client = self.mod.BedrockClient(
            exit_stack=exit_stack, region_name="us-east-1"
        )
        client._system_prompt = MagicMock(return_value=None)
        client._template_messages = MagicMock(
            return_value=[{"role": "user", "content": []}]
        )
        client._inference_config = MagicMock(return_value=None)
        client._tool_config = MagicMock(return_value=None)

        with self.assertRaises(ValueError) as exc_info:
            await client(
                "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
                [],
                GenerationSettings(),
            )

        self.assertIn(
            "Amazon Bedrock model identifier "
            "'us.anthropic.claude-3-5-sonnet-20241022-v2:0' is no "
            "longer usable",
            str(exc_info.exception),
        )
        self.assertIn(
            "aws bedrock list-inference-profiles --region us-east-1",
            str(exc_info.exception),
        )
        await exit_stack.aclose()

    async def test_client_stream_end_of_life_model_hint_with_geo_prefix(
        self,
    ):
        self.client.converse_stream.side_effect = FakeBedrockError(
            "ResourceNotFoundException",
            "This model version has reached the end of its life.",
        )
        exit_stack = AsyncExitStack()
        client = self.mod.BedrockClient(
            exit_stack=exit_stack, region_name="eu-west-1"
        )
        client._system_prompt = MagicMock(return_value=None)
        client._template_messages = MagicMock(
            return_value=[{"role": "user", "content": []}]
        )
        client._inference_config = MagicMock(return_value=None)
        client._tool_config = MagicMock(return_value=None)

        with self.assertRaises(ValueError) as exc_info:
            await client(
                "anthropic.claude-3-5-sonnet-20241022-v2:0",
                [],
                GenerationSettings(),
            )

        self.assertIn(
            "Try an active 'eu.'-prefixed profile.",
            str(exc_info.exception),
        )
        await exit_stack.aclose()

    async def test_client_stream_missing_use_case_details_hint(self):
        self.client.converse_stream.side_effect = FakeBedrockError(
            "ResourceNotFoundException",
            "Model use case details have not been submitted for this "
            "account. Fill out the Anthropic use case details form "
            "before using the model.",
        )
        exit_stack = AsyncExitStack()
        client = self.mod.BedrockClient(
            exit_stack=exit_stack, region_name="us-east-1"
        )
        client._system_prompt = MagicMock(return_value=None)
        client._template_messages = MagicMock(
            return_value=[{"role": "user", "content": []}]
        )
        client._inference_config = MagicMock(return_value=None)
        client._tool_config = MagicMock(return_value=None)

        with self.assertRaises(ValueError) as exc_info:
            await client(
                "us.anthropic.claude-sonnet-4-6",
                [],
                GenerationSettings(),
            )

        self.assertIn(
            "Anthropic use-case details have not been submitted",
            str(exc_info.exception),
        )
        self.assertIn(
            "aws bedrock get-use-case-for-model-access --region us-east-1",
            str(exc_info.exception),
        )
        await exit_stack.aclose()

    async def test_client_stream_requires_inference_profile_hint(self):
        self.client.converse_stream.side_effect = FakeBedrockError(
            "ValidationException",
            "Invocation of model ID anthropic.claude-sonnet-4-6 with "
            "on-demand throughput isn’t supported. Retry your request "
            "with the ID or ARN of an inference profile that contains "
            "this model.",
        )
        exit_stack = AsyncExitStack()
        client = self.mod.BedrockClient(
            exit_stack=exit_stack, region_name="us-east-1"
        )
        client._system_prompt = MagicMock(return_value=None)
        client._template_messages = MagicMock(
            return_value=[{"role": "user", "content": []}]
        )
        client._inference_config = MagicMock(return_value=None)
        client._tool_config = MagicMock(return_value=None)

        with self.assertRaises(ValueError) as exc_info:
            await client(
                "anthropic.claude-sonnet-4-6",
                [],
                GenerationSettings(),
            )

        self.assertIn(
            "cannot be invoked directly with on-demand throughput",
            str(exc_info.exception),
        )
        self.assertIn(
            "'us.anthropic.claude-sonnet-4-6' or "
            "'global.anthropic.claude-sonnet-4-6'",
            str(exc_info.exception),
        )
        await exit_stack.aclose()

    async def test_client_without_stream(self):
        self.client.converse.return_value = {
            "output": {
                "message": {
                    "content": [
                        {"text": {"text": "hello"}},
                        {"text": {"text": " world"}},
                    ]
                }
            },
            "usage": {"inputTokens": 3},
        }
        exit_stack = AsyncExitStack()
        client = self.mod.BedrockClient(exit_stack=exit_stack)
        client._system_prompt = MagicMock(return_value=None)
        client._template_messages = MagicMock(
            return_value=[{"role": "user", "content": []}]
        )
        client._inference_config = MagicMock(return_value=None)
        client._tool_config = MagicMock(return_value=None)

        result = await client(
            "model",
            [],
            GenerationSettings(),
            use_async_generator=False,
        )

        items = [item async for item in result]
        self.assertEqual(
            accumulate_canonical_stream_items(items).answer_text,
            "hello world",
        )
        self.assertEqual(result.provider_family, "bedrock")
        self.assertEqual(result.usage, {"inputTokens": 3})
        self.client.converse.assert_awaited_once()
        await exit_stack.aclose()

    def test_inference_config_full(self):
        client = self.mod.BedrockClient(exit_stack=AsyncExitStack())
        settings = GenerationSettings(
            max_new_tokens=64,
            temperature=0.5,
            top_p=0.9,
            top_k=4,
            stop_strings="END",
        )
        config = client._inference_config(settings)
        self.assertEqual(
            config,
            {
                "maxTokens": 64,
                "temperature": 0.5,
                "topP": 0.9,
                "topK": 4,
                "stopSequences": ["END"],
            },
        )
        self.assertIsNone(client._inference_config(None))

    def test_tool_config_none(self):
        client = self.mod.BedrockClient(exit_stack=AsyncExitStack())
        tool = MagicMock()
        tool.json_schemas.return_value = None
        self.assertIsNone(client._tool_config(tool))

    def test_response_text_handles_missing_content(self):
        client = self.mod.BedrockClient(exit_stack=AsyncExitStack())
        self.assertEqual(client._response_text({}), "")
        response = {"output": {"message": {"content": ["invalid"]}}}
        self.assertEqual(client._response_text(response), "")

    def test_template_messages_excludes_roles_and_tool_calls(self):
        client = self.mod.BedrockClient(exit_stack=AsyncExitStack())
        tool_call = ToolCall(id="c1", name="pkg.tool", arguments=[])
        message = Message(
            role=MessageRole.ASSISTANT,
            content=MessageContentText(type="text", text="ok"),
            tool_calls=[tool_call],
        )
        templated = client._template_messages(
            [
                Message(role=MessageRole.SYSTEM, content="sys"),
                message,
            ],
            ["system"],
        )
        self.assertEqual(len(templated), 1)
        self.assertEqual(templated[0]["role"], "assistant")
        tool_block = templated[0]["content"][-1]["toolUse"]
        self.assertEqual(tool_block["name"], "avl_cGtnLnRvb2w")
        self.assertEqual(tool_block["toolUseId"], "c1")

    def test_format_content_image_base64(self):
        client = self.mod.BedrockClient(exit_stack=AsyncExitStack())
        image = MessageContentImage(
            type="image_url",
            image_url={"data": "YWJj", "mime_type": "image/jpeg"},
        )
        blocks = client._format_content(image)
        source = blocks[0]["image"]["source"]
        self.assertEqual(source["type"], "base64")
        self.assertEqual(source["mediaType"], "image/jpeg")
        self.assertEqual(source["data"], "YWJj")
        self.assertEqual(client._format_content(None), [])
        fallback = client._image_source({"uri": "blob"})
        self.assertEqual(fallback, {"type": "url", "url": "blob"})
        other = client._format_content(123)
        self.assertEqual(other, [{"text": "123"}])

    def test_format_content_document(self):
        client = self.mod.BedrockClient(exit_stack=AsyncExitStack())
        blocks = client._format_content(
            MessageContentFile(
                type="file",
                file={
                    "citations": True,
                    "context": "Focus on the appendix",
                    "file_data": "YWJj",
                    "filename": "Quarterly.Report.pdf",
                    "mime_type": "application/pdf",
                },
            )
        )
        self.assertEqual(blocks[0], {"text": ""})
        self.assertEqual(
            blocks[1],
            {
                "document": {
                    "citations": {"enabled": True},
                    "context": "Focus on the appendix",
                    "format": "pdf",
                    "name": "Quarterly Report",
                    "source": {"bytes": b"abc"},
                }
            },
        )

    def test_format_content_document_in_list(self):
        client = self.mod.BedrockClient(exit_stack=AsyncExitStack())
        blocks = client._format_content(
            [
                MessageContentFile(
                    type="file",
                    file={
                        "file_data": "YWJj",
                        "filename": "report.pdf",
                        "mime_type": "application/pdf",
                    },
                )
            ]
        )

        self.assertEqual(blocks[0], {"text": ""})
        self.assertEqual(blocks[1]["document"]["source"], {"bytes": b"abc"})

    def test_document_source_variants(self):
        client = self.mod.BedrockClient(exit_stack=AsyncExitStack())
        self.assertEqual(
            client._document_source(
                {
                    "file_url": "s3://bucket/path/report.pdf",
                    "bucket_owner": "123456789012",
                }
            ),
            {
                "s3Location": {
                    "uri": "s3://bucket/path/report.pdf",
                    "bucketOwner": "123456789012",
                }
            },
        )
        self.assertEqual(
            client._document_source(
                {
                    "file_data": "plain text",
                    "mime_type": "text/plain",
                }
            ),
            {"text": "plain text"},
        )
        self.assertEqual(
            client._document_source(
                {
                    "file_data": b64encode(b"alpha,beta").decode("ascii"),
                    "mime_type": "text/csv",
                }
            ),
            {"text": "alpha,beta"},
        )
        self.assertEqual(
            client._document_source({"file_data": b"abc"}),
            {"bytes": b"abc"},
        )
        with self.assertRaises(AssertionError):
            client._document_source(
                {"file_url": "https://example.com/report.pdf"}
            )

    def test_document_helper_variants(self):
        client = self.mod.BedrockClient(exit_stack=AsyncExitStack())

        self.assertEqual(
            client._document_format({"filename": "summary.MD"}), "md"
        )
        self.assertEqual(
            client._document_format({"title": "sheet.XLSX"}), "xlsx"
        )
        self.assertIsNone(client._document_format({"title": "notes"}))
        self.assertIsNone(client._file_uri({}))

    def test_tool_schemas_ignore_non_function(self):
        client = self.mod.BedrockClient(exit_stack=AsyncExitStack())
        tool = MagicMock()
        tool.json_schemas.return_value = [{"type": "noop"}]
        self.assertIsNone(client._tool_schemas(tool))

    def test_template_messages_and_tool_config(self):
        client = self.mod.BedrockClient(exit_stack=AsyncExitStack())

        @dataclass
        class Payload:
            value: int

        tool_call = ToolCall(id="id1", name="pkg.tool", arguments={"a": 1})
        tool_result = ToolCallResult(
            id="id1", name="pkg.tool", call=tool_call, result=Payload(2)
        )
        tool_error = ToolCallError(
            id="id2",
            name="pkg.tool",
            call=tool_call,
            error=ValueError("bad"),
            message="bad",
        )

        messages = [
            Message(role=MessageRole.USER, content="hello"),
            Message(
                role=MessageRole.DEVELOPER,
                content=MessageContentText(type="text", text="dev"),
            ),
            Message(
                role=MessageRole.ASSISTANT,
                content=[
                    MessageContentText(type="text", text="chunk"),
                    MessageContentImage(
                        type="image_url",
                        image_url={"url": "https://example.com"},
                    ),
                ],
            ),
            Message(role=MessageRole.TOOL, tool_call_result=tool_result),
            Message(role=MessageRole.TOOL, tool_call_error=tool_error),
        ]

        templated = client._template_messages(messages)
        self.assertEqual(templated[0]["role"], "user")
        self.assertEqual(templated[0]["content"][0]["text"], "hello")
        self.assertEqual(templated[1]["role"], "user")
        self.assertEqual(templated[1]["content"][0]["text"], "dev")
        self.assertEqual(
            templated[2]["content"][1]["image"]["source"]["url"],
            "https://example.com",
        )
        self.assertEqual(
            templated[3]["content"][0]["toolResult"]["toolUseId"],
            "id1",
        )
        self.assertEqual(
            templated[3]["content"][0]["toolResult"]["content"][0]["text"],
            '{"value": 2}',
        )
        self.assertEqual(
            templated[4]["content"][0]["toolResult"]["status"],
            "error",
        )
        self.assertEqual(
            templated[4]["content"][0]["toolResult"]["content"][0]["text"],
            '"bad"',
        )

        tool_manager = MagicMock()
        tool_manager.json_schemas.return_value = [
            {
                "type": "function",
                "function": {
                    "name": "pkg.tool",
                    "description": "desc",
                    "parameters": {"type": "object"},
                },
            }
        ]
        config = client._tool_config(tool_manager)
        self.assertEqual(
            config["tools"][0]["toolSpec"]["name"],
            "avl_cGtnLnRvb2w",
        )

    def test_template_messages_tool_diagnostic(self):
        client = self.mod.BedrockClient(exit_stack=AsyncExitStack())
        diagnostic = ToolCallDiagnostic(
            id="diag1",
            call_id="call1",
            requested_name="missing",
            code=ToolCallDiagnosticCode.UNKNOWN_TOOL,
            stage=ToolCallDiagnosticStage.RESOLVE,
            message="Tool is unknown.",
        )

        templated = client._template_messages(
            [
                Message(
                    role=MessageRole.TOOL,
                    name="missing",
                    arguments={"a": 1},
                    tool_call_diagnostic=diagnostic,
                )
            ]
        )

        tool_result = templated[0]["content"][0]["toolResult"]
        self.assertEqual(tool_result["toolUseId"], "call1")
        self.assertEqual(tool_result["status"], "error")
        payload = loads(tool_result["content"][0]["text"])
        self.assertEqual(payload["code"], "tool.unknown")
        self.assertEqual(payload["requested_name"], "missing")

    def test_template_messages_unanchored_tool_diagnostic(self):
        client = self.mod.BedrockClient(exit_stack=AsyncExitStack())
        diagnostic = ToolCallDiagnostic(
            id="diag1",
            code=ToolCallDiagnosticCode.MALFORMED_CALL,
            stage=ToolCallDiagnosticStage.PARSE,
            message="Tool call could not be parsed.",
        )

        templated = client._template_messages(
            [Message(role=MessageRole.TOOL, tool_call_diagnostic=diagnostic)]
        )

        self.assertEqual(templated[0]["role"], "assistant")
        payload = loads(templated[0]["content"][0]["text"])
        self.assertEqual(payload["code"], "tool_call.malformed")

    def test_model_loads_client(self):
        settings = TransformerEngineSettings(
            access_token="https://endpoint",
            base_url="us-east-1",
        )
        model = self.mod.BedrockModel("model", settings)
        client = model._load_model()
        self.assertIsInstance(client, self.mod.BedrockClient)
        self.assertEqual(client._region_name, "us-east-1")
        self.assertEqual(client._endpoint_url, "https://endpoint")


class BedrockStreamHelpersTest(TestCase):
    def test_error_and_region_helpers_false_paths(self):
        _, _, patcher = patch_bedrock_imports()
        self.addCleanup(patcher.stop)
        module = import_module("avalan.model.nlp.text.vendor.bedrock")

        plain_error = Exception("plain")
        self.assertIsNone(module._bedrock_error_code(plain_error))
        self.assertEqual(module._bedrock_error_message(plain_error), "plain")

        missing_details_error = Exception("missing details")
        missing_details_error.response = {"Error": "invalid"}
        self.assertIsNone(module._bedrock_error_code(missing_details_error))

        numeric_code_error = Exception("numeric code")
        numeric_code_error.response = {"Error": {"Code": 123}}
        self.assertIsNone(module._bedrock_error_code(numeric_code_error))

        numeric_message_error = Exception("numeric message")
        numeric_message_error.response = {"Error": {"Message": 123}}
        self.assertEqual(
            module._bedrock_error_message(numeric_message_error),
            "numeric message",
        )
        self.assertIsNone(module._geo_inference_prefix(None))
        self.assertEqual(module._geo_inference_prefix("eu-west-1"), "eu.")
        self.assertIsNone(module._geo_inference_prefix("ap-southeast-1"))

    def test_string_helper(self):
        _, _, patcher = patch_bedrock_imports()
        self.addCleanup(patcher.stop)
        module = import_module("avalan.model.nlp.text.vendor.bedrock")
        self.assertEqual(module._string({"text": {"text": "value"}}), "value")
        self.assertIsNone(module._string(123))

    def test_string_helper_reasoning_and_string(self):
        _, _, patcher = patch_bedrock_imports()
        self.addCleanup(patcher.stop)
        module = import_module("avalan.model.nlp.text.vendor.bedrock")
        reasoning = {"reasoningText": {"string": "done"}}
        self.assertEqual(module._string(reasoning), "done")
        wrapped = {"string": {"text": "inner"}}
        self.assertEqual(module._string(wrapped), "inner")

    def test_get_helper_reads_attributes(self):
        _, _, patcher = patch_bedrock_imports()
        self.addCleanup(patcher.stop)
        module = import_module("avalan.model.nlp.text.vendor.bedrock")
        obj = SimpleNamespace(value="x")
        self.assertEqual(module._get(obj, "value"), "x")

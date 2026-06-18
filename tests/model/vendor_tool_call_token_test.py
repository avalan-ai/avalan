from asyncio import run
from json import loads
from typing import Any, cast
from unittest import TestCase
from uuid import uuid4

from avalan.entities import (
    Message,
    MessageContentFile,
    MessageContentImage,
    MessageContentText,
    MessageRole,
    ToolCall,
    ToolCallToken,
)
from avalan.model.stream import (
    CanonicalStreamItem,
    StreamItemKind,
    StreamTerminalOutcome,
    StreamValidationError,
    stream_channel_for_kind,
)
from avalan.model.vendor import (
    TextGenerationVendor,
    TextGenerationVendorStream,
)


def _canonical_item(
    kind: StreamItemKind,
    sequence: int,
    *,
    text_delta: str | None = None,
    usage: object | None = None,
    metadata: dict[str, object] | None = None,
) -> CanonicalStreamItem:
    return CanonicalStreamItem(
        stream_session_id="source-stream",
        run_id="source-run",
        turn_id="source-turn",
        sequence=sequence,
        kind=kind,
        channel=stream_channel_for_kind(kind),
        text_delta=text_delta,
        usage=cast(Any, usage),
        terminal_outcome=(
            StreamTerminalOutcome.COMPLETED
            if kind is StreamItemKind.STREAM_COMPLETED
            else None
        ),
        metadata=cast(dict[str, Any], metadata or {}),
    )


class VendorBuildToolCallTokenTestCase(TestCase):
    def test_call_is_abstract_boundary(self) -> None:
        with self.assertRaises(NotImplementedError):
            run(TextGenerationVendor()("model", []))

    def test_system_prompt_returns_string_content(self) -> None:
        prompt = TextGenerationVendor()._system_prompt(
            [
                Message(role=MessageRole.USER, content="skip"),
                Message(role=MessageRole.SYSTEM, content="system"),
            ]
        )

        self.assertEqual(prompt, "system")

    def test_system_prompt_returns_text_content_or_none(self) -> None:
        vendor = TextGenerationVendor()

        self.assertEqual(
            vendor._system_prompt(
                [
                    Message(
                        role=MessageRole.SYSTEM,
                        content=MessageContentText(type="text", text="system"),
                    )
                ]
            ),
            "system",
        )
        self.assertIsNone(
            vendor._system_prompt(
                [
                    Message(
                        role=MessageRole.SYSTEM,
                        content=MessageContentImage(
                            type="image_url", image_url={"url": "image"}
                        ),
                    )
                ]
            )
        )
        self.assertIsNone(
            vendor._system_prompt(
                [Message(role=MessageRole.USER, content="skip")]
            )
        )

    def test_template_messages_wraps_content_blocks(self) -> None:
        messages = [
            Message(
                role=MessageRole.USER,
                content=MessageContentText(type="text", text="hello"),
            ),
            Message(
                role=MessageRole.USER,
                content=MessageContentImage(
                    type="image_url", image_url={"url": "image"}
                ),
            ),
            Message(
                role=MessageRole.USER,
                content=MessageContentFile(
                    type="file", file={"file_id": "file"}
                ),
            ),
            Message(role=MessageRole.USER, content=None),
        ]

        templated = TextGenerationVendor()._template_messages(messages)

        self.assertEqual(
            templated,
            [
                {"role": "user", "content": "hello"},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": "image"}}
                    ],
                },
                {
                    "role": "user",
                    "content": [{"type": "file", "file": {"file_id": "file"}}],
                },
                {"role": "user", "content": "None"},
            ],
        )

    def test_template_messages_wraps_strings_lists_and_excluded_roles(
        self,
    ) -> None:
        messages = [
            Message(role=MessageRole.SYSTEM, content="skip"),
            Message(role=MessageRole.USER, content="hello"),
            Message(
                role=MessageRole.USER,
                content=[
                    MessageContentText(type="text", text="listed"),
                    MessageContentImage(
                        type="image_url", image_url={"url": "image"}
                    ),
                ],
            ),
        ]

        templated = TextGenerationVendor()._template_messages(
            messages,
            exclude_roles=["system"],
        )

        self.assertEqual(
            templated,
            [
                {"role": "user", "content": "hello"},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "listed"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "image"},
                        },
                    ],
                },
            ],
        )

    def test_stream_exposes_provider_metadata(self) -> None:
        async def generator():
            if False:
                yield _canonical_item(StreamItemKind.STREAM_STARTED, 0)

        stream = TextGenerationVendorStream(
            generator(),
            provider_family="openai",
            usage={"tokens": 1},
        )

        self.assertEqual(stream.provider_family, "openai")
        self.assertEqual(stream.usage, {"tokens": 1})

    def test_stream_iterates_public_canonical_items_and_metadata(
        self,
    ) -> None:
        async def generator():
            yield _canonical_item(StreamItemKind.STREAM_STARTED, 0)
            yield _canonical_item(
                StreamItemKind.ANSWER_DELTA,
                1,
                text_delta="token",
            )
            yield _canonical_item(StreamItemKind.ANSWER_DONE, 2)
            yield _canonical_item(StreamItemKind.STREAM_COMPLETED, 3)
            yield _canonical_item(StreamItemKind.STREAM_CLOSED, 4)

        async def collect_items() -> list[CanonicalStreamItem]:
            stream = TextGenerationVendorStream(
                generator(),
                provider_family="openai",
                usage={"output_tokens": 1},
            )
            iterator = stream()
            self.assertIsNot(iterator, stream)
            return [item async for item in iterator]

        items = run(collect_items())

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
        self.assertEqual(items[0].stream_session_id, "vendor-stream")
        self.assertEqual(items[1].text_delta, "token")
        self.assertEqual({item.provider_family for item in items}, {"openai"})
        self.assertEqual(items[-2].usage, {"output_tokens": 1})

        with self.assertRaises(AssertionError):
            TextGenerationVendorStream(cast(Any, None)).__aiter__()

    def test_stream_rejects_legacy_generator_items(self) -> None:
        class Source:
            def __init__(self) -> None:
                self.close_count = 0

            async def aclose(self) -> None:
                self.close_count += 1

        async def generator():
            yield "token"

        async def collect_stream() -> Source:
            source = Source()
            stream = TextGenerationVendorStream(
                cast(Any, generator()),
                sources=(source,),
            )
            with self.assertRaisesRegex(
                StreamValidationError,
                "unsupported legacy vendor stream item",
            ):
                async for _ in stream():
                    pass
            return source

        source = run(collect_stream())

        self.assertEqual(source.close_count, 1)

    def test_stream_accepts_empty_canonical_generator(self) -> None:
        async def generator():
            if False:
                yield _canonical_item(StreamItemKind.STREAM_STARTED, 0)

        async def collect_stream() -> list[CanonicalStreamItem]:
            stream = TextGenerationVendorStream(generator())
            return [item async for item in stream()]

        self.assertEqual(run(collect_stream()), [])

    def test_stream_rejects_legacy_item_after_canonical_first_item(
        self,
    ) -> None:
        async def generator():
            yield _canonical_item(StreamItemKind.STREAM_STARTED, 0)
            yield "legacy"

        async def collect_stream() -> list[CanonicalStreamItem]:
            stream = TextGenerationVendorStream(cast(Any, generator()))
            items: list[CanonicalStreamItem] = []
            with self.assertRaisesRegex(
                StreamValidationError,
                "unsupported legacy vendor stream item",
            ):
                async for item in stream():
                    items.append(item)
            return items

        items = run(collect_stream())

        self.assertEqual(
            [item.kind for item in items],
            [StreamItemKind.STREAM_STARTED],
        )

    def test_stream_preserves_canonical_source_metadata_and_usage(
        self,
    ) -> None:
        async def generator():
            yield _canonical_item(
                StreamItemKind.STREAM_STARTED,
                0,
                metadata={"source": "canonical"},
            )
            yield _canonical_item(
                StreamItemKind.ANSWER_DELTA,
                1,
                text_delta="canonical",
            )
            yield _canonical_item(StreamItemKind.ANSWER_DONE, 2)
            yield _canonical_item(
                StreamItemKind.STREAM_COMPLETED,
                3,
                usage={"output_tokens": 2},
            )
            yield _canonical_item(StreamItemKind.STREAM_CLOSED, 4)

        async def collect_items() -> list[CanonicalStreamItem]:
            stream = TextGenerationVendorStream(
                generator(),
                provider_family="openai",
                usage={"output_tokens": 99},
            )
            return [item async for item in stream()]

        items = run(collect_items())

        self.assertEqual(items[0].metadata, {"source": "canonical"})
        self.assertEqual(items[1].text_delta, "canonical")
        self.assertEqual(items[-2].usage, {"output_tokens": 2})
        self.assertEqual({item.provider_family for item in items}, {"openai"})

    def test_stream_cancel_is_idempotent(self) -> None:
        class Source:
            def __init__(self) -> None:
                self.cancel_count = 0

            async def cancel(self) -> None:
                self.cancel_count += 1

        async def generator():
            yield "token"

        async def cancel_twice() -> Source:
            source = Source()
            stream = TextGenerationVendorStream(generator(), sources=(source,))
            await stream.cancel()
            await stream.cancel()
            return source

        source = run(cancel_twice())

        self.assertEqual(source.cancel_count, 1)

    def test_stream_iteration_closes_sources(self) -> None:
        class Source:
            def __init__(self) -> None:
                self.close_count = 0

            async def aclose(self) -> None:
                self.close_count += 1

        async def generator():
            yield _canonical_item(StreamItemKind.STREAM_STARTED, 0)
            yield _canonical_item(StreamItemKind.STREAM_COMPLETED, 1)

        async def collect_stream() -> Source:
            source = Source()
            stream = TextGenerationVendorStream(generator(), sources=(source,))
            async for _ in stream():
                pass
            return source

        source = run(collect_stream())

        self.assertEqual(source.close_count, 1)

    def test_stream_close_accepts_sync_sources_and_dedupes(self) -> None:
        class Source:
            def __init__(self) -> None:
                self.close_count = 0

            def aclose(self) -> None:
                self.close_count += 1

        async def generator():
            yield "token"

        async def close_stream() -> Source:
            source = Source()
            stream = TextGenerationVendorStream(
                generator(), sources=(source, source)
            )
            await stream.aclose()
            await stream.aclose()
            return source

        source = run(close_stream())

        self.assertEqual(source.close_count, 1)

    def test_stream_close_rejects_bad_sync_result(self) -> None:
        class Source:
            def aclose(self) -> object:
                return object()

        async def generator():
            yield "token"

        async def close_stream() -> None:
            stream = TextGenerationVendorStream(
                generator(), sources=(Source(),)
            )
            await stream.aclose()

        with self.assertRaises(AssertionError):
            run(close_stream())

    def test_stream_close_reports_single_source_error(self) -> None:
        class Source:
            async def aclose(self) -> None:
                raise RuntimeError("source close failed")

        async def generator():
            yield "token"

        async def close_stream() -> None:
            stream = TextGenerationVendorStream(
                generator(), sources=(Source(),)
            )
            await stream.aclose()

        with self.assertRaisesRegex(RuntimeError, "source close failed"):
            run(close_stream())

    def test_stream_close_reports_multiple_source_errors(self) -> None:
        class Source:
            def __init__(self, name: str) -> None:
                self._name = name

            async def aclose(self) -> None:
                raise RuntimeError(f"{self._name} close failed")

        async def generator():
            yield "token"

        async def close_stream() -> None:
            stream = TextGenerationVendorStream(
                generator(), sources=(Source("first"), Source("second"))
            )
            await stream.aclose()

        with self.assertRaises(ExceptionGroup) as context:
            run(close_stream())

        self.assertEqual(
            [str(error) for error in context.exception.exceptions],
            ["first close failed", "second close failed"],
        )

    def test_encode_tool_name_preserves_provider_safe_names(self) -> None:
        self.assertEqual(
            TextGenerationVendor.encode_tool_name("tool_name"),
            "tool_name",
        )

    def test_build_tool_call_token_from_string_json(self) -> None:
        token = TextGenerationVendor.build_tool_call_token(
            call_id="1",
            tool_name=TextGenerationVendor.encode_tool_name("pkg.tool"),
            arguments='{"a": 1}',
        )
        expected = ToolCallToken(
            token=(
                '<tool_call>{"name": "pkg.tool", "arguments": {"a":'
                ' 1}, "id": "1"}</tool_call>'
            ),
            call=ToolCall(
                id="1",
                name="pkg.tool",
                arguments={"a": 1},
                provider_name="avl_cGtnLnRvb2w",
                provider_name_encoded=True,
            ),
            provider_name="avl_cGtnLnRvb2w",
        )
        self.assertEqual(token, expected)

    def test_build_tool_call_token_preserves_plain_provider_name(self) -> None:
        token = TextGenerationVendor.build_tool_call_token(
            call_id="1",
            tool_name="pkg__tool",
            arguments='{"a": 1}',
        )
        expected = ToolCallToken(
            token=(
                '<tool_call>{"name": "pkg__tool", "arguments": {"a":'
                ' 1}, "id": "1"}</tool_call>'
            ),
            call=ToolCall(
                id="1",
                name="pkg__tool",
                arguments={"a": 1},
                provider_name="pkg__tool",
            ),
            provider_name="pkg__tool",
        )
        self.assertEqual(token, expected)

    def test_build_tool_call_token_keeps_call_as_executable_boundary(
        self,
    ) -> None:
        token = TextGenerationVendor.build_tool_call_token(
            call_id="call_1",
            tool_name=TextGenerationVendor.encode_tool_name("pkg.tool"),
            arguments={"value": 3},
        )

        self.assertEqual(
            token.call,
            ToolCall(
                id="call_1",
                name="pkg.tool",
                arguments={"value": 3},
                provider_name="avl_cGtnLnRvb2w",
                provider_name_encoded=True,
            ),
        )
        self.assertEqual(token.provider_name, "avl_cGtnLnRvb2w")
        payload = loads(
            token.token.removeprefix("<tool_call>").removesuffix(
                "</tool_call>"
            )
        )
        self.assertEqual(
            payload,
            {"name": "pkg.tool", "arguments": {"value": 3}, "id": "call_1"},
        )

    def test_build_tool_call_token_preserves_malformed_encoded_name(
        self,
    ) -> None:
        token = TextGenerationVendor.build_tool_call_token(
            call_id="call-1",
            tool_name="avl_notbase64",
            arguments={"value": 3},
        )

        self.assertEqual(
            token.call,
            ToolCall(
                id="call-1",
                name="avl_notbase64",
                arguments={"value": 3},
                provider_name="avl_notbase64",
                provider_name_encoded=True,
            ),
        )
        self.assertEqual(token.provider_name, "avl_notbase64")
        payload = loads(
            token.token.removeprefix("<tool_call>").removesuffix(
                "</tool_call>"
            )
        )
        self.assertEqual(
            payload,
            {
                "name": "avl_notbase64",
                "arguments": {"value": 3},
                "id": "call-1",
            },
        )

    def test_build_tool_call_token_handles_invalid_json_str(self) -> None:
        token = TextGenerationVendor.build_tool_call_token(
            call_id=None,
            tool_name="tool",
            arguments='{"a": }',
        )
        expected = ToolCallToken(
            token='<tool_call>{"name": "tool", "arguments": {}}</tool_call>',
            call=ToolCall(
                id=None,
                name="tool",
                arguments={},
                provider_name="tool",
                provider_arguments_malformed=True,
            ),
            provider_name="tool",
        )
        self.assertEqual(token, expected)

    def test_build_tool_call_token_marks_non_object_json_str(self) -> None:
        for arguments in ("null", '["a"]', '"text"'):
            with self.subTest(arguments=arguments):
                token = TextGenerationVendor.build_tool_call_token(
                    call_id="call-1",
                    tool_name="tool",
                    arguments=arguments,
                )

                self.assertEqual(token.call.arguments, {})
                self.assertTrue(token.call.provider_arguments_malformed)
                self.assertEqual(
                    token.token,
                    '<tool_call>{"name": "tool", "arguments": {},'
                    ' "id": "call-1"}</tool_call>',
                )

    def test_build_tool_call_token_keeps_empty_object_arguments_valid(
        self,
    ) -> None:
        token = TextGenerationVendor.build_tool_call_token(
            call_id="call-1",
            tool_name="tool",
            arguments="{}",
        )

        self.assertEqual(token.call.arguments, {})
        self.assertFalse(token.call.provider_arguments_malformed)

    def test_build_tool_call_token_from_dict(self) -> None:
        token = TextGenerationVendor.build_tool_call_token(
            call_id="2",
            tool_name=None,
            arguments={"b": 2},
        )
        expected = ToolCallToken(
            token=(
                '<tool_call>{"name": "", "arguments": {"b": 2},'
                ' "id": "2"}</tool_call>'
            ),
            call=ToolCall(id="2", name="", arguments={"b": 2}),
        )
        self.assertEqual(token, expected)

    def test_build_tool_call_token_preserves_object_call_id(self) -> None:
        call_id = uuid4()
        token = TextGenerationVendor.build_tool_call_token(
            call_id=call_id,
            tool_name="tool",
            arguments={"b": 2},
        )
        expected = ToolCallToken(
            token=(
                '<tool_call>{"name": "tool", "arguments": {"b": 2}, '
                f'"id": "{call_id}"}}</tool_call>'
            ),
            call=ToolCall(
                id=str(call_id),
                name="tool",
                arguments={"b": 2},
                provider_name="tool",
            ),
            provider_name="tool",
        )
        self.assertEqual(token, expected)

    def test_build_tool_call_token_rejects_invalid_provider_name(self) -> None:
        with self.assertRaises(AssertionError):
            TextGenerationVendor.build_tool_call_token(
                call_id="1",
                tool_name="pkg.tool",
                arguments={},
            )

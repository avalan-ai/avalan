from asyncio import run
from json import loads
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
from avalan.model.vendor import (
    TextGenerationVendor,
    TextGenerationVendorStream,
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

    def test_stream_exposes_provider_metadata(self) -> None:
        async def generator():
            yield "token"

        stream = TextGenerationVendorStream(
            generator(),
            provider_family="openai",
            usage={"tokens": 1},
        )

        self.assertEqual(stream.provider_family, "openai")
        self.assertEqual(stream.usage, {"tokens": 1})

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
            ),
            provider_name="tool",
        )
        self.assertEqual(token, expected)

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

"""Verify native provider encodings for transient tool result images."""

from base64 import b64decode
from contextlib import AsyncExitStack
from unittest import TestCase

from avalan.entities import (
    Message,
    MessageRole,
    ToolCall,
    ToolCallResult,
    ToolResultImage,
    ToolResultImageDeliveryError,
    ToolResultText,
)
from avalan.model.nlp.text.vendor.anthropic import AnthropicClient
from avalan.model.nlp.text.vendor.bedrock import BedrockClient
from avalan.model.nlp.text.vendor.google import GoogleClient
from avalan.model.nlp.text.vendor.huggingface import HuggingfaceClient
from avalan.model.nlp.text.vendor.litellm import LiteLLMClient


class ToolResultImageProviderTest(TestCase):
    """Require native, ordered image blocks or explicit delivery failure."""

    def test_anthropic_tool_result_retains_text_and_image_order(self) -> None:
        result, first, second = _result()

        message = AnthropicClient._tool_result_message(result)

        blocks = message["content"][0]["content"]
        self.assertEqual(
            [block["type"] for block in blocks],
            ["text", "text", "image", "text", "image"],
        )
        self.assertEqual(
            b64decode(blocks[2]["source"]["data"]),
            first,
        )
        self.assertEqual(
            b64decode(blocks[4]["source"]["data"]),
            second,
        )

    def test_bedrock_tool_result_retains_native_bytes_and_order(self) -> None:
        result, first, second = _result()
        client = BedrockClient(exit_stack=AsyncExitStack())

        message = client._tool_result_message(result)

        blocks = message["content"][0]["toolResult"]["content"]
        self.assertEqual(
            [next(iter(block)) for block in blocks],
            ["text", "text", "image", "text", "image"],
        )
        self.assertEqual(blocks[2]["image"]["source"]["bytes"], first)
        self.assertEqual(blocks[4]["image"]["source"]["bytes"], second)

    def test_google_tool_result_retains_native_order(self) -> None:
        result, first, second = _result()
        client = object.__new__(GoogleClient)

        messages = client._template_messages(
            [Message(role=MessageRole.TOOL, tool_call_result=result)]
        )

        parts = messages[0]["parts"]
        self.assertEqual(
            [next(iter(part)) for part in parts],
            [
                "function_response",
                "text",
                "inline_data",
                "text",
                "inline_data",
            ],
        )
        self.assertEqual(b64decode(parts[2]["inline_data"]["data"]), first)
        self.assertEqual(b64decode(parts[4]["inline_data"]["data"]), second)

    def test_openai_compatible_tool_result_fails_without_image_carrier(self):
        result, _first, _second = _result()
        client = LiteLLMClient()

        with self.assertRaises(ToolResultImageDeliveryError):
            client._template_messages(
                [Message(role=MessageRole.TOOL, tool_call_result=result)]
            )

    def test_base_provider_failure_is_explicit_for_tool_images(self) -> None:
        result, _first, _second = _result()
        client = object.__new__(HuggingfaceClient)

        with self.assertRaises(ToolResultImageDeliveryError):
            client._template_messages(
                [Message(role=MessageRole.TOOL, tool_call_result=result)]
            )


def _result() -> tuple[ToolCallResult, bytes, bytes]:
    first = b"provider first image"
    second = b"provider second image"
    return (
        ToolCallResult(
            id="result-1",
            name="shell.montage",
            arguments={},
            call=ToolCall(id="call-1", name="shell.montage", arguments={}),
            result="metadata",
            content=(
                ToolResultText(text="first metadata"),
                ToolResultImage(
                    data=first,
                    media_type="image/png",
                    width=1,
                    height=1,
                ),
                ToolResultText(text="second metadata"),
                ToolResultImage(
                    data=second,
                    media_type="image/jpeg",
                    width=1,
                    height=1,
                ),
            ),
        ),
        first,
        second,
    )

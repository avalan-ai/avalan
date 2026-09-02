"""Cover typed multimodal content returned through the generic tool manager."""

from unittest import IsolatedAsyncioTestCase, main

from avalan.entities import (
    ToolCall,
    ToolCallContext,
    ToolCallResult,
    ToolManagerSettings,
    ToolResult,
    ToolResultImage,
    ToolResultImageDetail,
    ToolResultText,
)
from avalan.tool import Tool, ToolSet
from avalan.tool.manager import ToolManager


class ToolResultTest(IsolatedAsyncioTestCase):
    """Require generic tools to preserve ordered text and image blocks."""

    async def test_manager_preserves_typed_content_and_text_only_tools(self):
        manager = ToolManager.create_instance(
            available_toolsets=[
                ToolSet(namespace="example", tools=[_Tool(), _TextTool()])
            ],
            enable_tools=["example.result", "example.text"],
            settings=ToolManagerSettings(),
        )

        multimodal = await manager.execute_call(
            ToolCall(id="result", name="example.result", arguments={}),
            context=ToolCallContext(),
        )
        text_only = await manager.execute_call(
            ToolCall(id="text", name="example.text", arguments={}),
            context=ToolCallContext(),
        )

        self.assertIsInstance(multimodal, ToolCallResult)
        assert isinstance(multimodal, ToolCallResult)
        self.assertEqual(multimodal.result, {"artifact": "sample"})
        self.assertEqual(
            [type(block) for block in multimodal.content],
            [ToolResultText, ToolResultImage, ToolResultText],
        )
        image = multimodal.content[1]
        assert isinstance(image, ToolResultImage)
        self.assertEqual(image.data, b"exact image bytes")
        self.assertIsInstance(text_only, ToolCallResult)
        assert isinstance(text_only, ToolCallResult)
        self.assertEqual(text_only.result, "unchanged")
        self.assertEqual(text_only.content, ())

    def test_typed_tool_content_rejects_invalid_text_and_image_blocks(
        self,
    ) -> None:
        """Reject malformed typed blocks before any continuation sees them."""
        valid = ToolResultImage(data=b"pixels", media_type="image/png")
        self.assertTrue(valid.available)
        unavailable = ToolResultImage(
            media_type="image/png",
            unavailable_reason="The execution target cannot retrieve pixels.",
        )
        self.assertFalse(unavailable.available)

        invalid_texts = (
            {"type": "other", "text": "text"},
            {"text": 1},
        )
        for values in invalid_texts:
            with self.subTest(values=values):
                with self.assertRaises((TypeError, ValueError)):
                    ToolResultText(**values)  # type: ignore[arg-type]

        invalid_images = (
            {"type": "other", "data": b"pixels", "media_type": "image/png"},
            {"data": b"pixels", "media_type": "text/plain"},
            {
                "data": b"pixels",
                "media_type": "image/png",
                "detail": "high",
            },
            {"data": "pixels", "media_type": "image/png"},
            {"media_type": "image/png"},
            {
                "data": b"pixels",
                "media_type": "image/png",
                "unavailable_reason": "unavailable",
            },
            {
                "media_type": "image/png",
                "unavailable_reason": 1,
            },
            {"data": b"pixels", "media_type": "image/png", "sha256": "bad"},
            {
                "data": b"pixels",
                "media_type": "image/png",
                "sha256": "0" * 64,
            },
            {"data": b"pixels", "media_type": "image/png", "width": 0},
            {"data": b"pixels", "media_type": "image/png", "height": True},
        )
        for values in invalid_images:
            with self.subTest(values=values):
                with self.assertRaises((TypeError, ValueError)):
                    ToolResultImage(**values)  # type: ignore[arg-type]

        self.assertEqual(valid.detail, ToolResultImageDetail.AUTO)

    def test_tool_result_content_requires_an_ordered_block_tuple(self) -> None:
        """Reject mutable or unknown blocks in tool continuation content."""
        call = ToolCall(id="call", name="example.result", arguments={})
        invalid_content = ([ToolResultText(text="text")], ("text",))
        for content in invalid_content:
            with self.subTest(content=content):
                with self.assertRaises(TypeError):
                    ToolResult(  # type: ignore[arg-type]
                        result="metadata",
                        content=content,
                    )
                with self.assertRaises(TypeError):
                    ToolCallResult(  # type: ignore[arg-type]
                        id="result",
                        name="example.result",
                        arguments={},
                        call=call,
                        result="metadata",
                        content=content,
                    )


class _Tool(Tool):
    """Return one typed result or a conventional text-only result.

    Returns:
        Typed content for the requested example tool.
    """

    def __init__(self) -> None:
        super().__init__()
        self.__name__ = "result"

    async def __call__(self, *, context: ToolCallContext) -> ToolResult:
        return ToolResult(
            result={"artifact": "sample"},
            content=(
                ToolResultText(text="first"),
                ToolResultImage(
                    data=b"exact image bytes",
                    media_type="image/png",
                    width=1,
                    height=1,
                ),
                ToolResultText(text="last"),
            ),
        )


class _TextTool(Tool):
    """Return one conventional text-only result.

    Returns:
        Conventional text output.
    """

    def __init__(self) -> None:
        super().__init__()
        self.__name__ = "text"

    async def __call__(self, *, context: ToolCallContext) -> str:
        return "unchanged"


if __name__ == "__main__":
    main()

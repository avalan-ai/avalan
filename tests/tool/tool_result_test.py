"""Cover typed multimodal content returned through the generic tool manager."""

from unittest import IsolatedAsyncioTestCase, main

from avalan.entities import (
    ToolCall,
    ToolCallContext,
    ToolCallResult,
    ToolManagerSettings,
    ToolResult,
    ToolResultImage,
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

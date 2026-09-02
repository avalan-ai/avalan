"""Shared validation for model-visible images returned by tools."""

from ....entities import (
    ToolCallResult,
    ToolResultContent,
    ToolResultImage,
    ToolResultImageDeliveryError,
    ToolResultText,
)


def tool_result_content(
    result: ToolCallResult,
) -> tuple[ToolResultContent, ...]:
    """Return ordered multimodal blocks retained on one tool result."""
    return result.content


def tool_result_images(result: ToolCallResult) -> tuple[ToolResultImage, ...]:
    """Return ordered image blocks retained on one tool result."""
    return tuple(
        item for item in result.content if isinstance(item, ToolResultImage)
    )


def tool_result_text_blocks(
    result: ToolCallResult,
) -> tuple[ToolResultText, ...]:
    """Return ordered text blocks retained on one tool result."""
    return tuple(
        item for item in result.content if isinstance(item, ToolResultText)
    )


def required_image_data(image: ToolResultImage) -> bytes:
    """Return transient pixels or fail without claiming the model saw them."""
    if image.data is not None:
        return image.data
    detail = image.unavailable_reason or "Image pixels are unavailable."
    raise ToolResultImageDeliveryError(detail)

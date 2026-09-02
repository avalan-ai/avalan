"""Build transient model-visible images from approved shell artifacts."""

from ...entities import (
    ToolResult,
    ToolResultImage,
    ToolResultImageDetail,
    ToolResultText,
)
from .entities import (
    GENERATED_FILE_MATERIALIZED_PATH_METADATA_KEY,
    GeneratedFile,
    ShellExecutionStatus,
    ShellFormattedResult,
)
from .filesystem import (
    ShellPathMetadata,
    inspect_path,
    read_validated_bytes,
    resolve_policy_path,
    validate_supported_image_bytes,
)
from .settings import ShellToolSettings

from base64 import b64decode
from binascii import Error as BinasciiError
from pathlib import Path


async def montage_tool_result(
    formatted: ShellFormattedResult,
    *,
    settings: ShellToolSettings,
    detail: ToolResultImageDetail = ToolResultImageDetail.AUTO,
) -> ToolResult:
    """Return text metadata and exact transient image blocks for a montage."""
    result = formatted.execution_result
    if result.status is not ShellExecutionStatus.COMPLETED:
        return ToolResult(result=formatted)

    images: list[ToolResultImage] = []
    for generated in result.generated_files:
        if (
            not generated.media_type.startswith("image/")
            or generated.truncated
        ):
            continue
        images.append(
            await _generated_image_block(
                generated,
                backend=result.backend,
                settings=settings,
                detail=detail,
            )
        )
    if not images:
        return ToolResult(result=formatted)
    return ToolResult(
        result=formatted,
        content=(ToolResultText(text=str(formatted)), *images),
    )


async def viewed_image_tool_result(
    path: Path,
    display_path: str,
    path_metadata: ShellPathMetadata,
    *,
    settings: ShellToolSettings,
    detail: ToolResultImageDetail,
) -> ToolResult:
    """Read one policy-approved local image into a transient tool result."""
    data = await read_validated_bytes(
        path,
        path_metadata,
        max_bytes=settings.max_ocr_input_bytes,
    )
    media_type, width, height = validate_supported_image_bytes(
        data,
        max_long_edge_pixels=settings.max_raster_long_edge_pixels,
        max_pixels=settings.max_ocr_pixels,
    )
    _validate_dimensions(width, height, settings)
    image = ToolResultImage(
        data=data,
        media_type=media_type,
        detail=detail,
        width=width,
        height=height,
    )
    text_metadata = (
        f"Viewed image: {display_path}\n"
        f"media_type: {media_type}\n"
        f"bytes: {len(data)}\n"
        f"sha256: {image.sha256}\n"
        f"width: {width}\n"
        f"height: {height}\n"
        f"detail: {detail.value}"
    )
    return ToolResult(
        result=text_metadata,
        content=(ToolResultText(text=text_metadata), image),
    )


async def _generated_image_block(
    generated: GeneratedFile,
    *,
    backend: str,
    settings: ShellToolSettings,
    detail: ToolResultImageDetail,
) -> ToolResultImage:
    data = await _generated_image_bytes(
        generated,
        backend=backend,
        settings=settings,
    )
    if data is None:
        return ToolResultImage(
            media_type=generated.media_type,
            detail=detail,
            sha256=generated.sha256,
            width=generated.width,
            height=generated.height,
            unavailable_reason=(
                "The execution target did not return image bytes for model "
                "delivery. Avalan will not read a host fallback."
            ),
        )
    media_type, width, height = validate_supported_image_bytes(
        data,
        max_long_edge_pixels=settings.max_raster_long_edge_pixels,
        max_pixels=settings.max_ocr_pixels,
    )
    if media_type != generated.media_type:
        raise ValueError("generated image media type does not match artifact")
    _validate_dimensions(width, height, settings)
    if generated.width != width or generated.height != height:
        raise ValueError("generated image dimensions do not match artifact")
    return ToolResultImage(
        data=data,
        media_type=media_type,
        detail=detail,
        sha256=generated.sha256,
        width=width,
        height=height,
    )


async def _generated_image_bytes(
    generated: GeneratedFile,
    *,
    backend: str,
    settings: ShellToolSettings,
) -> bytes | None:
    if generated.transient_content is not None:
        return generated.transient_content
    if generated.content_base64 is not None:
        try:
            return b64decode(generated.content_base64, validate=True)
        except (BinasciiError, ValueError):
            raise ValueError(
                "generated image artifact has invalid base64"
            ) from None
    if backend != "local":
        return None
    value = generated.metadata.get(
        GENERATED_FILE_MATERIALIZED_PATH_METADATA_KEY
    )
    if not isinstance(value, str) or not value:
        return None
    workspace_root = await resolve_policy_path(Path(settings.workspace_root))
    materialized_root = await resolve_policy_path(
        workspace_root / settings.materialized_input_files_dir
    )
    path = await resolve_policy_path(Path(value))
    if not _is_relative_to(path, materialized_root):
        return None
    metadata = await inspect_path(path)
    if (
        metadata.is_symlink
        or not metadata.is_file
        or not _is_relative_to(metadata.resolved_path, materialized_root)
        or metadata.size != generated.bytes
    ):
        return None
    return await read_validated_bytes(
        path,
        metadata,
        max_bytes=settings.max_ocr_input_bytes,
    )


def _validate_dimensions(
    width: int,
    height: int,
    settings: ShellToolSettings,
) -> None:
    if (
        max(width, height) > settings.max_raster_long_edge_pixels
        or width * height > settings.max_ocr_pixels
    ):
        raise ValueError("image dimensions exceed the configured limits")


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True

from ..entities import (
    GENERATED_OUTPUT_PREFIX_PLACEHOLDER,
    GeneratedOutputPlan,
    ShellCommandRequest,
    ShellExecutionErrorCode,
    ShellOutputKind,
)
from .base import (
    NormalizedPath,
    ShellCommandDefinition,
    ShellCommandPolicyContext,
    ShellDependencyGroup,
)
from .helpers import (
    _literal_option,
    _media_display_path_argument,
    _media_path_argument,
    _validate_known_options,
    policy_denied,
)

from collections.abc import Mapping
from math import ceil, sqrt
from pathlib import PurePosixPath
from re import compile as compile_pattern

MONTAGE_INPUT_DIMENSIONS_METADATA_KEY = "_montage_input_dimensions"
_DIMENSIONS_PATTERN = compile_pattern(r"([1-9][0-9]{0,5})x([1-9][0-9]{0,5})")
_GEOMETRY_PATTERN = compile_pattern(r"\+([0-9]{1,5})\+([0-9]{1,5})")
_OUTPUT_FILENAME_PATTERN = compile_pattern(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_CONTAINER_DEFAULT_FONT = "DejaVu-Sans"
_CONFIGURED_FONT_DISPLAY = "[configured_font]"
_JPEG_FORMATS = frozenset({"jpg", "jpeg"})


def _montage_paths(
    paths: tuple[NormalizedPath, ...],
    *,
    max_inputs: int,
) -> tuple[NormalizedPath, ...]:
    if len(paths) < 2:
        raise policy_denied(
            ShellExecutionErrorCode.INVALID_OPTION,
            "montage requires at least two input paths",
        )
    if len(paths) > max_inputs:
        raise policy_denied(
            ShellExecutionErrorCode.TOO_LARGE,
            "montage input count is too large",
        )
    for path in paths:
        if path.operand.kind != "image_file":
            raise policy_denied(
                ShellExecutionErrorCode.INVALID_OPTION,
                "unsupported montage path kind",
            )
    return paths


def _dimensions_option(
    options: Mapping[str, object],
    name: str,
) -> tuple[str | None, tuple[int, int] | None]:
    value = options.get(name)
    if value is None:
        return None, None
    if not isinstance(value, str):
        raise policy_denied(
            ShellExecutionErrorCode.INVALID_OPTION,
            f"{name} must be a dimension string",
        )
    match = _DIMENSIONS_PATTERN.fullmatch(value)
    if match is None:
        raise policy_denied(
            ShellExecutionErrorCode.INVALID_OPTION,
            f"{name} must use WIDTHxHEIGHT",
        )
    return value, (int(match.group(1)), int(match.group(2)))


def _tile_option(
    options: Mapping[str, object],
    *,
    input_count: int,
) -> tuple[str, tuple[int, int]]:
    value, dimensions = _dimensions_option(options, "tile")
    if value is None or dimensions is None:
        columns = ceil(sqrt(input_count))
        rows = ceil(input_count / columns)
        return f"{columns}x{rows}", (columns, rows)
    columns, rows = dimensions
    if columns * rows < input_count:
        raise policy_denied(
            ShellExecutionErrorCode.INVALID_OPTION,
            "tile capacity must include every montage input",
        )
    return value, dimensions


def _geometry_option(
    options: Mapping[str, object],
    *,
    max_spacing_pixels: int,
) -> tuple[str, tuple[int, int]]:
    value = options.get("geometry", "+0+0")
    if not isinstance(value, str):
        raise policy_denied(
            ShellExecutionErrorCode.INVALID_OPTION,
            "geometry must be a spacing string",
        )
    match = _GEOMETRY_PATTERN.fullmatch(value)
    if match is None:
        raise policy_denied(
            ShellExecutionErrorCode.INVALID_OPTION,
            "geometry must use +X+Y",
        )
    horizontal = int(match.group(1))
    vertical = int(match.group(2))
    if horizontal > max_spacing_pixels or vertical > max_spacing_pixels:
        raise policy_denied(
            ShellExecutionErrorCode.INVALID_OPTION,
            "montage geometry spacing is too large",
        )
    return value, (horizontal, vertical)


def _quality_option(options: Mapping[str, object]) -> int | None:
    value = options.get("quality")
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise policy_denied(
            ShellExecutionErrorCode.INVALID_OPTION,
            "quality must be an integer",
        )
    if value < 1 or value > 100:
        raise policy_denied(
            ShellExecutionErrorCode.INVALID_OPTION,
            "quality is out of range",
        )
    return value


def _output_filename(
    options: Mapping[str, object],
    output_format: str,
) -> tuple[str, str]:
    value = options.get("output_filename")
    default_suffix = f".{output_format}"
    if value is None:
        return "montage", default_suffix
    if not isinstance(value, str):
        raise policy_denied(
            ShellExecutionErrorCode.INVALID_OPTION,
            "output_filename must be a safe basename",
        )
    path = PurePosixPath(value)
    if path.name != value or value in {".", ".."}:
        raise policy_denied(
            ShellExecutionErrorCode.INVALID_OPTION,
            "output_filename must not contain a directory",
        )
    if _OUTPUT_FILENAME_PATTERN.fullmatch(value) is None:
        raise policy_denied(
            ShellExecutionErrorCode.INVALID_OPTION,
            "output_filename must be a safe basename",
        )
    suffix = path.suffix.lower()
    if not suffix or not _format_matches_suffix(output_format, suffix):
        raise policy_denied(
            ShellExecutionErrorCode.INVALID_OPTION,
            "output_filename suffix does not match output_format",
        )
    return value[: -len(path.suffix)], suffix


def _format_matches_suffix(output_format: str, suffix: str) -> bool:
    suffix_format = suffix.removeprefix(".")
    if output_format in _JPEG_FORMATS:
        return suffix_format in _JPEG_FORMATS
    return suffix_format == output_format


def _input_dimensions(
    metadata: Mapping[str, object],
    *,
    input_count: int,
) -> tuple[tuple[int, int], ...]:
    value = metadata.get(MONTAGE_INPUT_DIMENSIONS_METADATA_KEY)
    if not isinstance(value, tuple) or len(value) != input_count:
        raise policy_denied(
            ShellExecutionErrorCode.UNSUPPORTED_MEDIA_SIGNATURE,
            "montage image dimensions are unavailable",
        )
    dimensions: list[tuple[int, int]] = []
    for item in value:
        if (
            not isinstance(item, tuple)
            or len(item) != 2
            or not all(
                isinstance(component, int) and not isinstance(component, bool)
                for component in item
            )
        ):
            raise policy_denied(
                ShellExecutionErrorCode.UNSUPPORTED_MEDIA_SIGNATURE,
                "montage image dimensions are unavailable",
            )
        width, height = item
        if width <= 0 or height <= 0:
            raise policy_denied(
                ShellExecutionErrorCode.UNSUPPORTED_MEDIA_SIGNATURE,
                "montage image dimensions are invalid",
            )
        dimensions.append((width, height))
    return tuple(dimensions)


def _validate_projected_output(
    input_dimensions: tuple[tuple[int, int], ...],
    thumbnail_dimensions: tuple[int, int] | None,
    tile_dimensions: tuple[int, int],
    spacing: tuple[int, int],
    context: ShellCommandPolicyContext,
) -> None:
    if thumbnail_dimensions is None:
        cell_width = max(width for width, _ in input_dimensions)
        cell_height = max(height for _, height in input_dimensions)
    else:
        cell_width, cell_height = thumbnail_dimensions
    columns, rows = tile_dimensions
    horizontal, vertical = spacing
    width = columns * (cell_width + 2 * horizontal)
    height = rows * (cell_height + 2 * vertical)
    settings = context.settings
    if (
        max(width, height) > settings.max_raster_long_edge_pixels
        or width * height > settings.max_raster_pixels
    ):
        raise policy_denied(
            ShellExecutionErrorCode.GENERATED_OUTPUT_CAP_EXCEEDED,
            "projected montage dimensions exceed output limits",
        )
    context.metadata["montage_projected_dimensions"] = (width, height)


def _safe_imagemagick_path(path: str) -> str:
    if path.startswith(("/", "./", "../")):
        return path
    return f"./{path}"


def _first_imagemagick_scene(path: str) -> str:
    return f"{_safe_imagemagick_path(path)}[0]"


def _output_plan(
    context: ShellCommandPolicyContext,
    *,
    display_prefix: str,
    suffix: str,
    media_type: str,
) -> GeneratedOutputPlan:
    settings = context.settings
    return GeneratedOutputPlan(
        prefix_name="montage",
        display_prefix=display_prefix,
        allowed_suffixes=(suffix,),
        suffix_media_types={suffix: media_type},
        max_files=1,
        max_file_bytes=settings.max_output_file_bytes,
        max_total_bytes=settings.max_total_output_file_bytes,
        max_inline_bytes=settings.max_inline_output_file_bytes,
        max_raster_long_edge_pixels=settings.max_raster_long_edge_pixels,
        max_raster_pixels=settings.max_raster_pixels,
        output_path_suffix=suffix,
    )


def build_argv(
    context: ShellCommandPolicyContext,
) -> tuple[tuple[str, ...], tuple[str, ...], GeneratedOutputPlan]:
    request = context.request
    settings = context.settings
    _validate_known_options(
        request.options,
        allowed_options={
            "thumbnail",
            "tile",
            "geometry",
            "output_format",
            "output_filename",
            "quality",
        },
        command="montage",
    )
    paths = _montage_paths(
        context.paths,
        max_inputs=settings.max_montage_inputs,
    )
    thumbnail, thumbnail_dimensions = _dimensions_option(
        request.options,
        "thumbnail",
    )
    if (
        thumbnail_dimensions is not None
        and max(thumbnail_dimensions) > settings.max_raster_long_edge_pixels
    ):
        raise policy_denied(
            ShellExecutionErrorCode.INVALID_OPTION,
            "montage thumbnail dimensions are too large",
        )
    tile, tile_dimensions = _tile_option(
        request.options,
        input_count=len(paths),
    )
    geometry, spacing = _geometry_option(
        request.options,
        max_spacing_pixels=settings.max_montage_spacing_pixels,
    )
    quality = _quality_option(request.options)
    output_format = _literal_option(
        request.options,
        "output_format",
        default="jpg",
        allowed=tuple(settings.allowed_montage_output_formats),
    )
    display_prefix, suffix = _output_filename(
        request.options,
        output_format,
    )
    media_type = (
        "image/jpeg" if output_format in _JPEG_FORMATS else "image/png"
    )
    dimensions = _input_dimensions(
        context.metadata,
        input_count=len(paths),
    )
    _validate_projected_output(
        dimensions,
        thumbnail_dimensions,
        tile_dimensions,
        spacing,
        context,
    )
    output_plan = _output_plan(
        context,
        display_prefix=display_prefix,
        suffix=suffix,
        media_type=media_type,
    )
    input_arguments = tuple(
        _first_imagemagick_scene(
            _media_path_argument(context.workspace.cwd, path.path)
        )
        for path in paths
    )
    display_input_arguments = tuple(
        _first_imagemagick_scene(
            _media_display_path_argument(path.display_path)
        )
        for path in paths
    )
    argv_parts = [
        context.executable_name,
        "-define",
        "registry:filename:literal=true",
        "-limit",
        "memory",
        str(settings.max_montage_memory_bytes),
        "-limit",
        "map",
        str(settings.max_montage_map_bytes),
        "-limit",
        "disk",
        str(settings.max_montage_disk_bytes),
        "-limit",
        "thread",
        str(settings.max_montage_threads),
        "-limit",
        "list-length",
        str(len(paths)),
    ]
    configured_font = settings.montage_font
    font = configured_font
    if not font and settings.execution_mode == "container":
        font = _CONTAINER_DEFAULT_FONT
    font_argument_index: int | None = None
    if font:
        font_argument_index = len(argv_parts) + 1
        argv_parts.extend(("-font", font))
    start = len(argv_parts)
    argv_parts.extend(input_arguments)
    argv_parts.extend(("+set", "label", "+set", "caption"))
    if thumbnail is not None:
        argv_parts.extend(("-thumbnail", thumbnail))
    argv_parts.extend(("-tile", tile, "-geometry", geometry, "-strip"))
    if quality is not None:
        argv_parts.extend(("-quality", str(quality)))
    argv_parts.append(GENERATED_OUTPUT_PREFIX_PLACEHOLDER)
    display_parts = list(argv_parts)
    if configured_font and font_argument_index is not None:
        display_parts[font_argument_index] = _CONFIGURED_FONT_DISPLAY
    display_parts[start : start + len(paths)] = display_input_arguments
    display_parts[-1] = f"{display_prefix}{suffix}"
    context.metadata["montage_input_count"] = len(paths)
    context.metadata["montage_thumbnail"] = thumbnail
    context.metadata["montage_tile"] = tile
    context.metadata["montage_geometry"] = geometry
    context.metadata["montage_output_format"] = output_format
    context.metadata["generated_output_display_prefix"] = display_prefix
    return tuple(argv_parts), tuple(display_parts), output_plan


def output_contract(
    request: ShellCommandRequest,
) -> tuple[str, ShellOutputKind]:
    return "application/json", ShellOutputKind.GENERATED_FILES


COMMAND_DEFINITION = ShellCommandDefinition(
    logical_id="montage",
    executable_name="montage",
    dependency_group=ShellDependencyGroup.IMAGEMAGICK,
    container_package_hints=(
        "imagemagick",
        "imagemagick-jpeg",
        "font-dejavu",
    ),
    argv_builder=build_argv,
    output_contract=output_contract,
    media_risk=True,
    supports_double_dash=False,
)

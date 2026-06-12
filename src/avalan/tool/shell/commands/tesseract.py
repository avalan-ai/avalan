from ..entities import ShellExecutionErrorCode
from .base import (
    ShellCommandDefinition,
    ShellCommandPolicyContext,
    ShellDependencyGroup,
)
from .helpers import (
    _literal_option,
    _media_display_path_argument,
    _media_path_argument,
    _optional_bounded_int_option,
    _single_path,
    _validate_known_options,
    policy_denied,
)

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..settings import ShellToolSettings


def _ocr_mode_option(
    options: Mapping[str, object],
    name: str,
    *,
    default: int,
    min_value: int,
    max_value: int,
) -> int:
    value = options.get(name, default)
    if not isinstance(value, int) or isinstance(value, bool):
        raise policy_denied(
            ShellExecutionErrorCode.INVALID_OCR_MODE,
            f"{name} must be an integer",
        )
    if value < min_value or value > max_value:
        raise policy_denied(
            ShellExecutionErrorCode.INVALID_OCR_MODE,
            f"{name} is out of range",
        )
    return value


def _optional_ocr_mode_option(
    options: Mapping[str, object],
    name: str,
    *,
    min_value: int,
    max_value: int,
) -> int | None:
    if name not in options or options[name] is None:
        return None
    return _ocr_mode_option(
        options,
        name,
        default=min_value,
        min_value=min_value,
        max_value=max_value,
    )


def _tesseract_languages(
    options: Mapping[str, object],
    settings: "ShellToolSettings",
) -> tuple[str, ...]:
    value = options.get("languages")
    if value is None:
        return (settings.allowed_tesseract_languages[0],)
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise policy_denied(
            ShellExecutionErrorCode.UNSUPPORTED_OCR_LANGUAGE,
            "languages must be a sequence",
        )
    if not value or len(value) > settings.max_ocr_languages:
        raise policy_denied(
            ShellExecutionErrorCode.UNSUPPORTED_OCR_LANGUAGE,
            "language count is out of range",
        )
    languages: list[str] = []
    for language in value:
        if (
            not isinstance(language, str)
            or language not in settings.allowed_tesseract_languages
        ):
            raise policy_denied(
                ShellExecutionErrorCode.UNSUPPORTED_OCR_LANGUAGE,
                "unsupported OCR language",
            )
        languages.append(language)
    return tuple(languages)


def build_argv(
    context: ShellCommandPolicyContext,
) -> tuple[tuple[str, ...], tuple[str, ...], None]:
    request = context.request
    settings = context.settings
    _validate_known_options(
        request.options,
        allowed_options={
            "languages",
            "psm",
            "oem",
            "dpi",
            "output_format",
        },
        command="tesseract",
    )
    path = _single_path(
        context.paths,
        allowed_kinds=("image_file",),
        command="tesseract",
    )
    languages = _tesseract_languages(request.options, settings)
    output_format = _literal_option(
        request.options,
        "output_format",
        default="txt",
        allowed=tuple(settings.allowed_tesseract_output_formats),
    )
    psm = _ocr_mode_option(
        request.options,
        "psm",
        default=3,
        min_value=0,
        max_value=13,
    )
    oem = _optional_ocr_mode_option(
        request.options,
        "oem",
        min_value=0,
        max_value=3,
    )
    dpi = _optional_bounded_int_option(
        request.options,
        "dpi",
        min_value=1,
        max_value=settings.max_tesseract_dpi,
    )
    language_argument = "+".join(languages)
    context.metadata["ocr_languages"] = languages
    context.metadata["ocr_output_format"] = output_format
    context.metadata["ocr_thread_limit"] = settings.tesseract_thread_limit
    path_argument = _media_path_argument(context.workspace.cwd, path.path)
    display_path_argument = _media_display_path_argument(path.display_path)
    argv_parts = [
        context.executable_name,
        path_argument,
        "stdout",
        "-l",
        language_argument,
        "--psm",
        str(psm),
    ]
    if oem is not None:
        argv_parts.extend(("--oem", str(oem)))
    if dpi is not None:
        argv_parts.extend(("--dpi", str(dpi)))
    display_parts = list(argv_parts)
    display_parts[1] = display_path_argument
    return tuple(argv_parts), tuple(display_parts), None


COMMAND_DEFINITION = ShellCommandDefinition(
    logical_id="tesseract",
    executable_name="tesseract",
    dependency_group=ShellDependencyGroup.OCR,
    container_package_hints=("tesseract-ocr", "tesseract"),
    argv_builder=build_argv,
    media_risk=True,
    supports_double_dash=False,
)

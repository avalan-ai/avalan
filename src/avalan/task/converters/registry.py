from ..feature_gate import TaskFeature, feature_available
from . import (
    FileConverter,
    UnavailableFileConverter,
    markdown_converter_capability,
)
from .markdown import MarkdownFileConverter
from .text import TextFileConverter

from collections.abc import Mapping
from types import MappingProxyType


def default_file_converters() -> Mapping[str, FileConverter]:
    values: dict[str, FileConverter] = {"text": TextFileConverter()}
    markdown_capability = markdown_converter_capability()
    if feature_available(TaskFeature.DOCUMENT_CONVERSION):
        values["markdown"] = MarkdownFileConverter()
    else:
        values["markdown"] = UnavailableFileConverter(
            name="markdown",
            version="unavailable",
            capability=markdown_capability,
        )
    return MappingProxyType(values)

from importlib.metadata import metadata
from importlib.metadata import version as metadata_version
from typing import Any
from urllib.parse import ParseResult, urlparse

from packaging.version import Version, parse


def _config() -> dict[str, Any]:
    config = metadata("avalan")
    package_version = metadata_version("avalan")
    return {
        "name": config["Name"],
        "version": package_version,
        "license": config["License"],
        "url": "https://avalan.ai",
    }


config = _config()


def license() -> str:
    assert "license" in config
    return str(config["license"])


def name() -> str:
    assert "name" in config
    return str(config["name"])


def version() -> Version:
    assert "version" in config
    return parse(str(config["version"]))


def site() -> ParseResult:
    assert "url" in config
    return urlparse(str(config["url"]))

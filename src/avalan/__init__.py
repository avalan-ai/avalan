from importlib.metadata import metadata
from importlib.metadata import version as metadata_version
from urllib.parse import ParseResult, urlparse

from packaging.version import Version, parse


def _config() -> dict[str, str]:
    config = metadata("avalan")
    package_version = metadata_version("avalan")
    return {
        "name": str(config["Name"]),
        "version": package_version,
        "license": str(config["License"]),
        "url": "https://avalan.ai",
    }


config: dict[str, str] = _config()


def license() -> str:
    assert "license" in config
    return config["license"]


def name() -> str:
    assert "name" in config
    return config["name"]


def version() -> Version:
    assert "version" in config
    return parse(config["version"])


def site() -> ParseResult:
    assert "url" in config
    return urlparse(config["url"])

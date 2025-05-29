from packaging.version import parse, Version
from pathlib import Path
from tomllib import load
from urllib.parse import urlparse
from urllib.parse import ParseResult


def _config() -> dict:
    project_path = Path(__file__).resolve().parents[2] / "pyproject.toml"
    with open(project_path, "rb") as file:
        config = load(file)
        return config


config = _config()


def license() -> str:
    assert (
        "project" in config
        and "license" in config["project"]
        and "text" in config["project"]["license"]
    )
    return config["project"]["license"]["text"]


def name() -> str:
    assert "project" in config and "name" in config["project"]
    return config["project"]["name"]


def version() -> Version:
    assert (
        "tool" in config
        and "poetry" in config["tool"]
        and "version" in config["tool"]["poetry"]
    )
    version = config["tool"]["poetry"]["version"]
    return parse(version)


def site() -> ParseResult:
    assert (
        "project" in config
        and "urls" in config["project"]
        and "homepage" in config["project"]["urls"]
    )
    homepage = config["project"]["urls"]["homepage"]
    return urlparse(homepage)

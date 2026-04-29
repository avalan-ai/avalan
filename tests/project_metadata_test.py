import tomllib
from pathlib import Path


def test_vendors_extra_includes_bedrock_runtime_dependencies() -> None:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))

    optional_deps = data["project"]["optional-dependencies"]
    vendors = optional_deps["vendors"]

    assert "aioboto3>=15.0.0,<16.0.0" in vendors
    assert "diffusers>=0.37.1,<0.38.0" in vendors

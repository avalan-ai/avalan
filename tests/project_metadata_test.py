import tomllib
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


def test_vendors_extra_includes_bedrock_runtime_dependencies() -> None:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))

    optional_deps = data["project"]["optional-dependencies"]
    vendors = optional_deps["vendors"]

    assert "aioboto3>=15.0.0,<16.0.0" in vendors
    assert "diffusers>=0.37.1,<0.38.0" in vendors


def test_hosted_agent_extras_omit_local_runtime_dependencies() -> None:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))

    optional_deps = data["project"]["optional-dependencies"]
    selected_extras = ("agent", "server", "tool", "vendors")
    dependencies = {
        canonicalize_name(Requirement(requirement).name)
        for extra in selected_extras
        for requirement in optional_deps[extra]
    }

    assert not dependencies & {
        "accelerate",
        "bitsandbytes",
        "sentence-transformers",
        "torch",
        "torchaudio",
        "torchvision",
        "transformers",
        "vllm",
    }

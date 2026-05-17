import tomllib
from pathlib import Path

from packaging.markers import Marker
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name


def _pyproject() -> dict[str, object]:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    return tomllib.loads(pyproject.read_text(encoding="utf-8"))


def _optional_dependencies() -> dict[str, list[str]]:
    data = _pyproject()
    return data["project"]["optional-dependencies"]


def _test_group_dependencies() -> dict[str, object]:
    data = _pyproject()
    return data["tool"]["poetry"]["group"]["test"]["dependencies"]


def test_vendors_extra_includes_bedrock_runtime_dependencies() -> None:
    optional_deps = _optional_dependencies()
    vendors = optional_deps["vendors"]

    assert "aioboto3>=15.0.0,<16.0.0" in vendors
    assert "diffusers>=0.37.1,<0.38.0" in vendors


def test_hosted_agent_extras_omit_local_runtime_dependencies() -> None:
    optional_deps = _optional_dependencies()
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
        "pyds4",
        "vllm",
    }


def test_ds4_extra_declares_platform_scoped_pyds4_dependency() -> None:
    requirements = [
        Requirement(requirement)
        for requirement in _optional_dependencies()["ds4"]
    ]

    assert {
        canonicalize_name(requirement.name) for requirement in requirements
    } == {"pyds4"}
    assert all(
        requirement.specifier == SpecifierSet(">=1.0.2,<2.0.0")
        for requirement in requirements
    )
    assert all(requirement.url is None for requirement in requirements)
    assert all(requirement.marker is not None for requirement in requirements)
    assert any(
        requirement.marker is not None
        and requirement.marker.evaluate(
            {
                "platform_system": "Darwin",
                "platform_machine": "arm64",
            }
        )
        for requirement in requirements
    )
    assert any(
        requirement.marker is not None
        and requirement.marker.evaluate(
            {
                "platform_system": "Linux",
                "platform_machine": "x86_64",
            }
        )
        for requirement in requirements
    )
    assert not any(
        requirement.marker is not None
        and requirement.marker.evaluate(
            {
                "platform_system": "Darwin",
                "platform_machine": "x86_64",
            }
        )
        for requirement in requirements
    )
    assert not any(
        requirement.marker is not None
        and requirement.marker.evaluate(
            {
                "platform_system": "Windows",
                "platform_machine": "AMD64",
            }
        )
        for requirement in requirements
    )


def test_test_group_installs_pyds4_for_ds4_bridge_tests() -> None:
    dependency = _test_group_dependencies()["pyds4"]

    assert isinstance(dependency, dict)
    assert dependency["version"] == ">=1.0.2,<2.0.0"
    assert "markers" in dependency
    marker = Marker(str(dependency["markers"]))
    assert marker.evaluate(
        {
            "platform_system": "Linux",
            "platform_machine": "x86_64",
        }
    )
    assert marker.evaluate(
        {
            "platform_system": "Darwin",
            "platform_machine": "arm64",
        }
    )
    assert not marker.evaluate(
        {
            "platform_system": "Darwin",
            "platform_machine": "x86_64",
        }
    )
    assert not marker.evaluate(
        {
            "platform_system": "Windows",
            "platform_machine": "AMD64",
        }
    )


def test_core_dependencies_omit_optional_ds4_binding() -> None:
    data = _pyproject()
    dependencies = {
        canonicalize_name(Requirement(requirement).name)
        for requirement in data["project"]["dependencies"]
    }

    assert "pyds4" not in dependencies

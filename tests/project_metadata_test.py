import tomllib
from pathlib import Path
from typing import Any, cast

from packaging.markers import Marker
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name


def _pyproject() -> dict[str, object]:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    return tomllib.loads(pyproject.read_text(encoding="utf-8"))


def _poetry_lock() -> dict[str, Any]:
    lockfile = Path(__file__).resolve().parents[1] / "poetry.lock"
    return tomllib.loads(lockfile.read_text(encoding="utf-8"))


def _lock_packages_by_name() -> dict[str, dict[str, Any]]:
    packages = cast(list[dict[str, Any]], _poetry_lock()["package"])
    return {package["name"]: package for package in packages}


def _optional_dependencies() -> dict[str, list[str]]:
    data = _pyproject()
    return data["project"]["optional-dependencies"]


def _test_group_dependencies() -> dict[str, object]:
    data = _pyproject()
    return data["tool"]["poetry"]["group"]["test"]["dependencies"]


def _requirements(extra: str) -> list[Requirement]:
    return [
        Requirement(requirement)
        for requirement in _optional_dependencies()[extra]
    ]


def _requirements_by_name(extra: str, name: str) -> list[Requirement]:
    return [
        requirement
        for requirement in _requirements(extra)
        if canonicalize_name(requirement.name) == name
    ]


def test_vendors_extra_includes_bedrock_runtime_dependencies() -> None:
    optional_deps = _optional_dependencies()
    vendors = optional_deps["vendors"]

    assert "aioboto3>=15.0.0,<16.0.0" in vendors
    assert "diffusers>=0.38.0,<0.39.0" in vendors
    assert "safetensors>=0.8.0rc0,<0.9.0" in vendors


def test_project_metadata_advertises_python_314_support() -> None:
    data = _pyproject()
    project = data["project"]
    specifier = SpecifierSet(str(project["requires-python"]))

    assert "3.11" in specifier
    assert "3.14" in specifier
    assert "3.14.1" in specifier
    assert "3.15" not in specifier
    assert "Programming Language :: Python :: 3.14" in project["classifiers"]


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


def test_task_extra_declares_jsonschema_dependency() -> None:
    requirements = _requirements_by_name("task", "jsonschema")

    assert len(requirements) == 1
    assert requirements[0].specifier == SpecifierSet(">=4.26.0,<5.0.0")
    assert requirements[0].marker is None


def test_task_pgsql_extra_declares_postgresql_dependencies() -> None:
    psycopg_requirements = _requirements_by_name("task-pgsql", "psycopg")
    binary_requirements = _requirements_by_name(
        "task-pgsql",
        "psycopg-binary",
    )

    assert len(psycopg_requirements) == 1
    assert len(binary_requirements) == 1
    assert psycopg_requirements[0].specifier == SpecifierSet(">=3.2.9,<4.0.0")
    assert psycopg_requirements[0].extras == {"pool"}
    assert psycopg_requirements[0].marker is None
    assert binary_requirements[0].specifier == SpecifierSet(">=3.2.9,<4.0.0")

    binary_marker = binary_requirements[0].marker

    assert binary_marker is not None
    assert binary_marker.evaluate({"python_version": "3.13"})
    assert not binary_marker.evaluate({"python_version": "3.14"})


def test_task_pgsql_extra_omits_migration_dependencies() -> None:
    optional_deps = _optional_dependencies()
    task_pgsql_dependencies = {
        canonicalize_name(Requirement(requirement).name)
        for requirement in optional_deps["task-pgsql"]
    }

    assert "alembic" not in task_pgsql_dependencies
    assert "sqlalchemy" not in task_pgsql_dependencies


def test_vllm_extras_remain_scoped_below_python_314() -> None:
    optional_deps = _optional_dependencies()

    for extra in ("vllm", "nvidia"):
        requirements: list[Requirement] = []
        for requirement in optional_deps[extra]:
            parsed = Requirement(requirement)
            if canonicalize_name(parsed.name) == "vllm":
                requirements.append(parsed)

        assert len(requirements) == 1
        marker = requirements[0].marker

        assert marker is not None
        assert marker.evaluate(
            {
                "platform_system": "Linux",
                "python_version": "3.13",
            }
        )
        assert not marker.evaluate(
            {
                "platform_system": "Linux",
                "python_version": "3.14",
            }
        )
        assert not marker.evaluate(
            {
                "platform_system": "Darwin",
                "python_version": "3.13",
            }
        )


def test_vision_extra_scopes_torchvision_python_3141() -> None:
    requirements = _requirements_by_name("vision", "torchvision")

    assert len(requirements) == 1
    marker = requirements[0].marker

    assert marker is not None
    assert marker.evaluate({"python_full_version": "3.14.0"})
    assert not marker.evaluate({"python_full_version": "3.14.1"})
    assert marker.evaluate({"python_full_version": "3.14.5"})


def test_memory_extra_requires_python_314_faiss_release() -> None:
    faiss_requirements = _requirements_by_name("memory", "faiss-cpu")

    assert len(faiss_requirements) == 1
    assert faiss_requirements[0].specifier == SpecifierSet(">=1.14.2,<2.0.0")


def test_memory_extra_scopes_document_conversion_below_python_314() -> None:
    requirements = _requirements_by_name("memory", "markitdown")

    assert len(requirements) == 1
    marker = requirements[0].marker

    assert marker is not None
    assert marker.evaluate({"python_version": "3.13"})
    assert not marker.evaluate({"python_version": "3.14"})


def test_memory_extra_omits_psycopg_binary_on_python_314() -> None:
    psycopg_requirements = _requirements_by_name("memory", "psycopg")
    binary_requirements = _requirements_by_name("memory", "psycopg-binary")

    assert len(psycopg_requirements) == 1
    assert len(binary_requirements) == 1
    assert psycopg_requirements[0].specifier == SpecifierSet(">=3.2.9,<4.0.0")
    assert binary_requirements[0].specifier == SpecifierSet(">=3.2.9,<4.0.0")
    assert psycopg_requirements[0].extras == {"pool"}
    assert psycopg_requirements[0].marker is None

    binary_marker = binary_requirements[0].marker

    assert binary_marker is not None
    assert binary_marker.evaluate({"python_version": "3.13"})
    assert not binary_marker.evaluate({"python_version": "3.14"})


def test_lock_scopes_python_314_install_blockers() -> None:
    lock = _poetry_lock()
    packages = _lock_packages_by_name()
    memory_313 = {
        "extra": "memory",
        "implementation_name": "cpython",
        "python_version": "3.13",
    }
    memory_314 = {
        "extra": "memory",
        "implementation_name": "cpython",
        "python_version": "3.14",
    }

    assert lock["metadata"]["python-versions"] == ">=3.11,<3.15"

    for name in (
        "coloredlogs",
        "magika",
        "markitdown",
        "onnxruntime",
        "psycopg-binary",
    ):
        marker = Marker(str(packages[name]["markers"]))
        assert marker.evaluate(memory_313)
        assert not marker.evaluate(memory_314)

    marker = Marker(str(packages["torchvision"]["markers"]))

    assert marker.evaluate(
        {
            "extra": "vision",
            "platform_system": "Darwin",
            "python_full_version": "3.14.0",
            "python_version": "3.14",
        }
    )
    assert not marker.evaluate(
        {
            "extra": "vision",
            "platform_system": "Darwin",
            "python_full_version": "3.14.1",
            "python_version": "3.14",
        }
    )


def test_ds4_extra_declares_platform_scoped_pyds4_dependency() -> None:
    requirements = _requirements("ds4")

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

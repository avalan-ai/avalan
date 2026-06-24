import re
import tomllib
from pathlib import Path
from typing import Any, cast

from packaging.markers import Marker
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name


def _pyproject() -> dict[str, object]:
    pyproject = _repository_root() / "pyproject.toml"
    return tomllib.loads(pyproject.read_text(encoding="utf-8"))


def _poetry_lock() -> dict[str, Any]:
    lockfile = _repository_root() / "poetry.lock"
    return tomllib.loads(lockfile.read_text(encoding="utf-8"))


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _read_repository_text(path: str) -> str:
    return (_repository_root() / path).read_text(encoding="utf-8")


def _lock_packages_by_name() -> dict[str, dict[str, Any]]:
    packages = cast(list[dict[str, Any]], _poetry_lock()["package"])
    return {package["name"]: package for package in packages}


def _optional_dependencies() -> dict[str, list[str]]:
    data = _pyproject()
    return data["project"]["optional-dependencies"]


def _test_group_dependencies() -> dict[str, object]:
    data = _pyproject()
    return data["tool"]["poetry"]["group"]["test"]["dependencies"]


def _supported_python_versions() -> set[str]:
    data = _pyproject()
    project = cast(dict[str, object], data["project"])
    classifiers = cast(list[str], project["classifiers"])
    prefix = "Programming Language :: Python :: "
    return {
        classifier.removeprefix(prefix)
        for classifier in classifiers
        if classifier.startswith(prefix)
    }


def _workflow_python_versions(workflow: str) -> list[set[str]]:
    matrices: list[set[str]] = []
    for match in re.finditer(r"python:\s*\[([^\]]+)\]", workflow):
        versions = {
            version.strip().strip("'\"")
            for version in match.group(1).split(",")
            if version.strip()
        }
        matrices.append(versions)
    return matrices


def _workflow_declares_event(workflow: str, event: str) -> bool:
    return re.search(rf"(?m)^  {re.escape(event)}:\s*$", workflow) is not None


def _makefile_enforces_coverage_fail_under(makefile: str) -> bool:
    return "PYTEST_ARGS += --cov=src/ --cov-report=xml" in makefile and (
        "PYTEST_ARGS += --cov-fail-under=99.995 --cov-precision=2" in makefile
    )


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


def test_test_workflow_covers_supported_matrix_and_build_gates() -> None:
    workflow = _read_repository_text(".github/workflows/test.yml")
    matrix_versions = _workflow_python_versions(workflow)

    assert _workflow_declares_event(workflow, "push")
    assert _workflow_declares_event(workflow, "pull_request")
    assert _workflow_declares_event(workflow, "workflow_dispatch")
    assert matrix_versions == [
        _supported_python_versions(),
        _supported_python_versions(),
    ]
    assert (
        "if: matrix.target.os == 'ubuntu-latest' && matrix.python != '3.14'"
        in workflow
    )
    assert "run: make test-pgsql coverage no-install" in workflow
    assert (
        "if: matrix.target.os == 'ubuntu-latest' && matrix.python == '3.14'"
        in workflow
    )
    assert "run: make test-pgsql coverage-report no-install" in workflow
    assert (
        "if: matrix.target.os != 'ubuntu-latest' && matrix.python != '3.14'"
        in workflow
    )
    assert "run: make test coverage no-install" in workflow
    assert (
        "if: matrix.target.os != 'ubuntu-latest' && matrix.python == '3.14'"
        in workflow
    )
    assert "run: make test coverage-report no-install" in workflow
    assert "run: poetry build --format wheel --clean" in workflow
    assert "path: dist/*.whl" in workflow


def test_workflow_matrix_detection_rejects_partial_python_support() -> None:
    workflow = "matrix:\n  python: ['3.11', '3.12']\n"

    assert _workflow_python_versions(workflow) != [
        _supported_python_versions()
    ]


def test_workflow_event_detection_rejects_missing_pull_request() -> None:
    workflow = "on:\n  push:\n  workflow_dispatch:\n"

    assert not _workflow_declares_event(workflow, "pull_request")


def test_make_coverage_command_enforces_fail_under_gate() -> None:
    makefile = _read_repository_text("Makefile")

    assert _makefile_enforces_coverage_fail_under(makefile)


def test_make_coverage_gate_detection_rejects_upload_only_coverage() -> None:
    makefile = "PYTEST_ARGS += --cov=src/ --cov-report=xml\n"

    assert not _makefile_enforces_coverage_fail_under(makefile)


def test_make_coverage_gate_detection_requires_precision() -> None:
    makefile = (
        "PYTEST_ARGS += --cov=src/ --cov-report=xml\n"
        "PYTEST_ARGS += --cov-fail-under=99.995\n"
    )

    assert not _makefile_enforces_coverage_fail_under(makefile)


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


def test_youtube_extra_and_transcript_dependency_are_removed() -> None:
    optional_deps = _optional_dependencies()
    locked_packages = _lock_packages_by_name()

    assert "youtube" not in optional_deps
    assert "youtube-transcript-api" not in locked_packages


def test_task_extra_declares_jsonschema_dependency() -> None:
    requirements = _requirements_by_name("task", "jsonschema")

    assert len(requirements) == 1
    assert requirements[0].specifier == SpecifierSet(">=4.26.0,<5.0.0")
    assert requirements[0].marker is None


def test_task_documents_extra_declares_document_dependencies() -> None:
    markitdown_requirements = _requirements_by_name(
        "task-documents",
        "markitdown",
    )
    markdownify_requirements = _requirements_by_name(
        "task-documents",
        "markdownify",
    )

    assert len(markitdown_requirements) == 1
    assert len(markdownify_requirements) == 1
    assert markitdown_requirements[0].specifier == SpecifierSet(
        ">=0.1.2,<0.2.0"
    )
    assert markitdown_requirements[0].extras == {"pdf"}
    assert markdownify_requirements[0].specifier == SpecifierSet(
        ">=1.1.0,<2.0.0"
    )
    assert markdownify_requirements[0].marker is None

    markitdown_marker = markitdown_requirements[0].marker

    assert markitdown_marker is not None
    assert markitdown_marker.evaluate({"python_version": "3.13"})
    assert not markitdown_marker.evaluate({"python_version": "3.14"})


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


def test_task_prometheus_extra_declares_prometheus_dependency() -> None:
    requirements = _requirements_by_name(
        "task-prometheus",
        "prometheus-client",
    )

    assert len(requirements) == 1
    assert requirements[0].specifier == SpecifierSet(">=0.23.0,<1.0.0")
    assert requirements[0].marker is None


def test_task_otel_extra_declares_opentelemetry_dependency() -> None:
    requirements = _requirements_by_name(
        "task-otel",
        "opentelemetry-sdk",
    )

    assert len(requirements) == 1
    assert requirements[0].specifier == SpecifierSet(">=1.41.1,<2.0.0")
    assert requirements[0].marker is None


def test_task_pgsql_extra_omits_migration_dependencies() -> None:
    optional_deps = _optional_dependencies()
    task_pgsql_dependencies = {
        canonicalize_name(Requirement(requirement).name)
        for requirement in optional_deps["task-pgsql"]
    }

    assert "alembic" not in task_pgsql_dependencies
    assert "sqlalchemy" not in task_pgsql_dependencies


def test_task_pgsql_extra_omits_memory_vector_dependencies() -> None:
    optional_deps = _optional_dependencies()
    task_pgsql_dependencies = {
        canonicalize_name(Requirement(requirement).name)
        for requirement in optional_deps["task-pgsql"]
    }

    assert "pgvector" not in task_pgsql_dependencies


def test_memory_extra_omits_migration_dependencies() -> None:
    optional_deps = _optional_dependencies()
    memory_dependencies = {
        canonicalize_name(Requirement(requirement).name)
        for requirement in optional_deps["memory"]
    }

    assert "alembic" not in memory_dependencies
    assert "sqlalchemy" not in memory_dependencies


def test_vllm_extras_omit_vulnerable_runtime_dependency() -> None:
    optional_deps = _optional_dependencies()

    for extra in ("vllm", "nvidia"):
        dependencies = {
            canonicalize_name(Requirement(requirement).name)
            for requirement in optional_deps[extra]
        }

        assert "vllm" not in dependencies
        assert "diskcache" not in dependencies


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

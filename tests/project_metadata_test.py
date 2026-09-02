import ast
import re
import tomllib
from pathlib import Path
from typing import Any, cast

from packaging.markers import Marker
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name

_PATCH_WORKER_BASE_IMAGE = (
    "python:3.11-slim-bookworm@sha256:"
    "2e32f7d302adc1c37428355c1e646897c0c53f4fd60b6a551245fb90ee129f91"
)
_PHASE11_TEST_PREFIX = "tests/patch/phase_11_contract_test.py::"
_MACOS_DOCKER_INTEGRATION_NODES = tuple(
    _PHASE11_TEST_PREFIX + name
    for name in (
        "test_patch_phase_11_reuses_shared_context_contract_corpus",
        "test_patch_phase_11_subroot_commit_and_root_replacement_race",
        "test_patch_phase_11_commits_container_move_with_private_artifact_cleanup",
        "test_patch_phase_11_scopes_container_mutation_to_trusted_cwd",
        "test_patch_phase_11_inactive_profile_never_starts_or_advertises_runtime",
        "test_patch_phase_11_missing_container_endpoint_fails_closed",
        "test_patch_phase_11_wrong_container_binder_reaps_runtime",
        "test_patch_phase_11_requirements",
        "test_patch_phase_11_recovers_authenticated_lease_across_process_restart",
        "test_patch_phase_11_serializes_initial_volume_creation_across_processes",
        "test_patch_phase_11_dispose_fails_closed_while_reclaim_owns_guard",
        "test_patch_phase_11_failed_start_cleanup_never_deletes_reclaimed_volume",
        "test_patch_phase_11_failed_start_cleanup_defers_to_live_volume",
        "test_patch_phase_11_e2e_020_reconciles_cancelled_multifile_apply",
        "test_patch_phase_11_e2e_021_fences_replaced_plan_bound_context",
        "test_patch_phase_11_reconciles_post_dispatch_stale_after_first_effect",
        "test_patch_phase_11_rejects_forged_replayed_and_out_of_order_channel",
        "test_patch_phase_11_preserves_container_representation_and_metadata",
        "test_patch_phase_11_rejects_hostile_container_volume_topology",
        "test_patch_phase_11_container_public_lifecycle_and_surfaces_are_redacted",
        "test_patch_phase_11_serializes_container_contexts_in_one_domain",
        "test_patch_phase_11_rejects_mismatched_domain_for_one_persistent_volume",
        "test_patch_phase_11_rejects_destination_race_at_container_fence",
        "test_patch_phase_11_service_loss_at_fence_stays_pending_without_effect",
        "test_patch_phase_11_container_service_has_only_its_sealed_authority",
    )
) + (
    (
        "tests/patch/phase_14_tri_profile_test.py::"
        "test_patch_e2e_035_container_shared_root_is_test_only_and_physical"
    ),
    (
        "tests/patch/phase_15_hardening_test.py::"
        "test_patch_e2e_039_local_sandbox_container_conformance"
    ),
)


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


def _workflow_enforces_input_gates(workflow: str) -> bool:
    type_gate = (
        "      - name: Verify structured-input type contracts\n"
        "        if: matrix.target.os == 'ubuntu-latest' && "
        "matrix.python == '3.11'\n"
        "        run: |\n"
        "          make lint-check\n"
        "          make typecheck-input-contract INPUT_PHASE=5\n"
    )
    test_gate = "        run: make test no-install\n"
    metadata_gate = (
        "      - name: Verify clean generated metadata\n"
        "        run: git diff --check\n"
    )
    return all(
        gate in workflow
        for gate in (
            type_gate,
            test_gate,
            metadata_gate,
        )
    )


def _workflow_limits_pushes_to_main(workflow: str) -> bool:
    return "  push:\n    branches:\n      - main\n" in workflow


def _workflow_enforces_pinned_worker_image_preflight(
    workflow: str, *, condition: str | None
) -> bool:
    """Return whether a workflow pulls the exact PATCH worker image."""
    image_step = "      - name: Pre-pull PATCH worker base image\n"
    if condition is not None:
        image_step += f"        if: {condition}\n"
    image_step += f"        run: docker pull {_PATCH_WORKER_BASE_IMAGE}\n"
    return (
        workflow.count(f"docker pull {_PATCH_WORKER_BASE_IMAGE}") == 1
        and image_step in workflow
    )


def _phase11_real_docker_nodes() -> set[str]:
    """Return Phase 11 tests that build and invoke a real Docker worker."""
    path = "tests/patch/phase_11_contract_test.py"
    source = _read_repository_text(path)
    tree = ast.parse(source)
    nodes: set[str] = set()
    for node in tree.body:
        if not isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)):
            continue
        if not node.name.startswith("test_"):
            continue
        test_source = ast.get_source_segment(source, node) or ""
        if (
            "_test_image(" in test_source
            and 'monkeypatch.setattr(_CONTAINER, "_docker_output", output)'
            not in test_source
        ):
            nodes.add(f"{path}::{node.name}")
    return nodes


def _real_docker_integration_nodes() -> set[str]:
    """Return the exact test functions that need a Docker Linux runtime."""
    return _phase11_real_docker_nodes() | {
        (
            "tests/patch/phase_14_tri_profile_test.py::"
            "test_patch_e2e_035_container_shared_root_is_test_only_and_physical"
        ),
        (
            "tests/patch/phase_15_hardening_test.py::"
            "test_patch_e2e_039_local_sandbox_container_conformance"
        ),
    }


def _workflow_enforces_macos_non_docker_policy(workflow: str) -> bool:
    """Return whether macOS deselects only real Docker integrations."""
    macos_step = workflow.split(
        "      - name: Run non-Docker tests (macOS lacks Docker/Linux VM)\n",
        maxsplit=1,
    )
    linux_step = (
        "      - name: Run tests\n"
        "        if: matrix.target.os == 'ubuntu-latest' && "
        "matrix.python != '3.11'\n"
        "        run: make test no-install\n"
    )
    return (
        len(macos_step) == 2
        and "        if: matrix.target.os == 'macos-15'\n" in macos_step[1]
        and "# GitHub-hosted macos-15 has no Docker daemon or nested Linux VM."
        in macos_step[1]
        and (
            "# Keep every portable test; deselect only real Docker "
            "integrations."
            in macos_step[1]
        )
        and macos_step[1].count("--deselect=")
        == len(_MACOS_DOCKER_INTEGRATION_NODES)
        and all(
            macos_step[1].count("--deselect=" + node) == 1
            for node in _MACOS_DOCKER_INTEGRATION_NODES
        )
        and "--ignore=" not in macos_step[1]
        and linux_step in workflow
    )


def _makefile_lint_check_is_non_mutating(makefile: str) -> bool:
    lint_check = makefile.split("lint-check:\n", maxsplit=1)[1].split(
        "\n\ntest:\n",
        maxsplit=1,
    )[0]
    return (
        "poetry run ruff check $(LINT_PATHS)" in lint_check
        and (
            "poetry run black --check --preview "
            "--enable-unstable-feature=string_processing $(LINT_PATHS)"
            in lint_check
        )
        and "poetry run mypy\n" in lint_check
        and "poetry run mypy $(INPUT_CONTRACT_SCRIPTS)" in lint_check
        and "poetry run ruff check $(CONVERSATION_CONTRACT_SCRIPTS)"
        in lint_check
        and (
            "poetry run black --check --preview "
            "--enable-unstable-feature=string_processing "
            "$(CONVERSATION_CONTRACT_SCRIPTS)"
            in lint_check
        )
        and "poetry run mypy $(CONVERSATION_CONTRACT_SCRIPTS)" in lint_check
        and "ruff format" not in lint_check
        and "--fix" not in lint_check
        and lint_check.index(
            "poetry run black --check --preview "
            "--enable-unstable-feature=string_processing $(LINT_PATHS)"
        )
        < lint_check.index("poetry run ruff check $(LINT_PATHS)")
        and lint_check.index(
            "poetry run black --check --preview "
            "--enable-unstable-feature=string_processing "
            "$(CONVERSATION_CONTRACT_SCRIPTS)"
        )
        < lint_check.index(
            "poetry run ruff check $(CONVERSATION_CONTRACT_SCRIPTS)"
        )
    )


def _makefile_lint_uses_black_as_formatter(makefile: str) -> bool:
    """Return whether mutating lint leaves Black as the formatter."""
    lint = makefile.split("lint:\n", maxsplit=1)[1].split(
        "\n\nlint-check:\n",
        maxsplit=1,
    )[0]
    lint_black = (
        "poetry run black --preview "
        "--enable-unstable-feature=string_processing"
    )
    return (
        "ruff format" not in lint
        and "poetry run ruff check --fix $(LINT_PATHS)" in lint
        and f"{lint_black} $(LINT_PATHS)" in lint
        and "poetry run ruff check --fix $(CONVERSATION_CONTRACT_SCRIPTS)"
        in lint
        and f"{lint_black} $(CONVERSATION_CONTRACT_SCRIPTS)" in lint
        and lint.index(f"{lint_black} $(LINT_PATHS)")
        < lint.index("poetry run ruff check --fix $(LINT_PATHS)")
        and lint.index(f"{lint_black} $(CONVERSATION_CONTRACT_SCRIPTS)")
        < lint.index(
            "poetry run ruff check --fix $(CONVERSATION_CONTRACT_SCRIPTS)"
        )
    )


def _workflow_enforces_single_postgresql_lane(workflow: str) -> bool:
    condition = (
        "matrix.target.os == 'ubuntu-latest' && matrix.python == '3.11'"
    )
    return (
        workflow.count(f"        if: {condition}\n") == 3
        and "      - name: Start PostgreSQL\n" in workflow
        and "sudo systemctl start postgresql.service" in workflow
        and "      - name: Run tests with PostgreSQL\n" in workflow
        and (
            "AVALAN_TASK_TEST_POSTGRESQL_ADMIN_DSN: "
            "postgresql://postgres:postgres@127.0.0.1:5432/postgres"
            in workflow
        )
        and (
            "if: matrix.target.os == 'ubuntu-latest' && "
            "matrix.python != '3.11'"
            in workflow
        )
        and workflow.count("AVALAN_TASK_TEST_POSTGRESQL_ADMIN_DSN") == 1
        and "AVALAN_TASK_TEST_POSTGRESQL_DOCKER" not in workflow
        and "make test-pgsql" not in workflow
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


def test_mypy_skips_opencv_runtime_stubs() -> None:
    """Keep vendor-installed OpenCV stubs out of project type contracts."""
    mypy = _pyproject()["tool"]["mypy"]
    overrides = mypy["overrides"]

    assert {"module": ["cv2", "cv2.*"], "follow_imports": "skip"} in overrides


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

    assert _workflow_limits_pushes_to_main(workflow)
    assert _workflow_declares_event(workflow, "pull_request")
    assert _workflow_declares_event(workflow, "workflow_dispatch")
    assert matrix_versions == [
        _supported_python_versions(),
        _supported_python_versions(),
    ]
    assert _workflow_enforces_input_gates(workflow)
    assert _real_docker_integration_nodes() == set(
        _MACOS_DOCKER_INTEGRATION_NODES
    )
    assert _workflow_enforces_macos_non_docker_policy(workflow)
    assert workflow.count("          make lint-check\n") == 1
    assert "          make lint\n" not in workflow
    assert _workflow_enforces_single_postgresql_lane(workflow)
    assert _makefile_lint_check_is_non_mutating(
        _read_repository_text("Makefile")
    )
    assert _makefile_lint_uses_black_as_formatter(
        _read_repository_text("Makefile")
    )
    coverage_workflow = _read_repository_text(
        ".github/workflows/code-coverage.yml"
    )
    assert _workflow_enforces_pinned_worker_image_preflight(
        workflow,
        condition="matrix.target.os == 'ubuntu-latest'",
    )
    assert _workflow_enforces_pinned_worker_image_preflight(
        coverage_workflow,
        condition=None,
    )
    assert (
        _read_repository_text(
            "tests/fixtures/patch/container_worker.Dockerfile"
        ).splitlines()[0]
        == f"FROM {_PATCH_WORKER_BASE_IMAGE}"
    )
    assert "run: make test no-install coverage" in coverage_workflow
    assert "--deselect=" not in coverage_workflow
    assert "run: make test coverage" not in coverage_workflow
    assert (
        "tests/project_metadata_test.py::"
        "test_test_workflow_covers_supported_matrix_and_build_gates"
        in coverage_workflow
    )
    assert "run: make test-conversation-current-exact" not in coverage_workflow
    assert "make test-conversation-exact" not in coverage_workflow
    assert "make test-conversation-pgsql-exact" not in coverage_workflow
    assert "sudo systemctl start postgresql.service" in coverage_workflow
    assert (
        "AVALAN_TASK_TEST_POSTGRESQL_ADMIN_DSN: "
        "postgresql://postgres:postgres@127.0.0.1:5432/postgres"
        in coverage_workflow
    )
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


def test_workflow_rejects_non_main_push_fanout() -> None:
    workflow = _read_repository_text(".github/workflows/test.yml").replace(
        "      - main\n",
        "      - '**'\n",
        1,
    )

    assert not _workflow_limits_pushes_to_main(workflow)


def test_workflow_exact_gate_detection_rejects_partial_coverage() -> None:
    workflow = _read_repository_text(".github/workflows/test.yml").replace(
        "make test no-install",
        "make install",
    )

    assert not _workflow_enforces_input_gates(workflow)


def test_workflow_exact_gate_detection_rejects_matrix_fanout() -> None:
    workflow = _read_repository_text(".github/workflows/test.yml").replace(
        "if: matrix.target.os == 'ubuntu-latest' && matrix.python == '3.11'",
        "if: matrix.python == '3.11'",
        1,
    )

    assert not _workflow_enforces_input_gates(workflow)


def test_workflow_rejects_broad_or_partial_macos_docker_exclusions() -> None:
    workflow = _read_repository_text(".github/workflows/test.yml").replace(
        "--deselect=tests/patch/phase_11_contract_test.py::"
        "test_patch_phase_11_requirements",
        "--ignore=tests/patch",
        1,
    )

    assert not _workflow_enforces_macos_non_docker_policy(workflow)


def test_makefile_rejects_mutating_ci_lint_check() -> None:
    makefile = _read_repository_text("Makefile").replace(
        "poetry run ruff check $(LINT_PATHS)",
        "poetry run ruff check --fix $(LINT_PATHS)",
        1,
    )

    assert not _makefile_lint_check_is_non_mutating(makefile)


def test_makefile_rejects_mutating_black_ci_lint_check() -> None:
    makefile = _read_repository_text("Makefile").replace(
        "poetry run black --check --preview ",
        "poetry run black --preview ",
        1,
    )

    assert not _makefile_lint_check_is_non_mutating(makefile)


def test_makefile_rejects_ruff_formatter() -> None:
    makefile = _read_repository_text("Makefile").replace(
        "lint:\n",
        "lint:\n\tpoetry run ruff format --preview $(LINT_PATHS)\n",
        1,
    )

    assert not _makefile_lint_uses_black_as_formatter(makefile)


def test_makefile_rejects_ruff_before_black() -> None:
    makefile = _read_repository_text("Makefile").replace(
        "poetry run black --preview"
        " --enable-unstable-feature=string_processing $(LINT_PATHS)\n\tpoetry"
        " run ruff check --fix $(LINT_PATHS)",
        "poetry run ruff check --fix $(LINT_PATHS)\n\tpoetry run black"
        " --preview --enable-unstable-feature=string_processing $(LINT_PATHS)",
        1,
    )

    assert not _makefile_lint_uses_black_as_formatter(makefile)


def test_makefile_rejects_ruff_check_before_black() -> None:
    makefile = _read_repository_text("Makefile").replace(
        "poetry run black --check --preview "
        "--enable-unstable-feature=string_processing $(LINT_PATHS)\n\t"
        "poetry run ruff check $(LINT_PATHS)",
        "poetry run ruff check $(LINT_PATHS)\n\t"
        "poetry run black --check --preview "
        "--enable-unstable-feature=string_processing $(LINT_PATHS)",
        1,
    )

    assert not _makefile_lint_check_is_non_mutating(makefile)


def test_workflow_rejects_worker_image_digest_drift() -> None:
    workflow = _read_repository_text(".github/workflows/test.yml").replace(
        _PATCH_WORKER_BASE_IMAGE,
        _PATCH_WORKER_BASE_IMAGE.removesuffix("1") + "0",
        1,
    )

    assert not _workflow_enforces_pinned_worker_image_preflight(
        workflow,
        condition="matrix.target.os == 'ubuntu-latest'",
    )


def test_workflow_postgresql_gate_detection_rejects_matrix_fanout() -> None:
    workflow = _read_repository_text(".github/workflows/test.yml").replace(
        "matrix.target.os == 'ubuntu-latest' && matrix.python == '3.11'",
        "matrix.target.os == 'ubuntu-latest'",
    )

    assert not _workflow_enforces_single_postgresql_lane(workflow)


def test_workflow_postgresql_gate_detection_rejects_docker() -> None:
    workflow = _read_repository_text(".github/workflows/test.yml").replace(
        "AVALAN_TASK_TEST_POSTGRESQL_ADMIN_DSN",
        "AVALAN_TASK_TEST_POSTGRESQL_DOCKER",
    )

    assert not _workflow_enforces_single_postgresql_lane(workflow)


def test_pgsql_conformance_defers_driver_imports_until_owned_dsn() -> None:
    source = _read_repository_text(
        "tests/conversation/pgsql_conformance_test.py"
    )
    dsn = source.index('_DSN = environ.get("AVALAN_TASK_TEST_POSTGRESQL_DSN")')
    guard = source.index("if _DSN is None:")
    skip = source.index("pytest.skip(", guard)
    allow_module_skip = source.index("allow_module_level=True", skip)
    helper_import = source.index("from durable_codec_test import")
    driver_import = source.index("from psycopg.errors import")
    pgsql_import = source.index("from avalan.pgsql import")

    assert dsn < guard < skip < allow_module_skip < helper_import
    assert helper_import < driver_import
    assert allow_module_skip < pgsql_import
    assert "pytest.mark.skipif" not in source
    assert "except ImportError" not in source


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


def test_shell_code_container_documents_git_support() -> None:
    dockerfile = _read_repository_text("docs/containers/shell-code/Dockerfile")
    readme = _read_repository_text("docs/containers/shell-code/README.md")

    assert re.search(r"(?m)^\s+git\s*\\?$", dockerfile)
    assert "- `git`" in readme
    assert "--tool shell.git_log" in readme
    assert "--tool shell.git_status" in readme


def test_shell_code_container_includes_process_tool_support() -> None:
    dockerfile = _read_repository_text("docs/containers/shell-code/Dockerfile")
    readme = _read_repository_text("docs/containers/shell-code/README.md")

    assert re.search(r"(?m)^\s+procps-ng\s*\\?$", dockerfile)
    assert re.search(r"(?m)^\s+lsof\s*\\?$", dockerfile)
    assert "- `pgrep`" in readme
    assert "- `ps`" in readme
    assert "- `lsof`" in readme
    assert "- `kill`" in readme
    assert "--tool shell.pgrep" in readme
    assert "--tool shell.ps" in readme
    assert "--tool shell.lsof" in readme
    assert "--tool-shell-allow-process-tools" in readme
    assert "--tool-shell-allow-process-control" in readme


def test_shell_code_container_includes_date_support() -> None:
    dockerfile = _read_repository_text("docs/containers/shell-code/Dockerfile")
    readme = _read_repository_text("docs/containers/shell-code/README.md")

    assert re.search(r"(?m)^\s+coreutils\s*\\?$", dockerfile)
    assert "- `date`" in readme
    assert "--tool shell.date" in readme


def test_shell_code_container_includes_shasum_support() -> None:
    dockerfile = _read_repository_text("docs/containers/shell-code/Dockerfile")
    readme = _read_repository_text("docs/containers/shell-code/README.md")

    assert re.search(r"(?m)^\s+perl-utils\s*\\?$", dockerfile)
    assert "- `shasum`" in readme
    assert "--tool shell.shasum" in readme


def test_shell_code_container_includes_montage_support() -> None:
    dockerfile = _read_repository_text("docs/containers/shell-code/Dockerfile")
    readme = _read_repository_text("docs/containers/shell-code/README.md")

    assert re.search(r"(?m)^\s+imagemagick\s*\\?$", dockerfile)
    assert re.search(r"(?m)^\s+imagemagick-jpeg\s*\\?$", dockerfile)
    assert re.search(r"(?m)^\s+font-dejavu\s*\\?$", dockerfile)
    assert "- `montage`" in readme
    assert "--tool shell.montage" in readme
    assert "--tool-shell-allow-media-tools" in readme


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
    alembic_requirements = _requirements_by_name("task-pgsql", "alembic")
    psycopg_requirements = _requirements_by_name("task-pgsql", "psycopg")
    binary_requirements = _requirements_by_name(
        "task-pgsql",
        "psycopg-binary",
    )
    sqlalchemy_requirements = _requirements_by_name(
        "task-pgsql",
        "sqlalchemy",
    )

    assert len(alembic_requirements) == 1
    assert len(psycopg_requirements) == 1
    assert len(binary_requirements) == 1
    assert len(sqlalchemy_requirements) == 1
    assert alembic_requirements[0].specifier == SpecifierSet(">=1.14.0,<2.0.0")
    assert alembic_requirements[0].marker is None
    assert psycopg_requirements[0].specifier == SpecifierSet(">=3.2.9,<4.0.0")
    assert psycopg_requirements[0].extras == {"pool"}
    assert psycopg_requirements[0].marker is None
    assert binary_requirements[0].specifier == SpecifierSet(">=3.2.9,<4.0.0")
    assert sqlalchemy_requirements[0].specifier == SpecifierSet(
        ">=2.0.0,<3.0.0"
    )
    assert sqlalchemy_requirements[0].marker is None

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


def test_task_pgsql_extra_includes_migration_dependencies() -> None:
    optional_deps = _optional_dependencies()
    task_pgsql_dependencies = {
        canonicalize_name(Requirement(requirement).name)
        for requirement in optional_deps["task-pgsql"]
    }

    assert "alembic" in task_pgsql_dependencies
    assert "sqlalchemy" in task_pgsql_dependencies


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

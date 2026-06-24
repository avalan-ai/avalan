from unittest import TestCase, main

from avalan.container import (
    CONFORMANCE_PLAN,
    ContainerBackend,
    ContainerDiagnostic,
    ContainerDiagnosticCategory,
    ContainerDiagnosticCode,
    ContainerExecutionScope,
    ContainerExecutionSettings,
    ContainerSurface,
    assert_container_syntax_supported,
    container_syntax_diagnostics,
    resolve_container_backend,
)
from avalan.container.conformance import _path_values


class ContainerConformanceTest(TestCase):
    def test_default_settings_disable_containers(self) -> None:
        settings = ContainerExecutionSettings()

        self.assertFalse(settings.enabled)
        self.assertEqual(settings.backend, ContainerBackend.NONE)
        self.assertEqual(
            settings.scope,
            ContainerExecutionScope.SHELL_CONTAINER_EXECUTION,
        )
        resolution = resolve_container_backend(settings)
        self.assertTrue(resolution.ok)
        self.assertIsNone(resolution.backend)
        self.assertTrue(resolution.direct_execution_allowed)

    def test_phase_zero_locks_release_targets(self) -> None:
        self.assertEqual(
            CONFORMANCE_PLAN.first_release_scopes,
            (
                ContainerExecutionScope.CORE_CONTRACT,
                ContainerExecutionScope.SHELL_CONTAINER_EXECUTION,
            ),
        )
        self.assertEqual(
            CONFORMANCE_PLAN.later_scopes,
            (
                ContainerExecutionScope.DURABLE_WORKFLOW,
                ContainerExecutionScope.BACKEND_BREADTH,
                ContainerExecutionScope.RUNTIME_ENVELOPE,
                ContainerExecutionScope.MODEL_BACKEND,
                ContainerExecutionScope.ADVANCED_RUNTIME_FEATURES,
            ),
        )
        self.assertFalse(CONFORMANCE_PLAN.backward_compatibility_required)
        self.assertTrue(CONFORMANCE_PLAN.reject_unknown_container_sections)
        self.assertIn(
            "container fake-e2e no-fallback tests",
            CONFORMANCE_PLAN.default_ci_jobs,
        )
        self.assertIn(
            "docker runtime e2e",
            CONFORMANCE_PLAN.optional_runtime_ci_jobs,
        )
        self.assertEqual(
            CONFORMANCE_PLAN.promoted_integration_backends,
            (ContainerBackend.DOCKER,),
        )
        self.assertEqual(
            CONFORMANCE_PLAN.optional_integration_backends,
            (
                ContainerBackend.PODMAN,
                ContainerBackend.NERDCTL,
                ContainerBackend.WINDOWS_DOCKER,
            ),
        )
        self.assertEqual(
            CONFORMANCE_PLAN.opt_in_integration_backends,
            (ContainerBackend.APPLE_CONTAINER,),
        )

    def test_diagnostic_serializes_to_public_dictionary(self) -> None:
        diagnostic = ContainerDiagnostic(
            code=ContainerDiagnosticCode.UNSUPPORTED_SYNTAX,
            path="tool.container",
            message="Unsupported.",
            hint="Remove it.",
            category=ContainerDiagnosticCategory.UNSUPPORTED,
        )

        self.assertEqual(
            diagnostic.as_dict(),
            {
                "code": "container.unsupported_syntax",
                "path": "tool.container",
                "category": "unsupported",
                "message": "Unsupported.",
                "hint": "Remove it.",
            },
        )

    def test_agent_toml_container_sections_are_supported(self) -> None:
        diagnostics = container_syntax_diagnostics(
            ContainerSurface.AGENT_TOML,
            {
                "tool": {
                    "container": {"backend": "auto"},
                    "shell": {
                        "backend": "container",
                        "container": {"profile": "workspace-readonly"},
                    },
                }
            },
        )

        self.assertEqual(diagnostics, ())

    def test_unsupported_container_sections_are_diagnosed(self) -> None:
        diagnostics = container_syntax_diagnostics(
            ContainerSurface.TASK_TOML,
            {"execution": {"container": {"profile": "workspace-readonly"}}},
        )

        self.assertEqual(
            [diagnostic.path for diagnostic in diagnostics],
            [
                "execution.container",
            ],
        )
        self.assertTrue(
            all(
                diagnostic.code is ContainerDiagnosticCode.UNSUPPORTED_SYNTAX
                for diagnostic in diagnostics
            )
        )
        self.assertTrue(
            all(
                diagnostic.category is ContainerDiagnosticCategory.UNSUPPORTED
                for diagnostic in diagnostics
            )
        )

    def test_all_declared_unsupported_surface_paths_are_diagnosed(
        self,
    ) -> None:
        unsupported_paths = CONFORMANCE_PLAN.unsupported_surface_paths
        for surface, paths in unsupported_paths.items():
            for path in paths:
                with self.subTest(surface=surface.value, path=path):
                    raw = _raw_with_path(
                        path,
                        (
                            "container"
                            if path == "tool.shell.backend"
                            else {"enabled": True}
                        ),
                    )
                    diagnostics = container_syntax_diagnostics(surface, raw)

                    self.assertEqual(len(diagnostics), 1)
                    self.assertEqual(
                        diagnostics[0].code,
                        ContainerDiagnosticCode.UNSUPPORTED_SYNTAX,
                    )
                    self.assertEqual(
                        diagnostics[0].path,
                        path.replace("*", "selected"),
                    )

    def test_flow_node_container_syntax_matches_named_nodes(self) -> None:
        diagnostics = container_syntax_diagnostics(
            ContainerSurface.FLOW_TOML,
            {
                "nodes": {
                    "search": {
                        "runtime": {
                            "container": {"profile": "workspace-readonly"}
                        }
                    }
                }
            },
        )

        self.assertEqual(diagnostics, ())

    def test_missing_and_non_mapping_paths_do_not_match(self) -> None:
        self.assertEqual(
            container_syntax_diagnostics(
                ContainerSurface.AGENT_TOML,
                {"tool": "not-a-table"},
            ),
            (),
        )
        self.assertEqual(
            container_syntax_diagnostics(ContainerSurface.TASK_TOML, {}),
            (),
        )
        self.assertEqual(
            container_syntax_diagnostics(
                ContainerSurface.FLOW_TOML,
                {"nodes": {1: {"runtime": {"container": {}}}}},
            ),
            (),
        )
        self.assertEqual(
            container_syntax_diagnostics(
                ContainerSurface.FLOW_TOML,
                {"runtime": "not-a-table"},
            ),
            (),
        )

    def test_wildcard_path_values_skip_non_mapping_branches(self) -> None:
        matches = _path_values(
            {
                "nodes": {
                    "search": {
                        "runtime": {
                            "container": {"profile": "workspace-readonly"}
                        }
                    },
                    "note": "not-a-table",
                    1: {"runtime": {"container": {"profile": "ignored"}}},
                }
            },
            "nodes.*.runtime.container",
        )

        self.assertEqual(
            matches,
            (
                (
                    "nodes.search.runtime.container",
                    {"profile": "workspace-readonly"},
                ),
            ),
        )

    def test_string_surface_names_are_supported(self) -> None:
        diagnostics = container_syntax_diagnostics(
            "server",
            {"request": {"container": {"profile": "workspace-readonly"}}},
        )

        self.assertEqual(len(diagnostics), 1)
        self.assertEqual(diagnostics[0].path, "request.container")

    def test_assert_container_syntax_supported_rejects_diagnostics(
        self,
    ) -> None:
        assert_container_syntax_supported(
            ContainerSurface.AGENT_TOML,
            {"tool": {"container": {"backend": "auto"}}},
        )

        with self.assertRaisesRegex(AssertionError, "execution.container"):
            assert_container_syntax_supported(
                ContainerSurface.TASK_TOML,
                {"execution": {"container": {"profile": "p"}}},
            )

    def test_required_backend_never_falls_back_to_host_execution(self) -> None:
        settings = ContainerExecutionSettings(
            backend=ContainerBackend.DOCKER,
            required=True,
            profile="workspace-readonly",
        )

        resolution = resolve_container_backend(
            settings,
            available_backends=(),
        )

        self.assertFalse(resolution.ok)
        self.assertIsNone(resolution.backend)
        self.assertFalse(resolution.direct_execution_allowed)
        self.assertEqual(
            resolution.diagnostics[0].code,
            ContainerDiagnosticCode.BACKEND_UNAVAILABLE,
        )

    def test_available_backend_resolution_uses_container_backend(self) -> None:
        settings = ContainerExecutionSettings(
            backend="docker",
            required=True,
            scope="runtime_envelope",
            profile="workspace-readonly",
        )

        resolution = resolve_container_backend(
            settings,
            available_backends=("docker",),
        )

        self.assertTrue(settings.enabled)
        self.assertEqual(settings.backend, ContainerBackend.DOCKER)
        self.assertEqual(
            settings.scope,
            ContainerExecutionScope.RUNTIME_ENVELOPE,
        )
        self.assertTrue(resolution.ok)
        self.assertEqual(resolution.backend, ContainerBackend.DOCKER)
        self.assertFalse(resolution.direct_execution_allowed)

    def test_required_disabled_backend_fails_closed(self) -> None:
        resolution = resolve_container_backend(
            ContainerExecutionSettings(required=True),
        )

        self.assertFalse(resolution.ok)
        self.assertIsNone(resolution.backend)
        self.assertFalse(resolution.direct_execution_allowed)
        self.assertEqual(
            resolution.diagnostics[0].code,
            ContainerDiagnosticCode.BACKEND_REQUIRED,
        )

    def test_local_is_not_a_container_backend(self) -> None:
        with self.assertRaises(AssertionError):
            ContainerExecutionSettings(backend="local")

    def test_invalid_string_settings_are_rejected(self) -> None:
        with self.assertRaises(AssertionError):
            ContainerExecutionSettings(scope="unknown")
        with self.assertRaises(AssertionError):
            container_syntax_diagnostics("unknown", {})
        with self.assertRaises(AssertionError):
            resolve_container_backend(
                ContainerExecutionSettings(),
                available_backends=("local",),
            )


def _raw_with_path(path: str, value: object) -> dict[str, object]:
    raw: dict[str, object] = {}
    current = raw
    parts = path.replace("*", "selected").split(".")
    for part in parts[:-1]:
        child: dict[str, object] = {}
        current[part] = child
        current = child
    current[parts[-1]] = value
    return raw


if __name__ == "__main__":
    main()

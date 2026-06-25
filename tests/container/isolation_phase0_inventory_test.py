from pathlib import Path
from tomllib import load
from unittest import TestCase, main

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_INVENTORY_PATH = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "isolation_phase0_inventory.toml"
)
_IGNORED_INVENTORY_PATH = (
    _REPOSITORY_ROOT / "specs" / "isolation-phase0-inventory.toml"
)
_SCANNED_ROOTS = ("src", "tests", "docs", ".github")
_SCANNED_SUFFIXES = {".md", ".py", ".toml", ".yaml", ".yml"}
_SKIPPED_PARTS = {"__pycache__", "artifacts", "recording"}
_LEGACY_REFERENCE_TERMS = (
    "Podman",
    "podman",
    "PODMAN",
    "nerdctl",
    "NERDCTL",
    "containerd",
    "Windows Docker",
    "windows-docker",
    "WINDOWS_DOCKER",
    "Windows containers",
    "Microsoft Windows containers",
    "Microsoft containers",
    "Linux-on-WSL2",
    "WSL2",
    "Hyper-V",
    "AVALAN_CONTAINER_PODMAN_E2E",
    "AVALAN_CONTAINER_NERDCTL_E2E",
    "AVALAN_CONTAINER_WINDOWS_DOCKER_E2E",
    "ContainerBackend.AUTO",
)


class IsolationPhase0InventoryTest(TestCase):
    def test_target_public_values_are_locked(self) -> None:
        inventory = _inventory()
        target = inventory["target"]

        self.assertEqual(
            tuple(target["shell_execution_modes"]),
            ("container", "sandbox", "local"),
        )
        self.assertEqual(
            tuple(target["container_backends"]),
            ("apple-container", "docker"),
        )
        self.assertEqual(
            tuple(target["sandbox_backends"]),
            ("seatbelt", "bubblewrap"),
        )

    def test_target_contract_rejects_unknown_and_legacy_strings(self) -> None:
        inventory = _inventory()
        target = inventory["target"]
        rejected = inventory["rejected_values"]
        allowed_modes = set(target["shell_execution_modes"])
        allowed_container_backends = set(target["container_backends"])
        allowed_sandbox_backends = set(target["sandbox_backends"])

        for value in rejected["unknown_shell_execution_modes"]:
            with self.subTest(value=value):
                self.assertNotIn(value, allowed_modes)
        for value in rejected["legacy_container_backends"]:
            with self.subTest(value=value):
                self.assertNotIn(value, allowed_container_backends)
                self.assertNotIn(value, allowed_sandbox_backends)
        for value in rejected["unknown_container_backends"]:
            with self.subTest(value=value):
                self.assertNotIn(value, allowed_container_backends)
        for value in rejected["unknown_sandbox_backends"]:
            with self.subTest(value=value):
                self.assertNotIn(value, allowed_sandbox_backends)

    def test_current_runtime_reality_is_recorded(self) -> None:
        inventory = _inventory()
        current_reality = inventory["current_reality"]
        container_sources = "\n".join(
            path.read_text(encoding="utf-8", errors="ignore")
            for path in (
                _REPOSITORY_ROOT / "src" / "avalan" / "container"
            ).glob("*.py")
        )

        self.assertEqual(
            current_reality["apple_container"]["status"],
            "implemented",
        )
        self.assertIn("class AppleContainerBackend", container_sources)
        self.assertEqual(current_reality["docker"]["status"], "target-only")
        self.assertFalse(
            current_reality["docker"]["concrete_backend_exists"],
        )
        self.assertNotIn("class DockerContainerBackend", container_sources)

    def test_ci_jobs_are_defined_for_target_runtimes(self) -> None:
        inventory = _inventory()
        ci = inventory["ci"]
        default_jobs = tuple(ci["default"]["jobs"])
        optional_jobs = {
            job["runtime"]: job for job in ci["optional_real_runtime_jobs"]
        }

        self.assertIn("isolation phase0 static inventory", default_jobs)
        self.assertIn("container fake-e2e no-fallback", default_jobs)
        self.assertEqual(
            set(optional_jobs),
            {"docker", "apple-container", "seatbelt", "bubblewrap"},
        )
        self.assertEqual(
            optional_jobs["docker"]["gate"],
            "AVALAN_CONTAINER_DOCKER_E2E=1",
        )
        self.assertEqual(
            optional_jobs["seatbelt"]["gate"],
            "AVALAN_ISOLATION_SEATBELT_E2E=1",
        )

    def test_inventory_has_explicit_removal_and_keep_lists(self) -> None:
        inventory = _inventory()
        removal_terms = _terms(inventory["removal_references"])
        keep_terms = _terms(inventory["keep_references"])

        self.assertTrue(inventory["removal_references"])
        self.assertTrue(inventory["keep_references"])
        self.assertTrue(
            {
                "PODMAN",
                "NERDCTL",
                "WINDOWS_DOCKER",
                "podman",
                "nerdctl",
                "windows-docker",
                "WSL2",
                "Hyper-V",
            }.issubset(removal_terms)
        )
        self.assertIn("AppleContainerBackend", keep_terms)
        self.assertIn("microsoft/Phi-4-mini-instruct", keep_terms)

    def test_legacy_runtime_reference_paths_are_classified(self) -> None:
        inventory = _inventory()
        classified_paths = {
            entry["path"]
            for section in ("removal_references", "keep_references")
            for entry in inventory[section]
        }
        discovered_paths = _legacy_runtime_reference_paths()

        self.assertTrue(discovered_paths)
        self.assertTrue(
            discovered_paths.issubset(classified_paths),
            sorted(discovered_paths - classified_paths),
        )

    def test_shell_mode_site_inventory_covers_public_surfaces(self) -> None:
        inventory = _inventory()
        entries = inventory["local_container_shell_mode_sites"]
        surfaces = {entry["surface"] for entry in entries}
        paths = {entry["path"] for entry in entries}

        self.assertTrue(entries)
        self.assertTrue(
            {"source", "cli", "agent_toml", "docs", "specs", "tests"}.issubset(
                surfaces
            )
        )
        self.assertIn("src/avalan/tool/shell/settings.py", paths)
        self.assertIn("src/avalan/tool/shell/container.py", paths)
        self.assertIn("src/avalan/agent/loader.py", paths)
        self.assertIn("src/avalan/cli/__main__.py", paths)
        self.assertIn("tests/agent/loader_test.py", paths)
        self.assertIn("docs/CONTAINERS.md", paths)
        for entry in entries:
            with self.subTest(path=entry["path"]):
                if entry.get("local_only", False):
                    self.assertEqual(entry["surface"], "specs")
                else:
                    self.assertTrue(
                        (_REPOSITORY_ROOT / entry["path"]).exists()
                    )
                self.assertTrue(entry["values"])
                self.assertTrue(entry["action"])

    def test_fake_e2e_snapshot_exit_condition_is_partial(self) -> None:
        inventory = _inventory()
        condition = inventory["phase0_exit_conditions"][
            "fake_e2e_sdk_cli_toml_snapshots"
        ]

        self.assertEqual(condition["status"], "partial")
        self.assertIn("Phase 1", condition["reason"])
        self.assertIn("Phase 2", condition["reason"])
        self.assertIn("sandbox", condition["blocking_missing_values"])
        self.assertIn("podman", condition["blocking_values"])


def _inventory() -> dict[str, object]:
    with _INVENTORY_PATH.open("rb") as inventory_file:
        return load(inventory_file)


def _terms(entries: list[dict[str, object]]) -> set[str]:
    return {term for entry in entries for term in entry["terms"]}


def _legacy_runtime_reference_paths() -> set[str]:
    paths: set[str] = set()
    for root_name in _SCANNED_ROOTS:
        root = _REPOSITORY_ROOT / root_name
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not _should_scan(path):
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            if any(term in text for term in _LEGACY_REFERENCE_TERMS):
                paths.add(path.relative_to(_REPOSITORY_ROOT).as_posix())
    return paths


def _should_scan(path: Path) -> bool:
    if not path.is_file():
        return False
    if path in {
        _INVENTORY_PATH,
        _IGNORED_INVENTORY_PATH,
        Path(__file__).resolve(),
    }:
        return False
    if path.suffix not in _SCANNED_SUFFIXES:
        return False
    return not _SKIPPED_PARTS.intersection(path.parts)


if __name__ == "__main__":
    main()

from pathlib import Path
from unittest import TestCase, main

FIXTURE_ROOT = Path(__file__).parent / "fixtures"

FIXTURE_DIRECTORIES = (
    "policy",
    "filesystem",
    "formatting",
    "executor",
    "cli_toml_e2e",
    "real_subprocess_smoke",
    "find_roots",
    "filters",
    "json",
    "media",
    "ocr",
)

NEGATIVE_FIXTURE_MATRIX = (
    "traversal",
    "host_absolute_path",
    "hidden_path",
    "sensitive_path",
    "symlink",
    "special_file",
    "binary_content",
    "invalid_utf8",
    "dash_prefixed_name",
    "large_file",
    "oversized_glob",
    "malformed_option",
)

REQUIRED_FIXTURE_FILES = (
    "filesystem/visible.txt",
    "filesystem/.hidden.txt",
    "filesystem/-dash-prefixed.txt",
    "filesystem/binary.bin",
    "filesystem/invalid-utf8.bin",
    "filters/table.csv",
    "filters/table.tsv",
    "filters/lines.txt",
    "json/valid.json",
    "json/malformed.json",
    "json/oversized-template.json",
    "media/small.pdf",
    "media/problem.pdf",
    "media/oversized-template.pdf",
    "media/generated-raster.ppm",
    "ocr/small.pgm",
    "ocr/unsupported-signature.dat",
    "policy/workspace.toml",
    "formatting/redaction.txt",
    "executor/stream-output.txt",
    "cli_toml_e2e/agent-shell.toml",
    "real_subprocess_smoke/README.md",
    "find_roots/!/visible.txt",
    "find_roots/(/visible.txt",
)

PERFORMANCE_PASS_CRITERIA = {
    "retained_stream_bytes": "cap + chunk_size",
    "max_active_subprocesses": "configured_limit",
    "shared_process_limiter": "configured_limit",
    "redaction_growth": "captured_bytes",
    "post_filter_growth": "captured_bytes",
    "path_normalization_calls": "input_paths",
    "glob_normalization_calls": "input_globs",
    "executable_resolution_calls": "unique_command_ids",
}

TOCTOU_BOUNDARY = (
    "local_preflight_is_best_effort",
    "executor_revalidates_before_sensitive_reads",
    "container_is_future_defense_in_depth",
)


class ShellFixtureContractTest(TestCase):
    def test_fixture_directories_exist(self) -> None:
        for directory in FIXTURE_DIRECTORIES:
            with self.subTest(directory=directory):
                self.assertTrue((FIXTURE_ROOT / directory).is_dir())

    def test_negative_fixture_matrix_is_locked(self) -> None:
        self.assertEqual(
            NEGATIVE_FIXTURE_MATRIX,
            (
                "traversal",
                "host_absolute_path",
                "hidden_path",
                "sensitive_path",
                "symlink",
                "special_file",
                "binary_content",
                "invalid_utf8",
                "dash_prefixed_name",
                "large_file",
                "oversized_glob",
                "malformed_option",
            ),
        )

    def test_representative_fixture_files_exist(self) -> None:
        for relative_path in REQUIRED_FIXTURE_FILES:
            with self.subTest(relative_path=relative_path):
                self.assertTrue((FIXTURE_ROOT / relative_path).is_file())

    def test_fixtures_stay_small_and_deterministic(self) -> None:
        max_fixture_bytes = 8192

        for relative_path in REQUIRED_FIXTURE_FILES:
            with self.subTest(relative_path=relative_path):
                fixture_path = FIXTURE_ROOT / relative_path
                self.assertLessEqual(
                    fixture_path.stat().st_size,
                    max_fixture_bytes,
                )

    def test_binary_and_encoding_fixtures_match_labels(self) -> None:
        self.assertIn(
            b"\x00",
            (FIXTURE_ROOT / "filesystem" / "binary.bin").read_bytes(),
        )
        invalid_bytes = (
            FIXTURE_ROOT / "filesystem" / "invalid-utf8.bin"
        ).read_bytes()
        with self.assertRaises(UnicodeDecodeError):
            invalid_bytes.decode()

    def test_performance_pass_criteria_are_bounded_resources(self) -> None:
        self.assertEqual(
            PERFORMANCE_PASS_CRITERIA,
            {
                "retained_stream_bytes": "cap + chunk_size",
                "max_active_subprocesses": "configured_limit",
                "shared_process_limiter": "configured_limit",
                "redaction_growth": "captured_bytes",
                "post_filter_growth": "captured_bytes",
                "path_normalization_calls": "input_paths",
                "glob_normalization_calls": "input_globs",
                "executable_resolution_calls": "unique_command_ids",
            },
        )
        for criterion in PERFORMANCE_PASS_CRITERIA:
            self.assertNotIn("seconds", criterion)

    def test_local_executor_boundary_is_explicit(self) -> None:
        self.assertEqual(
            TOCTOU_BOUNDARY,
            (
                "local_preflight_is_best_effort",
                "executor_revalidates_before_sensitive_reads",
                "container_is_future_defense_in_depth",
            ),
        )


if __name__ == "__main__":
    main()

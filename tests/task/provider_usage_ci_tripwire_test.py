from pathlib import Path
from unittest import TestCase, main


class ProviderUsageCiTripwireTest(TestCase):
    def test_tracked_poc_fixtures_do_not_reference_local_smoke_inputs(
        self,
    ) -> None:
        root = (
            Path(__file__).parents[2]
            / "docs"
            / "examples"
            / "tasks"
            / "poc_extraction"
        )
        fixture_files = tuple(
            path
            for path in root.iterdir()
            if path.is_file() and path.name != "README.md"
        )

        self.assertIn(root / "sample.pdf", fixture_files)
        self.assertNotIn(root / "artifacts", fixture_files)
        for path in fixture_files:
            with self.subTest(path=path.name):
                self.assertFalse(path.name.startswith("."))
                self.assertNotIn("LA_Checkmate", path.name)
                self.assertNotIn("customer", path.name.lower())
                self.assertNotIn("specs", path.parts)
                if path.suffix != ".pdf":
                    text = path.read_text(encoding="utf-8")
                    self.assertNotIn("specs/poc", text)
                    self.assertNotIn("pulumi", text.lower())
                    self.assertNotIn("staging", text.lower())
                    self.assertNotIn("data:application/pdf", text)
                    self.assertNotIn("data:image/", text)

    def test_default_poc_usage_smoke_is_documented_as_fake_provider_ci(
        self,
    ) -> None:
        readme = (
            Path(__file__).parents[2]
            / "docs"
            / "examples"
            / "tasks"
            / "poc_extraction"
            / "README.md"
        )
        text = readme.read_text(encoding="utf-8")

        self.assertIn("Default CI uses fake-provider tests", text)
        self.assertIn("tests/task/direct_client_e2e_test.py", text)
        self.assertIn("tests/task/full_e2e_matrix_test.py", text)
        self.assertIn("tests/cli/task_test.py", text)
        self.assertIn("outside the default CI path", text)
        self.assertIn("without customer documents", text)
        self.assertNotIn("specs/poc", text)
        self.assertNotIn("pulumi", text.lower())
        self.assertNotIn("staging", text.lower())


if __name__ == "__main__":
    main()

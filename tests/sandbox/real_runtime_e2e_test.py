from asyncio import run as run_async
from os import environ
from unittest import TestCase, main

from avalan.sandbox import (
    BubblewrapSandboxBackend,
    SandboxBackendDiagnosticCode,
    SeatbeltSandboxBackend,
)


class SandboxRealRuntimeE2ETest(TestCase):
    def test_seatbelt_live_runtime_gate_skips_with_diagnostic(self) -> None:
        if environ.get("AVALAN_ISOLATION_SEATBELT_E2E") != "1":
            self.skipTest(
                "set AVALAN_ISOLATION_SEATBELT_E2E=1 to run seatbelt e2e"
            )

        probe = run_async(SeatbeltSandboxBackend().probe(timeout_seconds=2))
        if not probe.available:
            diagnostic = probe.diagnostics[0]
            self.assertIn(
                diagnostic.code,
                (
                    SandboxBackendDiagnosticCode.BACKEND_UNAVAILABLE,
                    SandboxBackendDiagnosticCode.CAPABILITY_MISMATCH,
                    SandboxBackendDiagnosticCode.TIMEOUT,
                ),
            )
            self.skipTest(diagnostic.message)

        self.assertTrue(probe.ok)

    def test_bubblewrap_live_runtime_gate_skips_with_diagnostic(self) -> None:
        if environ.get("AVALAN_ISOLATION_BUBBLEWRAP_E2E") != "1":
            self.skipTest(
                "set AVALAN_ISOLATION_BUBBLEWRAP_E2E=1 to run bubblewrap e2e"
            )

        probe = run_async(BubblewrapSandboxBackend().probe(timeout_seconds=2))
        if not probe.available:
            diagnostic = probe.diagnostics[0]
            self.assertIn(
                diagnostic.code,
                (
                    SandboxBackendDiagnosticCode.BACKEND_UNAVAILABLE,
                    SandboxBackendDiagnosticCode.CAPABILITY_MISMATCH,
                    SandboxBackendDiagnosticCode.TIMEOUT,
                ),
            )
            self.skipTest(diagnostic.message)

        self.assertTrue(probe.ok)


if __name__ == "__main__":
    main()

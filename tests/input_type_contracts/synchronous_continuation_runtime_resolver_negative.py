"""Reject a synchronous continuation runtime loader."""

from avalan.interaction import (
    ContinuationRevisionBinding,
    ExecutionDefinitionRef,
    ResolvedContinuationRuntime,
    TrustedContinuationRuntimeLoader,
)


class SynchronousRuntimeLoader:
    """Return a fresh runtime synchronously instead of through an awaitable."""

    trusted_continuation_runtime_loader = True

    def load_continuation_runtime(
        self,
        definition: ExecutionDefinitionRef,
        revision_binding: ContinuationRevisionBinding,
    ) -> ResolvedContinuationRuntime:
        return ResolvedContinuationRuntime(
            definition=definition,
            revision_binding=revision_binding,
            runtime=object(),
            operation=object(),
            model=object(),
            tools=object(),
            capabilities=object(),
            credentials_reloaded_from_trusted_config=True,
        )


loader: TrustedContinuationRuntimeLoader = SynchronousRuntimeLoader()

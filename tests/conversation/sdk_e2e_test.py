"""Exercise the public direct conversation SDK boundary."""

from inspect import iscoroutinefunction

import avalan
import avalan.conversation as conversation
import avalan.sdk as sdk


def test_normative_sdk_contract() -> None:
    """Expose one strictly asynchronous typed SDK surface."""
    public_types = (
        "ConversationBranchIntent",
        "ConversationResetIntent",
        "DirectConversationClient",
        "DirectConversationOutputDelta",
        "DirectConversationResult",
        "DirectConversationRuntime",
        "DirectConversationStream",
        "DirectConversationStreamState",
        "DirectConversationStreamTerminal",
        "OneShotConversationSettings",
        "StandaloneCompactRequest",
        "StatelessConversationSettings",
        "StoredConversationSettings",
    )
    for name in public_types:
        root_value = getattr(avalan, name)
        assert root_value is getattr(sdk, name)
        assert root_value is getattr(conversation, name)

    assert avalan.GenerationSettings is sdk.GenerationSettings
    assert avalan.ReasoningSettings is sdk.ReasoningSettings

    methods = (
        avalan.DirectConversationClient.create,
        avalan.DirectConversationClient.continue_conversation,
        avalan.DirectConversationClient.branch,
        avalan.DirectConversationClient.reset,
        avalan.DirectConversationClient.compact,
    )
    assert all(iscoroutinefunction(method) for method in methods)

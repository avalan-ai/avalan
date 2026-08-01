"""Reject dynamic object mappings at canonical provider item boundaries."""

from avalan.conversation import (
    ConversationCodecVersion,
    ConversationModelCallId,
    ProviderItem,
    ProviderItemCaller,
    ProviderItemId,
    ProviderItemIndex,
    ProviderItemKind,
    ProviderItemOrder,
    ProviderItemPhase,
    ProviderLaneId,
)

UNTYPED_PAYLOAD: dict[str, object] = {"payload": object()}
INVALID_ITEM = ProviderItem(
    item_id=ProviderItemId("item"),
    lane_id=ProviderLaneId("lane"),
    model_call_id=ConversationModelCallId("model-call"),
    kind=ProviderItemKind.MESSAGE,
    order=ProviderItemOrder(0),
    provider_index=ProviderItemIndex(0),
    phase=ProviderItemPhase.INPUT,
    caller=ProviderItemCaller.CALLER,
    canonical_input=UNTYPED_PAYLOAD,
    normalization_version=ConversationCodecVersion(1),
)

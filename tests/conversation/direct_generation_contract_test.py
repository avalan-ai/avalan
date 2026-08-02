"""Verify typed generation and model-call conversation boundaries."""

from collections.abc import Callable, Mapping
from datetime import timedelta
from typing import Any, cast

import pytest
from phase2_fixtures import (
    NOW,
    authority,
    binding,
    reasoning,
    retention,
    root_identity,
)

from avalan.agent import Specification
from avalan.conversation import (
    AtomicCommitReceipt,
    AuthorityScope,
    CheckpointKind,
    CheckpointLifecycle,
    CheckpointTimestamps,
    ChildLaneRetentionPolicy,
    ConversationBindingDriftError,
    ConversationCapabilityError,
    ConversationCheckpoint,
    ConversationCodecVersion,
    ConversationProviderStateSink,
    ConversationRunRequest,
    ConversationValidationError,
    MultiLaneCheckpointContent,
    ProviderItemLedger,
    ProviderLaneBinding,
    ProviderLaneLifecycle,
    StatelessConversationSettings,
    StatelessProviderLaneSnapshot,
    VisibleTranscript,
)
from avalan.entities import (
    EngineUri,
    GenerationSettings,
    Modality,
    Operation,
    OperationAudioParameters,
    OperationParameters,
    OperationTextParameters,
    OperationVisionParameters,
    merge_generation_settings_options,
)
from avalan.model import ModelCallContext
from avalan.model.call import validate_native_model_call_context
from avalan.model.modalities.audio import AudioClassificationModality
from avalan.model.modalities.text import TextGenerationModality
from avalan.model.modalities.vision import VisionImageClassificationModality

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    """Run direct generation tests on asyncio only."""
    return "asyncio"


class _Coordinator:
    async def execute(
        self,
        request: ConversationRunRequest,
    ) -> AtomicCommitReceipt:
        raise AssertionError(request)

    async def stream(
        self,
        request: ConversationRunRequest,
    ) -> AtomicCommitReceipt:
        raise AssertionError(request)

    async def stream_with_sink(
        self,
        request: ConversationRunRequest,
        sink: ConversationProviderStateSink,
    ) -> AtomicCommitReceipt:
        raise AssertionError((request, sink))

    async def compact(
        self,
        request: ConversationRunRequest,
    ) -> AtomicCommitReceipt:
        raise AssertionError(request)


class _Model:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    async def __call__(
        self,
        *args: object,
        **kwargs: object,
    ) -> str:
        self.calls.append((args, kwargs))
        return "visible-result"


def _operation(settings: GenerationSettings | None) -> Operation:
    return Operation(
        generation_settings=settings,
        input="visible-input",
        modality=Modality.TEXT_GENERATION,
        parameters=OperationParameters(
            text=OperationTextParameters(
                system_prompt="system",
                developer_prompt="developer",
            )
        ),
        requires_input=True,
    )


def _engine_uri() -> EngineUri:
    return EngineUri(
        host=None,
        port=None,
        user=None,
        password=None,
        vendor="openai",
        model_id="fake-phase4-model",
        params={},
    )


def _active_context() -> ModelCallContext:
    return ModelCallContext(
        specification=Specification(role=None, goal=None),
        input="visible-input",
        conversation_coordinator=_Coordinator(),
        conversation_authority=authority(),
        conversation_lane=binding("lane-native-dormant"),
    )


def _checkpoint(
    scope: AuthorityScope,
    lane: ProviderLaneBinding,
) -> ConversationCheckpoint:
    return ConversationCheckpoint(
        identity=root_identity("model-call-context"),
        kind=CheckpointKind.COMPLETED_OUTWARD_TURN,
        lifecycle=CheckpointLifecycle.COMMITTED,
        authority=scope,
        content=MultiLaneCheckpointContent(
            visible_transcript=VisibleTranscript(entries=()),
            lanes=(
                StatelessProviderLaneSnapshot(
                    binding=lane,
                    ledger=ProviderItemLedger(
                        lane_id=lane.lane_id,
                        normalization_version=ConversationCodecVersion(1),
                        items=(),
                    ),
                    reasoning=reasoning(),
                    lifecycle=ProviderLaneLifecycle.COMMITTED,
                    retention_policy=ChildLaneRetentionPolicy.RETAIN,
                ),
            ),
        ),
        timestamps=CheckpointTimestamps(
            created_at=NOW,
            committed_at=NOW,
            expires_at=NOW + timedelta(hours=1),
        ),
        retention=retention(),
    )


async def test_ordinary_generation_preserves_one_shot_call_shape(
    record_property: Callable[[str, object], None],
) -> None:
    """Preserve the generic one-shot provider invocation exactly."""
    record_property("conversation_acceptance_evidence", "public")
    modality = TextGenerationModality()
    omitted_model = _Model()
    explicit_model = _Model()

    omitted = await modality(
        _engine_uri(),
        cast(Any, omitted_model),
        _operation(None),
    )
    explicit = await modality(
        _engine_uri(),
        cast(Any, explicit_model),
        _operation(GenerationSettings()),
    )

    assert omitted == explicit == "visible-result"
    assert omitted_model.calls == explicit_model.calls
    assert len(omitted_model.calls) == 1
    _, kwargs = omitted_model.calls[0]
    settings = kwargs["settings"]
    assert type(settings) is GenerationSettings


async def test_active_native_conversation_is_rejected_before_model_call() -> (
    None
):
    """Keep all native provider profiles dormant in Phase 4."""
    modality = TextGenerationModality()
    model = _Model()
    settings = GenerationSettings()
    with pytest.raises(ConversationCapabilityError):
        await modality(
            _engine_uri(),
            cast(Any, model),
            _operation(settings),
            context=_active_context(),
        )
    assert model.calls == []


@pytest.mark.parametrize(
    ("modality", "operation"),
    (
        (
            AudioClassificationModality(),
            Operation(
                generation_settings=GenerationSettings(),
                input=None,
                modality=Modality.AUDIO_CLASSIFICATION,
                parameters=OperationParameters(
                    audio=OperationAudioParameters(
                        path="private-audio.wav",
                        sampling_rate=16_000,
                    )
                ),
            ),
        ),
        (
            VisionImageClassificationModality(),
            Operation(
                generation_settings=GenerationSettings(),
                input=None,
                modality=Modality.VISION_IMAGE_CLASSIFICATION,
                parameters=OperationParameters(
                    vision=OperationVisionParameters(path="private-image.png")
                ),
            ),
        ),
    ),
)
async def test_active_context_rejects_non_text_modalities_before_dispatch(
    modality: AudioClassificationModality | VisionImageClassificationModality,
    operation: Operation,
) -> None:
    """Apply the single registry guard to representative non-text calls."""
    model = _Model()

    with pytest.raises(ConversationCapabilityError):
        await modality(
            _engine_uri(),
            cast(Any, model),
            operation,
            context=_active_context(),
        )
    assert model.calls == []


async def test_inactive_context_preserves_non_text_one_shot_dispatch() -> None:
    """Preserve the ordinary context shape when no conversation is active."""
    model = _Model()
    operation = Operation(
        generation_settings=GenerationSettings(),
        input=None,
        modality=Modality.AUDIO_CLASSIFICATION,
        parameters=OperationParameters(
            audio=OperationAudioParameters(
                path="ordinary-audio.wav",
                sampling_rate=16_000,
            )
        ),
    )

    result = await AudioClassificationModality()(
        _engine_uri(),
        cast(Any, model),
        operation,
        context=ModelCallContext(
            specification=Specification(role=None, goal=None),
            input=None,
        ),
    )

    assert result == "visible-result"
    assert model.calls == [
        (
            (),
            {
                "path": "ordinary-audio.wav",
                "sampling_rate": 16_000,
            },
        )
    ]


@pytest.mark.parametrize(
    "overrides",
    (
        {"conversation": StatelessConversationSettings()},
        {"authority": "caller-authority"},
        {"upstream_response_id": "upstream-secret"},
        {"provider_ledger": ()},
        {"envelope": "caller-state"},
        {"checkpoint_id": "checkpoint-secret"},
    ),
)
def test_generic_generation_overrides_reject_conversation_state(
    overrides: Mapping[str, object],
) -> None:
    """Keep authority and continuation state out of generic mappings."""
    with pytest.raises(ConversationValidationError):
        merge_generation_settings_options({}, overrides)


def test_model_call_context_requires_complete_exact_binding() -> None:
    """Reject partial run-scoped authority and mismatched lane ownership."""
    scope: AuthorityScope = authority()
    lane: ProviderLaneBinding = binding("lane-context")
    context = ModelCallContext(
        specification=Specification(role=None, goal=None),
        input=None,
        conversation_coordinator=_Coordinator(),
        conversation_authority=scope,
        conversation_lane=lane,
    )
    assert context.conversation_authority == scope
    assert context.conversation_lane == lane

    with pytest.raises(ConversationValidationError):
        validate_native_model_call_context(cast(ModelCallContext, object()))

    with pytest.raises(ConversationValidationError):
        ModelCallContext(
            specification=Specification(role=None, goal=None),
            input=None,
            conversation_authority=scope,
        )
    with pytest.raises(ConversationValidationError):
        ModelCallContext(
            specification=Specification(role=None, goal=None),
            input=None,
            conversation_coordinator=_Coordinator(),
            conversation_authority=scope,
            conversation_lane=binding(
                "lane-wrong-agent",
                agent="different-agent",
            ),
        )


def test_model_call_context_rejects_invalid_runtime_components() -> None:
    """Reject impostor coordinator, authority, lane, and checkpoint values."""
    scope = authority()
    lane = binding("lane-runtime-components")
    coordinator = _Coordinator()

    for values in (
        {
            "conversation_coordinator": object(),
            "conversation_authority": scope,
            "conversation_lane": lane,
        },
        {
            "conversation_coordinator": coordinator,
            "conversation_authority": object(),
            "conversation_lane": lane,
        },
        {
            "conversation_coordinator": coordinator,
            "conversation_authority": scope,
            "conversation_lane": object(),
        },
        {
            "conversation_coordinator": coordinator,
            "conversation_authority": scope,
            "conversation_lane": lane,
            "conversation_checkpoint": object(),
        },
    ):
        with pytest.raises(ConversationValidationError):
            ModelCallContext(
                specification=Specification(role=None, goal=None),
                input=None,
                conversation_coordinator=cast(
                    Any,
                    values["conversation_coordinator"],
                ),
                conversation_authority=cast(
                    AuthorityScope,
                    values["conversation_authority"],
                ),
                conversation_lane=cast(
                    ProviderLaneBinding,
                    values["conversation_lane"],
                ),
                conversation_checkpoint=cast(
                    ConversationCheckpoint | None,
                    values.get("conversation_checkpoint"),
                ),
            )


def test_model_call_context_binds_checkpoint_authority_and_lane() -> None:
    """Reject a checkpoint from another authority, lane, or transport."""
    coordinator = _Coordinator()
    scope = authority()
    lane = binding("lane-checkpoint-context")
    checkpoint = _checkpoint(scope, lane)

    context = ModelCallContext(
        specification=Specification(role=None, goal=None),
        input=None,
        conversation_coordinator=coordinator,
        conversation_authority=scope,
        conversation_lane=lane,
        conversation_checkpoint=checkpoint,
    )
    assert context.conversation_checkpoint == checkpoint

    for (
        current_scope,
        current_lane,
        current_checkpoint,
        error_type,
    ) in (
        (
            scope,
            lane,
            _checkpoint(authority("other-principal"), lane),
            ConversationValidationError,
        ),
        (
            scope,
            lane,
            _checkpoint(scope, binding("different-checkpoint-lane")),
            ConversationValidationError,
        ),
        (
            scope,
            binding("lane-checkpoint-context", streaming=True),
            checkpoint,
            ConversationBindingDriftError,
        ),
    ):
        with pytest.raises(error_type):
            ModelCallContext(
                specification=Specification(role=None, goal=None),
                input=None,
                conversation_coordinator=coordinator,
                conversation_authority=current_scope,
                conversation_lane=current_lane,
                conversation_checkpoint=current_checkpoint,
            )

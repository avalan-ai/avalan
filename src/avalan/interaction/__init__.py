"""Expose the canonical structured task-interaction domain."""

from .codec import (
    canonical_continuation_snapshot_bytes as canonical_continuation_snapshot_bytes,  # noqa: E501
)
from .codec import (
    canonical_interaction_snapshot_bytes as canonical_interaction_snapshot_bytes,  # noqa: E501
)
from .codec import (
    canonical_resolution_digest as canonical_resolution_digest,
)
from .codec import (
    continuation_snapshot_digest as continuation_snapshot_digest,
)
from .codec import (
    decode_continuation_snapshot as decode_continuation_snapshot,
)
from .codec import (
    decode_execution_origin as decode_execution_origin,
)
from .codec import (
    decode_input_answer as decode_input_answer,
)
from .codec import (
    decode_input_model_result as decode_input_model_result,
)
from .codec import (
    decode_input_question as decode_input_question,
)
from .codec import (
    decode_input_request as decode_input_request,
)
from .codec import (
    decode_input_required_result as decode_input_required_result,
)
from .codec import (
    decode_input_resolution as decode_input_resolution,
)
from .codec import (
    decode_interaction_snapshot as decode_interaction_snapshot,
)
from .codec import (
    encode_continuation_outcome as encode_continuation_outcome,
)
from .codec import (
    encode_continuation_snapshot as encode_continuation_snapshot,
)
from .codec import (
    encode_execution_origin as encode_execution_origin,
)
from .codec import (
    encode_input_answer as encode_input_answer,
)
from .codec import (
    encode_input_model_result as encode_input_model_result,
)
from .codec import (
    encode_input_question as encode_input_question,
)
from .codec import (
    encode_input_request as encode_input_request,
)
from .codec import (
    encode_input_required_result as encode_input_required_result,
)
from .codec import (
    encode_input_resolution as encode_input_resolution,
)
from .codec import (
    encode_interaction_snapshot as encode_interaction_snapshot,
)
from .codec import (
    interaction_snapshot_digest as interaction_snapshot_digest,
)
from .codec import (
    semantic_request_fingerprint as semantic_request_fingerprint,
)
from .entities import (
    REQUEST_AGGREGATE_CONTENT_MEMBERS as REQUEST_AGGREGATE_CONTENT_MEMBERS,
)
from .entities import (
    RESERVED_INPUT_CAPABILITY_NAME as RESERVED_INPUT_CAPABILITY_NAME,
)
from .entities import (
    AgentId as AgentId,
)
from .entities import (
    AnsweredResolution as AnsweredResolution,
)
from .entities import (
    AnswerProvenance as AnswerProvenance,
)
from .entities import (
    BranchId as BranchId,
)
from .entities import (
    CancellationScope as CancellationScope,
)
from .entities import (
    CancelledResolution as CancelledResolution,
)
from .entities import (
    CapabilityRevision as CapabilityRevision,
)
from .entities import (
    Choice as Choice,
)
from .entities import (
    ChoiceValue as ChoiceValue,
)
from .entities import (
    ConfirmationAnswer as ConfirmationAnswer,
)
from .entities import (
    ConfirmationQuestion as ConfirmationQuestion,
)
from .entities import (
    ContinuationDisposition as ContinuationDisposition,
)
from .entities import (
    ContinuationId as ContinuationId,
)
from .entities import (
    ContinuationRevisionBinding as ContinuationRevisionBinding,
)
from .entities import (
    ContinuationSnapshot as ContinuationSnapshot,
)
from .entities import (
    DeclinedResolution as DeclinedResolution,
)
from .entities import (
    ExecutionDefinitionRef as ExecutionDefinitionRef,
)
from .entities import (
    ExecutionOrigin as ExecutionOrigin,
)
from .entities import (
    ExpiredResolution as ExpiredResolution,
)
from .entities import (
    FreeFormOther as FreeFormOther,
)
from .entities import (
    HostCapabilities as HostCapabilities,
)
from .entities import (
    HostHandling as HostHandling,
)
from .entities import (
    InputAnswer as InputAnswer,
)
from .entities import (
    InputAnsweredResult as InputAnsweredResult,
)
from .entities import (
    InputCancelledResult as InputCancelledResult,
)
from .entities import (
    InputContinuationOutcome as InputContinuationOutcome,
)
from .entities import (
    InputDeclinedResult as InputDeclinedResult,
)
from .entities import (
    InputModelResult as InputModelResult,
)
from .entities import (
    InputQuestion as InputQuestion,
)
from .entities import (
    InputRequest as InputRequest,
)
from .entities import (
    InputRequestId as InputRequestId,
)
from .entities import (
    InputRequiredResult as InputRequiredResult,
)
from .entities import (
    InputResolution as InputResolution,
)
from .entities import (
    InputResultKind as InputResultKind,
)
from .entities import (
    InputTimedOutResult as InputTimedOutResult,
)
from .entities import (
    InputUnavailableResult as InputUnavailableResult,
)
from .entities import (
    InteractionClass as InteractionClass,
)
from .entities import (
    InteractionSnapshot as InteractionSnapshot,
)
from .entities import (
    ModelCallId as ModelCallId,
)
from .entities import (
    ModelConfigRevision as ModelConfigRevision,
)
from .entities import (
    ModelId as ModelId,
)
from .entities import (
    MultilineTextAnswer as MultilineTextAnswer,
)
from .entities import (
    MultilineTextQuestion as MultilineTextQuestion,
)
from .entities import (
    MultipleSelectionAnswer as MultipleSelectionAnswer,
)
from .entities import (
    MultipleSelectionQuestion as MultipleSelectionQuestion,
)
from .entities import (
    ParticipantId as ParticipantId,
)
from .entities import (
    PresentationHint as PresentationHint,
)
from .entities import (
    PrincipalScope as PrincipalScope,
)
from .entities import (
    ProviderConfigRevision as ProviderConfigRevision,
)
from .entities import (
    ProviderFamilyName as ProviderFamilyName,
)
from .entities import (
    ProviderIdempotencyKey as ProviderIdempotencyKey,
)
from .entities import (
    QuestionId as QuestionId,
)
from .entities import (
    QuestionType as QuestionType,
)
from .entities import (
    RequestState as RequestState,
)
from .entities import (
    RequirementMode as RequirementMode,
)
from .entities import (
    ResolutionStatus as ResolutionStatus,
)
from .entities import (
    ResumeInputContinuation as ResumeInputContinuation,
)
from .entities import (
    RunId as RunId,
)
from .entities import (
    SelectedChoice as SelectedChoice,
)
from .entities import (
    SelectionValidationConstraints as SelectionValidationConstraints,
)
from .entities import (
    SelectionValue as SelectionValue,
)
from .entities import (
    SelectionValueType as SelectionValueType,
)
from .entities import (
    SessionId as SessionId,
)
from .entities import (
    SingleSelectionAnswer as SingleSelectionAnswer,
)
from .entities import (
    SingleSelectionQuestion as SingleSelectionQuestion,
)
from .entities import (
    StateRevision as StateRevision,
)
from .entities import (
    StreamSessionId as StreamSessionId,
)
from .entities import (
    SupersededResolution as SupersededResolution,
)
from .entities import (
    TaskId as TaskId,
)
from .entities import (
    TenantId as TenantId,
)
from .entities import (
    TerminateInputContinuation as TerminateInputContinuation,
)
from .entities import (
    TextAnswer as TextAnswer,
)
from .entities import (
    TextQuestion as TextQuestion,
)
from .entities import (
    TextValidationConstraints as TextValidationConstraints,
)
from .entities import (
    TimedOutResolution as TimedOutResolution,
)
from .entities import (
    TurnId as TurnId,
)
from .entities import (
    UnavailableResolution as UnavailableResolution,
)
from .entities import (
    UserId as UserId,
)
from .entities import (
    create_input_request as create_input_request,
)
from .error import (
    InputCodecError as InputCodecError,
)
from .error import (
    InputContractError as InputContractError,
)
from .error import (
    InputErrorCode as InputErrorCode,
)
from .error import (
    InputSnapshotError as InputSnapshotError,
)
from .error import (
    InputValidationError as InputValidationError,
)
from .state import (
    InputTransitionApplied as InputTransitionApplied,
)
from .state import (
    InputTransitionError as InputTransitionError,
)
from .state import (
    InputTransitionRejected as InputTransitionRejected,
)
from .state import (
    InputTransitionResult as InputTransitionResult,
)
from .state import (
    TransitionResultType as TransitionResultType,
)
from .state import (
    mark_request_pending as mark_request_pending,
)
from .state import (
    project_resolution_to_model as project_resolution_to_model,
)
from .state import (
    resolve_request as resolve_request,
)
from .validation import (
    MAX_REQUEST_CHARACTERS as MAX_REQUEST_CHARACTERS,
)
from .validation import (
    MAX_REQUEST_UTF8_BYTES as MAX_REQUEST_UTF8_BYTES,
)
from .validation import (
    MAX_STATE_REVISION as MAX_STATE_REVISION,
)

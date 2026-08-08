"""Provide the dormant typed mutation-domain boundary.

The package defines immutable value and truth contracts only.  It neither
registers a tool nor opens an effectful integration surface.
"""

from avalan.patch.codec import (
    decode_diagnostic as decode_diagnostic,
)
from avalan.patch.codec import (
    decode_event as decode_event,
)
from avalan.patch.codec import (
    decode_pending as decode_pending,
)
from avalan.patch.codec import (
    decode_public_pending as decode_public_pending,
)
from avalan.patch.codec import (
    decode_result as decode_result,
)
from avalan.patch.codec import (
    encode_diagnostic as encode_diagnostic,
)
from avalan.patch.codec import (
    encode_event as encode_event,
)
from avalan.patch.codec import (
    encode_pending as encode_pending,
)
from avalan.patch.codec import (
    encode_public_pending as encode_public_pending,
)
from avalan.patch.codec import (
    encode_result as encode_result,
)
from avalan.patch.domain import (
    AlgorithmDigest as AlgorithmDigest,
)
from avalan.patch.domain import (
    ApprovalGrant as ApprovalGrant,
)
from avalan.patch.domain import (
    ApprovalMode as ApprovalMode,
)
from avalan.patch.domain import (
    ArtifactState as ArtifactState,
)
from avalan.patch.domain import (
    Audience as Audience,
)
from avalan.patch.domain import (
    ByteSize as ByteSize,
)
from avalan.patch.domain import (
    CommitGraph as CommitGraph,
)
from avalan.patch.domain import (
    CommitStepJournal as CommitStepJournal,
)
from avalan.patch.domain import (
    CommitStepState as CommitStepState,
)
from avalan.patch.domain import (
    CommitTruth as CommitTruth,
)
from avalan.patch.domain import (
    ContextKind as ContextKind,
)
from avalan.patch.domain import (
    DomainFacade as DomainFacade,
)
from avalan.patch.domain import (
    DurationTicks as DurationTicks,
)
from avalan.patch.domain import (
    ErrorStage as ErrorStage,
)
from avalan.patch.domain import (
    FileMode as FileMode,
)
from avalan.patch.domain import (
    LifecyclePhase as LifecyclePhase,
)
from avalan.patch.domain import (
    LineageJournal as LineageJournal,
)
from avalan.patch.domain import (
    LineageState as LineageState,
)
from avalan.patch.domain import (
    LogicalPath as LogicalPath,
)
from avalan.patch.domain import (
    MutationPlan as MutationPlan,
)
from avalan.patch.domain import (
    MutationScope as MutationScope,
)
from avalan.patch.domain import (
    MutationState as MutationState,
)
from avalan.patch.domain import (
    PatchApprovalId as PatchApprovalId,
)
from avalan.patch.domain import (
    PatchContextId as PatchContextId,
)
from avalan.patch.domain import (
    PatchDomainId as PatchDomainId,
)
from avalan.patch.domain import (
    PatchErrorCode as PatchErrorCode,
)
from avalan.patch.domain import (
    PatchEventId as PatchEventId,
)
from avalan.patch.domain import (
    PatchFingerprint as PatchFingerprint,
)
from avalan.patch.domain import (
    PatchGrantId as PatchGrantId,
)
from avalan.patch.domain import (
    PatchInvocationOutcome as PatchInvocationOutcome,
)
from avalan.patch.domain import (
    PatchLifecycleEvent as PatchLifecycleEvent,
)
from avalan.patch.domain import (
    PatchLineageId as PatchLineageId,
)
from avalan.patch.domain import (
    PatchOperationId as PatchOperationId,
)
from avalan.patch.domain import (
    PatchPending as PatchPending,
)
from avalan.patch.domain import (
    PatchPendingOperationId as PatchPendingOperationId,
)
from avalan.patch.domain import (
    PatchPlanId as PatchPlanId,
)
from avalan.patch.domain import (
    PatchRequest as PatchRequest,
)
from avalan.patch.domain import (
    PatchRequestId as PatchRequestId,
)
from avalan.patch.domain import (
    PatchResult as PatchResult,
)
from avalan.patch.domain import (
    PatchStatus as PatchStatus,
)
from avalan.patch.domain import (
    PatchStepId as PatchStepId,
)
from avalan.patch.domain import (
    PatchTargetId as PatchTargetId,
)
from avalan.patch.domain import (
    PatchValidationError as PatchValidationError,
)
from avalan.patch.domain import (
    PatchWorkspaceId as PatchWorkspaceId,
)
from avalan.patch.domain import (
    PublicPendingProjection as PublicPendingProjection,
)
from avalan.patch.domain import (
    SequenceNumber as SequenceNumber,
)
from avalan.patch.domain import (
    coarsen_error_code as coarsen_error_code,
)
from avalan.patch.domain import (
    derive_commit_truth as derive_commit_truth,
)
from avalan.patch.parser import (
    DORMANT_PARAMETER_DESCRIPTORS as DORMANT_PARAMETER_DESCRIPTORS,
)
from avalan.patch.parser import (
    AddDeclarationSyntax as AddDeclarationSyntax,
)
from avalan.patch.parser import (
    CanonicalPatchRequest as CanonicalPatchRequest,
)
from avalan.patch.parser import (
    DeleteDeclarationSyntax as DeleteDeclarationSyntax,
)
from avalan.patch.parser import (
    DormantParameterDescriptor as DormantParameterDescriptor,
)
from avalan.patch.parser import (
    PatchDeclarationSyntax as PatchDeclarationSyntax,
)
from avalan.patch.parser import (
    PatchDocumentSyntax as PatchDocumentSyntax,
)
from avalan.patch.parser import (
    PatchHunkSyntax as PatchHunkSyntax,
)
from avalan.patch.parser import (
    PatchInputAccumulator as PatchInputAccumulator,
)
from avalan.patch.parser import (
    PatchInputError as PatchInputError,
)
from avalan.patch.parser import (
    PatchInputErrorCode as PatchInputErrorCode,
)
from avalan.patch.parser import (
    PatchInputLimits as PatchInputLimits,
)
from avalan.patch.parser import (
    PatchLineSyntax as PatchLineSyntax,
)
from avalan.patch.parser import (
    PatchRequestParser as PatchRequestParser,
)
from avalan.patch.parser import (
    RawPatchIngress as RawPatchIngress,
)
from avalan.patch.parser import (
    RawPatchInputKind as RawPatchInputKind,
)
from avalan.patch.parser import (
    RawPatchInputState as RawPatchInputState,
)
from avalan.patch.parser import (
    RawProviderIngressAdapter as RawProviderIngressAdapter,
)
from avalan.patch.parser import (
    RawProviderProfile as RawProviderProfile,
)
from avalan.patch.parser import (
    RawToolCallId as RawToolCallId,
)
from avalan.patch.parser import (
    StructuredEditSyntax as StructuredEditSyntax,
)
from avalan.patch.parser import (
    TextEditSyntax as TextEditSyntax,
)
from avalan.patch.parser import (
    UpdateDeclarationSyntax as UpdateDeclarationSyntax,
)

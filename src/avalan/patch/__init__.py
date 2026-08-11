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
from avalan.patch.coordinator import (
    ArtifactJournal as ArtifactJournal,
)
from avalan.patch.coordinator import (
    CommitLease as CommitLease,
)
from avalan.patch.coordinator import (
    CommitWorker as CommitWorker,
)
from avalan.patch.coordinator import (
    CoordinatorBoundary as CoordinatorBoundary,
)
from avalan.patch.coordinator import (
    CoordinatorError as CoordinatorError,
)
from avalan.patch.coordinator import (
    CoordinatorErrorCode as CoordinatorErrorCode,
)
from avalan.patch.coordinator import (
    CoordinatorRegistry as CoordinatorRegistry,
)
from avalan.patch.coordinator import (
    InMemoryCoordinatorStore as InMemoryCoordinatorStore,
)
from avalan.patch.coordinator import (
    InMemoryLeaseManager as InMemoryLeaseManager,
)
from avalan.patch.coordinator import (
    InMemoryPatchCoordinator as InMemoryPatchCoordinator,
)
from avalan.patch.coordinator import (
    JournalStep as JournalStep,
)
from avalan.patch.coordinator import (
    LockFootprint as LockFootprint,
)
from avalan.patch.coordinator import (
    RetransmissionKey as RetransmissionKey,
)
from avalan.patch.coordinator import (
    RevalidationFact as RevalidationFact,
)
from avalan.patch.coordinator import (
    RevalidationField as RevalidationField,
)
from avalan.patch.coordinator import (
    RevalidationSnapshot as RevalidationSnapshot,
)
from avalan.patch.coordinator import (
    RuntimeIdentity as RuntimeIdentity,
)
from avalan.patch.coordinator import (
    ScriptedCommitWorker as ScriptedCommitWorker,
)
from avalan.patch.coordinator import (
    ScriptedFaultController as ScriptedFaultController,
)
from avalan.patch.coordinator import (
    ScriptedReconciler as ScriptedReconciler,
)
from avalan.patch.coordinator import (
    SettlementJournal as SettlementJournal,
)
from avalan.patch.coordinator import (
    WorkerReport as WorkerReport,
)
from avalan.patch.coordinator import (
    WorkerState as WorkerState,
)
from avalan.patch.coordinator import (
    footprint_for as footprint_for,
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
from avalan.patch.planner import (
    BoundedPlannerWorker as BoundedPlannerWorker,
)
from avalan.patch.planner import (
    LogicalText as LogicalText,
)
from avalan.patch.planner import (
    Match as Match,
)
from avalan.patch.planner import (
    MatchKind as MatchKind,
)
from avalan.patch.planner import (
    PlannerCandidate as PlannerCandidate,
)
from avalan.patch.planner import (
    PlannerError as PlannerError,
)
from avalan.patch.planner import (
    PlannerErrorCode as PlannerErrorCode,
)
from avalan.patch.planner import (
    PlannerFacade as PlannerFacade,
)
from avalan.patch.planner import (
    PlannerFile as PlannerFile,
)
from avalan.patch.planner import (
    PlannerLimits as PlannerLimits,
)
from avalan.patch.planner import (
    PlannerParentMount as PlannerParentMount,
)
from avalan.patch.planner import (
    PlannerWorkspace as PlannerWorkspace,
)
from avalan.patch.planner import (
    StructuredDiff as StructuredDiff,
)
from avalan.patch.planner import (
    TextRepresentation as TextRepresentation,
)
from avalan.patch.planner import (
    apply_replacements as apply_replacements,
)
from avalan.patch.planner import (
    find_match as find_match,
)
from avalan.patch.planner import (
    plan as plan,
)
from avalan.patch.planner import (
    render_review_diff as render_review_diff,
)
from avalan.patch.planner import (
    supported_text as supported_text,
)
from avalan.patch.policy import (
    ApprovalRequirements as ApprovalRequirements,
)
from avalan.patch.policy import (
    ApprovalService as ApprovalService,
)
from avalan.patch.policy import (
    BrokerDecision as BrokerDecision,
)
from avalan.patch.policy import (
    PlanApprovalBroker as PlanApprovalBroker,
)
from avalan.patch.policy import (
    PlanReviewRequest as PlanReviewRequest,
)
from avalan.patch.policy import (
    PolicyAuthorizer as PolicyAuthorizer,
)
from avalan.patch.policy import (
    RuntimeGrantStore as RuntimeGrantStore,
)
from avalan.patch.policy import (
    RuntimePlanStore as RuntimePlanStore,
)
from avalan.patch.policy import (
    TrustedPatchPolicy as TrustedPatchPolicy,
)
from avalan.patch.policy import (
    seal_plan as seal_plan,
)
from avalan.patch.target import (
    AliasMode as AliasMode,
)
from avalan.patch.target import (
    CommitUnavailable as CommitUnavailable,
)
from avalan.patch.target import (
    EphemeralWorkerWitness as EphemeralWorkerWitness,
)
from avalan.patch.target import (
    FileIdentity as FileIdentity,
)
from avalan.patch.target import (
    ForeignWriterGuarantee as ForeignWriterGuarantee,
)
from avalan.patch.target import (
    InspectionBatch as InspectionBatch,
)
from avalan.patch.target import (
    InspectionRequest as InspectionRequest,
)
from avalan.patch.target import (
    LocalInspectionTarget as LocalInspectionTarget,
)
from avalan.patch.target import (
    LocalPlatformProfile as LocalPlatformProfile,
)
from avalan.patch.target import (
    MetadataClassification as MetadataClassification,
)
from avalan.patch.target import (
    MutationTarget as MutationTarget,
)
from avalan.patch.target import (
    ParentWitness as ParentWitness,
)
from avalan.patch.target import (
    PrimitiveProbe as PrimitiveProbe,
)
from avalan.patch.target import (
    ProbeState as ProbeState,
)
from avalan.patch.target import (
    ResolvedMutationScope as ResolvedMutationScope,
)
from avalan.patch.target import (
    RootWitness as RootWitness,
)
from avalan.patch.target import (
    ScopeResolver as ScopeResolver,
)
from avalan.patch.target import (
    ScopeSelection as ScopeSelection,
)
from avalan.patch.target import (
    TargetErrorCode as TargetErrorCode,
)
from avalan.patch.target import (
    TargetHandshake as TargetHandshake,
)
from avalan.patch.target import (
    TargetIdentity as TargetIdentity,
)
from avalan.patch.target import (
    TargetIncapableReason as TargetIncapableReason,
)
from avalan.patch.target import (
    TargetInspectionError as TargetInspectionError,
)
from avalan.patch.target import (
    TargetPrimitive as TargetPrimitive,
)
from avalan.patch.target import (
    TargetSnapshot as TargetSnapshot,
)
from avalan.patch.target import (
    WorkerIsolationPolicy as WorkerIsolationPolicy,
)
from avalan.patch.toolset import (
    PATCH_APPLY_SCHEMA as PATCH_APPLY_SCHEMA,
)
from avalan.patch.toolset import (
    PATCH_EDIT_SCHEMA as PATCH_EDIT_SCHEMA,
)
from avalan.patch.toolset import (
    InMemoryPatchLifecycleService as InMemoryPatchLifecycleService,
)
from avalan.patch.toolset import (
    PatchAdmissionDecision as PatchAdmissionDecision,
)
from avalan.patch.toolset import (
    PatchAdmissionFilter as PatchAdmissionFilter,
)
from avalan.patch.toolset import (
    PatchAdmissionView as PatchAdmissionView,
)
from avalan.patch.toolset import (
    PatchCapabilitySnapshot as PatchCapabilitySnapshot,
)
from avalan.patch.toolset import (
    PatchInvocationCapability as PatchInvocationCapability,
)
from avalan.patch.toolset import (
    PatchRuntimeBinder as PatchRuntimeBinder,
)
from avalan.patch.toolset import (
    PatchRuntimeBinding as PatchRuntimeBinding,
)
from avalan.patch.toolset import (
    PatchSdkHost as PatchSdkHost,
)
from avalan.patch.toolset import (
    PatchSdkService as PatchSdkService,
)
from avalan.patch.toolset import (
    PatchTestHostProfile as PatchTestHostProfile,
)
from avalan.patch.toolset import (
    PatchToolError as PatchToolError,
)
from avalan.patch.toolset import (
    PatchToolLoader as PatchToolLoader,
)
from avalan.patch.toolset import (
    PatchToolManagerBundle as PatchToolManagerBundle,
)
from avalan.patch.toolset import (
    PatchToolSet as PatchToolSet,
)
from avalan.patch.toolset import (
    project_model_result as project_model_result,
)

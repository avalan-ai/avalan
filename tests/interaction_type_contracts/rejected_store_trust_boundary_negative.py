"""Keep caller-authored store mutations outside sealed trust authority."""

from typing import cast

from avalan.interaction import (
    AnswerProvenance,
    CancelInteractionCommand,
    DeclinedResolution,
    InteractionActor,
    InteractionCorrelation,
    InteractionExecutionScope,
    ResolutionStatus,
    StateRevision,
    SupersedeInteractionScopeCommand,
    TaskInputClassification,
    TaskInputClassificationDecision,
    TaskInputClassificationRequest,
    TerminalizeInteractionCommand,
    TerminalizeInteractionScopeCommand,
    TrustedDefaultResolutionCommand,
    TrustedDefaultResolutionRequest,
)

actor = cast(InteractionActor, object())
correlation = cast(InteractionCorrelation, object())
scope = cast(InteractionExecutionScope, object())
revision = cast(StateRevision, object())
classification_request = cast(TaskInputClassificationRequest, object())
decline = cast(DeclinedResolution, object())
forged_command_type = TrustedDefaultResolutionCommand

TaskInputClassification._from_classifier(
    classification_request,
    decision=TaskInputClassificationDecision.ALLOW,
    classification_id="forged-classification",
)
TrustedDefaultResolutionRequest(
    actor=actor,
    correlation=correlation,
    expected_state_revision=revision,
    proposed_resolution=decline,
)
CancelInteractionCommand(
    actor=actor,
    correlation=correlation,
    provenance=AnswerProvenance.POLICY,
)
TerminalizeInteractionCommand(
    actor=actor,
    correlation=correlation,
    status=ResolutionStatus.UNAVAILABLE,
    provenance=AnswerProvenance.TRUSTED_DEFAULT,
)
TerminalizeInteractionScopeCommand(
    actor=actor,
    scope=scope,
    provenance=AnswerProvenance.POLICY,
)
SupersedeInteractionScopeCommand(
    actor=actor,
    scope=scope,
    provenance=AnswerProvenance.TRUSTED_DEFAULT,
)

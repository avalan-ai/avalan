"""Expose Avalan's root API after initializing package metadata."""

# ruff: noqa: E402

from .agent.orchestrator_contract import Orchestrator as Orchestrator
from .entities import GenerationSettings as GenerationSettings
from .entities import Input as Input
from .entities import Message as Message
from .entities import ReasoningSettings as ReasoningSettings
from .interaction.entities import AnswerProvenance as AnswerProvenance
from .interaction.entities import Choice as Choice
from .interaction.entities import ChoiceValue as ChoiceValue
from .interaction.entities import ConfirmationAnswer as ConfirmationAnswer
from .interaction.entities import ConfirmationQuestion as ConfirmationQuestion
from .interaction.entities import FreeFormOther as FreeFormOther
from .interaction.entities import InputAnswer as InputAnswer
from .interaction.entities import InputQuestion as InputQuestion
from .interaction.entities import MultilineTextAnswer as MultilineTextAnswer
from .interaction.entities import (
    MultilineTextQuestion as MultilineTextQuestion,
)
from .interaction.entities import (
    MultipleSelectionAnswer as MultipleSelectionAnswer,
)
from .interaction.entities import (
    MultipleSelectionQuestion as MultipleSelectionQuestion,
)
from .interaction.entities import QuestionId as QuestionId
from .interaction.entities import RequestState as RequestState
from .interaction.entities import RequirementMode as RequirementMode
from .interaction.entities import (
    ResolutionIdempotencyKey as ResolutionIdempotencyKey,
)
from .interaction.entities import SelectedChoice as SelectedChoice
from .interaction.entities import (
    SingleSelectionAnswer as SingleSelectionAnswer,
)
from .interaction.entities import (
    SingleSelectionQuestion as SingleSelectionQuestion,
)
from .interaction.entities import StateRevision as StateRevision
from .interaction.entities import TextAnswer as TextAnswer
from .interaction.entities import TextQuestion as TextQuestion
from .interaction.error import (
    InputAlreadyResolvedError as InputAlreadyResolvedError,
)
from .interaction.error import (
    InputAuthorizationError as InputAuthorizationError,
)
from .interaction.error import InputContractError as InputContractError
from .interaction.error import InputErrorCode as InputErrorCode
from .interaction.error import InputExpiredError as InputExpiredError
from .interaction.error import InputNotFoundError as InputNotFoundError
from .interaction.error import (
    InputSupersededError as InputSupersededError,
)
from .interaction.error import InputValidationError as InputValidationError
from .interaction.policy import InteractionPolicy as InteractionPolicy
from .interaction.policy import (
    TaskInputCapabilityState as TaskInputCapabilityState,
)

from importlib.metadata import metadata
from importlib.metadata import version as metadata_version
from typing import Any
from urllib.parse import ParseResult, urlparse

from packaging.version import Version, parse


def _config() -> dict[str, Any]:
    config = metadata("avalan")
    package_version = metadata_version("avalan")
    return {
        "name": config["Name"],
        "version": package_version,
        "license": config["License"],
        "url": "https://avalan.ai",
    }


config = _config()


def license() -> str:
    assert "license" in config
    return str(config["license"])


def name() -> str:
    assert "name" in config
    return str(config["name"])


def version() -> Version:
    assert "version" in config
    return parse(str(config["version"]))


def site() -> ParseResult:
    assert "url" in config
    return urlparse(str(config["url"]))


from .sdk import ActiveConversationSettings as ActiveConversationSettings
from .sdk import AgentHeadlessInputPolicy as AgentHeadlessInputPolicy
from .sdk import AgentInteractionRuntime as AgentInteractionRuntime
from .sdk import AgentRunCancelled as AgentRunCancelled
from .sdk import AgentRunCompleted as AgentRunCompleted
from .sdk import AgentRunFailed as AgentRunFailed
from .sdk import AgentRunInputRequired as AgentRunInputRequired
from .sdk import AgentRunResult as AgentRunResult
from .sdk import AgentRunResultKind as AgentRunResultKind
from .sdk import (
    AttachedInputCancellationHandler as AttachedInputCancellationHandler,
)
from .sdk import AttachedInputContext as AttachedInputContext
from .sdk import AttachedInputDetached as AttachedInputDetached
from .sdk import (
    AttachedInputDisconnected as AttachedInputDisconnected,
)
from .sdk import (
    AttachedInputDisconnectReason as AttachedInputDisconnectReason,
)
from .sdk import AttachedInputHandler as AttachedInputHandler
from .sdk import AttachedInputOutcome as AttachedInputOutcome
from .sdk import CompactionPolicy as CompactionPolicy
from .sdk import (
    ConversationBranchIntent as ConversationBranchIntent,
)
from .sdk import ConversationHandle as ConversationHandle
from .sdk import (
    ConversationHandleUnavailableError as ConversationHandleUnavailableError,
)
from .sdk import ConversationMode as ConversationMode
from .sdk import (
    ConversationModeChangeOperation as ConversationModeChangeOperation,
)
from .sdk import ConversationModeConversion as ConversationModeConversion
from .sdk import ConversationModeReset as ConversationModeReset
from .sdk import ConversationParent as ConversationParent
from .sdk import ConversationResetIntent as ConversationResetIntent
from .sdk import ConversationSettings as ConversationSettings
from .sdk import (
    DirectConversationCancelledError as DirectConversationCancelledError,
)
from .sdk import DirectConversationClient as DirectConversationClient
from .sdk import (
    DirectConversationOutputDelta as DirectConversationOutputDelta,
)
from .sdk import DirectConversationResult as DirectConversationResult
from .sdk import DirectConversationRuntime as DirectConversationRuntime
from .sdk import DirectConversationStream as DirectConversationStream
from .sdk import (
    DirectConversationStreamError as DirectConversationStreamError,
)
from .sdk import (
    DirectConversationStreamItem as DirectConversationStreamItem,
)
from .sdk import (
    DirectConversationStreamState as DirectConversationStreamState,
)
from .sdk import (
    DirectConversationStreamTerminal as DirectConversationStreamTerminal,
)
from .sdk import DirectDeletionResult as DirectDeletionResult
from .sdk import DisabledCompaction as DisabledCompaction
from .sdk import DurableInputBridge as DurableInputBridge
from .sdk import (
    DurableInputContinuationPayload as DurableInputContinuationPayload,
)
from .sdk import DurableInputIntegration as DurableInputIntegration
from .sdk import (
    DurableInputPersistenceAccepted as DurableInputPersistenceAccepted,
)
from .sdk import (
    DurableInputPersistenceRequest as DurableInputPersistenceRequest,
)
from .sdk import DurableInputRequestPayload as DurableInputRequestPayload
from .sdk import EffectiveReasoningContext as EffectiveReasoningContext
from .sdk import EffectiveReasoningMetadata as EffectiveReasoningMetadata
from .sdk import InlineCompaction as InlineCompaction
from .sdk import InputAnswerSubmission as InputAnswerSubmission
from .sdk import InputContinuationRef as InputContinuationRef
from .sdk import InputController as InputController
from .sdk import InputControllerBridge as InputControllerBridge
from .sdk import InputControllerClient as InputControllerClient
from .sdk import InputDeclineSubmission as InputDeclineSubmission
from .sdk import InputInspection as InputInspection
from .sdk import InputInspectionRequest as InputInspectionRequest
from .sdk import InputPolicyValueProvider as InputPolicyValueProvider
from .sdk import InputPrincipal as InputPrincipal
from .sdk import InputRequestRef as InputRequestRef
from .sdk import InputRequestView as InputRequestView
from .sdk import InputResolutionAccepted as InputResolutionAccepted
from .sdk import InputResolutionRequest as InputResolutionRequest
from .sdk import InputResolutionResult as InputResolutionResult
from .sdk import InputSubmission as InputSubmission
from .sdk import InputValidationFeedback as InputValidationFeedback
from .sdk import ModeTransitionAuthority as ModeTransitionAuthority
from .sdk import NamedHeadParent as NamedHeadParent
from .sdk import (
    OneShotConversationSettings as OneShotConversationSettings,
)
from .sdk import ProviderUsage as ProviderUsage
from .sdk import ReasoningContext as ReasoningContext
from .sdk import StandaloneCompactRequest as StandaloneCompactRequest
from .sdk import StandaloneCompactResult as StandaloneCompactResult
from .sdk import (
    StatelessConversationHandle as StatelessConversationHandle,
)
from .sdk import (
    StatelessConversationSettings as StatelessConversationSettings,
)
from .sdk import StatelessParent as StatelessParent
from .sdk import StoredConversationHandle as StoredConversationHandle
from .sdk import (
    StoredConversationSettings as StoredConversationSettings,
)
from .sdk import StoredParent as StoredParent
from .sdk import (
    create_attached_input_runtime as create_attached_input_runtime,
)
from .sdk import create_decline_input_policy as create_decline_input_policy
from .sdk import (
    create_durable_input_integration as create_durable_input_integration,
)
from .sdk import create_external_controller_input_policy as _external_policy
from .sdk import create_input_controller as create_input_controller
from .sdk import (
    create_policy_value_input_policy as create_policy_value_input_policy,
)
from .sdk import (
    create_predeclared_input_policy as create_predeclared_input_policy,
)
from .sdk import (
    create_trusted_default_input_policy as create_trusted_default_input_policy,
)
from .sdk import (
    create_unavailable_input_policy as create_unavailable_input_policy,
)
from .sdk import inspect_input as inspect_input
from .sdk import resolve_input as resolve_input
from .sdk import run_agent as run_agent

create_external_controller_input_policy = _external_policy

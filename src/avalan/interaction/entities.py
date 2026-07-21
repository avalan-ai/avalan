"""Define immutable values for structured task interactions."""

from ..types import JsonValue
from .error import InputErrorCode, InputValidationError
from .validation import (
    MAX_STATE_REVISION,
    validate_aware_datetime,
    validate_bool,
    validate_choice_value,
    validate_int,
    validate_multiline_text,
    validate_opaque_id,
    validate_other_text,
    validate_presentation_text,
    validate_question_id,
    validate_single_line_text,
    validate_state_revision,
    validate_total_request_content,
)

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from json import dumps
from math import isfinite
from re import compile as compile_pattern
from types import MappingProxyType
from typing import ClassVar, Literal, NewType, NoReturn, TypeAlias, final
from unicodedata import normalize

RESERVED_INPUT_CAPABILITY_NAME = "request_user_input"
REQUEST_AGGREGATE_CONTENT_MEMBERS = (
    "interaction_class",
    "mode",
    "reason",
    "ordered question identifiers, types, prompts, and requirements",
    "headers, help text, choices, recommendations, and defaults",
    "free-form permissions, validation constraints, and presentation hints",
)

RunId = NewType("RunId", str)
TurnId = NewType("TurnId", str)
TaskId = NewType("TaskId", str)
AgentId = NewType("AgentId", str)
BranchId = NewType("BranchId", str)
ModelCallId = NewType("ModelCallId", str)
InputRequestId = NewType("InputRequestId", str)
ContinuationId = NewType("ContinuationId", str)
StreamSessionId = NewType("StreamSessionId", str)
UserId = NewType("UserId", str)
TenantId = NewType("TenantId", str)
ParticipantId = NewType("ParticipantId", str)
SessionId = NewType("SessionId", str)
QuestionId = NewType("QuestionId", str)
ChoiceValue = NewType("ChoiceValue", str)
StateRevision = NewType("StateRevision", int)
ProviderFamilyName = NewType("ProviderFamilyName", str)
ModelId = NewType("ModelId", str)
ProviderConfigRevision = NewType("ProviderConfigRevision", str)
ModelConfigRevision = NewType("ModelConfigRevision", str)
CapabilityRevision = NewType("CapabilityRevision", str)
ProviderIdempotencyKey = NewType("ProviderIdempotencyKey", str)

_SNAPSHOT_KIND_PATTERN = compile_pattern(r"^[a-z][a-z0-9._-]{0,63}$")
_PROHIBITED_SNAPSHOT_KEY_SUFFIXES = (
    ("api", "key"),
    ("api", "keys"),
    ("authentication", "challenge"),
    ("authentication", "challenges"),
    ("authorization",),
    ("authorizations",),
    ("credential",),
    ("credentials",),
    ("mfa", "challenge"),
    ("mfa", "challenges"),
    ("password",),
    ("passwords",),
    ("payment", "card"),
    ("payment", "cards"),
    ("private", "key"),
    ("private", "keys"),
    ("secret",),
    ("secrets",),
    ("token",),
    ("tokens",),
)
_PROHIBITED_SNAPSHOT_COMPACT_SUFFIXES = tuple(
    "".join(suffix) for suffix in _PROHIBITED_SNAPSHOT_KEY_SUFFIXES
)
_SNAPSHOT_CREDENTIAL_PAYLOAD_QUALIFIERS = frozenset(
    {
        "header",
        "headers",
        "value",
        "values",
    }
)
_SAFE_SNAPSHOT_TOKEN_METRIC_KEYS = frozenset(
    {
        "completion_tokens",
        "input_tokens",
        "max_tokens",
        "output_tokens",
        "prompt_tokens",
        "total_tokens",
    }
)


class InteractionClass(StrEnum):
    """Identify interaction state machines that must remain separate."""

    TASK_INPUT = "task_input"
    ACTION_APPROVAL = "action_approval"
    STEERING = "steering"
    AUTHENTICATION = "authentication"


class QuestionType(StrEnum):
    """Identify the semantic type of an input question."""

    CONFIRMATION = "confirmation"
    TEXT = "text"
    MULTILINE_TEXT = "multiline_text"
    SINGLE_SELECTION = "single_selection"
    MULTIPLE_SELECTION = "multiple_selection"


class RequirementMode(StrEnum):
    """Identify whether a request can time out while its run exists."""

    REQUIRED = "required"
    ADVISORY = "advisory"


class RequestState(StrEnum):
    """Identify the canonical lifecycle state of an input request."""

    CREATED = "created"
    PENDING = "pending"
    ANSWERED = "answered"
    DECLINED = "declined"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"
    UNAVAILABLE = "unavailable"
    EXPIRED = "expired"
    SUPERSEDED = "superseded"


class ResolutionStatus(StrEnum):
    """Identify an exactly-once terminal resolution category."""

    ANSWERED = "answered"
    DECLINED = "declined"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"
    UNAVAILABLE = "unavailable"
    EXPIRED = "expired"
    SUPERSEDED = "superseded"


class AnswerProvenance(StrEnum):
    """Identify who or what supplied a resolution value."""

    HUMAN = "human"
    TRUSTED_DEFAULT = "trusted_default"
    POLICY = "policy"
    EXTERNAL_CONTROLLER = "external_controller"


class PresentationHint(StrEnum):
    """Express an advisory native presentation preference."""

    COMPACT = "compact"
    EXPANDED = "expanded"
    RADIO = "radio"
    LIST = "list"
    CHECKBOX = "checkbox"
    SINGLE_LINE = "single_line"
    EDITOR = "editor"


class HostHandling(StrEnum):
    """Identify how an interaction-capable host can handle requests."""

    ATTACHED = "attached"
    DETACHED = "detached"
    UNAVAILABLE = "unavailable"


class SelectionValueType(StrEnum):
    """Identify a structured selection answer value."""

    SELECTED_CHOICE = "selected_choice"
    FREE_FORM_OTHER = "free_form_other"


class InputResultKind(StrEnum):
    """Identify a model-facing or suspension result variant."""

    INPUT_REQUIRED = "input_required"
    ANSWERED = "answered"
    DECLINED = "declined"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"
    UNAVAILABLE = "unavailable"


class ContinuationDisposition(StrEnum):
    """Identify whether a resolution resumes or terminates execution."""

    RESUME = "resume"
    TERMINATE = "terminate"


class CancellationScope(StrEnum):
    """Distinguish request cancellation from containing-run cancellation."""

    REQUEST = "request"
    CONTAINING_RUN = "containing_run"


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class ExecutionDefinitionRef:
    """Identify the trusted immutable definition needed for continuation."""

    agent_definition_locator: str
    agent_definition_revision: str
    operation_id: str
    operation_index: int
    model_config_reference: str
    tool_revision: str
    capability_revision: str

    def __post_init__(self) -> None:
        bounds = {
            "agent_definition_locator": (2_048, 8_192),
            "agent_definition_revision": (128, 512),
            "operation_id": (128, 512),
            "model_config_reference": (512, 2_048),
            "tool_revision": (128, 512),
            "capability_revision": (128, 512),
        }
        for name, (maximum_characters, maximum_bytes) in bounds.items():
            object.__setattr__(
                self,
                name,
                validate_opaque_id(
                    getattr(self, name),
                    name,
                    maximum_characters=maximum_characters,
                    maximum_bytes=maximum_bytes,
                ),
            )
        object.__setattr__(
            self,
            "operation_index",
            validate_int(
                self.operation_index,
                "operation_index",
                minimum=0,
                maximum=MAX_STATE_REVISION,
            ),
        )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class PrincipalScope:
    """Carry optional identity scope supplied by a trusted host."""

    user_id: UserId | None = None
    tenant_id: TenantId | None = None
    participant_id: ParticipantId | None = None
    session_id: SessionId | None = None

    def __post_init__(self) -> None:
        for name, constructor in (
            ("user_id", UserId),
            ("tenant_id", TenantId),
            ("participant_id", ParticipantId),
            ("session_id", SessionId),
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(
                    self,
                    name,
                    constructor(validate_opaque_id(value, name)),
                )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class ExecutionOrigin:
    """Correlate an interaction with one logical execution."""

    run_id: RunId
    turn_id: TurnId
    agent_id: AgentId
    branch_id: BranchId
    model_call_id: ModelCallId
    stream_session_id: StreamSessionId
    definition: ExecutionDefinitionRef
    principal: PrincipalScope = field(default_factory=PrincipalScope)
    task_id: TaskId | None = None
    parent_branch_id: BranchId | None = None

    def __post_init__(self) -> None:
        for name, constructor in (
            ("run_id", RunId),
            ("turn_id", TurnId),
            ("agent_id", AgentId),
            ("branch_id", BranchId),
            ("model_call_id", ModelCallId),
            ("stream_session_id", StreamSessionId),
        ):
            object.__setattr__(
                self,
                name,
                constructor(validate_opaque_id(getattr(self, name), name)),
            )
        if self.task_id is not None:
            object.__setattr__(
                self,
                "task_id",
                TaskId(validate_opaque_id(self.task_id, "task_id")),
            )
        if self.parent_branch_id is not None:
            object.__setattr__(
                self,
                "parent_branch_id",
                BranchId(
                    validate_opaque_id(
                        self.parent_branch_id,
                        "parent_branch_id",
                    )
                ),
            )
        if not isinstance(self.definition, ExecutionDefinitionRef):
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "definition",
                "value must be an execution definition reference",
            )
        if not isinstance(self.principal, PrincipalScope):
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "principal",
                "value must be a principal scope",
            )
        if self.parent_branch_id == self.branch_id:
            raise InputValidationError(
                InputErrorCode.INVALID_FORMAT,
                "parent_branch_id",
                "parent branch must differ from the active branch",
            )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class HostCapabilities:
    """Describe host-owned attached and durable resolution support."""

    attached_resolution: bool = False
    durable_resolution: bool = False

    def __post_init__(self) -> None:
        validate_bool(self.attached_resolution, "attached_resolution")
        validate_bool(self.durable_resolution, "durable_resolution")

    @property
    def handling(self) -> HostHandling:
        """Return the strongest handling mode available to the host."""
        if self.attached_resolution:
            return HostHandling.ATTACHED
        if self.durable_resolution:
            return HostHandling.DETACHED
        return HostHandling.UNAVAILABLE

    @property
    def can_advertise(self) -> bool:
        """Return whether the reserved capability can be advertised."""
        return self.handling is not HostHandling.UNAVAILABLE


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class ContinuationRevisionBinding:
    """Bind a continuation snapshot to trusted provider revisions."""

    provider_family: ProviderFamilyName
    model_id: ModelId
    provider_config_revision: ProviderConfigRevision
    model_config_revision: ModelConfigRevision
    capability_revision: CapabilityRevision

    def __post_init__(self) -> None:
        values = (
            (
                "provider_family",
                ProviderFamilyName,
                self.provider_family,
                64,
                256,
            ),
            ("model_id", ModelId, self.model_id, 512, 2_048),
            (
                "provider_config_revision",
                ProviderConfigRevision,
                self.provider_config_revision,
                128,
                512,
            ),
            (
                "model_config_revision",
                ModelConfigRevision,
                self.model_config_revision,
                128,
                512,
            ),
            (
                "capability_revision",
                CapabilityRevision,
                self.capability_revision,
                128,
                512,
            ),
        )
        for (
            name,
            constructor,
            value,
            maximum_characters,
            maximum_bytes,
        ) in values:
            object.__setattr__(
                self,
                name,
                constructor(
                    validate_opaque_id(
                        value,
                        name,
                        maximum_characters=maximum_characters,
                        maximum_bytes=maximum_bytes,
                    )
                ),
            )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class ContinuationSnapshot:
    """Store validated JSON provider state for durable continuation."""

    snapshot_kind: str
    revision_binding: ContinuationRevisionBinding
    model_call_id: ModelCallId
    provider_idempotency_key: ProviderIdempotencyKey
    payload: Mapping[str, JsonValue]
    version: int = 1

    def __post_init__(self) -> None:
        if self.version != 1 or isinstance(self.version, bool):
            raise InputValidationError(
                InputErrorCode.SNAPSHOT_UNSUPPORTED,
                "continuation_snapshot.version",
                "snapshot version is unsupported",
            )
        object.__setattr__(
            self,
            "snapshot_kind",
            _validate_snapshot_kind(self.snapshot_kind),
        )
        if not isinstance(self.revision_binding, ContinuationRevisionBinding):
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "continuation_snapshot.revision_binding",
                "value must be a continuation revision binding",
            )
        object.__setattr__(
            self,
            "model_call_id",
            ModelCallId(
                validate_opaque_id(
                    self.model_call_id,
                    "continuation_snapshot.model_call_id",
                )
            ),
        )
        object.__setattr__(
            self,
            "provider_idempotency_key",
            ProviderIdempotencyKey(
                validate_opaque_id(
                    self.provider_idempotency_key,
                    "continuation_snapshot.provider_idempotency_key",
                    maximum_characters=256,
                    maximum_bytes=1_024,
                )
            ),
        )
        if not isinstance(self.payload, Mapping):
            raise InputValidationError(
                InputErrorCode.SNAPSHOT_INVALID,
                "continuation_snapshot.payload",
                "snapshot payload must be a JSON object",
            )
        object.__setattr__(
            self,
            "payload",
            _freeze_snapshot_object(
                self.payload,
                "continuation_snapshot.payload",
            ),
        )


def _validate_snapshot_kind(value: object) -> str:
    snapshot_kind = validate_opaque_id(
        value,
        "continuation_snapshot.snapshot_kind",
        maximum_characters=64,
        maximum_bytes=256,
    )
    if _SNAPSHOT_KIND_PATTERN.fullmatch(snapshot_kind) is None:
        raise InputValidationError(
            InputErrorCode.SNAPSHOT_UNSUPPORTED,
            "continuation_snapshot.snapshot_kind",
            "snapshot kind has an invalid format",
        )
    return snapshot_kind


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class Choice:
    """Pair a stable machine value with mutable display wording."""

    value: ChoiceValue
    label: str
    description: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "value",
            ChoiceValue(validate_choice_value(self.value)),
        )
        object.__setattr__(
            self,
            "label",
            validate_presentation_text(
                self.label,
                "choice.label",
                minimum=1,
                maximum=80,
                maximum_bytes=320,
            ),
        )
        if self.label == "Other":
            raise InputValidationError(
                InputErrorCode.INVALID_FORMAT,
                "choice.label",
                "the free-form alternative is not a literal choice",
            )
        if self.description is not None:
            object.__setattr__(
                self,
                "description",
                validate_presentation_text(
                    self.description,
                    "choice.description",
                    minimum=1,
                    maximum=240,
                    maximum_bytes=960,
                ),
            )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class TextValidationConstraints:
    """Bound accepted text length within the semantic type limit."""

    minimum_length: int = 0
    maximum_length: int = 4_096

    def __post_init__(self) -> None:
        minimum = validate_int(
            self.minimum_length,
            "constraints.minimum_length",
            minimum=0,
            maximum=65_536,
        )
        maximum = validate_int(
            self.maximum_length,
            "constraints.maximum_length",
            minimum=0,
            maximum=65_536,
        )
        if minimum > maximum:
            raise InputValidationError(
                InputErrorCode.OUT_OF_BOUNDS,
                "constraints",
                "minimum length must not exceed maximum length",
            )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class SelectionValidationConstraints:
    """Bound the cardinality of a multiple-selection answer."""

    minimum: int = 0
    maximum: int = 20

    def __post_init__(self) -> None:
        minimum = validate_int(
            self.minimum,
            "constraints.minimum",
            minimum=0,
            maximum=20,
        )
        maximum = validate_int(
            self.maximum,
            "constraints.maximum",
            minimum=0,
            maximum=20,
        )
        if minimum > maximum:
            raise InputValidationError(
                InputErrorCode.INVALID_CARDINALITY,
                "constraints",
                "minimum selection count must not exceed maximum",
            )


_DEFAULT_SELECTION_CONSTRAINTS = SelectionValidationConstraints()


def _default_selection_constraints() -> SelectionValidationConstraints:
    return _DEFAULT_SELECTION_CONSTRAINTS


@dataclass(frozen=True, slots=True, kw_only=True)
class InputQuestion(ABC):
    """Store the fields shared by every semantic question variant."""

    question_id: QuestionId
    prompt: str
    required: bool
    header: str | None = None
    help_text: str | None = None
    presentation_hint: PresentationHint | None = None

    @property
    @abstractmethod
    def kind(self) -> QuestionType:
        """Return the semantic question discriminator."""

    def __post_init__(self) -> None:
        if not _is_input_question_variant(self):
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "question",
                "value must be a supported input question variant",
            )
        object.__setattr__(
            self,
            "question_id",
            QuestionId(validate_question_id(self.question_id)),
        )
        object.__setattr__(
            self,
            "prompt",
            validate_presentation_text(
                self.prompt,
                "question.prompt",
                minimum=1,
                maximum=500,
                maximum_bytes=2_000,
            ),
        )
        validate_bool(self.required, "question.required")
        if self.header is not None:
            object.__setattr__(
                self,
                "header",
                validate_presentation_text(
                    self.header,
                    "question.header",
                    minimum=1,
                    maximum=40,
                    maximum_bytes=160,
                ),
            )
        if self.help_text is not None:
            object.__setattr__(
                self,
                "help_text",
                validate_presentation_text(
                    self.help_text,
                    "question.help_text",
                    minimum=1,
                    maximum=1_000,
                    maximum_bytes=4_000,
                ),
            )
        if self.presentation_hint is not None and not isinstance(
            self.presentation_hint,
            PresentationHint,
        ):
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "question.presentation_hint",
                "value must be a presentation hint",
            )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class ConfirmationQuestion(InputQuestion):
    """Ask for a strict boolean confirmation."""

    default_value: bool | None = None
    kind: ClassVar[QuestionType] = QuestionType.CONFIRMATION

    def __post_init__(self) -> None:
        InputQuestion.__post_init__(self)
        if self.default_value is not None:
            validate_bool(self.default_value, "question.default_value")


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class TextQuestion(InputQuestion):
    """Ask for bounded single-line text."""

    default_value: str | None = None
    constraints: TextValidationConstraints = field(
        default_factory=TextValidationConstraints
    )
    kind: ClassVar[QuestionType] = QuestionType.TEXT

    def __post_init__(self) -> None:
        InputQuestion.__post_init__(self)
        _validate_text_constraints(self.constraints, 4_096)
        if self.default_value is not None:
            default = validate_single_line_text(
                self.default_value,
                "question.default_value",
            )
            _validate_text_cardinality(
                default,
                self.constraints,
                self.required,
                "question.default_value",
                InputErrorCode.INVALID_DEFAULT,
            )
            object.__setattr__(self, "default_value", default)


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class MultilineTextQuestion(InputQuestion):
    """Ask for bounded multiline text."""

    default_value: str | None = None
    constraints: TextValidationConstraints = field(
        default_factory=lambda: TextValidationConstraints(
            maximum_length=65_536
        )
    )
    kind: ClassVar[QuestionType] = QuestionType.MULTILINE_TEXT

    def __post_init__(self) -> None:
        InputQuestion.__post_init__(self)
        _validate_text_constraints(self.constraints, 65_536)
        if self.default_value is not None:
            default = validate_multiline_text(
                self.default_value,
                "question.default_value",
            )
            _validate_text_cardinality(
                default,
                self.constraints,
                self.required,
                "question.default_value",
                InputErrorCode.INVALID_DEFAULT,
            )
            object.__setattr__(self, "default_value", default)


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class SingleSelectionQuestion(InputQuestion):
    """Ask for one stable choice or a tagged free-form alternative."""

    choices: tuple[Choice, ...]
    allow_other: bool = False
    recommended_choice: ChoiceValue | None = None
    default_value: ChoiceValue | None = None
    kind: ClassVar[QuestionType] = QuestionType.SINGLE_SELECTION

    def __post_init__(self) -> None:
        InputQuestion.__post_init__(self)
        choices = _validate_choices(self.choices)
        validate_bool(self.allow_other, "question.allow_other")
        object.__setattr__(self, "choices", choices)
        if self.recommended_choice is not None:
            recommendation = _validate_choice_reference(
                self.recommended_choice,
                choices,
                "question.recommended_choice",
                InputErrorCode.INVALID_RECOMMENDATION,
            )
            object.__setattr__(
                self,
                "recommended_choice",
                recommendation,
            )
        if self.default_value is not None:
            default = _validate_choice_reference(
                self.default_value,
                choices,
                "question.default_value",
                InputErrorCode.INVALID_DEFAULT,
            )
            object.__setattr__(self, "default_value", default)


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class MultipleSelectionQuestion(InputQuestion):
    """Ask for an ordered tuple of stable selection values."""

    choices: tuple[Choice, ...]
    allow_other: bool = False
    recommended_choice: ChoiceValue | None = None
    default_value: tuple[ChoiceValue, ...] | None = None
    constraints: SelectionValidationConstraints = field(
        default_factory=_default_selection_constraints
    )
    kind: ClassVar[QuestionType] = QuestionType.MULTIPLE_SELECTION

    def __post_init__(self) -> None:
        InputQuestion.__post_init__(self)
        choices = _validate_choices(self.choices)
        validate_bool(self.allow_other, "question.allow_other")
        if not isinstance(self.constraints, SelectionValidationConstraints):
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "question.constraints",
                "value must be selection constraints",
            )
        available = len(choices) + int(self.allow_other)
        if self.constraints is _DEFAULT_SELECTION_CONSTRAINTS:
            object.__setattr__(
                self,
                "constraints",
                SelectionValidationConstraints(maximum=min(20, available)),
            )
        elif self.constraints.maximum > min(20, available):
            raise InputValidationError(
                InputErrorCode.INVALID_CARDINALITY,
                "question.constraints.maximum",
                "maximum exceeds the available selections",
            )
        object.__setattr__(self, "choices", choices)
        if self.recommended_choice is not None:
            recommendation = _validate_choice_reference(
                self.recommended_choice,
                choices,
                "question.recommended_choice",
                InputErrorCode.INVALID_RECOMMENDATION,
            )
            object.__setattr__(
                self,
                "recommended_choice",
                recommendation,
            )
        if self.default_value is not None:
            default = _validate_choice_references(
                self.default_value,
                choices,
                "question.default_value",
                InputErrorCode.INVALID_DEFAULT,
            )
            minimum = max(self.constraints.minimum, int(self.required))
            if (
                len(default) < minimum
                or len(default) > self.constraints.maximum
            ):
                raise InputValidationError(
                    InputErrorCode.INVALID_DEFAULT,
                    "question.default_value",
                    "default has invalid selection cardinality",
                )
            object.__setattr__(self, "default_value", default)


_INPUT_QUESTION_VARIANT_TYPES: tuple[type[InputQuestion], ...] = (
    ConfirmationQuestion,
    TextQuestion,
    MultilineTextQuestion,
    SingleSelectionQuestion,
    MultipleSelectionQuestion,
)


def _is_input_question_variant(value: object) -> bool:
    return type(value) in _INPUT_QUESTION_VARIANT_TYPES


@dataclass(frozen=True, slots=True, kw_only=True)
class SelectionValue(ABC):
    """Store one tagged value in a selection answer."""

    @property
    @abstractmethod
    def kind(self) -> SelectionValueType:
        """Return the selection-value discriminator."""

    def __post_init__(self) -> None:
        if not _is_selection_value_variant(self):
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "selection",
                "value must be a supported selection value variant",
            )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class SelectedChoice(SelectionValue):
    """Select a stable choice value."""

    value: ChoiceValue
    kind: ClassVar[SelectionValueType] = SelectionValueType.SELECTED_CHOICE

    def __post_init__(self) -> None:
        SelectionValue.__post_init__(self)
        object.__setattr__(
            self,
            "value",
            ChoiceValue(validate_choice_value(self.value, "answer.value")),
        )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class FreeFormOther(SelectionValue):
    """Enter a tagged free-form alternative to stable choices."""

    text: str
    kind: ClassVar[SelectionValueType] = SelectionValueType.FREE_FORM_OTHER

    def __post_init__(self) -> None:
        SelectionValue.__post_init__(self)
        object.__setattr__(self, "text", validate_other_text(self.text))


_SELECTION_VALUE_VARIANT_TYPES: tuple[type[SelectionValue], ...] = (
    SelectedChoice,
    FreeFormOther,
)


def _is_selection_value_variant(value: object) -> bool:
    return type(value) in _SELECTION_VALUE_VARIANT_TYPES


@dataclass(frozen=True, slots=True, kw_only=True)
class InputAnswer(ABC):
    """Store the fields shared by typed answers."""

    question_id: QuestionId
    provenance: AnswerProvenance

    @property
    @abstractmethod
    def question_type(self) -> QuestionType:
        """Return the corresponding semantic question type."""

    def __post_init__(self) -> None:
        if not _is_input_answer_variant(self):
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "answer",
                "value must be a supported input answer variant",
            )
        object.__setattr__(
            self,
            "question_id",
            QuestionId(validate_question_id(self.question_id)),
        )
        if not isinstance(self.provenance, AnswerProvenance):
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "answer.provenance",
                "value must be answer provenance",
            )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class ConfirmationAnswer(InputAnswer):
    """Provide a strict boolean confirmation answer."""

    value: bool
    question_type: ClassVar[QuestionType] = QuestionType.CONFIRMATION

    def __post_init__(self) -> None:
        InputAnswer.__post_init__(self)
        validate_bool(self.value, "answer.value")


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class TextAnswer(InputAnswer):
    """Provide a bounded single-line text answer."""

    value: str
    question_type: ClassVar[QuestionType] = QuestionType.TEXT

    def __post_init__(self) -> None:
        InputAnswer.__post_init__(self)
        object.__setattr__(
            self,
            "value",
            validate_single_line_text(self.value, "answer.value"),
        )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class MultilineTextAnswer(InputAnswer):
    """Provide a bounded normalized multiline answer."""

    value: str
    question_type: ClassVar[QuestionType] = QuestionType.MULTILINE_TEXT

    def __post_init__(self) -> None:
        InputAnswer.__post_init__(self)
        object.__setattr__(
            self,
            "value",
            validate_multiline_text(self.value, "answer.value"),
        )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class SingleSelectionAnswer(InputAnswer):
    """Provide one tagged selection answer."""

    value: SelectionValue
    question_type: ClassVar[QuestionType] = QuestionType.SINGLE_SELECTION

    def __post_init__(self) -> None:
        InputAnswer.__post_init__(self)
        if not _is_selection_value_variant(self.value):
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "answer.value",
                "value must be a tagged selection value",
            )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class MultipleSelectionAnswer(InputAnswer):
    """Provide an immutable ordered tuple of tagged selections."""

    values: tuple[SelectionValue, ...]
    question_type: ClassVar[QuestionType] = QuestionType.MULTIPLE_SELECTION

    def __post_init__(self) -> None:
        InputAnswer.__post_init__(self)
        if not isinstance(self.values, tuple):
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "answer.values",
                "value must be a tuple",
            )
        if len(self.values) > 20:
            raise InputValidationError(
                InputErrorCode.INVALID_CARDINALITY,
                "answer.values",
                "selection count exceeds its maximum",
            )
        selected: set[ChoiceValue] = set()
        other_count = 0
        for value in self.values:
            if type(value) is SelectedChoice:
                if value.value in selected:
                    raise InputValidationError(
                        InputErrorCode.DUPLICATE,
                        "answer.values",
                        "selected choice values must be unique",
                    )
                selected.add(value.value)
            elif type(value) is FreeFormOther:
                other_count += 1
            else:
                raise InputValidationError(
                    InputErrorCode.INVALID_TYPE,
                    "answer.values",
                    "values must be tagged selections",
                )
        if other_count > 1:
            raise InputValidationError(
                InputErrorCode.INVALID_CARDINALITY,
                "answer.values",
                "only one free-form alternative is permitted",
            )


_INPUT_ANSWER_VARIANT_TYPES: tuple[type[InputAnswer], ...] = (
    ConfirmationAnswer,
    TextAnswer,
    MultilineTextAnswer,
    SingleSelectionAnswer,
    MultipleSelectionAnswer,
)


def _is_input_answer_variant(value: object) -> bool:
    return type(value) in _INPUT_ANSWER_VARIANT_TYPES


@dataclass(frozen=True, slots=True, kw_only=True)
class InputResolution(ABC):
    """Store the fields shared by every terminal resolution."""

    request_id: InputRequestId
    provenance: AnswerProvenance
    resolved_at: datetime

    @property
    @abstractmethod
    def status(self) -> ResolutionStatus:
        """Return the terminal resolution discriminator."""

    def __post_init__(self) -> None:
        if not _is_input_resolution_variant(self):
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "resolution",
                "value must be a supported input resolution variant",
            )
        object.__setattr__(
            self,
            "request_id",
            InputRequestId(
                validate_opaque_id(self.request_id, "resolution.request_id")
            ),
        )
        if not isinstance(self.provenance, AnswerProvenance):
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "resolution.provenance",
                "value must be answer provenance",
            )
        object.__setattr__(
            self,
            "resolved_at",
            validate_aware_datetime(
                self.resolved_at,
                "resolution.resolved_at",
            ),
        )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class AnsweredResolution(InputResolution):
    """Resolve a request with typed answers."""

    answers: tuple[InputAnswer, ...]
    status: ClassVar[ResolutionStatus] = ResolutionStatus.ANSWERED

    def __post_init__(self) -> None:
        InputResolution.__post_init__(self)
        if not isinstance(self.answers, tuple):
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "resolution.answers",
                "answers must be a tuple",
            )
        seen: set[QuestionId] = set()
        for answer in self.answers:
            if not _is_input_answer_variant(answer):
                raise InputValidationError(
                    InputErrorCode.INVALID_TYPE,
                    "resolution.answers",
                    "answers must contain typed answer values",
                )
            if answer.question_id in seen:
                raise InputValidationError(
                    InputErrorCode.DUPLICATE,
                    "resolution.answers",
                    "answer question identifiers must be unique",
                )
            seen.add(answer.question_id)


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class DeclinedResolution(InputResolution):
    """Resolve a request as intentionally declined."""

    status: ClassVar[ResolutionStatus] = ResolutionStatus.DECLINED


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class CancelledResolution(InputResolution):
    """Resolve a request as explicitly cancelled."""

    scope: CancellationScope = CancellationScope.REQUEST
    status: ClassVar[ResolutionStatus] = ResolutionStatus.CANCELLED

    def __post_init__(self) -> None:
        InputResolution.__post_init__(self)
        if not isinstance(self.scope, CancellationScope):
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "resolution.scope",
                "value must be a cancellation scope",
            )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class TimedOutResolution(InputResolution):
    """Resolve an advisory request after its waiting budget."""

    status: ClassVar[ResolutionStatus] = ResolutionStatus.TIMED_OUT


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class UnavailableResolution(InputResolution):
    """Resolve a request when the active host loses capability."""

    status: ClassVar[ResolutionStatus] = ResolutionStatus.UNAVAILABLE


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class ExpiredResolution(InputResolution):
    """Resolve a request whose continuation is no longer valid."""

    status: ClassVar[ResolutionStatus] = ResolutionStatus.EXPIRED


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class SupersededResolution(InputResolution):
    """Resolve a request invalidated by newer branch state."""

    status: ClassVar[ResolutionStatus] = ResolutionStatus.SUPERSEDED


_INPUT_RESOLUTION_VARIANT_TYPES: tuple[type[InputResolution], ...] = (
    AnsweredResolution,
    DeclinedResolution,
    CancelledResolution,
    TimedOutResolution,
    UnavailableResolution,
    ExpiredResolution,
    SupersededResolution,
)


def _is_input_resolution_variant(value: object) -> bool:
    return type(value) in _INPUT_RESOLUTION_VARIANT_TYPES


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class InputRequest:
    """Represent one canonical structured task-input request."""

    request_id: InputRequestId
    continuation_id: ContinuationId
    origin: ExecutionOrigin
    mode: RequirementMode
    reason: str
    questions: tuple[InputQuestion, ...]
    created_at: datetime
    continuation_ttl_seconds: int = 86_400
    advisory_wait_seconds: int | None = None
    advisory_deadline: datetime | None = None
    state: RequestState = RequestState.CREATED
    state_revision: StateRevision = StateRevision(0)
    resolution: InputResolution | None = None
    interaction_class: InteractionClass = field(
        init=False,
        default=InteractionClass.TASK_INPUT,
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "request_id",
            InputRequestId(validate_opaque_id(self.request_id, "request_id")),
        )
        object.__setattr__(
            self,
            "continuation_id",
            ContinuationId(
                validate_opaque_id(self.continuation_id, "continuation_id")
            ),
        )
        if not isinstance(self.origin, ExecutionOrigin):
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "origin",
                "value must be an execution origin",
            )
        if not isinstance(self.mode, RequirementMode):
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "mode",
                "value must be a requirement mode",
            )
        object.__setattr__(
            self,
            "reason",
            validate_presentation_text(
                self.reason,
                "reason",
                minimum=1,
                maximum=500,
                maximum_bytes=2_000,
            ),
        )
        if not isinstance(self.questions, tuple):
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "questions",
                "questions must be a tuple",
            )
        if len(self.questions) < 1 or len(self.questions) > 3:
            raise InputValidationError(
                InputErrorCode.OUT_OF_BOUNDS,
                "questions",
                "a request must contain one to three questions",
            )
        question_ids: set[QuestionId] = set()
        for question in self.questions:
            if not _is_input_question_variant(question):
                raise InputValidationError(
                    InputErrorCode.INVALID_TYPE,
                    "questions",
                    "questions must contain typed question variants",
                )
            if question.question_id in question_ids:
                raise InputValidationError(
                    InputErrorCode.DUPLICATE,
                    "questions",
                    "question identifiers must be unique",
                )
            question_ids.add(question.question_id)
        object.__setattr__(
            self,
            "created_at",
            validate_aware_datetime(self.created_at, "created_at"),
        )
        object.__setattr__(
            self,
            "continuation_ttl_seconds",
            validate_int(
                self.continuation_ttl_seconds,
                "continuation_ttl_seconds",
                minimum=60,
                maximum=604_800,
            ),
        )
        if self.mode is RequirementMode.REQUIRED:
            if self.advisory_wait_seconds is not None:
                raise InputValidationError(
                    InputErrorCode.INVALID_FORMAT,
                    "advisory_wait_seconds",
                    "required requests cannot have an advisory wait",
                )
        else:
            advisory_wait = (
                60
                if self.advisory_wait_seconds is None
                else validate_int(
                    self.advisory_wait_seconds,
                    "advisory_wait_seconds",
                    minimum=1,
                    maximum=3_600,
                )
            )
            object.__setattr__(
                self,
                "advisory_wait_seconds",
                advisory_wait,
            )
        if not isinstance(self.state, RequestState):
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "state",
                "value must be a request state",
            )
        if self.advisory_deadline is not None:
            object.__setattr__(
                self,
                "advisory_deadline",
                validate_aware_datetime(
                    self.advisory_deadline,
                    "advisory_deadline",
                ),
            )
        _validate_request_advisory_timing(self)
        object.__setattr__(
            self,
            "state_revision",
            StateRevision(
                validate_state_revision(self.state_revision, "state_revision")
            ),
        )
        _validate_request_resolution(self)
        validate_total_request_content(_request_content(self))

    @property
    def required(self) -> bool:
        """Return whether the request forbids timeout resolution."""
        return self.mode is RequirementMode.REQUIRED


def create_input_request(
    *,
    request_id: InputRequestId,
    continuation_id: ContinuationId,
    origin: ExecutionOrigin,
    mode: RequirementMode,
    reason: str,
    questions: tuple[InputQuestion, ...],
    created_at: datetime,
    continuation_ttl_seconds: int = 86_400,
    advisory_wait_seconds: int | None = None,
) -> InputRequest:
    """Create an initial request whose class is always task input."""
    return InputRequest(
        request_id=request_id,
        continuation_id=continuation_id,
        origin=origin,
        mode=mode,
        reason=reason,
        questions=questions,
        created_at=created_at,
        continuation_ttl_seconds=continuation_ttl_seconds,
        advisory_wait_seconds=advisory_wait_seconds,
    )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class InputRequiredResult:
    """Report suspension of one transport segment for input."""

    request_id: InputRequestId
    continuation_id: ContinuationId
    detached_resumption_available: bool
    kind: Literal[InputResultKind.INPUT_REQUIRED] = field(
        init=False,
        default=InputResultKind.INPUT_REQUIRED,
    )

    def __post_init__(self) -> None:
        if type(self) is not InputRequiredResult:
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "result",
                "value must be an input-required result",
            )
        object.__setattr__(
            self,
            "request_id",
            InputRequestId(validate_opaque_id(self.request_id, "request_id")),
        )
        object.__setattr__(
            self,
            "continuation_id",
            ContinuationId(
                validate_opaque_id(self.continuation_id, "continuation_id")
            ),
        )
        validate_bool(
            self.detached_resumption_available,
            "detached_resumption_available",
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class _InputModelResultBase:
    """Store one explicit result returned to a pending model request."""

    request_id: InputRequestId
    provenance: AnswerProvenance
    resolved_at: datetime

    def __post_init__(self) -> None:
        if not _is_input_model_result_variant(self):
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "result",
                "value must be a supported input model result variant",
            )
        object.__setattr__(
            self,
            "request_id",
            InputRequestId(validate_opaque_id(self.request_id, "request_id")),
        )
        if not isinstance(self.provenance, AnswerProvenance):
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "provenance",
                "value must be answer provenance",
            )
        object.__setattr__(
            self,
            "resolved_at",
            validate_aware_datetime(self.resolved_at, "resolved_at"),
        )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class InputAnsweredResult(_InputModelResultBase):
    """Return validated typed answers to the pending model request."""

    answers: tuple[InputAnswer, ...]
    kind: Literal[InputResultKind.ANSWERED] = field(
        init=False,
        default=InputResultKind.ANSWERED,
    )

    def __post_init__(self) -> None:
        _InputModelResultBase.__post_init__(self)
        if not isinstance(self.answers, tuple) or not all(
            _is_input_answer_variant(answer) for answer in self.answers
        ):
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "answers",
                "answers must be a tuple of typed answer values",
            )
        question_ids = tuple(answer.question_id for answer in self.answers)
        if len(set(question_ids)) != len(question_ids):
            raise InputValidationError(
                InputErrorCode.DUPLICATE,
                "answers",
                "answer question identifiers must be unique",
            )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class InputDeclinedResult(_InputModelResultBase):
    """Return an explicit declined outcome to the pending model request."""

    kind: Literal[InputResultKind.DECLINED] = field(
        init=False,
        default=InputResultKind.DECLINED,
    )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class InputCancelledResult(_InputModelResultBase):
    """Return an explicit request-cancelled outcome when the run exists."""

    kind: Literal[InputResultKind.CANCELLED] = field(
        init=False,
        default=InputResultKind.CANCELLED,
    )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class InputTimedOutResult(_InputModelResultBase):
    """Return an advisory timeout without manufacturing an answer."""

    kind: Literal[InputResultKind.TIMED_OUT] = field(
        init=False,
        default=InputResultKind.TIMED_OUT,
    )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class InputUnavailableResult(_InputModelResultBase):
    """Return an explicit unavailable outcome when the run exists."""

    kind: Literal[InputResultKind.UNAVAILABLE] = field(
        init=False,
        default=InputResultKind.UNAVAILABLE,
    )


InputModelResult: TypeAlias = (
    InputAnsweredResult
    | InputDeclinedResult
    | InputCancelledResult
    | InputTimedOutResult
    | InputUnavailableResult
)


_INPUT_MODEL_RESULT_VARIANT_TYPES: tuple[type[_InputModelResultBase], ...] = (
    InputAnsweredResult,
    InputDeclinedResult,
    InputCancelledResult,
    InputTimedOutResult,
    InputUnavailableResult,
)


def _is_input_model_result_variant(value: object) -> bool:
    return type(value) in _INPUT_MODEL_RESULT_VARIANT_TYPES


@dataclass(frozen=True, slots=True, kw_only=True)
class _InputContinuationOutcomeBase:
    """Discriminate model resumption from logical-run termination."""

    request_id: InputRequestId

    def __post_init__(self) -> None:
        if not _is_input_continuation_outcome_variant(self):
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "outcome",
                "value must be a supported continuation outcome variant",
            )
        object.__setattr__(
            self,
            "request_id",
            InputRequestId(validate_opaque_id(self.request_id, "request_id")),
        )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class ResumeInputContinuation(_InputContinuationOutcomeBase):
    """Resume the same logical execution with one explicit model result."""

    result: InputModelResult
    disposition: Literal[ContinuationDisposition.RESUME] = field(
        init=False,
        default=ContinuationDisposition.RESUME,
    )

    def __post_init__(self) -> None:
        _InputContinuationOutcomeBase.__post_init__(self)
        if not _is_input_model_result_variant(self.result):
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "result",
                "value must be an input model result",
            )
        if self.request_id != self.result.request_id:
            raise InputValidationError(
                InputErrorCode.CORRELATION_MISMATCH,
                "request_id",
                "result does not match the continuation request",
            )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class TerminateInputContinuation(_InputContinuationOutcomeBase):
    """Terminate a stale, expired, or cancelled logical execution."""

    status: ResolutionStatus
    disposition: Literal[ContinuationDisposition.TERMINATE] = field(
        init=False,
        default=ContinuationDisposition.TERMINATE,
    )

    def __post_init__(self) -> None:
        _InputContinuationOutcomeBase.__post_init__(self)
        if not isinstance(self.status, ResolutionStatus):
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "status",
                "value must be a resolution status",
            )


InputContinuationOutcome: TypeAlias = (
    ResumeInputContinuation | TerminateInputContinuation
)


_INPUT_CONTINUATION_OUTCOME_VARIANT_TYPES: tuple[
    type[_InputContinuationOutcomeBase], ...
] = (
    ResumeInputContinuation,
    TerminateInputContinuation,
)


def _is_input_continuation_outcome_variant(value: object) -> bool:
    return type(value) in _INPUT_CONTINUATION_OUTCOME_VARIANT_TYPES


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class InteractionSnapshot:
    """Store one versioned JSON-safe interaction snapshot."""

    request: InputRequest
    version: int = 1

    def __post_init__(self) -> None:
        if self.version != 1 or isinstance(self.version, bool):
            raise InputValidationError(
                InputErrorCode.SNAPSHOT_UNSUPPORTED,
                "snapshot.version",
                "snapshot version is unsupported",
            )
        if type(self.request) is not InputRequest:
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "snapshot.request",
                "value must be an input request",
            )


def _validate_text_constraints(
    constraints: TextValidationConstraints,
    maximum: int,
) -> None:
    if not isinstance(constraints, TextValidationConstraints):
        raise InputValidationError(
            InputErrorCode.INVALID_TYPE,
            "question.constraints",
            "value must be text constraints",
        )
    if constraints.maximum_length > maximum:
        raise InputValidationError(
            InputErrorCode.OUT_OF_BOUNDS,
            "question.constraints.maximum_length",
            "constraint exceeds the semantic type limit",
        )


def _validate_text_cardinality(
    value: str,
    constraints: TextValidationConstraints,
    required: bool,
    path: str,
    code: InputErrorCode,
) -> None:
    minimum = max(constraints.minimum_length, int(required))
    if len(value) < minimum or len(value) > constraints.maximum_length:
        raise InputValidationError(
            code,
            path,
            "text length violates its question constraints",
        )


def _validate_choices(value: object) -> tuple[Choice, ...]:
    if not isinstance(value, tuple):
        raise InputValidationError(
            InputErrorCode.INVALID_TYPE,
            "question.choices",
            "choices must be a tuple",
        )
    if len(value) < 2 or len(value) > 20:
        raise InputValidationError(
            InputErrorCode.OUT_OF_BOUNDS,
            "question.choices",
            "selection questions require two to twenty choices",
        )
    if not all(isinstance(choice, Choice) for choice in value):
        raise InputValidationError(
            InputErrorCode.INVALID_TYPE,
            "question.choices",
            "choices must contain typed choice values",
        )
    choices = value
    seen: set[ChoiceValue] = set()
    for choice in choices:
        if choice.value in seen:
            raise InputValidationError(
                InputErrorCode.DUPLICATE,
                "question.choices",
                "choice values must be unique",
            )
        seen.add(choice.value)
    return choices


def _validate_choice_reference(
    value: object,
    choices: tuple[Choice, ...],
    path: str,
    code: InputErrorCode,
) -> ChoiceValue:
    normalized = ChoiceValue(validate_choice_value(value, path))
    if normalized not in {choice.value for choice in choices}:
        raise InputValidationError(
            code,
            path,
            "value does not reference an available choice",
        )
    return normalized


def _validate_choice_references(
    value: object,
    choices: tuple[Choice, ...],
    path: str,
    code: InputErrorCode,
) -> tuple[ChoiceValue, ...]:
    if not isinstance(value, tuple):
        raise InputValidationError(
            InputErrorCode.INVALID_TYPE,
            path,
            "value must be a tuple",
        )
    normalized = tuple(
        _validate_choice_reference(item, choices, path, code) for item in value
    )
    if len(set(normalized)) != len(normalized):
        raise InputValidationError(
            InputErrorCode.DUPLICATE,
            path,
            "choice values must be unique",
        )
    return normalized


def _validate_request_resolution(request: InputRequest) -> None:
    terminal = request.state not in {
        RequestState.CREATED,
        RequestState.PENDING,
    }
    if terminal != (request.resolution is not None):
        raise InputValidationError(
            InputErrorCode.ILLEGAL_TRANSITION,
            "resolution",
            "request state and resolution do not agree",
        )
    expected_revision = {
        RequestState.CREATED: 0,
        RequestState.PENDING: 1,
    }.get(request.state, 2)
    if request.state_revision != expected_revision:
        raise InputValidationError(
            InputErrorCode.ILLEGAL_TRANSITION,
            "state_revision",
            "request state and revision do not agree",
        )
    if request.resolution is None:
        return
    _validate_resolution_against_request(request, request.resolution)
    if request.state.value != request.resolution.status.value:
        raise InputValidationError(
            InputErrorCode.ILLEGAL_TRANSITION,
            "resolution.status",
            "resolution status does not match request state",
        )


def _validate_resolution_against_request(
    request: InputRequest,
    resolution: InputResolution,
) -> None:
    if not _is_input_resolution_variant(resolution):
        raise InputValidationError(
            InputErrorCode.INVALID_TYPE,
            "resolution",
            "value must be a supported input resolution variant",
        )
    if resolution.request_id != request.request_id:
        raise InputValidationError(
            InputErrorCode.CORRELATION_MISMATCH,
            "resolution.request_id",
            "resolution does not match the pending request",
        )
    if resolution.resolved_at < request.created_at:
        raise InputValidationError(
            InputErrorCode.INVALID_FORMAT,
            "resolution.resolved_at",
            "resolution predates request creation",
        )
    if (
        type(resolution) is TimedOutResolution
        and request.mode is RequirementMode.REQUIRED
    ):
        raise InputValidationError(
            InputErrorCode.TIMED_OUT_REQUIRED,
            "resolution.status",
            "required requests cannot time out",
        )
    if type(resolution) is TimedOutResolution:
        if request.advisory_deadline is None:
            raise InputValidationError(
                InputErrorCode.INVALID_FORMAT,
                "advisory_deadline",
                "advisory timeout requires a stored deadline",
            )
        if resolution.resolved_at < request.advisory_deadline:
            raise InputValidationError(
                InputErrorCode.INVALID_FORMAT,
                "resolution.resolved_at",
                "timeout resolution predates its stored advisory deadline",
            )
    if type(resolution) is AnsweredResolution:
        _validate_answers_against_request(request, resolution.answers)


def _validate_request_advisory_timing(request: InputRequest) -> None:
    deadline = request.advisory_deadline
    if request.mode is RequirementMode.REQUIRED:
        if deadline is not None:
            raise InputValidationError(
                InputErrorCode.INVALID_FORMAT,
                "advisory_deadline",
                "required requests cannot store an advisory deadline",
            )
        return
    if request.state is RequestState.CREATED:
        if deadline is not None:
            raise InputValidationError(
                InputErrorCode.ILLEGAL_TRANSITION,
                "advisory_deadline",
                "created advisory requests cannot store a deadline",
            )
        return
    if deadline is None:
        raise InputValidationError(
            InputErrorCode.ILLEGAL_TRANSITION,
            "advisory_deadline",
            "pending and terminal advisory requests require a deadline",
        )
    assert request.advisory_wait_seconds is not None
    if (
        deadline - request.created_at
    ).total_seconds() < request.advisory_wait_seconds:
        raise InputValidationError(
            InputErrorCode.INVALID_FORMAT,
            "advisory_deadline",
            "advisory deadline does not include the declared waiting budget",
        )


def _validate_answers_against_request(
    request: InputRequest,
    answers: tuple[InputAnswer, ...],
) -> None:
    if not all(
        _is_input_question_variant(question) for question in request.questions
    ):
        raise InputValidationError(
            InputErrorCode.INVALID_TYPE,
            "questions",
            "questions must contain supported input question variants",
        )
    questions = {
        question.question_id: question for question in request.questions
    }
    answered_ids: set[object] = set()
    for answer in answers:
        if not _is_input_answer_variant(answer):
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "resolution.answers",
                "answers must contain supported input answer variants",
            )
        question = questions.get(answer.question_id)
        if question is None:
            raise InputValidationError(
                InputErrorCode.UNKNOWN_QUESTION,
                "resolution.answers",
                "answer references an unknown question",
            )
        answered_ids.add(answer.question_id)
        if type(question) is ConfirmationQuestion:
            if type(answer) is not ConfirmationAnswer:
                _raise_answer_type()
        elif type(question) is TextQuestion:
            if type(answer) is not TextAnswer:
                _raise_answer_type()
            _validate_text_answer(question, answer)
        elif type(question) is MultilineTextQuestion:
            if type(answer) is not MultilineTextAnswer:
                _raise_answer_type()
            _validate_text_answer(question, answer)
        elif type(question) is SingleSelectionQuestion:
            if type(answer) is not SingleSelectionAnswer:
                _raise_answer_type()
            _validate_selection_value(question, answer.value)
        else:
            assert type(question) is MultipleSelectionQuestion
            if type(answer) is not MultipleSelectionAnswer:
                _raise_answer_type()
            minimum = max(question.constraints.minimum, int(question.required))
            if (
                len(answer.values) < minimum
                or len(answer.values) > question.constraints.maximum
            ):
                raise InputValidationError(
                    InputErrorCode.INVALID_CARDINALITY,
                    "answer.values",
                    "answer has invalid selection cardinality",
                )
            for value in answer.values:
                _validate_selection_value(question, value)
        _validate_trusted_default(question, answer)
    for question in request.questions:
        if question.required and question.question_id not in answered_ids:
            raise InputValidationError(
                InputErrorCode.MISSING_REQUIRED_ANSWER,
                "resolution.answers",
                "a required answer is missing",
            )


def _validate_text_answer(
    question: TextQuestion | MultilineTextQuestion,
    answer: TextAnswer | MultilineTextAnswer,
) -> None:
    minimum = max(question.constraints.minimum_length, int(question.required))
    if (
        len(answer.value) < minimum
        or len(answer.value) > question.constraints.maximum_length
    ):
        raise InputValidationError(
            InputErrorCode.INVALID_CARDINALITY,
            "answer.value",
            "answer violates its text length constraints",
        )


def _validate_selection_value(
    question: SingleSelectionQuestion | MultipleSelectionQuestion,
    value: SelectionValue,
) -> None:
    if type(value) is SelectedChoice:
        if value.value not in {choice.value for choice in question.choices}:
            raise InputValidationError(
                InputErrorCode.UNKNOWN_CHOICE,
                "answer.value",
                "answer references an unknown choice",
            )
        return
    if type(value) is not FreeFormOther:
        raise InputValidationError(
            InputErrorCode.INVALID_TYPE,
            "answer.value",
            "value must be a tagged selection value",
        )
    if not question.allow_other:
        raise InputValidationError(
            InputErrorCode.OTHER_NOT_ALLOWED,
            "answer.value",
            "question does not permit a free-form alternative",
        )


def _validate_trusted_default(
    question: object,
    answer: InputAnswer,
) -> None:
    if not _is_input_question_variant(question):
        raise InputValidationError(
            InputErrorCode.INVALID_TYPE,
            "question",
            "value must be a supported input question variant",
        )
    if not _is_input_answer_variant(answer):
        raise InputValidationError(
            InputErrorCode.INVALID_TYPE,
            "answer",
            "value must be a supported input answer variant",
        )
    if answer.provenance is not AnswerProvenance.TRUSTED_DEFAULT:
        return
    if (
        type(question) is ConfirmationQuestion
        and type(answer) is ConfirmationAnswer
    ):
        matches = question.default_value == answer.value
    elif type(question) is TextQuestion and type(answer) is TextAnswer:
        matches = question.default_value == answer.value
    elif (
        type(question) is MultilineTextQuestion
        and type(answer) is MultilineTextAnswer
    ):
        matches = question.default_value == answer.value
    elif (
        type(question) is SingleSelectionQuestion
        and type(answer) is SingleSelectionAnswer
    ):
        matches = (
            type(answer.value) is SelectedChoice
            and question.default_value == answer.value.value
        )
    elif (
        type(question) is MultipleSelectionQuestion
        and type(answer) is MultipleSelectionAnswer
    ):
        selected = tuple(
            value.value
            for value in answer.values
            if type(value) is SelectedChoice
        )
        matches = (
            len(selected) == len(answer.values)
            and question.default_value == selected
        )
    else:
        matches = False
    if not matches:
        raise InputValidationError(
            InputErrorCode.INVALID_DEFAULT,
            "answer.provenance",
            "trusted-default provenance requires the declared default",
        )


def _raise_answer_type() -> NoReturn:
    raise InputValidationError(
        InputErrorCode.ANSWER_TYPE_MISMATCH,
        "answer",
        "answer type does not match its question",
    )


def _request_content(request: InputRequest) -> tuple[str, ...]:
    questions: list[dict[str, object]] = []
    for question in request.questions:
        item: dict[str, object] = {
            "question_id": str(question.question_id),
            "kind": question.kind.value,
            "prompt": question.prompt,
            "required": question.required,
        }
        if question.header is not None:
            item["header"] = question.header
        if question.help_text is not None:
            item["help"] = question.help_text
        if question.presentation_hint is not None:
            item["presentation_hint"] = question.presentation_hint.value
        if type(question) is ConfirmationQuestion:
            if question.default_value is not None:
                item["default_value"] = question.default_value
        elif type(question) is TextQuestion:
            _add_text_question_content(item, question)
        elif type(question) is MultilineTextQuestion:
            _add_text_question_content(item, question)
        elif type(question) is SingleSelectionQuestion:
            _add_selection_question_content(item, question)
            if question.default_value is not None:
                item["default_value"] = str(question.default_value)
        else:
            assert type(question) is MultipleSelectionQuestion
            _add_selection_question_content(item, question)
            if question.default_value is not None:
                item["default_value"] = [
                    str(value) for value in question.default_value
                ]
            item["constraints"] = {
                "minimum": question.constraints.minimum,
                "maximum": question.constraints.maximum,
            }
        questions.append(item)
    payload = {
        "interaction_class": request.interaction_class.value,
        "mode": request.mode.value,
        "reason": request.reason,
        "questions": questions,
    }
    return (
        dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ),
    )


def _add_text_question_content(
    item: dict[str, object],
    question: TextQuestion | MultilineTextQuestion,
) -> None:
    if question.default_value is not None:
        item["default_value"] = question.default_value
    item["constraints"] = {
        "minimum_length": question.constraints.minimum_length,
        "maximum_length": question.constraints.maximum_length,
    }


def _add_selection_question_content(
    item: dict[str, object],
    question: SingleSelectionQuestion | MultipleSelectionQuestion,
) -> None:
    item["choices"] = [
        {
            "value": str(choice.value),
            "label": choice.label,
            **(
                {"description": choice.description}
                if choice.description is not None
                else {}
            ),
        }
        for choice in question.choices
    ]
    item["allow_other"] = question.allow_other
    if question.recommended_choice is not None:
        item["recommended_choice"] = str(question.recommended_choice)


def _freeze_snapshot_object(
    value: Mapping[str, object],
    path: str,
) -> Mapping[str, JsonValue]:
    frozen: dict[str, JsonValue] = {}
    for key, item in value.items():
        normalized_key = validate_opaque_id(
            key,
            f"{path}.key",
            maximum_characters=256,
            maximum_bytes=1_024,
        )
        if _is_prohibited_snapshot_key(
            normalized_key
        ) and not _is_safe_snapshot_token_metric(normalized_key, item):
            raise InputValidationError(
                InputErrorCode.SNAPSHOT_SECRET_PROHIBITED,
                f"{path}.key",
                "snapshot payload contains prohibited credential material",
            )
        if normalized_key in frozen:
            raise InputValidationError(
                InputErrorCode.SNAPSHOT_INVALID,
                f"{path}.key",
                "snapshot payload contains duplicate normalized keys",
            )
        frozen[normalized_key] = _freeze_snapshot_value(
            item,
            f"{path}.value",
        )
    return MappingProxyType(frozen)


def _is_prohibited_snapshot_key(value: str) -> bool:
    parts = _snapshot_key_tokens(value)
    compact_key = "".join(parts)
    while parts and parts[-1] in _SNAPSHOT_CREDENTIAL_PAYLOAD_QUALIFIERS:
        parts = parts[:-1]
    if any(
        parts[-len(suffix) :] == suffix
        for suffix in _PROHIBITED_SNAPSHOT_KEY_SUFFIXES
    ):
        return True
    compact_candidates = (compact_key,) + tuple(
        compact_key[: -len(qualifier)]
        for qualifier in _SNAPSHOT_CREDENTIAL_PAYLOAD_QUALIFIERS
        if compact_key.endswith(qualifier)
    )
    return any(
        candidate.endswith(suffix)
        for candidate in compact_candidates
        for suffix in _PROHIBITED_SNAPSHOT_COMPACT_SUFFIXES
    )


def _is_safe_snapshot_token_metric(key: str, value: object) -> bool:
    normalized_key = normalize("NFKC", key).casefold()
    if normalized_key not in _SAFE_SNAPSHOT_TOKEN_METRIC_KEYS:
        return False
    if isinstance(value, bool) or not isinstance(value, int | float):
        return False
    return isfinite(value) and value >= 0


def _snapshot_key_tokens(value: str) -> tuple[str, ...]:
    normalized_value = normalize("NFKC", value)
    tokens: list[str] = []
    current: list[str] = []
    for index, character in enumerate(normalized_value):
        if not character.isalnum():
            if current:
                tokens.append("".join(current).casefold())
                current = []
            continue
        if current and _is_snapshot_key_camel_boundary(
            normalized_value,
            index,
        ):
            tokens.append("".join(current).casefold())
            current = []
        current.append(character)
    if current:
        tokens.append("".join(current).casefold())
    return tuple(tokens)


def _is_snapshot_key_camel_boundary(value: str, index: int) -> bool:
    character = value[index]
    if not character.isupper():
        return False
    previous = value[index - 1]
    if previous.islower() or previous.isdigit():
        return True
    return (
        previous.isupper()
        and index + 1 < len(value)
        and value[index + 1].islower()
    )


def _freeze_snapshot_value(value: object, path: str) -> JsonValue:
    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float):
        if not isfinite(value):
            raise InputValidationError(
                InputErrorCode.SNAPSHOT_INVALID,
                path,
                "snapshot number must be finite",
            )
        return value
    if isinstance(value, Mapping):
        return _freeze_snapshot_object(value, path)
    if isinstance(value, tuple | list):
        return tuple(
            _freeze_snapshot_value(item, f"{path}.item") for item in value
        )
    raise InputValidationError(
        InputErrorCode.SNAPSHOT_INVALID,
        path,
        "snapshot payload contains a non-JSON value",
    )

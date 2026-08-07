"""Run authorized, redacted native Responses live conformance probes."""

from argparse import ArgumentParser, Namespace
from asyncio import CancelledError, run
from collections.abc import Awaitable, Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from enum import StrEnum
from hashlib import sha256
from importlib import import_module
from inspect import signature
from json import dumps, loads
from os import environ
from pathlib import Path
from re import fullmatch
from typing import Protocol, cast, final
from urllib.parse import urlsplit

from httpx import Request
from openai import AsyncOpenAI, DefaultAsyncHttpxClient, Omit, omit
from openai import __version__ as openai_version
from openai.resources.responses.responses import AsyncResponses
from openai.types.responses import (
    CompactedResponse,
    FunctionToolParam,
    Response,
    ResponseCompactionItem,
    ResponseFunctionToolCall,
    ResponseIncludable,
    ResponseInputItemParam,
    ResponseOutputMessage,
    ResponseReasoningItem,
    ResponseStreamEvent,
)
from openai.types.responses.parsed_response import (
    ParsedResponseFunctionToolCall,
)
from openai.types.responses.response_create_params import ContextManagement
from openai.types.responses.response_output_item import ResponseOutputItem
from openai.types.shared_params import Reasoning
from packaging.specifiers import SpecifierSet
from packaging.version import Version

from avalan.conversation.activation import ProviderApiForm
from avalan.conversation.binding import ProviderFamily
from avalan.conversation.errors import ConversationValidationError
from avalan.conversation.providers.openai import (
    NativeOpenAIEncryptedContentPolicy,
    _reject_json_constant,
    _replay_item_to_input_item,
    _unique_json_object,
)
from avalan.conversation.value import (
    canonical_json_bytes,
    freeze_json_value,
    validate_identifier,
)

_FIXTURE_DIRECTORY = Path("tests/fixtures/conversation")
_PROVIDER_EVIDENCE = _FIXTURE_DIRECTORY / "provider_evidence.phase12.json"
_ACTIVATION_MANIFEST = (
    _FIXTURE_DIRECTORY / "activation_preflight_manifest.phase12.json"
)
_AUTHORITY_TOKEN = "authorize-phase12-live-conformance"
_COST_TOKEN = "accept-phase12-provider-costs"
_MAX_EVIDENCE_AGE_DAYS = 30
_OPENAI_ENDPOINT = "https://api.openai.com/v1"
_OPENAI_API_REVISION = "openapi-2.3.0"
_ENCRYPTED_CONTENT_INCLUDE: ResponseIncludable = "reasoning.encrypted_content"
_INLINE_COMPACTION_MAX_OUTPUT_TOKENS = 512
_MAX_CLEANUP_REFERENCES = 8
_MAX_CLEANUP_ATTEMPTS = 3
_SDK_MAX_RETRIES = 0
_PROBE_TOOL_NAME = "phase12_probe"
_PROBE_CURRENT_VALUE = "1943"
_PROBE_STREAM_VALUE = "2505"
_PROBE_TOOL_OUTPUT = '{"accepted":true}'
_AZURE_ENDPOINT_PATTERN = (
    r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.openai\.azure\.com"
)
_MODEL_PATTERN = r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}"
_PLACEHOLDER_SECRETS = frozenset(
    {
        "changeme",
        "placeholder",
        "test",
        "your-api-key",
        "your_api_key",
    }
)
_OPENAI_DOCUMENTATION_URLS = (
    "https://developers.openai.com/api/docs/guides/compaction",
    "https://developers.openai.com/api/docs/guides/conversation-state",
    (
        "https://developers.openai.com/api/docs/guides/"
        "reasoning#preserve-reasoning-across-calls"
    ),
    (
        "https://developers.openai.com/api/reference/resources/"
        "responses/methods/compact"
    ),
    (
        "https://developers.openai.com/api/reference/resources/"
        "responses/methods/create"
    ),
    (
        "https://developers.openai.com/api/reference/resources/"
        "responses/methods/delete"
    ),
    (
        "https://developers.openai.com/api/reference/resources/"
        "responses/methods/retrieve"
    ),
)
_AZURE_DOCUMENTATION_URLS = (
    "https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/responses",
    (
        "https://learn.microsoft.com/en-us/rest/api/"
        "microsoft-foundry/azureopenai/responses"
    ),
)
_TYPED_ASYNC_METHODS = (
    "responses.compact",
    "responses.create",
    "responses.delete",
    "responses.retrieve",
    "responses.stream",
)
_TYPED_CREATE_FIELDS = (
    "context_management",
    "include",
    "input",
    "model",
    "previous_response_id",
    "reasoning",
    "store",
    "stream",
    "tools",
)
_TYPED_SYMBOLS = (
    ("openai.types.responses.CompactedResponse", CompactedResponse),
    ("openai.types.responses.Response", Response),
    (
        "openai.types.responses.ResponseCompactionItem",
        ResponseCompactionItem,
    ),
    (
        "openai.types.responses.ResponseFunctionToolCall",
        ResponseFunctionToolCall,
    ),
    (
        "openai.types.responses.ResponseInputItemParam",
        ResponseInputItemParam,
    ),
    (
        "openai.types.responses.ResponseOutputMessage",
        ResponseOutputMessage,
    ),
    (
        "openai.types.responses.ResponseReasoningItem",
        ResponseReasoningItem,
    ),
    (
        "openai.types.responses.ResponseStreamEvent",
        ResponseStreamEvent,
    ),
    (
        "openai.types.responses.response_create_params.ContextManagement",
        ContextManagement,
    ),
    ("openai.types.shared_params.Reasoning", Reasoning),
)


class LiveConformancePreflightError(RuntimeError):
    """Report a fail-closed condition before live provider dispatch."""


class LiveConformanceAssertionError(RuntimeError):
    """Report a provider observation that violates the required contract."""


class LiveConformanceExecutionError(RuntimeError):
    """Report a content-safe typed live SDK execution failure."""


class LiveConformanceCleanupError(RuntimeError):
    """Report a content-safe typed live cleanup failure."""


class LiveConformanceAccountingError(RuntimeError):
    """Report content-safe live transport accounting drift."""


class LiveProviderFamily(StrEnum):
    """Identify one exact native provider family allowed by the harness."""

    OPENAI = "openai"
    AZURE_OPENAI = "azure_openai"


class LiveConformanceCase(StrEnum):
    """Identify one required live native Responses behavior."""

    INLINE_COMPACTION = "inline_compaction"
    STANDALONE_COMPACTION = "standalone_compaction_and_unpruned_replay"
    STATELESS_ALL_TURNS = "stateless_all_turns_replay"
    STATELESS_CURRENT_TURN_TOOL = "stateless_current_turn_tool"
    STORED_CREATE = "stored_create"
    STORED_CHAIN = "stored_previous_response_chain"
    STORED_RETRIEVE_DELETE = "stored_retrieve_delete"
    STREAMING_TOOL = "streaming_tool"


REQUIRED_LIVE_CASES = tuple(LiveConformanceCase)
_EXECUTION_ORDER = (
    LiveConformanceCase.STATELESS_CURRENT_TURN_TOOL,
    LiveConformanceCase.STATELESS_ALL_TURNS,
    LiveConformanceCase.INLINE_COMPACTION,
    LiveConformanceCase.STANDALONE_COMPACTION,
    LiveConformanceCase.STORED_CREATE,
    LiveConformanceCase.STORED_CHAIN,
    LiveConformanceCase.STORED_RETRIEVE_DELETE,
    LiveConformanceCase.STREAMING_TOOL,
)


class _LiveOperationClass(StrEnum):
    """Identify one content-free live transport operation class."""

    CREATE_OR_STREAM = "create_or_stream"
    COMPACT = "compact"
    RETRIEVE = "retrieve"
    DELETE = "delete"


_EXPECTED_OPERATION_COUNTS = {
    _LiveOperationClass.CREATE_OR_STREAM: 7,
    _LiveOperationClass.COMPACT: 1,
    _LiveOperationClass.RETRIEVE: 1,
    _LiveOperationClass.DELETE: 2,
}


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class LiveConformanceConfig:
    """Hold explicit authority and exact native-provider identity."""

    provider_family: LiveProviderFamily
    endpoint: str
    api_form: str
    provider_api_revision: str
    model_or_deployment: str
    model_or_deployment_revision: str
    api_key: str = field(repr=False, compare=False)
    command_authority: bool
    environment_authority: str = field(repr=False, compare=False)
    command_cost_acknowledgement: bool
    environment_cost_acknowledgement: str = field(
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if self.provider_family not in {
            LiveProviderFamily.OPENAI,
            LiveProviderFamily.AZURE_OPENAI,
        }:
            raise LiveConformancePreflightError(
                "only exact native OpenAI provider families are allowed"
            )
        if (
            self.command_authority is not True
            or self.environment_authority != _AUTHORITY_TOKEN
        ):
            raise LiveConformancePreflightError(
                "explicit command and environment authority are required"
            )
        if (
            self.command_cost_acknowledgement is not True
            or self.environment_cost_acknowledgement != _COST_TOKEN
        ):
            raise LiveConformancePreflightError(
                "explicit command and environment cost acknowledgement "
                "are required"
            )
        secret = self.api_key.strip()
        if (
            not secret
            or secret != self.api_key
            or secret.casefold() in _PLACEHOLDER_SECRETS
        ):
            raise LiveConformancePreflightError(
                "an exact non-placeholder provider credential is required"
            )
        for value, label in (
            (self.model_or_deployment, "model or deployment"),
            (self.model_or_deployment_revision, "model revision"),
        ):
            if fullmatch(_MODEL_PATTERN, value) is None:
                raise LiveConformancePreflightError(
                    f"an exact {label} identifier is required"
                )
        if self.provider_family is LiveProviderFamily.OPENAI:
            if (
                self.endpoint != _OPENAI_ENDPOINT
                or self.api_form != ProviderApiForm.OPENAI_RESPONSES_V1.value
                or self.provider_api_revision != _OPENAI_API_REVISION
            ):
                raise LiveConformancePreflightError(
                    "native OpenAI endpoint and API identity are not exact"
                )
        else:
            _validate_azure_endpoint(self.endpoint)
            if (
                self.api_form
                not in {
                    ProviderApiForm.AZURE_OPENAI_V1.value,
                    ProviderApiForm.AZURE_OPENAI_V1_PREVIEW.value,
                }
                or self.provider_api_revision != self.api_form
            ):
                raise LiveConformancePreflightError(
                    "native Azure OpenAI API identity is not exact"
                )


def _encrypted_content_policy(
    provider_family: LiveProviderFamily,
) -> NativeOpenAIEncryptedContentPolicy:
    """Return the exact production-aligned encrypted-content policy."""
    if provider_family is LiveProviderFamily.OPENAI:
        return NativeOpenAIEncryptedContentPolicy.DEFAULT_RETURN
    if provider_family is LiveProviderFamily.AZURE_OPENAI:
        return NativeOpenAIEncryptedContentPolicy.EXPLICIT_INCLUDE
    raise LiveConformancePreflightError(
        "encrypted-content policy requires an exact native provider"
    )


def _encrypted_reasoning_include(
    config: LiveConformanceConfig,
) -> list[ResponseIncludable] | Omit:
    """Return the typed provider-specific encrypted-reasoning include."""
    if (
        _encrypted_content_policy(config.provider_family)
        is NativeOpenAIEncryptedContentPolicy.EXPLICIT_INCLUDE
    ):
        return [_ENCRYPTED_CONTENT_INCLUDE]
    return omit


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class LiveConformanceObservation:
    """Retain only redacted structural facts from one live operation."""

    case: LiveConformanceCase
    response_status: str
    parent_matches: bool
    reasoning_context: str | None
    item_kinds: tuple[str, ...]
    reported_model_identity: str = ""
    required_reasoning_items: int = 0
    encrypted_reasoning_items: int = 0
    compaction_items: int = 0
    opaque_compaction_items: int = 0
    tool_calls: int = 0
    required_tool_name_matches: bool = False
    required_tool_arguments_match: bool = False
    tool_correlation_matches: bool = False
    encrypted_reasoning_precedes_tool: bool = False
    preceding_reasoning_replayed: bool = False
    stream_event_kinds: tuple[str, ...] = ()
    complete_output_replayed: bool = False
    replayed_encrypted_reasoning_items: int = 0
    replayed_tool_outputs: int = 0
    compact_output_replayed: bool = False
    retrieved: bool = False
    deleted: bool = False


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class _RequiredToolEvidence:
    """Retain content-free facts about one required tool transition."""

    required_reasoning_items: int
    encrypted_reasoning_items: int
    tool_calls: int
    required_tool_name_matches: bool
    required_tool_arguments_match: bool
    tool_correlation_matches: bool
    encrypted_reasoning_precedes_tool: bool


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class LiveConformanceAccounting:
    """Retain only content-free final transport accounting."""

    create_or_stream_logical_operation_count: int
    compact_logical_operation_count: int
    retrieve_logical_operation_count: int
    delete_logical_operation_count: int
    create_or_stream_http_request_count: int
    compact_http_request_count: int
    retrieve_http_request_count: int
    delete_http_request_count: int
    unexpected_http_request_count: int
    sdk_configured_max_retries: int
    observed_sdk_retry_count: int
    request_path_class_mismatch_count: int
    cleanup_attempted: bool
    cleanup_completed: bool
    cleanup_delete_logical_operation_count: int
    cleanup_delete_http_request_count: int
    cleanup_pending_reference_count: int
    client_close_completed: bool

    @property
    def logical_operation_count(self) -> int:
        """Return the total number of SDK-level logical operations."""
        return sum(self._logical_operation_counts().values())

    @property
    def http_request_count(self) -> int:
        """Return the total number of observed HTTP requests."""
        return sum(self._http_request_counts().values())

    def redacted_payload(self) -> dict[str, object]:
        """Return content-free accounting without request material."""
        return {
            "logical_operation_count": self.logical_operation_count,
            "logical_operation_counts": self._logical_operation_counts(),
            "http_request_count": self.http_request_count,
            "http_request_counts": self._http_request_counts(),
            "sdk_configured_max_retries": self.sdk_configured_max_retries,
            "observed_sdk_retry_count": self.observed_sdk_retry_count,
            "unexpected_request_count": self.unexpected_http_request_count,
            "request_path_class_mismatch_count": (
                self.request_path_class_mismatch_count
            ),
            "cleanup_attempted": self.cleanup_attempted,
            "cleanup_completed": self.cleanup_completed,
            "cleanup_delete_logical_operation_count": (
                self.cleanup_delete_logical_operation_count
            ),
            "cleanup_delete_http_request_count": (
                self.cleanup_delete_http_request_count
            ),
            "cleanup_pending_reference_count": (
                self.cleanup_pending_reference_count
            ),
            "client_close_completed": self.client_close_completed,
            "successful_matrix_expected_counts_match": (
                _accounting_matches_expected_matrix(self)
            ),
        }

    def _logical_operation_counts(self) -> dict[str, int]:
        """Return logical counts keyed only by fixed operation classes."""
        return {
            "compact": self.compact_logical_operation_count,
            "create_or_stream": (
                self.create_or_stream_logical_operation_count
            ),
            "delete": self.delete_logical_operation_count,
            "retrieve": self.retrieve_logical_operation_count,
        }

    def _http_request_counts(self) -> dict[str, int]:
        """Return HTTP counts keyed only by fixed request classes."""
        return {
            "compact": self.compact_http_request_count,
            "create_or_stream": self.create_or_stream_http_request_count,
            "delete": self.delete_http_request_count,
            "retrieve": self.retrieve_http_request_count,
            "unexpected": self.unexpected_http_request_count,
        }


@final
class _TransportAccounting:
    """Generate content-free accounting from live transport activity."""

    def __init__(self) -> None:
        self._logical_counts = {
            operation: 0 for operation in _LiveOperationClass
        }
        self._http_counts = {operation: 0 for operation in _LiveOperationClass}
        self._unexpected_http_request_count = 0
        self._sdk_configured_max_retries = -1
        self._observed_sdk_retry_count = 0
        self._request_path_class_mismatch_count = 0
        self._active_operation: _LiveOperationClass | None = None
        self._active_http_request_count = 0
        self._cleanup_attempted = False
        self._cleanup_completed = False
        self._cleanup_active = False
        self._cleanup_delete_logical_operation_count = 0
        self._cleanup_delete_http_request_count = 0
        self._cleanup_pending_reference_count = 0
        self._client_close_completed = False

    def set_sdk_configured_max_retries(self, value: object) -> None:
        """Record the SDK client's actual configured retry limit."""
        if type(value) is not int or value < 0:
            raise LiveConformanceAccountingError(
                "live transport SDK retry configuration is invalid"
            )
        self._sdk_configured_max_retries = value

    @contextmanager
    def logical_operation(
        self,
        operation: _LiveOperationClass,
    ) -> Iterator[None]:
        """Count one SDK operation while correlating its HTTP requests."""
        if self._active_operation is not None:
            raise LiveConformanceAccountingError(
                "live transport logical operations overlap"
            )
        self._logical_counts[operation] += 1
        if self._cleanup_active and operation is _LiveOperationClass.DELETE:
            self._cleanup_delete_logical_operation_count += 1
        self._active_operation = operation
        self._active_http_request_count = 0
        try:
            yield
        finally:
            self._active_operation = None
            self._active_http_request_count = 0

    async def record_http_request(self, request: Request) -> None:
        """Count only method-derived path classes from one HTTP request."""
        operation = _classify_http_request(request.method, request.url.path)
        if operation is None:
            self._unexpected_http_request_count += 1
        else:
            self._http_counts[operation] += 1
            if (
                self._cleanup_active
                and operation is _LiveOperationClass.DELETE
            ):
                self._cleanup_delete_http_request_count += 1
        self._active_http_request_count += 1
        if self._active_http_request_count > 1:
            self._observed_sdk_retry_count += 1
        if operation is None or operation is not self._active_operation:
            self._request_path_class_mismatch_count += 1

    def begin_cleanup(self) -> None:
        """Mark the start of the one protected cleanup phase."""
        if self._cleanup_attempted or self._active_operation is not None:
            raise LiveConformanceAccountingError(
                "live transport cleanup accounting is invalid"
            )
        self._cleanup_attempted = True
        self._cleanup_active = True

    def finish_cleanup(
        self,
        *,
        completed: bool,
        pending_reference_count: int,
        client_close_completed: bool,
    ) -> None:
        """Record final cleanup state without retaining any identifier."""
        if (
            type(pending_reference_count) is not int
            or pending_reference_count < 0
        ):
            raise LiveConformanceAccountingError(
                "live transport cleanup count is invalid"
            )
        self._cleanup_active = False
        self._cleanup_pending_reference_count = pending_reference_count
        self._client_close_completed = client_close_completed
        self._cleanup_completed = (
            completed
            and pending_reference_count == 0
            and client_close_completed
        )

    def snapshot(self) -> LiveConformanceAccounting:
        """Return the final content-free accounting snapshot."""
        return LiveConformanceAccounting(
            create_or_stream_logical_operation_count=self._logical_counts[
                _LiveOperationClass.CREATE_OR_STREAM
            ],
            compact_logical_operation_count=self._logical_counts[
                _LiveOperationClass.COMPACT
            ],
            retrieve_logical_operation_count=self._logical_counts[
                _LiveOperationClass.RETRIEVE
            ],
            delete_logical_operation_count=self._logical_counts[
                _LiveOperationClass.DELETE
            ],
            create_or_stream_http_request_count=self._http_counts[
                _LiveOperationClass.CREATE_OR_STREAM
            ],
            compact_http_request_count=self._http_counts[
                _LiveOperationClass.COMPACT
            ],
            retrieve_http_request_count=self._http_counts[
                _LiveOperationClass.RETRIEVE
            ],
            delete_http_request_count=self._http_counts[
                _LiveOperationClass.DELETE
            ],
            unexpected_http_request_count=(
                self._unexpected_http_request_count
            ),
            sdk_configured_max_retries=self._sdk_configured_max_retries,
            observed_sdk_retry_count=self._observed_sdk_retry_count,
            request_path_class_mismatch_count=(
                self._request_path_class_mismatch_count
            ),
            cleanup_attempted=self._cleanup_attempted,
            cleanup_completed=self._cleanup_completed,
            cleanup_delete_logical_operation_count=(
                self._cleanup_delete_logical_operation_count
            ),
            cleanup_delete_http_request_count=(
                self._cleanup_delete_http_request_count
            ),
            cleanup_pending_reference_count=(
                self._cleanup_pending_reference_count
            ),
            client_close_completed=self._client_close_completed,
        )

    def __repr__(self) -> str:
        """Return a representation without request or provider material."""
        return "<content-free-live-transport-accounting>"


def _classify_http_request(
    method: str,
    path: str,
) -> _LiveOperationClass | None:
    """Return a fixed request class without retaining the request path."""
    if method == "POST" and path.endswith("/responses/compact"):
        return _LiveOperationClass.COMPACT
    if method == "POST" and path.endswith("/responses"):
        return _LiveOperationClass.CREATE_OR_STREAM
    if "/responses/" not in path:
        return None
    opaque_suffix = path.rsplit("/responses/", maxsplit=1)[1]
    if not opaque_suffix or "/" in opaque_suffix:
        return None
    if method == "GET":
        return _LiveOperationClass.RETRIEVE
    if method == "DELETE":
        return _LiveOperationClass.DELETE
    return None


def _accounting_matches_expected_matrix(
    accounting: LiveConformanceAccounting,
) -> bool:
    """Return whether one final snapshot matches the exact matrix."""
    logical = accounting._logical_operation_counts()
    http = accounting._http_request_counts()
    expected = {
        operation.value: count
        for operation, count in _EXPECTED_OPERATION_COUNTS.items()
    }
    return (
        logical == expected
        and http == {**expected, "unexpected": 0}
        and accounting.sdk_configured_max_retries == _SDK_MAX_RETRIES
        and accounting.observed_sdk_retry_count == 0
        and accounting.request_path_class_mismatch_count == 0
        and accounting.cleanup_attempted
        and accounting.cleanup_completed
        and accounting.cleanup_delete_logical_operation_count == 0
        and accounting.cleanup_delete_http_request_count == 0
        and accounting.cleanup_pending_reference_count == 0
        and accounting.client_close_completed
    )


def _validate_transport_accounting(
    accounting: LiveConformanceAccounting,
    *,
    require_complete_matrix: bool,
) -> None:
    """Fail closed on retries, unknown paths, mismatches, or cleanup drift."""
    values = (
        *accounting._logical_operation_counts().values(),
        *accounting._http_request_counts().values(),
        accounting.sdk_configured_max_retries,
        accounting.observed_sdk_retry_count,
        accounting.request_path_class_mismatch_count,
        accounting.cleanup_delete_logical_operation_count,
        accounting.cleanup_delete_http_request_count,
        accounting.cleanup_pending_reference_count,
    )
    if any(type(value) is not int or value < 0 for value in values):
        raise LiveConformanceAccountingError(
            "live transport accounting contains an invalid count"
        )
    if (
        accounting.sdk_configured_max_retries != _SDK_MAX_RETRIES
        or accounting.observed_sdk_retry_count != 0
        or accounting.unexpected_http_request_count != 0
        or accounting.request_path_class_mismatch_count != 0
        or not accounting.cleanup_attempted
        or not accounting.cleanup_completed
        or accounting.cleanup_pending_reference_count != 0
        or not accounting.client_close_completed
        or accounting.cleanup_delete_logical_operation_count
        != accounting.cleanup_delete_http_request_count
    ):
        raise LiveConformanceAccountingError(
            "live transport accounting failed closed validation"
        )
    logical = accounting._logical_operation_counts()
    http = accounting._http_request_counts()
    if any(
        http[operation.value] > logical[operation.value]
        for operation in _LiveOperationClass
    ):
        raise LiveConformanceAccountingError(
            "live transport HTTP accounting exceeds logical operations"
        )
    if require_complete_matrix and not _accounting_matches_expected_matrix(
        accounting
    ):
        raise LiveConformanceAccountingError(
            "live transport accounting does not match the complete matrix"
        )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class LiveConformanceReceipt:
    """Return a redacted, non-authoritative live conformance receipt."""

    provider_family: LiveProviderFamily
    endpoint_digest: str
    api_form: str
    provider_api_revision: str
    model_or_deployment: str
    model_or_deployment_revision: str
    model_identity_semantics: str
    observed_at: datetime
    provider_evidence_digest: str
    activation_manifest_digest: str
    structural_observations_digest: str
    completed_cases: tuple[LiveConformanceCase, ...]
    accounting: LiveConformanceAccounting

    def redacted_payload(self) -> dict[str, object]:
        """Return a JSON-safe receipt without credentials or opaque state."""
        return {
            "provider_family": self.provider_family.value,
            "endpoint_digest": self.endpoint_digest,
            "api_form": self.api_form,
            "provider_api_revision": self.provider_api_revision,
            "model_or_deployment": self.model_or_deployment,
            "model_or_deployment_revision": self.model_or_deployment_revision,
            "model_identity_semantics": self.model_identity_semantics,
            "observed_at": self.observed_at.isoformat(),
            "provider_evidence_digest": self.provider_evidence_digest,
            "activation_manifest_digest": self.activation_manifest_digest,
            "structural_observations_digest": (
                self.structural_observations_digest
            ),
            "completed_cases": [value.value for value in self.completed_cases],
            "accounting": self.accounting.redacted_payload(),
            "opaque_payloads_logged": False,
            "production_activation_granted": False,
        }


@final
class _ProtectedCleanupRetryChannel:
    """Retain bounded opaque cleanup references without exposing values."""

    def __init__(self) -> None:
        self._references: list[str] = []

    def ensure_capacity(self) -> None:
        """Reject a new stored effect before retry capacity is exhausted."""
        if len(self._references) >= _MAX_CLEANUP_REFERENCES:
            raise LiveConformanceCleanupError(
                "protected cleanup retry capacity is exhausted"
            )

    def retain(self, response_id: str) -> None:
        """Retain one opaque response reference in the protected channel."""
        if response_id in self._references:
            return
        self.ensure_capacity()
        self._references.append(response_id)

    def discard(self, response_id: str) -> None:
        """Discard one successfully deleted protected reference."""
        if response_id in self._references:
            self._references.remove(response_id)

    async def cleanup(
        self,
        delete_response: Callable[[str], Awaitable[None]],
    ) -> None:
        """Retry every retained deletion without disclosing identifiers."""
        for _ in range(_MAX_CLEANUP_ATTEMPTS):
            if not self._references:
                return
            for response_id in tuple(reversed(self._references)):
                try:
                    await delete_response(response_id)
                except CancelledError:
                    raise
                except Exception:
                    continue
                self.discard(response_id)
        if self._references:
            raise LiveConformanceCleanupError(
                "one or more protected cleanup operations failed"
            ) from None

    @property
    def pending_count(self) -> int:
        """Return only the content-free pending cleanup count."""
        return len(self._references)

    def __repr__(self) -> str:
        """Return a representation that never includes opaque references."""
        return "<protected-cleanup-retry-channel>"


class LiveConformanceTransport(Protocol):
    """Execute typed operations without exposing provider payloads."""

    async def execute(
        self,
        case: LiveConformanceCase,
    ) -> LiveConformanceObservation:
        """Execute one required live matrix case."""
        raise NotImplementedError

    async def aclose(self) -> None:
        """Delete retained live objects and close the provider client."""
        raise NotImplementedError

    def final_accounting(self) -> LiveConformanceAccounting:
        """Return content-free accounting after cleanup."""
        raise NotImplementedError


TransportFactory = Callable[
    [LiveConformanceConfig],
    LiveConformanceTransport,
]


def validate_provider_evidence(
    provider_evidence_path: Path,
    activation_manifest_path: Path,
    *,
    today: date,
    config: LiveConformanceConfig | None = None,
) -> tuple[str, str]:
    """Validate dated provider evidence and a strictly inactive manifest."""
    evidence, evidence_bytes = _load_json_object(provider_evidence_path)
    evidence_digest = _validate_canonical_digest(evidence)
    accessed_at = _date_field(evidence, "accessed_at")
    age = (today - accessed_at).days
    if age < 0 or age > _MAX_EVIDENCE_AGE_DAYS:
        raise LiveConformancePreflightError(
            "provider documentation evidence is stale or future-dated"
        )
    if evidence.get("production_activation_authority") is not False:
        raise LiveConformancePreflightError(
            "documentation evidence must not grant activation authority"
        )
    _validate_provider_sources(evidence, accessed_at)
    if config is not None:
        _validate_selected_provider_evidence(evidence, config)
    _validate_sdk_evidence(evidence)
    _validate_required_matrix(evidence.get("required_live_matrix"))

    manifest, _ = _load_json_object(activation_manifest_path)
    manifest_digest = _validate_canonical_digest(manifest)
    evidence_link = _object_field(manifest, "provider_evidence")
    if (
        evidence_link.get("path") != provider_evidence_path.name
        or evidence_link.get("byte_sha256")
        != sha256(evidence_bytes).hexdigest()
        or evidence_link.get("canonical_digest") != evidence_digest
    ):
        raise LiveConformancePreflightError(
            "activation manifest provider evidence link is invalid"
        )
    if (
        manifest.get("activation_state") != "inactive"
        or manifest.get("production_dispatch_enabled") is not False
        or manifest.get("production_advertisement_enabled") is not False
        or manifest.get("active_production_rows") != []
        or manifest.get("live_proof_ids") != []
    ):
        raise LiveConformancePreflightError(
            "phase 12 activation fixture must remain strictly inactive"
        )
    review = _object_field(manifest, "review")
    if (
        review.get("status") != "pending"
        or review.get("completed_review_ids") != []
    ):
        raise LiveConformancePreflightError(
            "inactive activation fixture must not claim completed review"
        )
    _validate_required_matrix(manifest.get("required_live_matrix"))
    return evidence_digest, manifest_digest


async def run_live_conformance(
    config: LiveConformanceConfig,
    *,
    provider_evidence_path: Path = _PROVIDER_EVIDENCE,
    activation_manifest_path: Path = _ACTIVATION_MANIFEST,
    cases: Sequence[LiveConformanceCase] = _EXECUTION_ORDER,
    transport_factory: TransportFactory,
    clock: Callable[[], Awaitable[datetime]],
) -> LiveConformanceReceipt:
    """Run the complete matrix only after every local preflight succeeds."""
    selected = tuple(cases)
    if selected != _EXECUTION_ORDER:
        raise LiveConformancePreflightError(
            "the complete ordered live conformance matrix is required"
        )
    preflight_now = await clock()
    if preflight_now.tzinfo is not UTC:
        raise LiveConformancePreflightError(
            "the live conformance clock must be UTC"
        )
    evidence_digest, activation_digest = validate_provider_evidence(
        provider_evidence_path,
        activation_manifest_path,
        today=preflight_now.date(),
        config=config,
    )
    try:
        transport = transport_factory(config)
    except Exception:
        raise LiveConformanceExecutionError(
            "live provider transport initialization failed"
        ) from None
    observations: list[LiveConformanceObservation] = []
    execution_failure: RuntimeError | None = None
    cancellation: CancelledError | None = None
    active_case: LiveConformanceCase | None = None
    try:
        for case in selected:
            active_case = case
            observation = await transport.execute(case)
            _assert_observation(case, observation)
            _assert_provider_identity(config, observation)
            observations.append(observation)
    except CancelledError as exc:
        cancellation = exc
    except (
        LiveConformanceAssertionError,
        LiveConformanceExecutionError,
    ) as exc:
        execution_failure = _case_failure(
            active_case,
            assertion=isinstance(exc, LiveConformanceAssertionError),
        )
    except Exception:
        execution_failure = _case_failure(active_case, assertion=False)
    cleanup_failure: LiveConformanceCleanupError | None = None
    try:
        await transport.aclose()
    except CancelledError:
        cleanup_failure = LiveConformanceCleanupError(
            "live provider cleanup was cancelled"
        )
    except LiveConformanceCleanupError as exc:
        cleanup_failure = exc
    except Exception:
        cleanup_failure = LiveConformanceCleanupError(
            "live provider cleanup failed"
        )
    accounting: LiveConformanceAccounting | None = None
    accounting_failure: LiveConformanceAccountingError | None = None
    try:
        accounting = transport.final_accounting()
        _validate_transport_accounting(
            accounting,
            require_complete_matrix=(
                cleanup_failure is None
                and cancellation is None
                and execution_failure is None
            ),
        )
    except LiveConformanceAccountingError as exc:
        accounting_failure = exc
    except Exception:
        accounting_failure = LiveConformanceAccountingError(
            "live transport final accounting is unavailable"
        )
    if cleanup_failure is not None:
        raise cleanup_failure from None
    if accounting_failure is not None:
        raise accounting_failure from None
    if cancellation is not None:
        raise cancellation from None
    if execution_failure is not None:
        raise execution_failure from None
    if accounting is None:
        raise LiveConformanceAccountingError(
            "live transport final accounting is unavailable"
        )
    observed_at = await clock()
    if observed_at.tzinfo is not UTC:
        raise LiveConformancePreflightError(
            "the live receipt clock must be UTC"
        )
    return LiveConformanceReceipt(
        provider_family=config.provider_family,
        endpoint_digest=sha256(config.endpoint.encode("utf-8")).hexdigest(),
        api_form=config.api_form,
        provider_api_revision=config.provider_api_revision,
        model_or_deployment=config.model_or_deployment,
        model_or_deployment_revision=config.model_or_deployment_revision,
        model_identity_semantics=_model_identity_semantics(config),
        observed_at=observed_at,
        provider_evidence_digest=evidence_digest,
        activation_manifest_digest=activation_digest,
        structural_observations_digest=(
            _structural_observations_digest(observations, accounting)
        ),
        completed_cases=tuple(value.case for value in observations),
        accounting=accounting,
    )


def _case_failure(
    case: LiveConformanceCase | None,
    *,
    assertion: bool,
) -> LiveConformanceAssertionError | LiveConformanceExecutionError:
    """Return one payload-free failure containing only its case name."""
    case_name = case.value if case is not None else "before_case_dispatch"
    message = f"live conformance case failed: {case_name}"
    if assertion:
        return LiveConformanceAssertionError(message)
    return LiveConformanceExecutionError(message)


@final
class OpenAISdkLiveConformanceTransport:
    """Execute the live matrix through typed AsyncOpenAI operations."""

    def __init__(self, config: LiveConformanceConfig) -> None:
        self._config = config
        self._accounting = _TransportAccounting()
        self._http_client = DefaultAsyncHttpxClient(
            event_hooks={
                "request": [self._accounting.record_http_request],
            }
        )
        self._client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.endpoint,
            default_query=(
                {"api-version": "preview"}
                if config.provider_api_revision
                == ProviderApiForm.AZURE_OPENAI_V1_PREVIEW.value
                else None
            ),
            http_client=self._http_client,
            max_retries=_SDK_MAX_RETRIES,
        )
        self._accounting.set_sdk_configured_max_retries(
            self._client.max_retries
        )
        self._stateless_output: list[ResponseInputItemParam] | None = None
        self._stateless_tool_outputs: list[ResponseInputItemParam] | None = (
            None
        )
        self._stored_first: Response | None = None
        self._stored_chained: Response | None = None
        self._cleanup_retries = _ProtectedCleanupRetryChannel()

    async def execute(
        self,
        case: LiveConformanceCase,
    ) -> LiveConformanceObservation:
        """Execute one typed native Responses operation."""
        try:
            match case:
                case LiveConformanceCase.STATELESS_CURRENT_TURN_TOOL:
                    return await self._stateless_current_turn_tool()
                case LiveConformanceCase.STATELESS_ALL_TURNS:
                    return await self._stateless_all_turns()
                case LiveConformanceCase.INLINE_COMPACTION:
                    return await self._inline_compaction()
                case LiveConformanceCase.STANDALONE_COMPACTION:
                    return await self._standalone_compaction()
                case LiveConformanceCase.STORED_CREATE:
                    return await self._stored_create()
                case LiveConformanceCase.STORED_CHAIN:
                    return await self._stored_chain()
                case LiveConformanceCase.STORED_RETRIEVE_DELETE:
                    return await self._stored_retrieve_delete()
                case LiveConformanceCase.STREAMING_TOOL:
                    return await self._streaming_tool()
            raise LiveConformanceExecutionError(
                "unsupported live conformance case"
            )
        except CancelledError:
            raise
        except LiveConformanceExecutionError as exc:
            raise exc from None
        except LiveConformanceAssertionError as exc:
            raise exc from None
        except Exception:
            raise LiveConformanceExecutionError(
                "typed live provider SDK operation failed"
            ) from None

    async def aclose(self) -> None:
        """Delete any retained stored responses and close the SDK client."""
        self._accounting.begin_cleanup()
        cancellation: CancelledError | None = None
        cleanup_failed = False
        try:
            await self._cleanup_retries.cleanup(self._delete_response)
        except CancelledError as exc:
            cancellation = exc
        except LiveConformanceCleanupError:
            cleanup_failed = True
        except Exception:
            cleanup_failed = True
        close_failed = False
        close_completed = False
        try:
            await self._client.close()
            close_completed = True
        except CancelledError as exc:
            cancellation = cancellation or exc
        except Exception:
            close_failed = True
        self._accounting.finish_cleanup(
            completed=(
                not cleanup_failed
                and not close_failed
                and cancellation is None
            ),
            pending_reference_count=self._cleanup_retries.pending_count,
            client_close_completed=close_completed,
        )
        if cleanup_failed or close_failed:
            raise LiveConformanceCleanupError(
                "protected live response cleanup or client close failed"
            ) from None
        if cancellation is not None:
            raise cancellation from None

    def final_accounting(self) -> LiveConformanceAccounting:
        """Return content-free accounting after protected cleanup."""
        return self._accounting.snapshot()

    async def _delete_response(self, response_id: str) -> None:
        """Delete one opaque response under logical request accounting."""
        with self._accounting.logical_operation(_LiveOperationClass.DELETE):
            await self._client.responses.delete(response_id)

    async def _stateless_current_turn_tool(
        self,
    ) -> LiveConformanceObservation:
        with self._accounting.logical_operation(
            _LiveOperationClass.CREATE_OR_STREAM
        ):
            response = await self._client.responses.create(
                model=self._config.model_or_deployment,
                input=(
                    "Compute ((37 times 41) plus (19 times 23) minus 11). "
                    "Then call phase12_probe exactly once with value equal to "
                    "the decimal result. Do not answer before the tool call."
                ),
                include=_encrypted_reasoning_include(self._config),
                reasoning=_reasoning("current_turn"),
                tools=[_tool()],
                tool_choice={"type": "function", "name": "phase12_probe"},
                store=False,
                stream=False,
            )
        self._stateless_output = _dump_output(
            response,
            provider_family=self._config.provider_family,
        )
        self._stateless_tool_outputs = _tool_outputs(response)
        tool_evidence = _required_tool_evidence(
            self._stateless_output,
            self._stateless_tool_outputs,
            expected_value=_PROBE_CURRENT_VALUE,
        )
        return _response_observation(
            LiveConformanceCase.STATELESS_CURRENT_TURN_TOOL,
            response,
            expected_parent=None,
            required_tool_evidence=tool_evidence,
        )

    async def _stateless_all_turns(self) -> LiveConformanceObservation:
        if (
            self._stateless_output is None
            or self._stateless_tool_outputs is None
        ):
            raise LiveConformanceAssertionError(
                "stateless current-turn output must precede all-turns replay"
            )
        replay = [
            *self._stateless_output,
            *self._stateless_tool_outputs,
            _input_message("Confirm the tool result in one sentence."),
        ]
        with self._accounting.logical_operation(
            _LiveOperationClass.CREATE_OR_STREAM
        ):
            response = await self._client.responses.create(
                model=self._config.model_or_deployment,
                input=replay,
                include=_encrypted_reasoning_include(self._config),
                reasoning=_reasoning("all_turns"),
                tools=[_tool()],
                store=False,
                stream=False,
            )
        complete = _complete_stateless_replay(
            self._stateless_output,
            self._stateless_tool_outputs,
            replay,
        )
        tool_evidence = _required_tool_evidence(
            self._stateless_output,
            self._stateless_tool_outputs,
            expected_value=_PROBE_CURRENT_VALUE,
        )
        return _response_observation(
            LiveConformanceCase.STATELESS_ALL_TURNS,
            response,
            expected_parent=None,
            complete_output_replayed=complete,
            replayed_encrypted_reasoning_items=(
                _encrypted_reasoning_input_items(self._stateless_output)
            ),
            replayed_tool_outputs=len(self._stateless_tool_outputs),
            required_tool_evidence=tool_evidence,
            preceding_reasoning_replayed=(
                _preceding_encrypted_reasoning_replayed(
                    self._stateless_output,
                    replay,
                )
            ),
        )

    async def _inline_compaction(self) -> LiveConformanceObservation:
        context_management: list[ContextManagement] = [
            {"type": "compaction", "compact_threshold": 2_048}
        ]
        with self._accounting.logical_operation(
            _LiveOperationClass.CREATE_OR_STREAM
        ):
            response = await self._client.responses.create(
                model=self._config.model_or_deployment,
                input=(
                    "Retain the marker phase12-inline and acknowledge it. "
                    + "bounded-context " * 4_096
                ),
                context_management=context_management,
                include=_encrypted_reasoning_include(self._config),
                reasoning=_reasoning("current_turn"),
                max_output_tokens=_INLINE_COMPACTION_MAX_OUTPUT_TOKENS,
                store=False,
                stream=False,
            )
        return _response_observation(
            LiveConformanceCase.INLINE_COMPACTION,
            response,
            expected_parent=None,
        )

    async def _standalone_compaction(self) -> LiveConformanceObservation:
        original: list[ResponseInputItemParam] = [
            _input_message(
                "Retain phase12-standalone. " + "compactable " * 4_096
            )
        ]
        with self._accounting.logical_operation(_LiveOperationClass.COMPACT):
            compacted = await self._client.responses.compact(
                model=self._config.model_or_deployment,
                input=original,
            )
        compact_output = _dump_compacted_output(
            compacted,
            provider_family=self._config.provider_family,
            original_input=original,
        )
        replay = [
            *compact_output,
            _input_message("Return only the retained marker."),
        ]
        with self._accounting.logical_operation(
            _LiveOperationClass.CREATE_OR_STREAM
        ):
            response = await self._client.responses.create(
                model=self._config.model_or_deployment,
                input=replay,
                include=_encrypted_reasoning_include(self._config),
                reasoning=_reasoning("current_turn"),
                store=False,
                stream=False,
            )
        observation = _response_observation(
            LiveConformanceCase.STANDALONE_COMPACTION,
            response,
            expected_parent=None,
            compact_output_replayed=(
                _complete_compact_replay(compact_output, replay)
            ),
        )
        compact_kinds = tuple(item.type for item in compacted.output)
        return LiveConformanceObservation(
            case=observation.case,
            response_status=observation.response_status,
            parent_matches=observation.parent_matches,
            reasoning_context=observation.reasoning_context,
            item_kinds=compact_kinds + observation.item_kinds,
            reported_model_identity=observation.reported_model_identity,
            required_reasoning_items=observation.required_reasoning_items,
            encrypted_reasoning_items=(observation.encrypted_reasoning_items),
            compaction_items=sum(
                isinstance(item, ResponseCompactionItem)
                for item in compacted.output
            ),
            opaque_compaction_items=sum(
                isinstance(item, ResponseCompactionItem)
                and bool(item.encrypted_content)
                for item in compacted.output
            ),
            tool_calls=observation.tool_calls,
            compact_output_replayed=observation.compact_output_replayed,
        )

    async def _stored_create(self) -> LiveConformanceObservation:
        self._cleanup_retries.ensure_capacity()
        with self._accounting.logical_operation(
            _LiveOperationClass.CREATE_OR_STREAM
        ):
            response = await self._client.responses.create(
                model=self._config.model_or_deployment,
                input="Remember the exact marker phase12-stored.",
                reasoning=_reasoning("current_turn"),
                store=True,
                stream=False,
            )
        self._stored_first = response
        self._cleanup_retries.retain(response.id)
        return _response_observation(
            LiveConformanceCase.STORED_CREATE,
            response,
            expected_parent=None,
        )

    async def _stored_chain(self) -> LiveConformanceObservation:
        if self._stored_first is None:
            raise LiveConformanceAssertionError(
                "stored create must precede previous-response chaining"
            )
        self._cleanup_retries.ensure_capacity()
        with self._accounting.logical_operation(
            _LiveOperationClass.CREATE_OR_STREAM
        ):
            response = await self._client.responses.create(
                model=self._config.model_or_deployment,
                input="Return only the marker from the previous turn.",
                previous_response_id=self._stored_first.id,
                reasoning=_reasoning("all_turns"),
                store=True,
                stream=False,
            )
        self._stored_chained = response
        self._cleanup_retries.retain(response.id)
        return _response_observation(
            LiveConformanceCase.STORED_CHAIN,
            response,
            expected_parent=self._stored_first.id,
        )

    async def _stored_retrieve_delete(self) -> LiveConformanceObservation:
        if self._stored_first is None or self._stored_chained is None:
            raise LiveConformanceAssertionError(
                "stored create and chain must precede lifecycle operations"
            )
        with self._accounting.logical_operation(_LiveOperationClass.RETRIEVE):
            retrieved = await self._client.responses.retrieve(
                self._stored_chained.id
            )
        if not isinstance(retrieved, Response):
            raise LiveConformanceAssertionError(
                "retrieve must return a typed Response"
            )
        retrieved_matches = retrieved.id == self._stored_chained.id
        await self._delete_response(self._stored_chained.id)
        self._cleanup_retries.discard(self._stored_chained.id)
        await self._delete_response(self._stored_first.id)
        self._cleanup_retries.discard(self._stored_first.id)
        return LiveConformanceObservation(
            case=LiveConformanceCase.STORED_RETRIEVE_DELETE,
            response_status="completed",
            parent_matches=True,
            reasoning_context=None,
            item_kinds=(),
            reported_model_identity=retrieved.model,
            retrieved=retrieved_matches,
            deleted=True,
        )

    async def _streaming_tool(self) -> LiveConformanceObservation:
        event_kinds: list[str] = []
        with self._accounting.logical_operation(
            _LiveOperationClass.CREATE_OR_STREAM
        ):
            async with self._client.responses.stream(
                model=self._config.model_or_deployment,
                input=(
                    "Compute ((43 times 47) plus (17 times 29) minus 9). "
                    "Then call phase12_probe exactly once with value equal to "
                    "the decimal result."
                ),
                include=_encrypted_reasoning_include(self._config),
                reasoning=_reasoning("current_turn"),
                tools=[_tool()],
                tool_choice={
                    "type": "function",
                    "name": "phase12_probe",
                },
                store=False,
            ) as stream:
                async for event in stream:
                    event_kinds.append(event.type)
                response = await stream.get_final_response()
        observation = _response_observation(
            LiveConformanceCase.STREAMING_TOOL,
            response,
            expected_parent=None,
            required_tool_evidence=_required_tool_evidence(
                _dump_output(
                    response,
                    provider_family=self._config.provider_family,
                ),
                _tool_outputs(response),
                expected_value=_PROBE_STREAM_VALUE,
            ),
        )
        return LiveConformanceObservation(
            case=observation.case,
            response_status=observation.response_status,
            parent_matches=observation.parent_matches,
            reasoning_context=observation.reasoning_context,
            item_kinds=observation.item_kinds,
            reported_model_identity=observation.reported_model_identity,
            required_reasoning_items=observation.required_reasoning_items,
            encrypted_reasoning_items=(observation.encrypted_reasoning_items),
            compaction_items=observation.compaction_items,
            opaque_compaction_items=observation.opaque_compaction_items,
            tool_calls=observation.tool_calls,
            required_tool_name_matches=(
                observation.required_tool_name_matches
            ),
            required_tool_arguments_match=(
                observation.required_tool_arguments_match
            ),
            tool_correlation_matches=observation.tool_correlation_matches,
            encrypted_reasoning_precedes_tool=(
                observation.encrypted_reasoning_precedes_tool
            ),
            stream_event_kinds=tuple(event_kinds),
        )


def _validate_azure_endpoint(endpoint: str) -> None:
    parsed = urlsplit(endpoint)
    if (
        parsed.scheme != "https"
        or parsed.port is not None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path != "/openai/v1"
        or parsed.hostname is None
        or fullmatch(_AZURE_ENDPOINT_PATTERN, parsed.hostname) is None
    ):
        raise LiveConformancePreflightError(
            "Azure OpenAI requires"
            " https://{resource}.openai.azure.com/openai/v1"
        )


def _load_json_object(path: Path) -> tuple[dict[str, object], bytes]:
    try:
        raw = path.read_bytes()
        value = loads(raw)
    except (OSError, UnicodeDecodeError, ValueError):
        raise LiveConformancePreflightError(
            f"unable to read evidence fixture {path.name}"
        ) from None
    if type(value) is not dict:
        raise LiveConformancePreflightError(
            f"evidence fixture {path.name} must be an object"
        )
    return cast(dict[str, object], value), raw


def _validate_canonical_digest(payload: dict[str, object]) -> str:
    digest = _object_field(payload, "canonical_digest")
    value = digest.get("value")
    if (
        digest.get("algorithm") != "sha256"
        or digest.get("encoding")
        != "utf-8 canonical JSON with sorted keys and compact separators"
        or digest.get("scope")
        != "all top-level fields except canonical_digest"
        or type(value) is not str
        or fullmatch(r"[0-9a-f]{64}", value) is None
    ):
        raise LiveConformancePreflightError(
            "fixture canonical digest declaration is invalid"
        )
    unsigned = dict(payload)
    unsigned.pop("canonical_digest")
    actual = sha256(
        dumps(
            unsigned,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    if actual != value:
        raise LiveConformancePreflightError(
            "fixture canonical digest does not match content"
        )
    return value


def _date_field(payload: Mapping[str, object], field_name: str) -> date:
    value = payload.get(field_name)
    if type(value) is not str:
        raise LiveConformancePreflightError(
            f"fixture {field_name} must be an ISO date"
        )
    try:
        return date.fromisoformat(value)
    except ValueError:
        raise LiveConformancePreflightError(
            f"fixture {field_name} must be an ISO date"
        ) from None


def _object_field(
    payload: Mapping[str, object],
    field_name: str,
) -> dict[str, object]:
    value = payload.get(field_name)
    if type(value) is not dict:
        raise LiveConformancePreflightError(
            f"fixture {field_name} must be an object"
        )
    return cast(dict[str, object], value)


def _validate_provider_sources(
    evidence: Mapping[str, object],
    accessed_at: date,
) -> None:
    providers = evidence.get("providers")
    if type(providers) is not list or len(providers) != 2:
        raise LiveConformancePreflightError(
            "provider evidence requires exact OpenAI and Azure rows"
        )
    expected = {
        "openai": (
            ProviderApiForm.OPENAI_RESPONSES_V1.value,
            _OPENAI_API_REVISION,
            "requested_model_and_response_model_revision_exact",
            "https://api.openai.com/v1",
            NativeOpenAIEncryptedContentPolicy.DEFAULT_RETURN.value,
            _OPENAI_DOCUMENTATION_URLS,
        ),
        "azure_openai": (
            ProviderApiForm.AZURE_OPENAI_V1_PREVIEW.value,
            ProviderApiForm.AZURE_OPENAI_V1_PREVIEW.value,
            "requested_deployment_response_model_and_revision_pin_exact",
            "https://{resource}.openai.azure.com/openai/v1",
            NativeOpenAIEncryptedContentPolicy.EXPLICIT_INCLUDE.value,
            _AZURE_DOCUMENTATION_URLS,
        ),
    }
    seen: set[str] = set()
    for raw_provider in providers:
        if type(raw_provider) is not dict:
            raise LiveConformancePreflightError(
                "provider evidence rows must be objects"
            )
        provider = cast(dict[str, object], raw_provider)
        family = provider.get("provider_family")
        if type(family) is not str or family not in expected or family in seen:
            raise LiveConformancePreflightError(
                "provider evidence family is missing, duplicate, or generic"
            )
        seen.add(family)
        (
            api_form,
            api_revision,
            model_semantics,
            endpoint_form,
            encrypted_content_policy,
            urls,
        ) = expected[family]
        if (
            provider.get("api_form") != api_form
            or provider.get("provider_api_revision") != api_revision
            or provider.get("model_identity_semantics") != model_semantics
            or provider.get("endpoint_form") != endpoint_form
            or provider.get("encrypted_content_policy")
            != encrypted_content_policy
        ):
            raise LiveConformancePreflightError(
                "provider evidence API form is not exact"
            )
        sources = provider.get("sources")
        if type(sources) is not list:
            raise LiveConformancePreflightError(
                "provider evidence sources must be a list"
            )
        actual_urls: list[str] = []
        for raw_source in sources:
            if type(raw_source) is not dict:
                raise LiveConformancePreflightError(
                    "provider documentation sources must be objects"
                )
            source = cast(dict[str, object], raw_source)
            url = source.get("url")
            facts = source.get("facts")
            if (
                type(url) is not str
                or _date_field(source, "accessed_at") != accessed_at
                or type(facts) is not list
                or not facts
                or any(type(fact) is not str or not fact for fact in facts)
            ):
                raise LiveConformancePreflightError(
                    "provider documentation source is incomplete"
                )
            actual_urls.append(url)
        if tuple(sorted(actual_urls)) != urls:
            raise LiveConformancePreflightError(
                "provider documentation source set has drifted"
            )


def _validate_selected_provider_evidence(
    evidence: Mapping[str, object],
    config: LiveConformanceConfig,
) -> None:
    """Bind the selected API and model semantics to reviewed evidence."""
    providers = evidence.get("providers")
    assert type(providers) is list
    selected = next(
        (
            cast(dict[str, object], value)
            for value in providers
            if type(value) is dict
            and value.get("provider_family") == config.provider_family.value
        ),
        None,
    )
    if selected is None or (
        selected.get("api_form") != config.api_form
        or selected.get("provider_api_revision")
        != config.provider_api_revision
        or selected.get("model_identity_semantics")
        != _model_identity_semantics(config)
        or selected.get("encrypted_content_policy")
        != _encrypted_content_policy(config.provider_family).value
    ):
        raise LiveConformancePreflightError(
            "selected provider identity is not bound to reviewed evidence"
        )


def _validate_sdk_evidence(evidence: Mapping[str, object]) -> None:
    sdk = _object_field(evidence, "sdk")
    supported_range = sdk.get("declared_supported_range")
    if (
        sdk.get("distribution") != "openai"
        or sdk.get("installed_version") != openai_version
        or supported_range != ">=2.42.0,<3.0.0"
        or Version(openai_version) not in SpecifierSet(supported_range)
        or sdk.get("client_type") != "openai.AsyncOpenAI"
        or sdk.get("typed_async_methods") != list(_TYPED_ASYNC_METHODS)
        or sdk.get("typed_request_fields") != list(_TYPED_CREATE_FIELDS)
        or sdk.get("typed_symbols") != [symbol for symbol, _ in _TYPED_SYMBOLS]
        or sdk.get("untyped_extra_body_permitted") is not False
    ):
        raise LiveConformancePreflightError(
            "installed or declared OpenAI SDK evidence has drifted"
        )
    create_parameters = signature(AsyncResponses.create).parameters
    typed_create_fields = tuple(
        field for field in create_parameters if field in _TYPED_CREATE_FIELDS
    )
    if typed_create_fields != _TYPED_CREATE_FIELDS:
        raise LiveConformancePreflightError(
            "installed SDK typed Responses fields have drifted"
        )
    for method_name in ("compact", "create", "delete", "retrieve", "stream"):
        if not callable(getattr(AsyncResponses, method_name, None)):
            raise LiveConformancePreflightError(
                "installed SDK is missing a required async lifecycle method"
            )
    for symbol, expected_symbol in _TYPED_SYMBOLS:
        module_name, attribute = symbol.rsplit(".", 1)
        if getattr(import_module(module_name), attribute, None) is not (
            expected_symbol
        ):
            raise LiveConformancePreflightError(
                "installed SDK typed symbol evidence has drifted"
            )


def _validate_required_matrix(value: object) -> None:
    expected = [case.value for case in REQUIRED_LIVE_CASES]
    if value != expected:
        raise LiveConformancePreflightError(
            "provider evidence must contain the complete exact live matrix"
        )


def _model_identity_semantics(config: LiveConformanceConfig) -> str:
    """Return the provider-specific exact model identity assertion."""
    if config.provider_family is LiveProviderFamily.OPENAI:
        return "requested_model_and_response_model_revision_exact"
    return "requested_deployment_response_model_and_revision_pin_exact"


def _assert_provider_identity(
    config: LiveConformanceConfig,
    observation: LiveConformanceObservation,
) -> None:
    """Require the exact provider-specific reported model identity."""
    expected = (
        config.model_or_deployment_revision
        if config.provider_family is LiveProviderFamily.OPENAI
        else config.model_or_deployment
    )
    if (
        fullmatch(_MODEL_PATTERN, observation.reported_model_identity) is None
        or observation.reported_model_identity != expected
    ):
        raise LiveConformanceAssertionError(
            "provider-reported model identity does not match the exact "
            "provider binding"
        )


def _structural_observations_digest(
    observations: Sequence[LiveConformanceObservation],
    accounting: LiveConformanceAccounting,
) -> str:
    """Digest redacted observations and generated transport accounting."""
    payload = {
        "accounting": accounting.redacted_payload(),
        "observations": [
            {
                "case": value.case.value,
                "response_status": value.response_status,
                "parent_matches": value.parent_matches,
                "reasoning_context": value.reasoning_context,
                "item_kinds": list(value.item_kinds),
                "reported_model_identity": value.reported_model_identity,
                "required_reasoning_items": value.required_reasoning_items,
                "encrypted_reasoning_items": value.encrypted_reasoning_items,
                "compaction_items": value.compaction_items,
                "opaque_compaction_items": value.opaque_compaction_items,
                "tool_calls": value.tool_calls,
                "required_tool_name_matches": (
                    value.required_tool_name_matches
                ),
                "required_tool_arguments_match": (
                    value.required_tool_arguments_match
                ),
                "tool_correlation_matches": value.tool_correlation_matches,
                "encrypted_reasoning_precedes_tool": (
                    value.encrypted_reasoning_precedes_tool
                ),
                "preceding_reasoning_replayed": (
                    value.preceding_reasoning_replayed
                ),
                "stream_event_kinds": list(value.stream_event_kinds),
                "complete_output_replayed": value.complete_output_replayed,
                "replayed_encrypted_reasoning_items": (
                    value.replayed_encrypted_reasoning_items
                ),
                "replayed_tool_outputs": value.replayed_tool_outputs,
                "compact_output_replayed": value.compact_output_replayed,
                "retrieved": value.retrieved,
                "deleted": value.deleted,
            }
            for value in observations
        ],
    }
    return sha256(
        dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _assert_observation(
    case: LiveConformanceCase,
    observation: LiveConformanceObservation,
) -> None:
    if observation.case is not case:
        raise LiveConformanceAssertionError(
            "transport returned an observation for the wrong case"
        )
    if case is LiveConformanceCase.STORED_RETRIEVE_DELETE:
        if not observation.retrieved or not observation.deleted:
            raise LiveConformanceAssertionError(
                "stored retrieve/delete lifecycle did not complete"
            )
        return
    if (
        observation.response_status != "completed"
        or not observation.parent_matches
    ):
        raise LiveConformanceAssertionError(
            "response status or immediate parent identity is invalid"
        )
    if case in {
        LiveConformanceCase.STATELESS_CURRENT_TURN_TOOL,
        LiveConformanceCase.STREAMING_TOOL,
    }:
        if (
            observation.required_reasoning_items < 1
            or observation.encrypted_reasoning_items
            != observation.required_reasoning_items
            or observation.tool_calls != 1
            or not observation.required_tool_name_matches
            or not observation.required_tool_arguments_match
            or not observation.tool_correlation_matches
            or not observation.encrypted_reasoning_precedes_tool
            or not _reasoning_precedes_tool(observation.item_kinds)
        ):
            raise LiveConformanceAssertionError(
                "the required encrypted reasoning and tool transition is "
                "invalid"
            )
    expected_context = (
        "all_turns"
        if case
        in {
            LiveConformanceCase.STATELESS_ALL_TURNS,
            LiveConformanceCase.STORED_CHAIN,
        }
        else "current_turn"
    )
    context_invalid = (
        observation.reasoning_context not in {None, "current_turn"}
        if case is LiveConformanceCase.STANDALONE_COMPACTION
        else observation.reasoning_context != expected_context
    )
    if context_invalid:
        raise LiveConformanceAssertionError(
            "provider did not preserve the requested reasoning context"
        )
    if case is LiveConformanceCase.STATELESS_ALL_TURNS and (
        not observation.complete_output_replayed
        or observation.required_reasoning_items < 1
        or observation.replayed_encrypted_reasoning_items
        != observation.required_reasoning_items
        or observation.replayed_tool_outputs != 1
        or observation.tool_calls != 1
        or not observation.required_tool_name_matches
        or not observation.required_tool_arguments_match
        or not observation.tool_correlation_matches
        or not observation.encrypted_reasoning_precedes_tool
        or not observation.preceding_reasoning_replayed
    ):
        raise LiveConformanceAssertionError(
            "all-turns stateless continuation did not replay complete "
            "encrypted output and its tool result"
        )
    if case is LiveConformanceCase.INLINE_COMPACTION and (
        observation.compaction_items < 1
        or observation.opaque_compaction_items != observation.compaction_items
    ):
        raise LiveConformanceAssertionError(
            "inline compaction did not emit an opaque compaction item"
        )
    if case is LiveConformanceCase.STANDALONE_COMPACTION and (
        observation.compaction_items < 1
        or observation.opaque_compaction_items != observation.compaction_items
        or not observation.compact_output_replayed
    ):
        raise LiveConformanceAssertionError(
            "standalone compact output was absent or pruned before replay"
        )
    if case is LiveConformanceCase.STREAMING_TOOL and (
        not observation.stream_event_kinds
        or observation.stream_event_kinds[-1] != "response.completed"
        or observation.stream_event_kinds.count("response.completed") != 1
        or sum(
            value
            in {
                "response.completed",
                "response.failed",
                "response.incomplete",
            }
            for value in observation.stream_event_kinds
        )
        != 1
    ):
        raise LiveConformanceAssertionError(
            "streaming did not terminate with response.completed"
        )


def _reasoning_precedes_tool(item_kinds: tuple[str, ...]) -> bool:
    try:
        return item_kinds.index("reasoning") < item_kinds.index(
            "function_call"
        )
    except ValueError:
        return False


def _encrypted_reasoning_input_items(
    items: Sequence[ResponseInputItemParam],
) -> int:
    """Count replay inputs that retain opaque encrypted reasoning."""
    return sum(
        item.get("type") == "reasoning"
        and type(item.get("encrypted_content")) is str
        and bool(item.get("encrypted_content"))
        for item in items
    )


def _required_tool_evidence(
    output: Sequence[ResponseInputItemParam],
    tool_outputs: Sequence[ResponseInputItemParam],
    *,
    expected_value: str,
) -> _RequiredToolEvidence:
    """Return exact structural facts for one required tool transition."""
    reasoning_positions = tuple(
        index
        for index, item in enumerate(output)
        if item.get("type") == "reasoning"
    )
    encrypted_positions = tuple(
        index
        for index in reasoning_positions
        if type(output[index].get("encrypted_content")) is str
        and bool(output[index].get("encrypted_content"))
    )
    function_positions = tuple(
        index
        for index, item in enumerate(output)
        if item.get("type") == "function_call"
    )
    function_call = (
        output[function_positions[0]] if len(function_positions) == 1 else None
    )
    tool_output = tool_outputs[0] if len(tool_outputs) == 1 else None
    call_id = (
        function_call.get("call_id") if function_call is not None else None
    )
    output_call_id = (
        tool_output.get("call_id") if tool_output is not None else None
    )
    return _RequiredToolEvidence(
        required_reasoning_items=len(reasoning_positions),
        encrypted_reasoning_items=len(encrypted_positions),
        tool_calls=len(function_positions),
        required_tool_name_matches=(
            function_call is not None
            and function_call.get("name") == _PROBE_TOOL_NAME
        ),
        required_tool_arguments_match=(
            function_call is not None
            and _tool_arguments_match(
                function_call.get("arguments"),
                expected_value,
            )
        ),
        tool_correlation_matches=(
            type(call_id) is str
            and bool(call_id)
            and output_call_id == call_id
            and tool_output is not None
            and tool_output.get("type") == "function_call_output"
            and tool_output.get("output") == _PROBE_TOOL_OUTPUT
        ),
        encrypted_reasoning_precedes_tool=(
            len(function_positions) == 1
            and any(
                index < function_positions[0] for index in encrypted_positions
            )
        ),
    )


def _tool_arguments_match(arguments: object, expected_value: str) -> bool:
    """Return whether strict JSON arguments equal the required object."""
    if type(arguments) is not str:
        return False
    try:
        decoded = loads(
            arguments,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (RecursionError, ValueError):
        return False
    return bool(decoded == {"value": expected_value})


def _preceding_encrypted_reasoning_replayed(
    output: Sequence[ResponseInputItemParam],
    replay: Sequence[ResponseInputItemParam],
) -> bool:
    """Prove an encrypted reasoning item before the tool was replayed."""
    function_positions = tuple(
        index
        for index, item in enumerate(output)
        if item.get("type") == "function_call"
    )
    if len(function_positions) != 1:
        return False
    function_position = function_positions[0]
    preceding_positions = tuple(
        index
        for index, item in enumerate(output[:function_position])
        if item.get("type") == "reasoning"
        and type(item.get("encrypted_content")) is str
        and bool(item.get("encrypted_content"))
    )
    return bool(preceding_positions) and all(
        index < len(replay) and replay[index] == output[index]
        for index in preceding_positions
    )


def _complete_stateless_replay(
    output: Sequence[ResponseInputItemParam],
    tool_outputs: Sequence[ResponseInputItemParam],
    replay: Sequence[ResponseInputItemParam],
) -> bool:
    """Prove exact prior output order, opaque state, and tool continuity."""
    output_count = len(output)
    tool_output_count = len(tool_outputs)
    tool_evidence = _required_tool_evidence(
        output,
        tool_outputs,
        expected_value=_PROBE_CURRENT_VALUE,
    )
    return (
        output_count > 0
        and tool_output_count == 1
        and len(replay) == output_count + tool_output_count + 1
        and replay[:output_count] == output
        and replay[output_count : output_count + tool_output_count]
        == tool_outputs
        and replay[-1].get("type") == "message"
        and replay[-1].get("role") == "user"
        and tool_evidence.required_reasoning_items >= 1
        and tool_evidence.encrypted_reasoning_items
        == tool_evidence.required_reasoning_items
        and tool_evidence.tool_calls == 1
        and tool_evidence.required_tool_name_matches
        and tool_evidence.required_tool_arguments_match
        and tool_evidence.tool_correlation_matches
        and tool_evidence.encrypted_reasoning_precedes_tool
        and _preceding_encrypted_reasoning_replayed(output, replay)
        and all("status" not in item for item in output)
        and all("status" not in item for item in tool_outputs)
    )


def _complete_compact_replay(
    output: Sequence[ResponseInputItemParam],
    replay: Sequence[ResponseInputItemParam],
) -> bool:
    """Prove the complete normalized opaque compact output was replayed."""
    compact_items = tuple(
        item for item in output if item.get("type") == "compaction"
    )
    return (
        bool(output)
        and len(replay) == len(output) + 1
        and replay[: len(output)] == output
        and replay[-1].get("type") == "message"
        and replay[-1].get("role") == "user"
        and bool(compact_items)
        and all(
            type(item.get("encrypted_content")) is str
            and bool(item.get("encrypted_content"))
            for item in compact_items
        )
        and all(
            "status" not in item and "created_by" not in item
            for item in output
        )
    )


def _tool() -> FunctionToolParam:
    return {
        "type": "function",
        "name": "phase12_probe",
        "description": "Return one bounded conformance marker.",
        "parameters": {
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
            "additionalProperties": False,
        },
        "strict": True,
    }


def _reasoning(context: str) -> Reasoning:
    if context == "current_turn":
        return {"context": "current_turn", "effort": "low"}
    if context == "all_turns":
        return {"context": "all_turns", "effort": "low"}
    raise LiveConformancePreflightError("unsupported reasoning context")


def _input_message(content: str) -> ResponseInputItemParam:
    return cast(
        ResponseInputItemParam,
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": content}],
        },
    )


def _provider_family(
    provider_family: LiveProviderFamily,
) -> ProviderFamily:
    """Return the exact production provider-family discriminator."""
    match provider_family:
        case LiveProviderFamily.OPENAI:
            return ProviderFamily.OPENAI
        case LiveProviderFamily.AZURE_OPENAI:
            return ProviderFamily.AZURE_OPENAI
    raise LiveConformancePreflightError(
        "replay normalization requires an exact native provider"
    )


def _dump_output(
    response: Response,
    *,
    provider_family: LiveProviderFamily,
) -> list[ResponseInputItemParam]:
    try:
        return [
            _normalize_response_output_item(
                item,
                provider_family=_provider_family(provider_family),
            )
            for item in response.output
        ]
    except ConversationValidationError:
        raise LiveConformanceAssertionError(
            "response output cannot be normalized for exact replay"
        ) from None


def _normalize_response_output_item(
    item: ResponseOutputItem,
    *,
    provider_family: ProviderFamily,
) -> ResponseInputItemParam:
    """Normalize one typed SDK response item for exact replay."""
    payload = item.model_dump(
        mode="json",
        exclude_none=True,
        warnings=False,
    )
    if isinstance(item, ParsedResponseFunctionToolCall):
        _strip_matching_parsed_arguments(item, payload)
    return _replay_item_to_input_item(
        payload,
        provider_family=provider_family,
    )


def _strip_matching_parsed_arguments(
    item: ParsedResponseFunctionToolCall,
    payload: dict[str, object],
) -> None:
    """Strip one SDK-only parsed argument after canonical equality proof."""
    if "parsed_arguments" not in payload:
        return
    if type(item).__api_exclude__ != {"parsed_arguments"}:
        raise ConversationValidationError()
    arguments = payload.get("arguments")
    parsed = payload.get("parsed_arguments")
    if type(arguments) is not str or type(parsed) is not dict:
        raise ConversationValidationError()
    try:
        decoded = loads(
            arguments,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
        decoded_bytes = canonical_json_bytes(freeze_json_value(decoded))
        parsed_bytes = canonical_json_bytes(freeze_json_value(parsed))
    except (ConversationValidationError, RecursionError, ValueError):
        raise ConversationValidationError() from None
    if decoded_bytes != parsed_bytes:
        raise ConversationValidationError()
    payload.pop("parsed_arguments")


def _dump_compacted_output(
    response: CompactedResponse,
    *,
    provider_family: LiveProviderFamily,
    original_input: Sequence[ResponseInputItemParam],
) -> list[ResponseInputItemParam]:
    if (
        not isinstance(response, CompactedResponse)
        or response.object != "response.compaction"
        or not response.output
    ):
        raise LiveConformanceAssertionError(
            "standalone compact must return a non-empty typed "
            "response.compaction"
        )
    try:
        if (
            not original_input
            or len(response.output) != len(original_input) + 1
        ):
            raise ConversationValidationError()
        retained = response.output[:-1]
        terminal = response.output[-1]
        if (
            not all(
                isinstance(item, ResponseOutputMessage) for item in retained
            )
            or not isinstance(terminal, ResponseCompactionItem)
            or sum(
                isinstance(item, ResponseCompactionItem)
                for item in response.output
            )
            != 1
        ):
            raise ConversationValidationError()
        family = _provider_family(provider_family)
        expected_messages = tuple(
            _normalize_original_compact_input(item, provider_family=family)
            for item in original_input
        )
        retained_messages = tuple(
            _normalize_compacted_user_message(
                item.model_dump(
                    mode="json",
                    exclude_none=True,
                    warnings=False,
                ),
                provider_family=family,
            )
            for item in retained
        )
        if tuple(
            canonical_json_bytes(freeze_json_value(item))
            for item in retained_messages
        ) != tuple(
            canonical_json_bytes(freeze_json_value(item))
            for item in expected_messages
        ):
            raise ConversationValidationError()
        compact_item = _replay_item_to_input_item(
            terminal.model_dump(
                mode="json",
                exclude_none=True,
                warnings=False,
            ),
            provider_family=family,
        )
        if (
            compact_item.get("type") != "compaction"
            or type(compact_item.get("encrypted_content")) is not str
            or not compact_item.get("encrypted_content")
        ):
            raise ConversationValidationError()
        return [*retained_messages, compact_item]
    except ConversationValidationError:
        raise LiveConformanceAssertionError(
            "compaction output cannot be normalized for exact replay"
        ) from None


def _normalize_original_compact_input(
    value: ResponseInputItemParam,
    *,
    provider_family: ProviderFamily,
) -> ResponseInputItemParam:
    """Return one closed original user input for compact-output binding."""
    if type(value) is not dict:
        raise ConversationValidationError()
    _validate_compact_user_message(value)
    normalized = _replay_item_to_input_item(
        value,
        provider_family=provider_family,
    )
    if canonical_json_bytes(
        freeze_json_value(normalized)
    ) != canonical_json_bytes(freeze_json_value(value)):
        raise ConversationValidationError()
    return normalized


def _normalize_compacted_user_message(
    value: Mapping[str, object],
    *,
    provider_family: ProviderFamily,
) -> ResponseInputItemParam:
    """Return one exact retained user message from compact SDK output."""
    if set(value) != {"content", "id", "role", "status", "type"}:
        raise ConversationValidationError()
    validate_identifier(value.get("id"), "compacted user message id")
    if value.get("status") != "completed":
        raise ConversationValidationError()
    normalized = dict(value)
    normalized.pop("id")
    normalized.pop("status")
    _validate_compact_user_message(normalized)
    return _replay_item_to_input_item(
        normalized,
        provider_family=provider_family,
    )


def _validate_compact_user_message(value: Mapping[str, object]) -> None:
    """Validate one closed text-only user message for compaction."""
    if (
        set(value) != {"content", "role", "type"}
        or value.get("role") != "user"
        or value.get("type") != "message"
    ):
        raise ConversationValidationError()
    content = value.get("content")
    if type(content) is not list or not content:
        raise ConversationValidationError()
    for part in content:
        if (
            type(part) is not dict
            or set(part) != {"text", "type"}
            or part.get("type") != "input_text"
            or type(part.get("text")) is not str
        ):
            raise ConversationValidationError()


def _tool_outputs(response: Response) -> list[ResponseInputItemParam]:
    outputs: list[ResponseInputItemParam] = []
    for item in response.output:
        if isinstance(item, ResponseFunctionToolCall):
            outputs.append(
                cast(
                    ResponseInputItemParam,
                    {
                        "type": "function_call_output",
                        "call_id": item.call_id,
                        "output": _PROBE_TOOL_OUTPUT,
                    },
                )
            )
    if len(outputs) != 1:
        raise LiveConformanceAssertionError(
            "exactly one required function tool call must be returned"
        )
    return outputs


def _response_observation(
    case: LiveConformanceCase,
    response: Response,
    *,
    expected_parent: str | None,
    complete_output_replayed: bool = False,
    replayed_encrypted_reasoning_items: int = 0,
    replayed_tool_outputs: int = 0,
    compact_output_replayed: bool = False,
    required_tool_evidence: _RequiredToolEvidence | None = None,
    preceding_reasoning_replayed: bool = False,
) -> LiveConformanceObservation:
    if not isinstance(response, Response):
        raise LiveConformanceAssertionError(
            "Responses create must return a typed Response"
        )
    reasoning_context = (
        response.reasoning.context if response.reasoning is not None else None
    )
    evidence = required_tool_evidence
    return LiveConformanceObservation(
        case=case,
        response_status=response.status or "",
        parent_matches=response.previous_response_id == expected_parent,
        reasoning_context=reasoning_context,
        item_kinds=tuple(item.type for item in response.output),
        reported_model_identity=response.model,
        required_reasoning_items=(
            evidence.required_reasoning_items if evidence is not None else 0
        ),
        encrypted_reasoning_items=sum(
            isinstance(item, ResponseReasoningItem)
            and bool(item.encrypted_content)
            for item in response.output
        ),
        compaction_items=sum(
            isinstance(item, ResponseCompactionItem)
            for item in response.output
        ),
        opaque_compaction_items=sum(
            isinstance(item, ResponseCompactionItem)
            and bool(item.encrypted_content)
            for item in response.output
        ),
        tool_calls=(
            evidence.tool_calls
            if evidence is not None
            else sum(
                isinstance(item, ResponseFunctionToolCall)
                for item in response.output
            )
        ),
        required_tool_name_matches=(
            evidence is not None and evidence.required_tool_name_matches
        ),
        required_tool_arguments_match=(
            evidence is not None and evidence.required_tool_arguments_match
        ),
        tool_correlation_matches=(
            evidence is not None and evidence.tool_correlation_matches
        ),
        encrypted_reasoning_precedes_tool=(
            evidence is not None and evidence.encrypted_reasoning_precedes_tool
        ),
        preceding_reasoning_replayed=preceding_reasoning_replayed,
        complete_output_replayed=complete_output_replayed,
        replayed_encrypted_reasoning_items=(
            replayed_encrypted_reasoning_items
        ),
        replayed_tool_outputs=replayed_tool_outputs,
        compact_output_replayed=compact_output_replayed,
    )


async def _utc_clock() -> datetime:
    return datetime.now(UTC)


def _build_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description=(
            "Run cost-bearing native Responses conformance with "
            "redacted output."
        ),
        epilog=(
            "Also requires AVALAN_LIVE_CONFORMANCE_AUTHORITY="
            "authorize-phase12-live-conformance and "
            "AVALAN_LIVE_CONFORMANCE_COST_ACK="
            "accept-phase12-provider-costs, plus exact provider identity "
            "and credential variables."
        ),
    )
    parser.add_argument(
        "--provider",
        required=True,
        choices=("openai", "azure_openai"),
    )
    parser.add_argument(
        "--authorize-live-provider-conformance",
        action="store_true",
    )
    parser.add_argument(
        "--acknowledge-provider-costs",
        action="store_true",
    )
    return parser


def _config_from_environment(
    arguments: Namespace,
    values: Mapping[str, str],
) -> LiveConformanceConfig:
    family = LiveProviderFamily(arguments.provider)
    if family is LiveProviderFamily.OPENAI:
        endpoint = values.get("OPENAI_BASE_URL", _OPENAI_ENDPOINT)
        api_form = ProviderApiForm.OPENAI_RESPONSES_V1.value
        provider_api_revision = _OPENAI_API_REVISION
        model = values.get("OPENAI_MODEL", "")
        revision = values.get("OPENAI_MODEL_REVISION", "")
        api_key = values.get("OPENAI_API_KEY", "")
    else:
        endpoint = values.get("AZURE_OPENAI_ENDPOINT", "")
        provider_api_revision = values.get(
            "AZURE_OPENAI_API_REVISION",
            "",
        )
        api_form = provider_api_revision
        model = values.get("AZURE_OPENAI_DEPLOYMENT", "")
        revision = values.get("AZURE_OPENAI_DEPLOYMENT_REVISION", "")
        api_key = values.get("AZURE_OPENAI_API_KEY", "")
    return LiveConformanceConfig(
        provider_family=family,
        endpoint=endpoint,
        api_form=api_form,
        provider_api_revision=provider_api_revision,
        model_or_deployment=model,
        model_or_deployment_revision=revision,
        api_key=api_key,
        command_authority=arguments.authorize_live_provider_conformance,
        environment_authority=values.get(
            "AVALAN_LIVE_CONFORMANCE_AUTHORITY", ""
        ),
        command_cost_acknowledgement=arguments.acknowledge_provider_costs,
        environment_cost_acknowledgement=values.get(
            "AVALAN_LIVE_CONFORMANCE_COST_ACK", ""
        ),
    )


async def _main(arguments: Namespace) -> int:
    config = _config_from_environment(arguments, environ)
    receipt = await run_live_conformance(
        config,
        transport_factory=OpenAISdkLiveConformanceTransport,
        clock=_utc_clock,
    )
    print(dumps(receipt.redacted_payload(), sort_keys=True))
    return 0


def main() -> int:
    """Run the command-line live conformance harness."""
    return run(_main(_build_parser().parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())

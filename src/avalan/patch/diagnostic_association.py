"""Associate a dormant pure diagnostic with immutable patch truth.

Only this trusted module creates the diagnostic host, its root signer, and
the projection host. It requires an already sealed plan that both selects a
diagnostic policy and discloses ``DIAGNOSTIC_ASSOCIATION``. The policy ID is
therefore derived from sealed policy truth, never from a caller argument.

Lower code receives only ``NewType`` byte values. They are copyable, but
contain no methods, host, store, plan, signer, callback, executor, process,
or retention authority. Importing or reflecting into this trusted module is
a trusted-host compromise, just as it is for the audience-projection root;
ordinary delivered values cannot reach this host or mint an identity.

The only execution is a final module-owned ``READ_ONLY_PROBE`` integrity check
over detached authenticated bytes. Its bounded outcome only attests that
internal check; it is not evidence of an external test, formatter, fixer, or
target verification. It has a fixed empty environment and arguments, a fixed
synthetic cwd, and no workspace, process, network, secret, hook,
configuration, plugin, formatter, fixer, interpreter, language-server, test,
or write port. Formatter/fixer output remains separately authorised and can
only begin a brand-new patch request through its typed protocol.
"""

from asyncio import (
    CancelledError,
    Task,
    create_task,
    current_task,
    shield,
    sleep,
    wait_for,
)
from base64 import b64decode, urlsafe_b64encode
from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
from hmac import compare_digest, digest
from json import dumps, loads
from secrets import token_bytes, token_urlsafe
from typing import Never, NewType, Protocol, final

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from avalan.patch.audience_projection import (
    AudienceProjectionError,
    AudienceProjectionHost,
    PatchAudienceProjectionSource,
)
from avalan.patch.codec import encode_result
from avalan.patch.domain import (
    LifecyclePhase,
    MutationState,
    PatchObserverCorrelationId,
    PatchRequestId,
    PatchStatus,
)
from avalan.patch.durable_store import (
    DurablePatchStore,
    DurableRequestAccess,
    DurableRequestSnapshot,
    DurableRetentionKind,
)
from avalan.patch.policy import (
    DiagnosticPolicyId,
    ExecutionSubject,
    PolicyDisclosure,
    SealedPlan,
    _validate_sealed_plan,
)


class DiagnosticAssociationErrorCode(str, Enum):
    """Name closed diagnostic-association boundary failures."""

    CAPABILITY_INVALID = "diagnostic.capability_invalid"
    APPROVAL_INVALID = "diagnostic.approval_invalid"
    ASSOCIATION_INVALID = "diagnostic.association_invalid"
    TERMINAL_UNAVAILABLE = "diagnostic.terminal_unavailable"
    TERMINAL_INVALID = "diagnostic.terminal_invalid"
    POLICY_INVALID = "diagnostic.policy_invalid"
    PROHIBITED_COMMAND = "diagnostic.prohibited_command"
    HOST_BUSY = "diagnostic.host_busy"


class DiagnosticAssociationError(RuntimeError):
    """Report one closed diagnostic-association boundary failure."""

    def __init__(self, code: DiagnosticAssociationErrorCode) -> None:
        """Initialize a bounded failure without source or process detail."""
        super().__init__(code.value)
        self.code = code


class DiagnosticCommandClass(str, Enum):
    """Name command classes rejected by the fixed diagnostic boundary."""

    READ_ONLY_PROBE = "read_only_probe"
    FORMATTER = "formatter"
    FIXER = "fixer"
    INTERPRETER = "interpreter"
    LANGUAGE_SERVER = "language_server"
    TEST = "test"
    REPOSITORY_HOOK = "repository_hook"
    WORKSPACE_CONFIGURATION = "workspace_configuration"
    WORKSPACE_PLUGIN = "workspace_plugin"
    NETWORK = "network"
    SECRET = "secret"
    WORKSPACE_WRITE = "workspace_write"
    EXTERNAL_WRITE = "external_write"
    ARBITRARY = "arbitrary"


class DiagnosticSandboxOutcome(str, Enum):
    """Name bounded independent diagnostic execution outcomes."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"
    UNAVAILABLE = "unavailable"


DiagnosticCapability = NewType("DiagnosticCapability", bytes)
"""Carry detached signed capability bytes with no trusted methods."""

DiagnosticApprovalReceipt = NewType("DiagnosticApprovalReceipt", bytes)
"""Carry detached separately-issued approval bytes with no trusted methods."""

DiagnosticAssociation = NewType("DiagnosticAssociation", bytes)
"""Carry detached signed association bytes with no trusted methods."""


@dataclass(frozen=True, slots=True, repr=False)
class _SealedReadOnlyProbeRequest:
    """Carry the exact final module-owned pure diagnostic request."""

    command: DiagnosticCommandClass
    arguments: tuple[str, ...]
    environment: tuple[tuple[str, str], ...]
    cwd: str
    detached_snapshot: bytes

    def __post_init__(self) -> None:
        """Reject every caller-configurable process or write input."""
        if (
            type(self.command) is not DiagnosticCommandClass
            or self.command is not DiagnosticCommandClass.READ_ONLY_PROBE
            or self.arguments != ()
            or self.environment != ()
            or self.cwd != "<detached-read-only-probe>"
            or type(self.detached_snapshot) is not bytes
            or len(self.detached_snapshot) != 32
        ):
            raise DiagnosticAssociationError(
                DiagnosticAssociationErrorCode.PROHIBITED_COMMAND
            )


@final
@dataclass(frozen=True, slots=True, repr=False, init=False)
class RemediationPatchAuthorization:
    """Carry separate authority to request a brand-new remediation patch."""

    def __init__(self, token: Never) -> None:
        """Reject public construction of remediation request authority."""
        del token
        raise DiagnosticAssociationError(
            DiagnosticAssociationErrorCode.CAPABILITY_INVALID
        )


@final
@dataclass(frozen=True, slots=True, repr=False, init=False)
class FormatterFixerResult:
    """Carry independently authorised formatter or fixer output opaquely."""

    def __init__(self, token: Never) -> None:
        """Reject public construction of formatter or fixer output."""
        del token
        raise DiagnosticAssociationError(
            DiagnosticAssociationErrorCode.CAPABILITY_INVALID
        )


class RemediationPatchRequester(Protocol):
    """Request a brand-new patch from separately authorised remediation."""

    async def begin_new_patch_request(
        self,
        authorization: RemediationPatchAuthorization,
        result: FormatterFixerResult,
    ) -> PatchRequestId:
        """Return a new patch request without a diagnostic-side write."""


def diagnostic_retention_kind() -> DurableRetentionKind:
    """Name existing retention purpose without granting retention authority."""
    return DurableRetentionKind.DIAGNOSTIC_ASSOCIATION


_ROOT_SIGNER = Ed25519PrivateKey.generate()
_ROOT_VERIFIER: Ed25519PublicKey = _ROOT_SIGNER.public_key()
_SNAPSHOT_BINDING_KEY = token_bytes(32)
_TRUSTED_READ_ONLY_PROBE_TIMEOUT_SECONDS = 0.050
_TRUSTED_PROBE_FINALIZATION_SECONDS = 0.050
_DIAGNOSTIC_AUDIENCE = "diagnostic_association"


@final
@dataclass(slots=True, repr=False)
class _TrustedDiagnosticHost:
    """Keep fixed token slots and every live task private.

    Repeated capability and approval requests return their one stored detached
    value. A receipt is consumed once and every later association attempt
    rejects. In-flight issuance and approval reservations prevent concurrent
    callers from minting a second identifier without retaining a registry.
    """

    _projection_host: AudienceProjectionHost
    _plan: SealedPlan
    _access: DurableRequestAccess
    _correlation: PatchObserverCorrelationId
    _policy_id: DiagnosticPolicyId
    _service_id: str
    _capability: DiagnosticCapability | None
    _capability_issuing: bool
    _approval_receipt: DiagnosticApprovalReceipt | None
    _approval_issuing: bool
    _consumed_receipt_id: str | None
    _generation: int
    _running_task: Task[DiagnosticSandboxOutcome] | None
    _finalizer: Task[None] | None

    async def _issue_capability(self) -> DiagnosticCapability:
        """Return the one reserved capability from inspected terminal truth."""
        if self._capability is not None:
            return self._capability
        if self._capability_issuing:
            raise DiagnosticAssociationError(
                DiagnosticAssociationErrorCode.CAPABILITY_INVALID
            )
        self._capability_issuing = True
        try:
            source = await self._source()
            snapshot = _terminal_truth(source, self._access)
            payload = _capability_payload(
                token_urlsafe(24),
                self._service_id,
                self._plan.binding.request.execution_id.value,
                self._plan.binding.subject,
                self._policy_id,
                snapshot,
                _terminal_binding(source),
            )
            self._capability = DiagnosticCapability(_sign_payload(payload))
            return self._capability
        finally:
            self._capability_issuing = False

    async def _approve(
        self, capability: DiagnosticCapability
    ) -> DiagnosticApprovalReceipt:
        """Return the one opaque receipt through the trusted approval path."""
        if (
            self._capability is None
            or type(capability) is not bytes
            or not compare_digest(capability, self._capability)
        ):
            raise DiagnosticAssociationError(
                DiagnosticAssociationErrorCode.APPROVAL_INVALID
            )
        if self._approval_receipt is not None:
            return self._approval_receipt
        if self._approval_issuing:
            raise DiagnosticAssociationError(
                DiagnosticAssociationErrorCode.APPROVAL_INVALID
            )
        self._approval_issuing = True
        try:
            payload = await self._verify_capability(capability)
            source = await self._source()
            snapshot = _terminal_truth(source, self._access)
            binding = _terminal_binding(source)
            _require_payload_terminal(payload, snapshot, binding, "capability")
            payload["association_id"] = token_urlsafe(24)
            payload["receipt_id"] = token_urlsafe(24)
            payload["kind"] = "approval"
            self._approval_receipt = DiagnosticApprovalReceipt(
                _sign_payload(payload)
            )
            return self._approval_receipt
        finally:
            self._approval_issuing = False

    async def _associate(
        self, receipt: DiagnosticApprovalReceipt
    ) -> DiagnosticAssociation:
        """Consume an approved receipt before await and run the fixed probe."""
        payload = self._receipt(receipt)
        receipt_id = _payload_string(payload, "receipt_id", "approval")
        if self._consumed_receipt_id is not None:
            raise DiagnosticAssociationError(
                DiagnosticAssociationErrorCode.APPROVAL_INVALID
            )
        self._consumed_receipt_id = receipt_id
        if self._has_live_work():
            raise DiagnosticAssociationError(
                DiagnosticAssociationErrorCode.HOST_BUSY
            )
        try:
            source = await self._source()
            snapshot = _terminal_truth(source, self._access)
            binding = _terminal_binding(source)
            _require_payload_terminal(payload, snapshot, binding, "approval")
            outcome = await self._run_probe(_sealed_probe_request(binding))
        except CancelledError:
            outcome = DiagnosticSandboxOutcome.CANCELLED
        except DiagnosticAssociationError:
            raise
        except Exception:
            outcome = DiagnosticSandboxOutcome.UNAVAILABLE
        payload["kind"] = "association"
        payload["outcome"] = outcome.value
        return DiagnosticAssociation(_sign_payload(payload))

    async def _aclose(self) -> None:
        """Retain and boundedly finalise any task owned by this host."""
        task = self._running_task
        if task is not None:
            finalizer = self._begin_finalization(self._generation, task)
            await self._wait_for_finalizer(finalizer)
        elif self._finalizer is not None:
            await self._wait_for_finalizer(self._finalizer)

    def _has_live_work(self) -> bool:
        """Reject a second execution while any owned work remains live."""
        return self._running_task is not None or (
            self._finalizer is not None and not self._finalizer.done()
        )

    async def _source(self) -> PatchAudienceProjectionSource:
        """Re-read authoritative durable truth through the sealed host."""
        try:
            witness = await self._projection_host.issue_access(
                self._access, self._correlation
            )
            return await self._projection_host.source(self._plan, witness)
        except AudienceProjectionError:
            raise DiagnosticAssociationError(
                DiagnosticAssociationErrorCode.TERMINAL_UNAVAILABLE
            ) from None
        except Exception:
            raise DiagnosticAssociationError(
                DiagnosticAssociationErrorCode.TERMINAL_UNAVAILABLE
            ) from None

    async def _verify_capability(self, value: object) -> dict[str, str]:
        """Verify exact signed capability scope for this private host."""
        if (
            self._capability is None
            or type(value) is not bytes
            or not compare_digest(value, self._capability)
        ):
            raise DiagnosticAssociationError(
                DiagnosticAssociationErrorCode.CAPABILITY_INVALID
            )
        payload = _verify_payload(value, "capability")
        if (
            _payload_string(payload, "audience", "capability")
            != _DIAGNOSTIC_AUDIENCE
            or _payload_string(payload, "service_id", "capability")
            != self._service_id
            or _payload_string(payload, "execution_id", "capability")
            != self._plan.binding.request.execution_id.value
            or _payload_string(payload, "request_id", "capability")
            != self._access.request_id.value
            or _payload_string(payload, "plan_id", "capability")
            != self._plan.plan_id.value
            or _payload_string(payload, "policy_id", "capability")
            != self._policy_id.value
        ):
            raise DiagnosticAssociationError(
                DiagnosticAssociationErrorCode.CAPABILITY_INVALID
            )
        _require_payload_subject(
            payload, self._plan.binding.subject, "capability"
        )
        source = await self._source()
        snapshot = _terminal_truth(source, self._access)
        _require_payload_terminal(
            payload,
            snapshot,
            _terminal_binding(source),
            "capability",
        )
        return payload

    def _receipt(self, value: object) -> dict[str, str]:
        """Verify exact signed approval receipt scope for this private host."""
        if (
            self._approval_receipt is None
            or type(value) is not bytes
            or not compare_digest(value, self._approval_receipt)
        ):
            raise DiagnosticAssociationError(
                DiagnosticAssociationErrorCode.APPROVAL_INVALID
            )
        payload = _verify_payload(value, "approval")
        if (
            _payload_string(payload, "audience", "approval")
            != _DIAGNOSTIC_AUDIENCE
            or _payload_string(payload, "service_id", "approval")
            != self._service_id
            or _payload_string(payload, "execution_id", "approval")
            != self._plan.binding.request.execution_id.value
            or _payload_string(payload, "request_id", "approval")
            != self._access.request_id.value
            or _payload_string(payload, "plan_id", "approval")
            != self._plan.plan_id.value
            or _payload_string(payload, "policy_id", "approval")
            != self._policy_id.value
        ):
            raise DiagnosticAssociationError(
                DiagnosticAssociationErrorCode.APPROVAL_INVALID
            )
        _require_payload_subject(
            payload, self._plan.binding.subject, "approval"
        )
        return payload

    async def _run_probe(
        self, request: _SealedReadOnlyProbeRequest
    ) -> DiagnosticSandboxOutcome:
        """Own one task until its callback has released it."""
        request.__post_init__()
        self._generation += 1
        generation = self._generation
        task: Task[DiagnosticSandboxOutcome] = create_task(
            _read_only_probe(request)
        )
        self._running_task = task
        task.add_done_callback(
            lambda completed: self._task_completed(generation, completed)
        )
        try:
            return await wait_for(
                shield(task), _TRUSTED_READ_ONLY_PROBE_TIMEOUT_SECONDS
            )
        except CancelledError:
            await self._finalize_if_owned(generation, task)
            return DiagnosticSandboxOutcome.CANCELLED
        except TimeoutError:
            await self._finalize_if_owned(generation, task)
            return DiagnosticSandboxOutcome.TIMED_OUT
        except Exception:
            await self._finalize_if_owned(generation, task)
            return DiagnosticSandboxOutcome.UNAVAILABLE

    async def _finalize_if_owned(
        self, generation: int, task: Task[DiagnosticSandboxOutcome]
    ) -> None:
        """Finalize a live task, leaving an already completed task released."""
        if task is self._running_task:
            await self._wait_for_finalizer(
                self._begin_finalization(generation, task)
            )

    def _begin_finalization(
        self, generation: int, task: Task[DiagnosticSandboxOutcome]
    ) -> Task[None]:
        """Create exactly one host-owned finalizer for the exact task."""
        if generation != self._generation or task is not self._running_task:
            raise DiagnosticAssociationError(
                DiagnosticAssociationErrorCode.HOST_BUSY
            )
        if self._finalizer is None or self._finalizer.done():
            self._finalizer = create_task(self._finalize(generation, task))
        return self._finalizer

    async def _wait_for_finalizer(self, finalizer: Task[None]) -> None:
        """Bound caller waiting without ever dropping host task ownership."""
        try:
            await wait_for(
                shield(finalizer), _TRUSTED_PROBE_FINALIZATION_SECONDS
            )
        except CancelledError:
            return
        except TimeoutError:
            return
        except Exception:
            return

    async def _finalize(
        self, generation: int, task: Task[DiagnosticSandboxOutcome]
    ) -> None:
        """Cancel and retain one task until it actually completes."""
        if not task.done():
            task.cancel()
        try:
            await shield(task)
        except CancelledError:
            pass
        except Exception:
            pass
        finally:
            if (
                generation == self._generation
                and task is self._running_task
                and current_task() is self._finalizer
            ):
                self._finalizer = None

    def _task_completed(
        self, generation: int, task: Task[DiagnosticSandboxOutcome]
    ) -> None:
        """Retrieve result and release only the matching task generation."""
        try:
            task.result()
        except CancelledError:
            pass
        except Exception:
            pass
        if generation == self._generation and task is self._running_task:
            self._running_task = None


async def _trusted_diagnostic_host(
    store: DurablePatchStore,
    plan: SealedPlan,
    access: DurableRequestAccess,
    correlation: PatchObserverCorrelationId,
) -> _TrustedDiagnosticHost:
    """Construct the trusted host only from policy-selected sealed truth."""
    if (
        type(plan) is not SealedPlan
        or type(access) is not DurableRequestAccess
        or type(correlation) is not PatchObserverCorrelationId
        or plan.binding.diagnostic_policy is None
        or PolicyDisclosure.DIAGNOSTIC_ASSOCIATION
        not in plan.binding.final.disclosures
    ):
        raise DiagnosticAssociationError(
            DiagnosticAssociationErrorCode.POLICY_INVALID
        )
    try:
        _validate_sealed_plan(plan)
    except Exception:
        raise DiagnosticAssociationError(
            DiagnosticAssociationErrorCode.POLICY_INVALID
        ) from None
    host = _TrustedDiagnosticHost(
        AudienceProjectionHost(store),
        plan,
        access,
        correlation,
        plan.binding.diagnostic_policy,
        token_urlsafe(24),
        None,
        False,
        None,
        False,
        None,
        0,
        None,
        None,
    )
    source = await host._source()
    _terminal_truth(source, access)
    return host


def _terminal_truth(
    source: object,
    access: object,
) -> DurableRequestSnapshot:
    """Validate complete committed durable truth before every diagnostic."""
    if (
        type(source) is not PatchAudienceProjectionSource
        or type(access) is not DurableRequestAccess
    ):
        raise DiagnosticAssociationError(
            DiagnosticAssociationErrorCode.TERMINAL_INVALID
        )
    assert isinstance(source, PatchAudienceProjectionSource)
    assert isinstance(access, DurableRequestAccess)
    try:
        snapshot = source._snapshot
        plan = source.plan
        terminal = snapshot.terminal
        identity = access.identity
        subject = plan.binding.subject
    except AttributeError:
        raise DiagnosticAssociationError(
            DiagnosticAssociationErrorCode.TERMINAL_INVALID
        ) from None
    if (
        type(snapshot) is not DurableRequestSnapshot
        or terminal is None
        or snapshot.pending is not None
        or snapshot.lifecycle is not LifecyclePhase.REQUEST_COMPLETED
        or snapshot.reservation.request_id != access.request_id
        or snapshot.reservation.identity != identity
        or identity.tenant_id != subject.tenant
        or identity.principal_id != subject.principal
        or identity.execution_id != plan.binding.request.execution_id
        or plan.binding.request.request_id != access.request_id
        or terminal.result.request_id != access.request_id
        or terminal.result.plan_id != plan.plan_id
        or terminal.result.status is not PatchStatus.COMMITTED
        or terminal.result.truth.mutation_state is not MutationState.COMMITTED
        or terminal.result.diagnostic is not None
        or terminal.pending_operation_id is not None
        or terminal.outbox.sequence != snapshot.event_cursor
        or terminal.outbox.correlation_id != source._access._correlation
    ):
        raise DiagnosticAssociationError(
            DiagnosticAssociationErrorCode.TERMINAL_INVALID
        )
    return snapshot


def _terminal_binding(source: PatchAudienceProjectionSource) -> bytes:
    """Return a private HMAC of complete canonical durable terminal truth."""
    return digest(
        _SNAPSHOT_BINDING_KEY,
        _canonical_terminal_snapshot(source.plan, source._snapshot),
        "sha256",
    )


def _canonical_terminal_snapshot(
    plan: SealedPlan, snapshot: DurableRequestSnapshot
) -> bytes:
    """Encode complete durable terminal facts without exposing a digest."""
    durable_plan = snapshot.plan
    terminal = snapshot.terminal
    if durable_plan is None or terminal is None:
        raise DiagnosticAssociationError(
            DiagnosticAssociationErrorCode.TERMINAL_INVALID
        )
    value = {
        "access": (
            snapshot.reservation.request_id.value,
            snapshot.reservation.identity.tenant_id.value,
            snapshot.reservation.identity.principal_id.value,
            snapshot.reservation.identity.execution_id.value,
            snapshot.reservation.identity.route_id.value,
            snapshot.reservation.identity.retransmission_key.value,
        ),
        "artifacts": tuple(
            (
                item.cursor.revision.value,
                item.artifact_id.value,
                item.state.value,
            )
            for item in snapshot.journal.artifacts
        ),
        "cancellation_requested": snapshot.cancellation_requested,
        "durable_plan": (
            durable_plan.plan_id.value,
            durable_plan.canonical_digest.value,
            durable_plan.fingerprint_digest.value,
            durable_plan.review_digest.value,
            durable_plan.context_id.value,
            durable_plan.workspace_id.value,
            durable_plan.domain_id.value,
            tuple(
                (item.step_id.value, item.lineage_id.value)
                for item in durable_plan.steps
            ),
        ),
        "event_cursor": snapshot.event_cursor.value,
        "journal_cursor": snapshot.journal.cursor.revision.value,
        "plan": (plan.plan_id.value, plan.binding.request_digest.value),
        "reservation_digest": snapshot.reservation.canonical_digest.value,
        "result": urlsafe_b64encode(encode_result(terminal.result)).decode(),
        "steps": tuple(
            (
                item.cursor.revision.value,
                item.step_id.value,
                item.lineage_id.value,
                item.state.value,
            )
            for item in snapshot.journal.steps
        ),
        "terminal_outbox": (
            terminal.outbox.event_id.value,
            terminal.outbox.request_id.value,
            terminal.outbox.sequence.value,
            terminal.outbox.lifecycle.value,
            terminal.outbox.correlation_id.value,
        ),
        "worker": (snapshot.worker_bound, snapshot.worker_reaped),
    }
    return dumps(
        value, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode()


def _capability_payload(
    capability_id: str,
    service_id: str,
    execution_id: str,
    subject: ExecutionSubject,
    policy_id: DiagnosticPolicyId,
    snapshot: DurableRequestSnapshot,
    binding: bytes,
) -> dict[str, str]:
    """Encode exact authenticated identity and durable association claims."""
    terminal = snapshot.terminal
    assert terminal is not None
    return {
        "association_id": "",
        "audience": _DIAGNOSTIC_AUDIENCE,
        "binding": _encode_binding(binding),
        "capability_id": capability_id,
        "execution_id": execution_id,
        "service_id": service_id,
        "kind": "capability",
        "plan_id": terminal.result.plan_id.value,
        "policy_id": policy_id.value,
        "principal": subject.principal.value,
        "receipt_id": "",
        "request_id": terminal.result.request_id.value,
        "run": subject.run.value,
        "session": subject.session.value,
        "task": subject.task.value,
        "tenant": subject.tenant.value,
        "agent": subject.agent.value,
        "version": "1",
    }


def _sign_payload(payload: dict[str, str]) -> bytes:
    """Sign canonical non-content claims at the trusted module root."""
    encoded = _canonical_payload(payload)
    return (
        urlsafe_b64encode(encoded)
        + b"."
        + urlsafe_b64encode(_ROOT_SIGNER.sign(encoded))
    )


def _verify_payload(token: object, kind: str) -> dict[str, str]:
    """Verify type and shape before decoding, coarsening malformed tokens."""
    code = _token_error_code(kind)
    try:
        if (
            type(token) is not bytes
            or type(kind) is not str
            or len(token) > 8192
        ):
            raise ValueError
        parts = token.split(b".")
        if len(parts) != 2 or not parts[0] or not parts[1]:
            raise ValueError
        encoded = b64decode(parts[0], altchars=b"-_", validate=True)
        signature = b64decode(parts[1], altchars=b"-_", validate=True)
        if not compare_digest(
            urlsafe_b64encode(encoded), parts[0]
        ) or not compare_digest(urlsafe_b64encode(signature), parts[1]):
            raise ValueError
        if not encoded or len(encoded) > 4096 or len(signature) != 64:
            raise ValueError
        _ROOT_VERIFIER.verify(signature, encoded)
        decoded: object = loads(encoded)
        if type(decoded) is not dict:
            raise ValueError
        payload: dict[str, str] = {}
        for key, value in decoded.items():
            if type(key) is not str or type(value) is not str:
                raise ValueError
            payload[key] = value
        if (
            payload.get("version") != "1"
            or payload.get("kind") != kind
            or encoded != _canonical_payload(payload)
        ):
            raise ValueError
        return payload
    except (InvalidSignature, ValueError, TypeError, UnicodeDecodeError):
        raise DiagnosticAssociationError(code) from None
    except Exception:
        raise DiagnosticAssociationError(code) from None


def _token_error_code(kind: str) -> DiagnosticAssociationErrorCode:
    """Return one fixed error class for a token family."""
    if kind == "capability":
        return DiagnosticAssociationErrorCode.CAPABILITY_INVALID
    if kind == "approval":
        return DiagnosticAssociationErrorCode.APPROVAL_INVALID
    return DiagnosticAssociationErrorCode.ASSOCIATION_INVALID


def _canonical_payload(payload: dict[str, str]) -> bytes:
    """Return one stable canonical signed-token encoding."""
    return dumps(
        payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode()


def _payload_string(payload: dict[str, str], name: str, kind: str) -> str:
    """Read one required bounded signed claim without raw token detail."""
    value = payload.get(name)
    if type(value) is not str or not value or len(value) > 128:
        raise DiagnosticAssociationError(_token_error_code(kind))
    return value


def _require_payload_subject(
    payload: dict[str, str], subject: ExecutionSubject, kind: str
) -> None:
    """Require signed claims to retain the exact sealed principal context."""
    if (
        _payload_string(payload, "principal", kind) != subject.principal.value
        or _payload_string(payload, "tenant", kind) != subject.tenant.value
        or _payload_string(payload, "run", kind) != subject.run.value
        or _payload_string(payload, "session", kind) != subject.session.value
        or _payload_string(payload, "task", kind) != subject.task.value
        or _payload_string(payload, "agent", kind) != subject.agent.value
    ):
        raise DiagnosticAssociationError(_token_error_code(kind))


def _require_payload_terminal(
    payload: dict[str, str],
    snapshot: DurableRequestSnapshot,
    binding: bytes,
    kind: str,
) -> None:
    """Require signed claims to match the fresh full durable snapshot."""
    terminal = snapshot.terminal
    assert terminal is not None
    if (
        _payload_string(payload, "request_id", kind)
        != terminal.result.request_id.value
        or _payload_string(payload, "plan_id", kind)
        != terminal.result.plan_id.value
        or _payload_string(payload, "binding", kind)
        != _encode_binding(binding)
    ):
        raise DiagnosticAssociationError(
            DiagnosticAssociationErrorCode.TERMINAL_INVALID
        )


def _encode_binding(value: bytes) -> str:
    """Encode a private HMAC without exposing a content digest."""
    return urlsafe_b64encode(value).decode()


def _sealed_probe_request(binding: bytes) -> _SealedReadOnlyProbeRequest:
    """Build the only executable diagnostic request inside this module."""
    return _SealedReadOnlyProbeRequest(
        DiagnosticCommandClass.READ_ONLY_PROBE,
        (),
        (),
        "<detached-read-only-probe>",
        binding,
    )


async def _read_only_probe(
    request: _SealedReadOnlyProbeRequest,
) -> DiagnosticSandboxOutcome:
    """Check detached binding integrity without external execution evidence."""
    request.__post_init__()
    await sleep(0)
    probe = sha256(
        b"avalan.phase12.read-only-probe\x00" + request.detached_snapshot
    ).digest()
    return (
        DiagnosticSandboxOutcome.SUCCEEDED
        if len(probe) == 32
        else DiagnosticSandboxOutcome.FAILED
    )

"""Exercise trusted Phase 12 projection delivery boundaries."""

import dataclasses
from asyncio import run
from collections.abc import Iterator
from copy import copy, deepcopy
from gc import collect
from inspect import getsource
from json import dumps as json_dumps
from json import loads as json_loads
from pathlib import Path
from pickle import dumps, loads
from runpy import run_path
from typing import Mapping
from weakref import ref

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
)

import avalan.patch.projection as projection_module
import avalan.patch.projection_codec as codec_module
from avalan.patch.domain import (
    ArtifactState,
    Audience,
    ByteSize,
    CommitTruth,
    ErrorStage,
    ExpiryTick,
    LifecyclePhase,
    LineageState,
    MutationState,
    PatchDiagnostic,
    PatchErrorCode,
    PatchEventId,
    PatchObserverCorrelationId,
    PatchPublicCorrelationId,
    PatchRequestId,
    PatchResult,
    PatchStatus,
    PostconditionState,
    RequestedEffectOccurrence,
    Retryability,
    SequenceNumber,
    WorkspaceChange,
)
from avalan.patch.durable_store import (
    DurableOutboxRecord,
    DurableTerminalRecord,
)
from avalan.patch.policy import (
    PolicyDisclosure,
    SealedPlan,
    cleanup_sealed_authorities,
)
from avalan.patch.projection import (
    ModelProjectionAuthority,
    PatchProjectionSource,
    ProjectionError,
    ProjectionOutputLimit,
    ProjectionPayload,
    create_approver_projection_boundary,
    create_audit_projection_boundary,
    create_model_projection_boundary,
)

_PHASE_FIVE = run_path(
    str(Path("tests/patch/phase_5_contract_test.py").resolve())
)
_SEAL_CLEANUP_TICK = ExpiryTick(2**63 - 1)


@pytest.fixture(autouse=True)
def _phase_twelve_seal_lifecycle() -> Iterator[None]:
    """Release test-local plan seals at each Phase 12 lifecycle boundary."""
    cleanup_sealed_authorities(_SEAL_CLEANUP_TICK)
    yield
    cleanup_sealed_authorities(_SEAL_CLEANUP_TICK)


async def _source(
    disclosures: frozenset[PolicyDisclosure],
) -> PatchProjectionSource:
    """Return one sealed complete plan with matching committed truth."""
    sealed_plan = await _PHASE_FIVE["_sealed_plan"](disclosures=disclosures)
    result = PatchResult(
        1,
        sealed_plan.binding.request.request_id,
        sealed_plan.plan_id,
        LifecyclePhase.REQUEST_COMPLETED,
        PatchStatus.COMMITTED,
        CommitTruth(
            MutationState.COMMITTED,
            LineageState.COMMITTED,
            RequestedEffectOccurrence.TRUE,
            ArtifactState.CLEANED,
            WorkspaceChange.CHANGED,
            True,
            PostconditionState.ESTABLISHED,
        ),
        None,
    )
    terminal = DurableTerminalRecord(
        result,
        DurableOutboxRecord(
            PatchEventId.new(),
            result.request_id,
            SequenceNumber(1),
            LifecyclePhase.REQUEST_COMPLETED,
            PatchObserverCorrelationId.new(),
        ),
        None,
    )
    return PatchProjectionSource(sealed_plan, terminal)


def _full_disclosures() -> frozenset[PolicyDisclosure]:
    """Return the test policy that authorizes every Phase 12 audience."""
    return frozenset(
        (
            PolicyDisclosure.MODEL_DIFF,
            PolicyDisclosure.MODEL_METADATA,
            PolicyDisclosure.COMPLETE_REVIEW,
        )
    )


def _lower_delivery(value: bytes) -> Mapping[str, object]:
    """Return untrusted lower-consumer JSON data from delivery bytes."""
    decoded = json_loads(value.decode("utf-8"))
    assert isinstance(decoded, dict)
    assert value == json_dumps(
        decoded,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return decoded


def _payload_mapping(delivery: Mapping[str, object]) -> Mapping[str, object]:
    """Return one lower delivery payload for test assertions."""
    payload = delivery["payload"]
    assert isinstance(payload, Mapping)
    return payload


def _diff_mapping(delivery: Mapping[str, object]) -> Mapping[str, object]:
    """Return one lower delivery diff payload for test assertions."""
    diff = _payload_mapping(delivery)["diff"]
    assert isinstance(diff, Mapping)
    return diff


def _host_model_artifact(
    payload: ProjectionPayload | None = None,
) -> projection_module._HostProjectionArtifact:
    """Return one trusted model artifact for direct verifier regressions."""
    signing_key = Ed25519PrivateKey.generate()
    correlation_id = "public_" + "a" * 16
    source_digest = "a" * 64
    terminal_digest = "b" * 64
    issuer_id = "issuer_" + "a" * 16
    trusted_payload = {} if payload is None else payload
    envelope = projection_module._encode_envelope(
        "model",
        correlation_id,
        source_digest,
        terminal_digest,
        issuer_id,
        signing_key,
        trusted_payload,
    )
    receipt = projection_module._verification_receipt(
        signing_key,
        "model",
        correlation_id,
        source_digest,
        terminal_digest,
        issuer_id,
    )
    return projection_module._HostProjectionArtifact(
        "model", envelope, receipt
    )


def _host_verify(
    artifact: projection_module._HostProjectionArtifact,
) -> Mapping[str, object]:
    """Run the trusted host verifier for direct negative regressions."""
    verified = projection_module._HOST_PROJECTION_ADAPTER.deliver(artifact)
    return {
        "audience": verified.audience,
        "correlation_id": verified.correlation_id,
        "source_digest": verified.source_digest,
        "terminal_digest": verified.terminal_digest,
        "issuer_id": verified.issuer_id,
        "payload": verified.payload,
    }


def _replace_artifact(
    artifact: projection_module._HostProjectionArtifact,
    *,
    envelope: bytes | None = None,
    receipt: bytes | None = None,
    audience: str | None = None,
) -> projection_module._HostProjectionArtifact:
    """Return one trusted-test artifact variant without lower delivery use."""
    return dataclasses.replace(
        artifact,
        envelope=artifact.envelope if envelope is None else envelope,
        receipt=artifact.receipt if receipt is None else receipt,
        audience=artifact.audience if audience is None else audience,
    )


def _ordinary_graph_contains(
    value: object,
    canary: bytes,
    seen: set[int] | None = None,
) -> bool:
    """Return whether ordinary object data reachability contains one canary."""
    active_seen = set() if seen is None else seen
    value_id = id(value)
    if value_id in active_seen:
        return False
    active_seen.add(value_id)
    if type(value) is bytes:
        return value == canary
    if isinstance(value, Mapping):
        return any(
            _ordinary_graph_contains(item, canary, active_seen)
            for item in value.values()
        )
    if isinstance(value, tuple):
        return any(
            _ordinary_graph_contains(item, canary, active_seen)
            for item in value
        )
    for owner in type(value).__mro__:
        slots = getattr(owner, "__slots__", ())
        names = (slots,) if type(slots) is str else slots
        for name in names:
            if name == "__weakref__" or not hasattr(value, name):
                continue
            if _ordinary_graph_contains(
                getattr(value, name), canary, active_seen
            ):
                return True
    return False


def test_patch_phase_12_requirements() -> None:
    """Deliver only host-verified detached audience bytes to consumers."""
    source = run(_source(_full_disclosures()))
    model_boundary = create_model_projection_boundary(source)
    approver_boundary = create_approver_projection_boundary(source)
    audit_boundary = create_audit_projection_boundary(source)
    model_delivery = model_boundary.project(
        model_boundary.authority(),
        ProjectionOutputLimit(source.plan.review.diff.size),
    )
    approver_delivery = approver_boundary.project(
        approver_boundary.authority()
    )
    audit_delivery = audit_boundary.project(audit_boundary.authority())
    model = _lower_delivery(model_delivery)
    approver = _lower_delivery(approver_delivery)
    audit = _lower_delivery(audit_delivery)

    assert type(model_delivery) is bytes
    assert type(approver_delivery) is bytes
    assert type(audit_delivery) is bytes
    assert model["audience"] == "model"
    assert approver["audience"] == "approver"
    assert audit["audience"] == "audit"
    assert str(model["correlation_id"]).startswith("public_")
    assert model["correlation_id"] != audit["correlation_id"]
    assert _diff_mapping(model)["complete"] is True
    assert _diff_mapping(model)["reason"] == "complete"
    assert _diff_mapping(model)["omitted_bytes"] == 0
    assert _diff_mapping(approver)["redacted"] is True
    assert _diff_mapping(audit)["redacted"] is True
    assert "review" not in _payload_mapping(model)
    assert "review" not in _payload_mapping(audit)
    assert "review" in _payload_mapping(approver)


def test_phase_12_model_diff_and_review_are_independent() -> None:
    """Bound model delivery without changing canonical review or truth."""
    source = run(_source(_full_disclosures()))
    model_boundary = create_model_projection_boundary(source)
    approver_boundary = create_approver_projection_boundary(source)
    fingerprint = source.plan.fingerprint
    plan = source.plan
    terminal = source.terminal
    review = source.plan.review
    initial_approver = approver_boundary.project(approver_boundary.authority())
    complete_model = _lower_delivery(
        model_boundary.project(
            model_boundary.authority(),
            ProjectionOutputLimit(source.plan.review.diff.size),
        )
    )
    model = _lower_delivery(
        model_boundary.project(
            model_boundary.authority(), ProjectionOutputLimit(ByteSize(1))
        )
    )
    approver = _lower_delivery(
        approver_boundary.project(approver_boundary.authority())
    )
    diff = _diff_mapping(model)
    complete = source.plan.review.diff.diff._value

    assert diff["truncated"] is True
    assert diff["complete"] is False
    assert diff["redacted"] is False
    assert diff["reason"] == "output_limit"
    assert len(str(diff["content"]).encode()) <= 1
    assert diff["omitted_bytes"] == len(complete) - len(
        str(diff["content"]).encode()
    )
    assert source.plan is plan
    assert source.plan.fingerprint is fingerprint
    assert source.terminal is terminal
    assert source.plan.review is review
    assert approver_boundary.project(approver_boundary.authority()) == (
        initial_approver
    )
    complete_payload = _payload_mapping(complete_model)
    truncated_payload = _payload_mapping(model)
    assert {
        key: value for key, value in truncated_payload.items() if key != "diff"
    } == {
        key: value for key, value in complete_payload.items() if key != "diff"
    }
    review = _payload_mapping(approver)["review"]
    assert isinstance(review, Mapping)
    review_diff = review["diff"]
    assert isinstance(review_diff, Mapping)
    complete_diff = review_diff["diff"]
    assert isinstance(complete_diff, Mapping)
    assert complete_diff["value"] == complete.decode()


def test_phase_12_host_verifier_rejects_tamper_and_substitution() -> None:
    """Reject malformed receipts and envelopes before lower delivery exists."""
    artifact = _host_model_artifact({"safe": ("value",)})
    verified = _host_verify(artifact)
    assert verified["audience"] == "model"
    assert isinstance(verified["payload"], Mapping)
    envelope = artifact.envelope
    receipt = artifact.receipt

    with pytest.raises(ProjectionError, match="host artifact"):
        getattr(projection_module._HOST_PROJECTION_ADAPTER, "deliver")(
            object()
        )
    with pytest.raises(ProjectionError, match="receipt"):
        _host_verify(_replace_artifact(artifact, audience="audit"))
    with pytest.raises(ProjectionError, match="invalid"):
        _host_verify(_replace_artifact(artifact, envelope=b"not-json"))
    with pytest.raises(ProjectionError, match="invalid"):
        _host_verify(_replace_artifact(artifact, receipt=b"{}"))
    with pytest.raises(ProjectionError, match="schema"):
        _host_verify(_replace_artifact(artifact, envelope=b"{}"))
    with pytest.raises(ProjectionError, match="invalid"):
        _host_verify(_replace_artifact(artifact, envelope=b"x" * 1_048_577))
    with pytest.raises(ProjectionError, match="invalid"):
        _host_verify(_replace_artifact(artifact, receipt=b"x" * 4_097))
    with pytest.raises(ProjectionError, match="invalid"):
        _host_verify(_replace_artifact(artifact, envelope=b'{"x":1,"x":1}'))
    with pytest.raises(ProjectionError, match="invalid"):
        _host_verify(_replace_artifact(artifact, envelope=b'{"payload":NaN}'))
    with pytest.raises(ProjectionError, match="invalid"):
        _host_verify(_replace_artifact(artifact, envelope=b" " + envelope))
    with pytest.raises(ProjectionError, match="invalid"):
        _host_verify(_replace_artifact(artifact, receipt=receipt + b" "))

    raw = json_loads(envelope.decode())
    raw["signature"] = "0" * 128
    with pytest.raises(ProjectionError, match="signature"):
        _host_verify(
            _replace_artifact(
                artifact,
                envelope=json_dumps(
                    raw, separators=(",", ":"), sort_keys=True
                ).encode(),
            )
        )
    raw = json_loads(envelope.decode())
    raw["signature"] = "A" * 128
    with pytest.raises(ProjectionError, match="signature"):
        _host_verify(
            _replace_artifact(
                artifact,
                envelope=json_dumps(
                    raw, separators=(",", ":"), sort_keys=True
                ).encode(),
            )
        )
    raw = json_loads(envelope.decode())
    raw["payload"] = []
    with pytest.raises(ProjectionError, match="payload"):
        _host_verify(
            _replace_artifact(
                artifact,
                envelope=json_dumps(
                    raw, separators=(",", ":"), sort_keys=True
                ).encode(),
            )
        )
    raw = json_loads(envelope.decode())
    raw["correlation_id"] = "not-public"
    with pytest.raises(ProjectionError, match="header"):
        _host_verify(
            _replace_artifact(
                artifact,
                envelope=json_dumps(
                    raw, separators=(",", ":"), sort_keys=True
                ).encode(),
            )
        )
    raw = json_loads(envelope.decode())
    raw["schema_version"] = 0
    with pytest.raises(ProjectionError, match="schema"):
        _host_verify(
            _replace_artifact(
                artifact,
                envelope=json_dumps(
                    raw, separators=(",", ":"), sort_keys=True
                ).encode(),
            )
        )
    raw = json_loads(envelope.decode())
    raw["audience"] = True
    with pytest.raises(ProjectionError, match="schema"):
        _host_verify(
            _replace_artifact(
                artifact,
                envelope=json_dumps(
                    raw, separators=(",", ":"), sort_keys=True
                ).encode(),
            )
        )
    raw = json_loads(envelope.decode())
    raw["source_digest"] = "c" * 64
    with pytest.raises(ProjectionError, match="receipt"):
        _host_verify(
            _replace_artifact(
                artifact,
                envelope=json_dumps(
                    raw, separators=(",", ":"), sort_keys=True
                ).encode(),
            )
        )
    receipt_raw = json_loads(receipt.decode())
    receipt_raw["public_key"] = "00"
    with pytest.raises(ProjectionError, match="key"):
        _host_verify(
            _replace_artifact(
                artifact,
                receipt=json_dumps(
                    receipt_raw, separators=(",", ":"), sort_keys=True
                ).encode(),
            )
        )
    receipt_raw = json_loads(receipt.decode())
    receipt_raw["schema_version"] = 0
    with pytest.raises(ProjectionError, match="schema"):
        _host_verify(
            _replace_artifact(
                artifact,
                receipt=json_dumps(
                    receipt_raw, separators=(",", ":"), sort_keys=True
                ).encode(),
            )
        )
    receipt_raw = json_loads(receipt.decode())
    receipt_raw["source_digest"] = "f" * 64
    with pytest.raises(ProjectionError, match="signature"):
        _host_verify(
            _replace_artifact(
                artifact,
                receipt=json_dumps(
                    receipt_raw, separators=(",", ":"), sort_keys=True
                ).encode(),
            )
        )
    with pytest.raises(ProjectionError, match="payload"):
        getattr(projection_module, "_host_freeze_payload")({1: "invalid"})
    with pytest.raises(ProjectionError, match="payload"):
        projection_module._host_freeze_value(object())


def test_phase_12_lower_codec_cannot_influence_host_delivery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep lower codec mutation and parsing outside host acceptance."""
    source = run(_source(frozenset()))
    raw_diff = source.plan.review.diff.diff._value
    source_reference = ref(source)
    boundary = create_model_projection_boundary(source)
    authority = boundary.authority()
    assert "_HOST_PROJECTION_ADAPTER.deliver" in getsource(
        projection_module.ModelProjectionBoundary.project
    )
    assert not any(
        name.startswith(("decode_", "verify_", "_host_"))
        for name in codec_module.__dict__
    )
    before = boundary.project(
        authority, ProjectionOutputLimit(source.plan.review.diff.size)
    )
    monkeypatch.setattr(
        codec_module,
        "decode_model_projection",
        lambda *args: b"attacker-controlled",
        raising=False,
    )
    monkeypatch.setattr(
        codec_module,
        "PROJECTION_ENVELOPE_SCHEMA_VERSION",
        999,
    )
    after = boundary.project(
        authority, ProjectionOutputLimit(source.plan.review.diff.size)
    )
    delivered = (after,)

    assert before == after
    assert type(after) is bytes
    assert not _ordinary_graph_contains(after, raw_diff)
    assert not hasattr(after, "root")
    assert not hasattr(after, "receipt")
    assert not hasattr(after, "adapter")
    assert not hasattr(after, "source")
    assert not hasattr(after, "terminal")
    assert not hasattr(after, "authority")
    assert not any(
        item is projection_module
        or isinstance(
            item, (PatchProjectionSource, SealedPlan, DurableTerminalRecord)
        )
        for item in codec_module.__dict__.values()
    )
    assert all(
        not hasattr(item, "__globals__")
        or item.__globals__ is not projection_module.__dict__
        for item in codec_module.__dict__.values()
    )

    del delivered
    del authority
    del boundary
    del source
    collect()
    assert source_reference() is None
    assert _lower_delivery(after)["audience"] == "model"


def test_phase_12_deliveries_are_immutable_and_never_host_reaccepted() -> None:
    """Keep lower delivery bytes immutable and outside the host input type."""
    source = run(_source(_full_disclosures()))
    boundary = create_model_projection_boundary(source)
    delivery = boundary.project(
        boundary.authority(),
        ProjectionOutputLimit(source.plan.review.diff.size),
    )

    assert not dataclasses.is_dataclass(delivery)
    with pytest.raises(TypeError):
        getattr(dataclasses, "asdict")(delivery)
    assert copy(delivery) is delivery
    assert deepcopy(delivery) is delivery
    assert loads(dumps(delivery)) == delivery
    with pytest.raises(AttributeError):
        object.__setattr__(delivery, "audience", "approver")
    with pytest.raises(TypeError):
        object.__new__(bytes)
    with pytest.raises(ProjectionError, match="host artifact"):
        getattr(projection_module._HOST_PROJECTION_ADAPTER, "deliver")(
            delivery
        )

    class ForgedDelivery(bytes):
        """Attempt to subclass one detached lower delivery at runtime."""

    with pytest.raises(ProjectionError, match="host artifact"):
        getattr(projection_module._HOST_PROJECTION_ADAPTER, "deliver")(
            ForgedDelivery(delivery)
        )


def test_phase_12_trusted_authorities_and_sources_fail_closed() -> None:
    """Reject reconstructed witnesses, sources, disclosures, and limits."""
    source = run(_source(_full_disclosures()))
    first = create_model_projection_boundary(source)
    second = create_model_projection_boundary(source)
    approver = create_approver_projection_boundary(source)
    audit = create_audit_projection_boundary(source)
    authority = first.authority()
    output_limit = ProjectionOutputLimit(source.plan.review.diff.size)
    forged = ModelProjectionAuthority(
        object(),
        authority.correlation_id,
        authority.source_digest,
        authority.terminal_digest,
    )

    with pytest.raises(ProjectionError, match="not issued"):
        first.project(forged, output_limit)
    with pytest.raises(ProjectionError, match="not issued"):
        second.project(authority, output_limit)
    original_correlation = authority.correlation_id
    object.__setattr__(
        authority, "correlation_id", PatchPublicCorrelationId.new()
    )
    with pytest.raises(ProjectionError, match="not issued"):
        first.project(authority, output_limit)
    object.__setattr__(authority, "correlation_id", original_correlation)
    with pytest.raises(ProjectionError, match="not issued"):
        getattr(approver, "project")(authority)
    with pytest.raises(ProjectionError, match="not issued"):
        getattr(audit, "project")(authority)
    assert not hasattr(first, "verification_receipt")
    assert not hasattr(first, "approver_authority")
    assert type(first.project(authority, output_limit)) is bytes

    diagnostic = PatchDiagnostic(
        stage=ErrorStage.COMMIT,
        code=PatchErrorCode.COMMIT_FAILED,
        retryability=Retryability.NOT_RETRYABLE,
    )
    substituted = PatchResult(
        1,
        PatchRequestId("request_" + "b" * 16),
        source.result.plan_id,
        LifecyclePhase.REQUEST_COMPLETED,
        PatchStatus.COMMIT_FAILED,
        CommitTruth(
            MutationState.NOT_COMMITTED,
            LineageState.NOT_COMMITTED,
            RequestedEffectOccurrence.FALSE,
            ArtifactState.ABSENT,
            WorkspaceChange.UNCHANGED,
            True,
            PostconditionState.UNKNOWN,
        ),
        diagnostic,
    )
    terminal = DurableTerminalRecord(
        substituted,
        DurableOutboxRecord(
            PatchEventId.new(),
            substituted.request_id,
            SequenceNumber(1),
            LifecyclePhase.REQUEST_COMPLETED,
            PatchObserverCorrelationId.new(),
        ),
        None,
    )
    with pytest.raises(ProjectionError, match="does not match"):
        PatchProjectionSource(source.plan, terminal)
    with pytest.raises(ProjectionError, match="output limit"):
        ProjectionOutputLimit(ByteSize(0))
    hidden = create_model_projection_boundary(run(_source(frozenset())))
    hidden_diff = _diff_mapping(
        _lower_delivery(
            hidden.project(
                hidden.authority(), ProjectionOutputLimit(ByteSize(1))
            )
        )
    )
    assert hidden_diff["redacted"] is True
    assert hidden_diff["omitted_bytes"] is None
    no_review = create_approver_projection_boundary(run(_source(frozenset())))
    with pytest.raises(ProjectionError, match="complete review"):
        no_review.project(no_review.authority())


def test_phase_12_trusted_construction_and_primitives_reject_duplication() -> (
    None
):
    """Reject copied trusted handles and preserve primitive conversion."""
    source = run(_source(_full_disclosures()))
    boundary = create_model_projection_boundary(source)
    authority = boundary.authority()
    signing_key = Ed25519PrivateKey.generate()

    assert repr(source) == "PatchProjectionSource(<redacted>)"
    assert repr(authority) == "ModelProjectionAuthority(<opaque>)"
    for value in (source, boundary, authority):
        with pytest.raises(ProjectionError):
            copy(value)
        with pytest.raises(ProjectionError):
            deepcopy(value)
        with pytest.raises(ProjectionError):
            dumps(value)
        with pytest.raises(ProjectionError):
            value.__reduce__()
    with pytest.raises(ProjectionError, match="authority is invalid"):
        ModelProjectionAuthority(
            object(), authority.correlation_id, "invalid", "invalid"
        )
    with pytest.raises(ProjectionError, match="source is invalid"):
        getattr(projection_module, "_terminal_payload")(
            object(), Audience.MODEL
        )
    object.__setattr__(source, "plan", object())
    with pytest.raises(ProjectionError, match="source is invalid"):
        create_audit_projection_boundary(source)
    with pytest.raises(ProjectionError, match="output limit"):
        getattr(boundary, "project")(authority, object())
    with pytest.raises(ProjectionError, match="complete review payload"):
        projection_module._review_payload(())
    with pytest.raises(ProjectionError, match="complete review contains"):
        projection_module._review_value(object())
    with pytest.raises(ProjectionError, match="source contains"):
        projection_module._canonical_value(object())
    assert projection_module._utf8_prefix("é".encode(), 1) == b""
    assert projection_module._utf8_prefix(b"plain", 2) == b"pl"
    with pytest.raises(ProjectionError, match="exceeds"):
        projection_module._encode_envelope(
            "model",
            "public_" + "a" * 16,
            "a" * 64,
            "b" * 64,
            "issuer_" + "a" * 16,
            signing_key,
            {"large": "x" * 1_048_576},
        )
    with pytest.raises(ProjectionError, match="header"):
        projection_module._encode_envelope(
            "invalid",
            "public_" + "a" * 16,
            "a" * 64,
            "b" * 64,
            "issuer_" + "a" * 16,
            signing_key,
            {},
        )

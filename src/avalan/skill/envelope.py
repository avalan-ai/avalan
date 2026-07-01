from .contract import SkillStatus
from .entities import (
    SkillDiagnosticInfo,
    SkillMatchResult,
    SkillMetadata,
    SkillModelValue,
    SkillProvenance,
    SkillReadCursor,
    SkillRegistryVersion,
    SkillResourceContent,
    model_dict,
)

from dataclasses import dataclass
from typing import TypeAlias

SkillEnvelopeItem: TypeAlias = SkillMetadata | SkillMatchResult


@dataclass(frozen=True, slots=True, kw_only=True)
class SkillResponseEnvelope:
    status: SkillStatus
    registry_version: SkillRegistryVersion
    items: tuple[SkillEnvelopeItem, ...] = ()
    content: SkillResourceContent | None = None
    diagnostics: tuple[SkillDiagnosticInfo, ...] = ()
    next_cursor: SkillReadCursor | None = None
    provenance: tuple[SkillProvenance, ...] = ()

    def __post_init__(self) -> None:
        assert isinstance(self.status, SkillStatus)
        assert isinstance(self.registry_version, SkillRegistryVersion)
        _assert_items(self.items)
        if self.content is not None:
            assert isinstance(self.content, SkillResourceContent)
            assert not self.items, "envelope must contain items or content"
        _assert_diagnostics(self.diagnostics)
        if self.status is not SkillStatus.OK:
            assert self.diagnostics, "non-ok envelopes require diagnostics"
            assert self.diagnostics[0].status == self.status
        if self.next_cursor is not None:
            assert isinstance(self.next_cursor, SkillReadCursor)
            assert self.content is not None, "cursor requires read content"
            assert (
                self.next_cursor.registry_version == self.registry_version
            ), "cursor registry version must match the envelope"
        _assert_provenance(self.provenance)
        for provenance in self.provenance:
            assert (
                provenance.registry_version == self.registry_version
            ), "provenance registry version must match the envelope"
        if self.content is not None:
            assert self.provenance, "read envelopes require provenance"
            if self.content.truncated:
                assert (
                    self.next_cursor is not None
                ), "truncated read envelopes require a cursor"
            else:
                assert (
                    self.next_cursor is None
                ), "untruncated read envelopes must not include a cursor"
            if self.next_cursor is not None:
                assert _cursor_matches_content(
                    self.next_cursor, self.content
                ), "cursor must continue the read content"
            assert any(
                _provenance_matches_content(provenance, self.content)
                for provenance in self.provenance
            ), "read provenance must match content"

    def as_model_dict(self) -> dict[str, SkillModelValue]:
        value: dict[str, object] = {
            "status": self.status.value,
            "registry_version": self.registry_version.as_model_value(),
            "diagnostics": tuple(
                diagnostic.as_model_dict() for diagnostic in self.diagnostics
            ),
            "provenance": tuple(
                provenance.as_model_dict() for provenance in self.provenance
            ),
        }
        if self.items:
            value["items"] = tuple(_model_item(item) for item in self.items)
        if self.content is not None:
            value["content"] = self.content.as_model_dict()
        if self.next_cursor is not None:
            value["next_cursor"] = self.next_cursor.as_model_value()
        return model_dict(value)


def _model_item(item: SkillEnvelopeItem) -> dict[str, SkillModelValue]:
    if isinstance(item, SkillMetadata):
        return item.as_model_dict()
    assert isinstance(item, SkillMatchResult)
    return item.as_model_dict()


def _assert_items(values: tuple[SkillEnvelopeItem, ...]) -> None:
    assert isinstance(values, tuple), "items must be a tuple"
    for value in values:
        assert isinstance(value, SkillMetadata | SkillMatchResult)


def _assert_diagnostics(values: tuple[SkillDiagnosticInfo, ...]) -> None:
    assert isinstance(values, tuple), "diagnostics must be a tuple"
    for value in values:
        assert isinstance(value, SkillDiagnosticInfo)


def _assert_provenance(values: tuple[SkillProvenance, ...]) -> None:
    assert isinstance(values, tuple), "provenance must be a tuple"
    for value in values:
        assert isinstance(value, SkillProvenance)


def _provenance_matches_content(
    provenance: SkillProvenance, content: SkillResourceContent
) -> bool:
    return (
        provenance.source_label == content.handle.source_label
        and provenance.skill_id == content.handle.skill_id
        and provenance.resource_id == content.handle.resource_id
    )


def _cursor_matches_content(
    cursor: SkillReadCursor, content: SkillResourceContent
) -> bool:
    return (
        cursor.source_label == content.handle.source_label
        and cursor.skill_id == content.handle.skill_id
        and cursor.resource_id == content.handle.resource_id
        and cursor.offset_bytes == content.end_byte
    )

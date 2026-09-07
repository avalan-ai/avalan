"""Materialize only explicit scalar bindings for one scheduled slot."""

from .definition import (
    BindingSource,
    JsonValue,
    TriggerInput,
    freeze_value,
    pointer_parts,
    utc_text,
)
from .records import OwnerScopeId, TriggerDefinition, occurrence_id

from collections.abc import Mapping
from datetime import datetime


def bind_input(
    value: TriggerInput,
    definition: TriggerDefinition,
    scheduled_at: datetime,
) -> JsonValue:
    """Return an immutable input snapshot with validated leaves replaced.

    Preserve all application strings literally. The caller must decrypt the
    registered input and validate the resulting task submission separately.
    """
    assert isinstance(definition, TriggerDefinition)
    return bind_values(
        value,
        owner=definition.owner_scope_id,
        trigger_id=definition.trigger_id,
        revision=definition.revision,
        scheduled_at=scheduled_at,
    )


def bind_values(
    value: TriggerInput,
    *,
    owner: OwnerScopeId,
    trigger_id: str,
    revision: int,
    scheduled_at: datetime,
) -> JsonValue:
    """Bind an allocated registration identity without inventing a record."""
    assert isinstance(value, TriggerInput)
    sources: Mapping[BindingSource, str | int] = {
        BindingSource.SCHEDULED_AT: utc_text(scheduled_at),
        BindingSource.OCCURRENCE_ID: occurrence_id(
            owner, trigger_id, revision, scheduled_at
        ),
        BindingSource.TRIGGER_ID: trigger_id,
        BindingSource.TRIGGER_REVISION: revision,
    }
    replacements = {
        pointer_parts(binding.path): sources[binding.source]
        for binding in value.bindings
    }

    def visit(item: JsonValue, path: tuple[str, ...]) -> JsonValue:
        if path in replacements:
            return replacements[path]
        if isinstance(item, Mapping):
            return freeze_value(
                {
                    key: visit(child, (*path, key))
                    for key, child in item.items()
                }
            )
        if isinstance(item, tuple | list):
            return tuple(
                visit(child, (*path, str(index)))
                for index, child in enumerate(item)
            )
        return item

    return visit(freeze_value(value.value), ())

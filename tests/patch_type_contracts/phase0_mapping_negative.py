"""Reject free-form mappings at the patch strict boundary."""


def reject_free_form_mapping() -> dict[str, object]:
    """Keep one deliberately unclosed mapping annotation for the fixture."""
    return {"untrusted": "value"}

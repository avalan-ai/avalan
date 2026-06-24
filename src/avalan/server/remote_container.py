from .authority import REMOTE_CONTAINER_PROFILE_SELECTOR_KEYS
from .container_policy import (
    RemoteContainerRequestError,
    remote_container_policy_from_state,
    validate_remote_container_arguments,
)

from collections.abc import Mapping

from fastapi import HTTPException, Request


async def validate_remote_container_profile_selection(
    request: Request,
) -> None:
    """Validate an operator-exposed remote container profile selector."""
    payload = await _json_payload(request)
    if not isinstance(payload, Mapping):
        return
    arguments = _profile_selector_arguments(payload)
    if not arguments:
        return
    try:
        container_request = validate_remote_container_arguments(
            arguments,
            policy=remote_container_policy_from_state(request.app.state),
        )
    except RemoteContainerRequestError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if container_request.profile is not None:
        request.state.remote_container_profile = container_request.profile


async def _json_payload(request: Request) -> object:
    try:
        return await request.json()
    except ValueError:
        return None


def _profile_selector_arguments(
    payload: Mapping[object, object],
) -> dict[str, object]:
    arguments: dict[str, object] = {}
    for key in REMOTE_CONTAINER_PROFILE_SELECTOR_KEYS:
        if key not in payload:
            continue
        value = payload[key]
        if key == "container" and not _is_container_profile_selector_value(
            value
        ):
            continue
        arguments[key] = value
    return arguments


def _is_container_profile_selector_value(value: object) -> bool:
    return (
        isinstance(value, Mapping)
        and "profile" in value
        and "profiles" not in value
    )

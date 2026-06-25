from .authority import (
    REMOTE_CONTAINER_PROFILE_SELECTOR_KEYS,
    reject_remote_runtime_authority_extra_fields,
)
from .container_policy import (
    RemoteContainerRequestError,
    remote_container_policy_from_state,
    validate_remote_container_arguments,
)

from collections.abc import Mapping

from fastapi import HTTPException, Request

_OPENAI_COMPATIBLE_REQUEST_FIELDS = frozenset(
    {
        "frequency_penalty",
        "input",
        "instructions",
        "logit_bias",
        "logprobs",
        "max_completion_tokens",
        "max_output_tokens",
        "max_tokens",
        "messages",
        "metadata",
        "model",
        "n",
        "parallel_tool_calls",
        "presence_penalty",
        "reasoning",
        "reasoning_effort",
        "response_format",
        "service_tier",
        "stop",
        "store",
        "stream",
        "stream_options",
        "temperature",
        "text",
        "tool_choice",
        "tools",
        "top_logprobs",
        "top_p",
        "truncation",
        "user",
    }
)


async def validate_remote_container_profile_selection(
    request: Request,
) -> None:
    """Validate remote authority fields and container profile selectors."""
    payload = await _json_payload(request)
    if not isinstance(payload, Mapping):
        return
    arguments = _profile_selector_arguments(payload)
    try:
        container_request = (
            validate_remote_container_arguments(
                arguments,
                policy=remote_container_policy_from_state(request.app.state),
            )
            if arguments
            else None
        )
        reject_remote_runtime_authority_extra_fields(
            payload,
            allowed_fields=_OPENAI_COMPATIBLE_REQUEST_FIELDS,
            allow_container_profile_selector=True,
            path="request",
        )
    except (RemoteContainerRequestError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if container_request is not None and container_request.profile is not None:
        # Server isolation remains operator-managed. The remote selector is
        # validation-only and records the approved profile for observability.
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

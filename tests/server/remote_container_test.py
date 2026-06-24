from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase

from fastapi import HTTPException

from avalan.server.container_policy import RemoteContainerRequestPolicy
from avalan.server.remote_container import (
    validate_remote_container_profile_selection,
)


class RemoteContainerProfileSelectionTestCase(IsolatedAsyncioTestCase):
    async def test_ignores_non_mapping_json_payload(self) -> None:
        request = _Request(["not", "an", "object"])

        await validate_remote_container_profile_selection(request)

        self.assertFalse(hasattr(request.state, "remote_container_profile"))

    async def test_ignores_invalid_json_payload(self) -> None:
        request = _Request(ValueError("invalid json"))

        await validate_remote_container_profile_selection(request)

        self.assertFalse(hasattr(request.state, "remote_container_profile"))

    async def test_allows_exposed_camel_case_profile_selector(self) -> None:
        request = _Request({"containerProfile": "workspace-readonly"})
        request.app.state.remote_container_policy = (
            RemoteContainerRequestPolicy(
                exposed_profiles=("workspace-readonly",)
            )
        )

        await validate_remote_container_profile_selection(request)

        self.assertEqual(
            request.state.remote_container_profile,
            "workspace-readonly",
        )

    async def test_rejects_unexposed_profile_selector(self) -> None:
        request = _Request({"container_profile": "workspace-readonly"})

        with self.assertRaises(HTTPException) as exc:
            await validate_remote_container_profile_selection(request)

        self.assertEqual(exc.exception.status_code, 400)
        self.assertIn("not exposed", str(exc.exception.detail))


class _Request:
    def __init__(self, payload: object) -> None:
        self._payload = payload
        self.app = SimpleNamespace(state=SimpleNamespace())
        self.state = SimpleNamespace()

    async def json(self) -> object:
        if isinstance(self._payload, ValueError):
            raise self._payload
        return self._payload

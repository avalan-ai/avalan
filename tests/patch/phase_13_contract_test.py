"""Exercise the authenticated remote patch continuation contract."""

from asyncio import run, sleep
from dataclasses import replace
from pathlib import Path
from runpy import run_path

import httpx
import pytest
from fastapi import FastAPI

from avalan.patch.domain import Capability, PatchContextId, PatchWorkspaceId
from avalan.patch.policy import (
    PatchAgentId,
    PatchPrincipalId,
    PatchRunId,
    PatchSessionId,
    PatchTaskId,
    PatchTenantId,
    PolicyDisclosure,
    PolicyRevision,
    PolicyRouteId,
)
from avalan.server.patch import (
    RemotePatchAuthority,
    RemotePatchController,
    RemotePatchEditPart,
    RemotePatchServerError,
    RemotePatchTestClient,
    RemotePatchTestServerConfiguration,
    _reject_forbidden_caller_fields,
    install_remote_patch_test_routes,
)


def _remote_helper(name: str) -> object:
    """Return one typed test-server harness helper by exact name."""
    helpers = run_path(
        str(Path("tests/server/patch_remote_test.py").resolve())
    )
    value = helpers.get(name)
    assert value is not None
    return value


def _authority() -> RemotePatchAuthority:
    """Return the existing fully bound remote test authority."""
    factory = _remote_helper("_authority")
    assert callable(factory)
    authority = factory()
    assert type(authority) is RemotePatchAuthority
    return authority


def _active_configuration(
    authority: RemotePatchAuthority,
) -> RemotePatchTestServerConfiguration:
    """Return the existing fully bound remote route configuration."""
    factory = _remote_helper("_active_configuration")
    assert callable(factory)
    values = factory(authority)
    assert isinstance(values, tuple) and len(values) == 3
    configuration = values[0]
    assert type(configuration) is RemotePatchTestServerConfiguration
    return configuration


def _configuration_for(
    authority: RemotePatchAuthority,
) -> RemotePatchTestServerConfiguration:
    """Return an inert binder configuration for handle-denial assertions."""
    binder_type = _remote_helper("_Binder")
    configuration = _remote_helper("_configuration")
    assert callable(binder_type) and callable(configuration)
    values = configuration(binder_type(), authority)
    assert type(values) is RemotePatchTestServerConfiguration
    return values


def test_patch_phase_13_requirements() -> None:
    """Advertise closed schemas and reserve remote edit/apply identities."""

    async def scenario() -> None:
        authority = _authority()
        configuration = _active_configuration(authority)
        app = FastAPI()
        controller = install_remote_patch_test_routes(app, configuration)
        assert type(controller) is RemotePatchController
        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(
                    app=app,
                    client=("127.0.0.1", 1),
                ),
                base_url="http://testserver",
            ) as http_client:
                client = RemotePatchTestClient(
                    http_client,
                    authority.correlation,
                )
                tools = await client.tools()
                edit = await client.edit(
                    "note.txt",
                    [
                        RemotePatchEditPart(
                            old_text="before",
                            new_text="after",
                        )
                    ],
                    "phase13-edit-key",
                )
                applied = await client.apply(
                    "\n".join(
                        (
                            "*** Begin Patch v1",
                            "*** Update File: note.txt",
                            "@@",
                            "-before",
                            "+after",
                            "*** End Patch",
                        )
                    ),
                    "phase13-apply-key",
                )
                replay = await client.edit(
                    "note.txt",
                    [
                        RemotePatchEditPart(
                            old_text="before",
                            new_text="after",
                        )
                    ],
                    "phase13-edit-key",
                )
            await sleep(0)
            return (
                [tool.name for tool in tools.data],
                [tool.strict for tool in tools.data],
                tools.data[0].parameters,
                tools.data[1].parameters,
                (edit.state, applied.state, replay.state),
                (edit.operation_handle, replay.operation_handle),
            )
        finally:
            await controller.close()

    observed = run(scenario())
    assert observed[0] == ["patch.edit", "patch.apply"]
    assert observed[1] == [True, True]
    assert observed[2] == {
        "additionalProperties": False,
        "properties": {
            "edits": {
                "items": {
                    "additionalProperties": False,
                    "properties": {
                        "new_text": {"type": "string"},
                        "old_text": {"minLength": 1, "type": "string"},
                    },
                    "required": ["old_text", "new_text"],
                    "type": "object",
                },
                "minItems": 1,
                "type": "array",
            },
            "path": {"type": "string"},
        },
        "required": ["path", "edits"],
        "type": "object",
    }
    assert observed[3] == {
        "additionalProperties": False,
        "properties": {"patch": {"type": "string"}},
        "required": ["patch"],
        "type": "object",
    }
    assert observed[4] == ("pending", "pending", "pending")
    assert observed[5][0] != observed[5][1]


def test_patch_phase_13_remote_authority_requirements() -> None:
    """Reject every caller authority replacement before remote dispatch."""
    authority = _authority()
    controller = RemotePatchController(_configuration_for(authority))
    candidates = (
        "approval",
        "approvals",
        "backend",
        "capabilities",
        "capability",
        "confirmation",
        "container_profile",
        "cwd",
        "disclosure",
        "limit",
        "limits",
        "matching_mode",
        "native_item_shape",
        "policy",
        "policy_version",
        "schema",
        "validator",
        "worker",
        "workspace",
    )
    for field in candidates:
        with pytest.raises(RemotePatchServerError):
            _reject_forbidden_caller_fields({field: "caller-controlled"})
    first = controller._operation(authority, "phase13-stable-key")
    second = controller._operation(authority, "phase13-stable-key")
    changed = controller._operation(authority, "phase13-changed-key")
    assert first == second
    assert first.request_id != changed.request_id
    assert first.identity.tenant_id == authority.tenant
    assert first.identity.principal_id == authority.principal
    assert first.identity.route_id == authority.route


def test_patch_phase_13_continuation_requirements() -> None:
    """Deny every wrong-authority opaque-handle continuation over HTTP."""

    async def scenario() -> None:
        authority = _authority()
        owner = RemotePatchController(_configuration_for(authority))
        handle = owner._seal_operation(
            owner._operation(authority, "phase13-continuation-key")
        )
        alternate_route = PolicyRouteId("route-phase13-other")
        authorities = (
            replace(authority, tenant=PatchTenantId("tenant-phase13-other")),
            replace(
                authority,
                principal=PatchPrincipalId("principal-phase13-other"),
            ),
            replace(authority, run=PatchRunId("run-phase13-other")),
            replace(
                authority, session=PatchSessionId("session-phase13-other")
            ),
            replace(authority, task=PatchTaskId("task-phase13-other")),
            replace(authority, agent=PatchAgentId("agent-phase13-other")),
            replace(
                authority,
                execution_scope="scope_phase13_other",
            ),
            replace(
                authority,
                route=alternate_route,
                approval_route=alternate_route,
            ),
            replace(
                authority,
                approval_route=PolicyRouteId("route-approval-phase13-other"),
            ),
            replace(
                authority,
                context=PatchContextId("context_" + "b" * 16),
            ),
            replace(
                authority,
                workspace=PatchWorkspaceId("workspace_" + "b" * 16),
            ),
            replace(
                authority,
                policy_revision=PolicyRevision("policy-phase13-other"),
            ),
            replace(
                authority,
                disclosures=frozenset((PolicyDisclosure.SERVER_EXACT_TRUTH,)),
            ),
            replace(
                authority,
                capabilities=frozenset((Capability.READ_FOR_MUTATION,)),
            ),
            replace(
                authority,
                correlation="correlation-phase13-other",
            ),
        )
        suffixes = (
            ("GET", ""),
            ("POST", "/await"),
            ("POST", "/cancel"),
            ("GET", "/events"),
        )
        observed: list[tuple[tuple[int, ...], tuple[bool, ...]]] = []
        for alternate in authorities:
            configuration = _configuration_for(alternate)
            controller = RemotePatchController(configuration)
            app = FastAPI()
            install_remote_patch_test_routes(
                app, configuration, controller=controller
            )
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(
                    app=app,
                    client=("127.0.0.1", 1),
                ),
                base_url="http://testserver",
            ) as client:
                responses = [
                    await client.request(
                        method,
                        "/__avalan_test__/patch/v1/operations/"
                        + handle
                        + suffix,
                        headers={
                            "X-Avalan-Correlation": alternate.correlation,
                        },
                    )
                    for method, suffix in suffixes
                ]
            observed.append(
                (
                    tuple(response.status_code for response in responses),
                    tuple(
                        response.json()
                        == {
                            "error": {
                                "code": "patch.operation_unavailable",
                                "message": "Patch operation unavailable.",
                            }
                        }
                        for response in responses
                    ),
                )
            )
            await controller.close()
        return tuple(observed)

    observed = run(scenario())
    assert len(observed) == 15
    assert all(statuses == (404, 404, 404, 404) for statuses, _ in observed)
    assert all(bodies == (True, True, True, True) for _, bodies in observed)

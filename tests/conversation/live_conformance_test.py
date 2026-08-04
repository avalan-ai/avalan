"""Run explicitly authorized native provider conformance probes."""

from collections.abc import Callable
from datetime import UTC, datetime
from importlib.util import module_from_spec, spec_from_file_location
from os import environ
from pathlib import Path
from sys import modules
from typing import Any

import pytest


def _load_live_module() -> Any:
    path = (
        Path(__file__).parents[2] / "scripts/conversation_live_conformance.py"
    )
    specification = spec_from_file_location(
        "conversation_live_conformance",
        path,
    )
    assert specification is not None
    assert specification.loader is not None
    module = module_from_spec(specification)
    modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


live = _load_live_module()

pytestmark = [
    pytest.mark.anyio,
    pytest.mark.skipif(
        environ.get("AVALAN_RUN_LIVE_CONFORMANCE") != "1",
        reason="cost-bearing provider conformance is explicitly opt-in",
    ),
]


@pytest.fixture
def anyio_backend() -> str:
    """Run live provider probes on asyncio only."""
    return "asyncio"


async def _clock() -> datetime:
    return datetime.now(UTC)


def _live_config(family: object) -> object:
    if family is live.LiveProviderFamily.OPENAI:
        endpoint = environ.get(
            "OPENAI_BASE_URL",
            "https://api.openai.com/v1",
        )
        model = environ.get("OPENAI_MODEL", "")
        revision = environ.get("OPENAI_MODEL_REVISION", "")
        api_key = environ.get("OPENAI_API_KEY", "")
        api_form = "openai_responses_v1"
        api_revision = "openapi-2.3.0"
    else:
        endpoint = environ.get("AZURE_OPENAI_ENDPOINT", "")
        model = environ.get("AZURE_OPENAI_DEPLOYMENT", "")
        revision = environ.get("AZURE_OPENAI_DEPLOYMENT_REVISION", "")
        api_key = environ.get("AZURE_OPENAI_API_KEY", "")
        api_revision = environ.get("AZURE_OPENAI_API_REVISION", "")
        api_form = api_revision
    return live.LiveConformanceConfig(
        provider_family=family,
        endpoint=endpoint,
        api_form=api_form,
        provider_api_revision=api_revision,
        model_or_deployment=model,
        model_or_deployment_revision=revision,
        api_key=api_key,
        command_authority=True,
        environment_authority=environ.get(
            "AVALAN_LIVE_CONFORMANCE_AUTHORITY", ""
        ),
        command_cost_acknowledgement=True,
        environment_cost_acknowledgement=environ.get(
            "AVALAN_LIVE_CONFORMANCE_COST_ACK", ""
        ),
    )


async def test_normative_completion_contract(
    record_property: Callable[[str, object], None],
) -> None:
    """Require both exact native providers to pass the complete live matrix."""
    record_property("conversation_acceptance_evidence", "live")
    completed: list[str] = []
    for family in (
        live.LiveProviderFamily.OPENAI,
        live.LiveProviderFamily.AZURE_OPENAI,
    ):
        receipt = await live.run_live_conformance(
            _live_config(family),
            transport_factory=live.OpenAISdkLiveConformanceTransport,
            clock=_clock,
        )
        assert receipt.completed_cases == live._EXECUTION_ORDER
        assert receipt.provider_family is family
        assert receipt.redacted_payload()["production_activation_granted"] is (
            False
        )
        completed.append(family.value)
    assert tuple(completed) == ("openai", "azure_openai")

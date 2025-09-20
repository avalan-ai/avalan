from typing import Any

from ...agent.orchestrator import Orchestrator
from .schema import dump_payload, validate_agent_card


def build_agent_card(
    orchestrator: Orchestrator,
    *,
    name: str,
    version: str,
    base_url: str,
    prefix: str,
) -> dict[str, Any]:
    """Return the Agent Card advertised over the A2A protocol."""

    capabilities: dict[str, bool] = {
        "messaging": True,
        "streaming": True,
    }
    if orchestrator.tool and not orchestrator.tool.is_empty:
        capabilities["tool-use"] = True

    normalized_prefix = prefix.strip()
    stripped_prefix = normalized_prefix.strip("/")
    if stripped_prefix:
        normalized_prefix = f"/{stripped_prefix}"
    else:
        normalized_prefix = ""

    base_path = f"{base_url}{normalized_prefix}"

    data = {
        "id": str(orchestrator.id),
        "name": orchestrator.name or name,
        "version": version,
        "summary": "Avalan agent orchestrated via the Avalan runtime.",
        "capabilities": capabilities,
        "interfaces": [
            {
                "type": "a2a.tasks",
                "url": f"{base_path}/tasks",
            },
            {
                "type": "a2a.events",
                "url": f"{base_path}/tasks/{{task_id}}/events",
            },
        ],
    }

    if orchestrator.model_ids:
        data["models"] = sorted(orchestrator.model_ids)

    return dump_payload(validate_agent_card(data))

#!/usr/bin/env -S PYTHONPATH=/runtime:/deps /usr/local/bin/python
"""Verify the mounted invocation before executing the test command."""

from json import dumps
from pathlib import Path
from sys import argv

from avalan.task.container_invocation import verify_container_invocation

invocation = verify_container_invocation()
context = invocation.context
assert tuple(argv[1:]) == ("agent", "agent.toml")
assert context.trigger is not None and context.deployment is not None
Path("/outputs/receipt.json").write_text(
    dumps(
        {
            "run_id": context.run_id,
            "attempt_id": context.attempt_id,
            "occurrence_id": context.trigger.occurrence_id,
            "deployment_id": context.deployment.execution_deployment_id,
            "dispatched": True,
            "input_value": invocation.input_value,
        }
    )
)

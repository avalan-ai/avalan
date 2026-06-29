# Shell Pipeline Traceability

This matrix maps the structured shell pipeline agenda to tracked tests and
docs. It is a release-readiness aid, not a substitute for the full test suite.

## Commands

| Purpose | Command |
| --- | --- |
| Focused shell pipeline smoke | `poetry run pytest tests/tool/shell/composition_entities_test.py tests/tool/shell/composition_policy_test.py tests/tool/shell/composition_executor_test.py tests/tool/shell/composition_backend_test.py tests/tool/shell/composition_tools_test.py tests/tool/shell/display_test.py` |
| Integration surfaces | `poetry run pytest tests/agent/loader_test.py tests/cli/main_test.py tests/cli/agent_test.py tests/cli/task_test.py tests/cli/tool_display_projection_e2e_test.py tests/flow/e2e_test.py tests/flow/validator_test.py tests/task/targets/agent_target_test.py tests/task/targets/flow_target_test.py tests/task/direct_client_e2e_test.py tests/task/queue_worker_e2e_test.py tests/tool/mcp_tool_test.py tests/server/a2a_v1_router_test.py tests/server/protocol_streaming_e2e_test.py` |
| Example validation | `poetry run pytest tests/agent/loader_test.py tests/flow/validator_test.py tests/task/task_loader_test.py` |
| Full release gate | `poetry run pytest --verbose -s` |
| Coverage gate | `poetry run pytest --verbose -s --cov=src/ --cov-fail-under=100` and `make test-coverage -- -100 src/` |
| Lint and type gate | `make lint` |

## Requirement Matrix

| Requirement or safety claim | Concrete tests and files | Status |
| --- | --- | --- |
| `shell.pipeline` is default-denied and appears only when selected and `allow_pipelines = true`. | `tests/tool/shell/opt_in_test.py`, `tests/tool/shell/toolset_test.py`, `tests/agent/loader_test.py`, `tests/cli/agent_test.py`, `tests/flow/validator_test.py`, `tests/task/targets/agent_target_test.py` | Covered |
| The public schema is structured data: `steps`, optional `mode`, timeout and byte caps, typed `stdin_from`, and no runtime authority fields. | `tests/tool/shell/composition_entities_test.py`, `tests/tool/shell/composition_tools_test.py`, `tests/tool/shell/display_test.py`, `tests/tool/mcp_tool_test.py`, `tests/server/authority_test.py` | Covered |
| Shell strings, arbitrary executables, public stdin, writes, path sentinel stdin, invalid refs, unsupported consumers, media mismatches, and unsafe paths are policy-denied before execution. | `tests/tool/shell/composition_policy_test.py`, `tests/tool/shell/policy_test.py`, `tests/tool/shell/guardrail_test.py`, `tests/tool/shell/filesystem_test.py`, `tests/tool/shell/tools_test.py` | Covered |
| Local byte pipelines preserve deterministic ordering, pipefail-style aggregate status, final stdout visibility, stage metadata, and bounded stream routing. | `tests/tool/shell/composition_executor_test.py`, `tests/tool/shell/composition_backend_test.py`, `tests/tool/shell/composition_tools_test.py`, `tests/flow/runtime_test.py` | Covered |
| Sandbox and container byte pipelines fail closed and do not fall back to host execution. | `tests/tool/shell/composition_backend_test.py`, `tests/tool/shell/container_test.py`, `tests/tool/shell/sandbox_test.py`, `tests/container/fail_closed_conformance_test.py` | Covered |
| Agent TOML, CLI flags, strict flows, tasks, MCP, A2A, server projection, and CLI themes cover positive and negative pipeline behavior. | `tests/agent/loader_test.py`, `tests/cli/main_test.py`, `tests/cli/agent_test.py`, `tests/cli/task_test.py`, `tests/cli/theme_basic_test.py`, `tests/cli/theme_fancy_test.py`, `tests/cli/tool_display_projection_e2e_test.py`, `tests/flow/e2e_test.py`, `tests/flow/validator_test.py`, `tests/task/direct_client_e2e_test.py`, `tests/task/queue_worker_e2e_test.py`, `tests/tool/mcp_tool_test.py`, `tests/server/a2a_v1_router_test.py`, `tests/server/protocol_streaming_e2e_test.py` | Covered |
| Performance and resource assertions cover bounded process count, memory and stream caps, deterministic ordering, and no host fallback. | `tests/tool/shell/composition_executor_test.py`, `tests/tool/shell/composition_backend_test.py`, `tests/tool/shell/composition_policy_test.py`, `tests/container/stress_conformance_test.py`, `tests/container/watchdog_conformance_test.py` | Covered |
| Documentation and examples match default-deny behavior and are loader-validated. | `docs/TOOLS.md`, `docs/CLI.md`, `docs/FLOWS.md`, `docs/TASKS.md`, `docs/ISOLATION.md`, `docs/CONTAINERS.md`, `docs/examples/agent_shell_pipeline.toml`, `docs/examples/flows/shell_pipeline.flow.toml`, `docs/examples/tasks/pipeline_agent.task.toml`, `docs/examples/tasks/pipeline_flow.task.toml`, `tests/agent/loader_test.py`, `tests/flow/validator_test.py`, `tests/task/task_loader_test.py` | Covered |

## Example Files

| Example | Validation path |
| --- | --- |
| `docs/examples/agent_shell_pipeline.toml` | Loaded by `tests/agent/loader_test.py`; asserts `shell.pipeline` selection, `allow_pipelines`, caps, and allowed commands. |
| `docs/examples/flows/shell_pipeline.flow.toml` | Loaded by `tests/flow/validator_test.py` through a `ToolManager` with `ShellToolSet(allow_pipelines=True)`. |
| `docs/examples/tasks/pipeline_agent.task.toml` | Loaded by `tests/task/task_loader_test.py`; asserts the agent target and direct run contract. |
| `docs/examples/tasks/pipeline_flow.task.toml` and `docs/examples/tasks/pipeline_flow.flow.toml` | Loaded by `tests/task/task_loader_test.py` and validated by flow tests with explicit pipeline runtime settings. |

## Known Scope

- Full byte-stream pipelines are local-only in v1.
- Sandbox and container byte pipelines require a future trusted structured
  runner and fail closed today.
- Historical specs under `specs/` are coordinator-owned planning artifacts and
  are not part of the tracked user-doc traceability inventory.

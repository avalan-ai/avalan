# Isolation Traceability

This matrix maps isolation safety claims to concrete verification files and
commands. It is a documentation aid, not a replacement for the full test suite.
"Partial" means existing tests cover part of the requirement and the remaining
gap is listed in the notes.

## Commands

| Purpose | Command |
| --- | --- |
| Phase 10 documentation and conformance smoke | `poetry run pytest tests/isolation tests/sandbox tests/container/release_conformance_test.py tests/container/fail_closed_conformance_test.py tests/container/watchdog_conformance_test.py tests/container/stress_conformance_test.py tests/tool/shell/container_test.py tests/tool/shell/sandbox_test.py` |
| Full release gate | `poetry run pytest --verbose -s` |
| Coverage gate | `make test-coverage` |
| Lint and type gate | `make lint` |
| Docker live gate | `AVALAN_CONTAINER_DOCKER_E2E=1 AVALAN_CONTAINER_DOCKER_E2E_IMAGE=<digest-pinned-image> poetry run pytest tests/container/docker_test.py tests/container/release_conformance_test.py` |
| Apple `container` live gate | `AVALAN_CONTAINER_APPLE_E2E=1 AVALAN_CONTAINER_APPLE_E2E_IMAGE=<digest-pinned-image> poetry run pytest tests/container/apple_test.py` |

## Phase Matrix

| Phase | Requirement or safety claim | Concrete tests and files | Status |
| --- | --- | --- | --- |
| Phase 0: inventory | Public modes are `local`, `sandbox`, and `container`; container backends are `docker` and `apple-container`; sandbox backends are `seatbelt` and `bubblewrap`; removed backend terms are inventoried. | `tests/container/isolation_phase0_inventory_test.py`, `tests/container/fixtures/isolation_phase0_inventory.toml` | Covered |
| Phase 1: backend cleanup | Removed container backend values are rejected before execution and CLI choices list only supported values. | `tests/container/conformance_test.py`, `tests/container/fail_closed_conformance_test.py`, `tests/cli/main_test.py` | Covered |
| Phase 2: settings model | Isolation settings are a tagged union; sandbox and container fields cannot be mixed; untrusted sources cannot define runtime authority. | `tests/isolation/isolation_settings_test.py`, `tests/agent/loader_test.py`, `tests/cli/agent_test.py` | Covered |
| Phase 3: sandbox contract | Fake sandbox backend covers success, denial, timeout, cancellation, stream truncation, output rejection, cleanup uncertainty, and concurrency limits. | `tests/sandbox/sandbox_phase3_test.py`, `tests/tool/shell/sandbox_test.py` | Covered |
| Phase 4: unified planning | Plans have deterministic fingerprints; required container does not fall back to sandbox or local without review; required sandbox does not fall back to local without review; approval reuse is mode and scope bound. | `tests/isolation/isolation_planning_test.py`, `tests/tool/shell/container_test.py`, `tests/tool/shell/sandbox_test.py` | Covered |
| Phase 5: Docker backend | Docker backend probes, lifecycle operations, capability mismatches, rootful authorization, image policy, output copy, cleanup, cancellation, timeout, and live-gated execution are tested. | `tests/container/docker_test.py`, `tests/container/backend_test.py`, `tests/container/lifecycle_test.py`, `tests/container/release_conformance_test.py` | Covered |
| Phase 6: real sandbox backends | Seatbelt and Bubblewrap profile generation, probes, path controls, network limits, execution, output, stream bounds, and cleanup behavior are tested with injected runners. Live-runtime probes are gated and skip with backend diagnostics when unavailable. | `tests/sandbox/sandbox_phase6_test.py`, `tests/sandbox/real_runtime_e2e_test.py` | Covered for injected runners and live-runtime probe gates. |
| Phase 7: shell integration | Shell toolsets use the selected isolation runtime and do not expose backend authority in model schemas. | `tests/tool/shell/container_test.py`, `tests/tool/shell/sandbox_test.py`, `tests/tool/shell/toolset_test.py`, `tests/tool/shell/public_api_test.py` | Covered |
| Phase 8: SDK, CLI, and agent TOML | Trusted SDK, CLI, and agent TOML settings normalize to equivalent plans; unsafe or mixed fields are rejected; CLI help choices are constrained. | `tests/isolation/isolation_settings_test.py`, `tests/agent/loader_test.py`, `tests/cli/agent_test.py`, `tests/cli/get_tool_settings_test.py`, `tests/cli/main_test.py` | Covered |
| Phase 9: durable and remote surfaces | Task, worker, flow, server, MCP, and A2A reject stale or widened isolation metadata and remote runtime authority. | `tests/task/container_execution_test.py`, `tests/task/task_loader_test.py`, `tests/task/worker_test.py`, `tests/flow/container_test.py`, `tests/flow/runtime_test.py`, `tests/server/authority_test.py`, `tests/server/container_policy_test.py`, `tests/server/remote_container_test.py`, `tests/server/mcp_router_test.py`, `tests/server/a2a_v1_router_test.py` | Covered for implemented container/profile-selection paths. Remote sandbox profile selectors are rejected today. |
| Phase 10: documentation | Supported modes, backends, policy fields, setup, examples, test gates, platform limits, security posture, fail-closed behavior, approval behavior, and diagnostics are documented. | `docs/ISOLATION.md`, `docs/CONTAINERS.md`, `docs/TOOLS.md`, `docs/INSTALL.md`, `docs/CLI.md` | Covered by documentation changes. |
| Phase 10: traceability | Requirement-to-test mapping exists for isolation phases and safety claims. | `docs/ISOLATION_TRACEABILITY.md` | Covered by documentation changes. |
| Phase 10: diagnostics | Stable diagnostic inventory, public dictionaries, audit metadata, and model-facing formatting are covered. | `tests/isolation/isolation_settings_test.py` | Covered by `stable_isolation_diagnostic_inventory()`. |
| Phase 10: removed-backend sweep | Active docs and CLI-help-adjacent files do not claim support for Podman, nerdctl, containerd/nerdctl, Microsoft containers, Windows Docker, WSL2, Hyper-V, or removed env gates. | Search command: `rg -n "podman|nerdctl|containerd|Windows Docker|WSL2|Hyper-V|Microsoft container|microsoft container|AVALAN_CONTAINER_PODMAN|AVALAN_CONTAINER_NERDCTL|AVALAN_CONTAINER_WINDOWS" docs README.md src/avalan/cli src/avalan/tool` | Covered for active docs and CLI help. Matches in `docs/ISOLATION.md` and this file are explicit non-support statements. Internal inventory and negative tests intentionally retain removed terms. |

## Phase 10 Safety Claims

| Safety claim | Evidence | Notes |
| --- | --- | --- |
| Unknown modes and unsupported backend strings fail closed. | `tests/isolation/isolation_settings_test.py`, `tests/container/conformance_test.py`, `tests/container/fail_closed_conformance_test.py`, `tests/cli/main_test.py` | Includes rejection of `auto`, `podman`, `nerdctl`, and `windows-docker`. |
| Sandbox and container policy fields cannot be mixed. | `tests/isolation/isolation_settings_test.py`, `tests/agent/loader_test.py`, `tests/task/container_execution_test.py`, `tests/flow/container_test.py` | Covers settings, trusted agent TOML, durable task metadata, and flow TOML rejection paths. |
| Model-visible schemas do not expose runtime authority. | `tests/tool/shell/container_test.py`, `tests/tool/shell/sandbox_test.py`, `tests/server/authority_test.py`, `tests/server/entities_test.py`, `tests/server/a2a_v1_router_test.py` | Remote inputs may select exposed container profiles only where policy permits it. |
| Required container execution does not run locally when the backend is missing, disabled, or denied. | `tests/tool/shell/container_test.py`, `tests/container/fail_closed_conformance_test.py`, `tests/container/policy_test.py` | Required paths report container errors instead of using the local executor. |
| Required sandbox execution does not run locally when the sandbox backend is missing, disabled, or denied. | `tests/tool/shell/sandbox_test.py`, `tests/sandbox/sandbox_phase3_test.py`, `tests/sandbox/sandbox_phase6_test.py`, `tests/sandbox/real_runtime_e2e_test.py` | Live sandbox gates prove probe availability and clear diagnostic skips; full live execution remains platform-dependent. |
| Plan fingerprints change when mode, backend, paths, images, roots, resources, or approval context changes. | `tests/isolation/isolation_planning_test.py`, `tests/container/planning_test.py`, `tests/container/policy_test.py` | Fingerprints are used by cached review, durable metadata, and audit. |
| Approvals cannot be reused across modes, scopes, attempts, policy versions, or broader permissions. | `tests/isolation/isolation_planning_test.py`, `tests/container/policy_test.py`, `tests/flow/container_test.py` | Covers stale, cross-mode, broader-scope, noninteractive, and mismatched durable approvals. |
| Durable workers and resumed flows revalidate stale or widened metadata before execution. | `tests/task/container_execution_test.py`, `tests/task/worker_test.py`, `tests/flow/container_test.py`, `tests/flow/serializer_test.py` | Covers stale fingerprint, stale policy, missing approval, and retry/resume behavior. |
| Diagnostics and audit redact sensitive values. | `tests/isolation/isolation_settings_test.py`, `tests/container/audit_test.py`, `tests/task/event_sanitization_test.py`, `tests/flow/flow_privacy_test.py`, `tests/flow/graph_test.py`, `tests/tool/shell/formatting_test.py` | Container diagnostic normalization has broad coverage, and the Phase 10 isolation inventory has stable public/audit/model-facing metadata coverage. |
| Streams, output, cleanup, probes, and concurrency are bounded. | `tests/container/watchdog_conformance_test.py`, `tests/container/stress_conformance_test.py`, `tests/container/output_test.py`, `tests/sandbox/sandbox_phase3_test.py`, `tests/sandbox/sandbox_phase6_test.py`, `tests/tool/shell/fixture_contract_test.py` | Server streaming has separate coverage in `tests/server/streaming_conformance_test.py` and `tests/server/streaming_latency_budget_test.py`. |

## Known Traceability Gaps

- The Phase 10 live sandbox gates currently prove probe availability and clear
  diagnostic skips. Full live execution coverage remains platform-dependent
  and is expected to run only on hosts with `sandbox-exec` or `bwrap`.
- Historical planning specs under `specs/` still describe removed container
  runtimes. They are not active user docs or CLI help, and this worker did not
  edit historical specs.

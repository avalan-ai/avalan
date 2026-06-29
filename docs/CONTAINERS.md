# Container Execution

Avalan container support is a policy-controlled execution path for selected
runtime scopes. It is disabled by default and must be enabled from trusted
operator configuration. Models and remote callers cannot choose arbitrary
images, mounts, runtime sockets, networks, devices, secrets, or backend flags.

This page documents current release readiness for the implemented container
execution surfaces. For the unified `local` / `sandbox` / `container` model,
see [Isolation execution](ISOLATION.md).

## Supported Scope

The first release conformance target is the core container contract plus shell
container execution. Some later-scope plumbing exists, but those paths are not
full conformance claims until their real-runtime and release gates close.

| Surface | Current status |
| --- | --- |
| Core contract | Supported for settings, trusted profile selection, merge/narrowing semantics, policy decisions, deterministic fingerprints, stable diagnostics, fake backend lifecycle, and audit records. |
| Shell tools | Supported as the primary command-scoped container path. Agent CLI and agent TOML can define trusted shell container policy. Execution requires a trusted `ContainerAsyncBackend`; without one, required container execution fails closed. |
| `code.search.ast.grep` | Can run through the same injected container runtime when SDK/operator code supplies settings and a backend. The model schema does not expose runtime authority. |
| Strict flows | Support trusted defaults, node narrowing, review records, and container events. Legacy flow paths reject container settings when they cannot enforce policy. |
| Direct and queued tasks | SDK/operator paths support `TaskContainerExecutionSettings`, durable plan metadata, worker revalidation, and injected container backends. Task TOML container syntax remains unsupported. |
| Server, MCP, and A2A | Remote callers may select only an operator-exposed safe container profile selector. Arbitrary runtime authority is rejected before tool execution. |
| Runtime envelopes | Planning and diagnostics exist for trusted agent, flow, task worker, server, and model backend envelopes. Actual envelope execution requires a trusted runner/loader; otherwise Avalan reports unavailable-envelope diagnostics. |
| Model backends | Envelope planning exists for eligible model backends. Host-native targets such as Metal remain host-native unless a trusted envelope implementation is supplied. |
| Backend breadth | Docker is the promoted backend. Apple `container` has an opt-in shell backend adapter. |

## Runtime Setup

Container policy is trusted configuration, not a model prompt feature. A
minimal shell setup needs:

- A digest-pinned image that contains the command binaries used by enabled
  shell tools.
- A trusted profile with the workspace mount, network policy, pull policy,
  resource limits, output policy, and review mode.
- A trusted `ContainerAsyncBackend` implementation injected by the hosting
  application, task worker, or runtime wrapper.

Agent TOML can declare the trusted policy:

```toml
[tool]
enable = ["shell.cat"]

[tool.container]
backend = "docker"
default_profile = "workspace-readonly"
allowed_profiles = ["workspace-readonly"]

[tool.container.profiles.workspace-readonly]
image = "ghcr.io/example/avalan-shell@sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
workspace_root = "."
network = "none"
pull_policy = "never"
platform = "linux/amd64"

[tool.container.profiles.workspace-readonly.resources]
cpu_count = 1
memory_bytes = 536870912
pids = 128
timeout_seconds = 30

[tool.shell]
backend = "container"

[tool.shell.container]
profile = "workspace-readonly"
required = true
```

The agent CLI exposes the same trusted shape:

```bash
avalan agent run \
  --engine-uri "ai://env:OPENAI_API_KEY@openai/gpt-4.1-mini" \
  --tool shell.cat \
  --tool-shell-backend container \
  --tool-shell-workspace-root . \
  --tool-container-backend docker \
  --tool-container-profile workspace-readonly \
  --tool-container-image "ghcr.io/example/avalan-shell@sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa" \
  --tool-container-pull-policy never \
  --tool-container-platform linux/amd64 \
  --tool-container-network-mode none \
  --tool-container-timeout-seconds 30 \
  --tool-shell-container-profile workspace-readonly \
  --tool-shell-container-required
```

These examples configure policy. They do not discover or start Docker by
themselves. The executable backend values are `docker` and
`apple-container`. Required execution fails closed instead of running on the
host when the selected backend adapter is unavailable.

## Optional Runtime Gates

Default tests use fake backends and catalog probes. They must not contact
Docker or Apple `container`.

The optional live-runtime gates are recorded as backend runtime requirements:

| Backend profile family | Gate |
| --- | --- |
| Docker Engine, Docker Desktop macOS | `AVALAN_CONTAINER_DOCKER_E2E=1`; backend lifecycle tests also require `AVALAN_CONTAINER_DOCKER_E2E_IMAGE=<digest-pinned-image>` |
| Apple `container` | `AVALAN_CONTAINER_APPLE_E2E=1` plus `AVALAN_CONTAINER_APPLE_E2E_IMAGE=<digest-pinned-image>` |

There is no repository-wide pytest marker for these gates today. The explicit
live-runtime tests are the Docker gate in
`tests/container/release_conformance_test.py` and the Apple `container` gate in
`tests/container/apple_test.py`; each calls `skipTest` with a diagnostic when
the matching environment variable is not set. Apple also requires
`AVALAN_CONTAINER_APPLE_E2E_IMAGE` so the live test can run the same
digest-pinned shell image used by the CLI tutorial. Missing optional runtimes
are not failures for the default suite.

Use the normal lightweight checks for conformance, policy, fake lifecycle, and
stress coverage:

```bash
poetry run pytest \
  tests/container/release_conformance_test.py \
  tests/container/fail_closed_conformance_test.py \
  tests/container/watchdog_conformance_test.py \
  tests/container/stress_conformance_test.py
```

The full merge gate remains:

```bash
poetry run pytest --verbose -s
make test-coverage
make lint
```

## Platform Limits

Docker Engine on Linux is the promoted catalog profile. Docker Desktop on
macOS is optional and VM-backed: shared mount prefixes, VM resource ceilings,
path normalization, case behavior, and signal behavior must be treated as
runtime-specific. Apple `container` is opt-in and available for shell tool
execution when the Apple CLI backend is selected.

Current catalog capabilities are CPU-only. Network allowlists, devices,
rootful engines, build support, resource controls, and VM-backed mounts are
accepted only when the selected backend reports compatible capabilities and
operator policy authorizes them. Metal is not exposed as a normal Linux
container GPU capability.

## Security Posture

Containers are defense in depth, not the whole security model. Avalan still
relies on host-side policy before any runtime starts:

- Typed tool schemas stay separate from runtime authority.
- Trusted profiles choose images, mounts, network, devices, resources,
  secrets, output policy, cleanup, and review behavior.
- Untrusted agent, flow, task, prompt, model, MCP, A2A, and HTTP request data
  can only narrow approved choices where a surface explicitly allows it.
- Host paths are normalized under allowed roots and sensitive paths,
  credential locations, runtime sockets, hidden paths, traversal, symlinks,
  special files, and remote paths are rejected or redacted by policy.
- Environment inheritance is denied by default; only explicit variables and
  allowlisted host environment names are planned.
- Secrets are referenced by approved names and redacted from diagnostics and
  audit output.
- Output streams and generated artifacts are bounded, typed, validated, and
  redacted before becoming tool, task, or audit output.

Do not treat containers as a complete multi-tenant isolation boundary without
rootless or user namespace hardening, VM-grade isolation, deployment-specific
runtime hardening, and host-level controls.

## Fail-Closed Behavior

Required container execution never silently falls back to host execution. A
missing backend, disabled required profile, unavailable runtime, capability
mismatch, denied image, denied pull/build, invalid mount, unsafe secret,
network denial, timeout, cancellation, output validation failure, cleanup
uncertainty, stale durable metadata, or unsupported surface produces a
structured failure.

Container shell execution supports individual commands and shell-local
`serial` or `parallel` compositions that do not route bytes between stages
when the selected profile can enforce each command. Full `shell.pipeline`
byte-stream mode, and any composition with `stdin_from`, is unsupported for
container execution. It fails closed with a policy-denied composition result.
Avalan does not translate container pipelines into shell strings or run them on
the host as a fallback.

Optional local fallback is allowed only when container execution is disabled
and not required. If `[tool.shell] backend = "container"` or
`--tool-shell-container-required` selects required container execution, a
missing backend returns a container error and leaves the local executor unused.

Remote request handling also fails closed: server, MCP, and A2A inputs may use
only an exposed profile selector such as `containerProfile` or
`container = {profile = "..."}`. Attempts to supply images, mounts, secrets,
network settings, devices, resources, backend flags, runtime envelopes, or
nested runtime authority are rejected before the request reaches a tool.

## Review Behavior

Profiles can deny escalation, require review, or preauthorize escalation.
Review behavior depends on the surface:

| Surface | Review mode |
| --- | --- |
| Interactive CLI | Interactive review can pause for approval. |
| Strict flow and queued task | Durable review records must match the exact plan fingerprint before resume or worker execution. |
| Direct task, server, MCP, and A2A | Review is not interactive, so review-required plans are denied. |

Approvals are scoped to the canonical plan fingerprint, profile, policy
version, review surface, and escalation triggers. A resumed flow or queued
task must revalidate the durable decision before execution.

## Diagnostics

Container diagnostics are structured with stable codes, paths, categories,
messages, hints, retryability, and privacy-safe metadata. Common codes include
`container.backend_required`, `container.backend_unavailable`,
`container.unsupported_syntax`, `container.backend.capability_mismatch`,
`container.backend.rootful_not_authorized`, `container.backend.image_denied`,
`container.backend.timeout`, `container.backend.cancelled`, and
`container.backend.cleanup_failed`.

Lifecycle and audit events use a shared vocabulary: policy evaluation, review
request, review decision, backend selection, image resolution, image pull,
mount preparation, container create/start, stdout/stderr/progress chunks,
stats, timeout, cancellation, exit, output copy, cleanup, denial, failure,
decision recorded, and result recorded. Raw stream bytes, host paths, prompts,
provider tokens, secret names or values, and large metadata fields are redacted
or truncated.

For task runs, sanitized events include `container_plan_verified`,
`container_lifecycle_completed`, and `container_worker_envelope_completed`
where the selected execution path reaches those phases.

## Known Risks And Deferred Non-Conformance

- Docker and Apple `container` backend adapters are implemented, but live
  runtime coverage is optional and environment-gated. Default verification
  still uses fake backends plus capability validation.
- Optional Docker and Apple real-runtime suites still need harness jobs that
  set the gates, provide digest-pinned images, probe runtimes, and skip with
  diagnostics when unavailable.
- Default conformance has release, fail-closed, watchdog, and stress tests,
  but it is not fully closed until real-runtime suites, broader
  platform performance gates, and every stable negative diagnostic audit are
  covered in CI.
- Backend breadth is not a release claim beyond the catalog. Apple `container`
  is explicitly opt-in with parity gaps, and VM-backed path behavior
  needs live validation on the target platform.
- Runtime envelopes and model backend containers require trusted
  loader/runner implementations. The default server and agent paths report
  unavailable-envelope diagnostics rather than pretending to run enveloped.
- Container byte pipelines are unsupported. `shell.pipeline` requests with
  `mode = "pipeline"` or `stdin_from` fail closed, even when individual shell
  commands can run in a container.
- Task TOML container syntax is explicitly unsupported. Use SDK/operator
  configuration for task container settings until task-file syntax is promoted.
- Generic profile-backed container tools, pooling, service reuse, broad build
  support, advanced provenance, and vulnerability enforcement are not default
  operational features.
- Shell path checks reduce filesystem race and leak risk but cannot remove
  every time-of-check/time-of-use race on mutable host filesystems.

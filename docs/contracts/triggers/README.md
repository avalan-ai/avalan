# Durable trigger contract, version 1

Status: Phase 0 design baseline. This document and its fixtures specify the
implementation; they do not advertise an available trigger API. Runtime,
parser conformance, persistence and deployment verification arrive in later
phases. `specs/TRIGGERS.md` is the implementation plan; this tracked contract
makes the decisions and fixtures available in a checkout without local specs.

## Configuration and types

All records are frozen, slotted, keyword-only dataclasses. Collections are
immutable snapshots. SDK identifiers are nonempty opaque strings, scoped by
an authenticated host-supplied `OwnerScopeId`; a loader never reads authority
from TOML/input. Unknown fields at every structural level are errors.

The only top-level TOML tables are `trigger`, `task`, `input`, `schedule`,
`policy`. Required fields: `trigger.schema_version = 1`, `trigger.name`,
`task.ref`, `schedule.type`. Names match `[a-z][a-z0-9-]{0,62}`.
`trigger.enabled` is a boolean, default true. Task refs are nonempty relative
paths resolved from the trigger directory and checked against configured
roots after resolving symlinks. Absolute and escaping refs fail validation.
The resolved task must pass ordinary task validation and have queue mode.

`TriggerSpec = CronTrigger | IntervalTrigger | AtTrigger` is a closed tagged
union. Cron fields: `type = "cron"`, required nonempty `expression`, optional
`timezone = "UTC"`. Interval fields: `type = "interval"`, required integer
`every_seconds` in 1..315360000, optional aware `start_at`. At fields:
`type = "at"`, required aware `at`. Fields from another variant are rejected.
Accepted timestamp input is a TOML offset datetime or string matching
`YYYY-MM-DDTHH:MM:SS[.ffffff](Z|+HH:MM|-HH:MM)` with 1..6 fractional digits,
valid calendar/offset components, no leap second and years 1..9999. Normalize
to aware UTC at microsecond precision; reject excess precision, overflow,
naive values and booleans in numeric fields, rather than truncate/coerce.

`input` is optional; absent input means null. Its only fields are `value`
(static JSON-compatible finite value) and `bindings` (array of tables).
TOML dates/datetimes in application values are not JSON values. Each binding
has exactly `path` and `source`. `path` is an RFC 6901 JSON Pointer to an
existing leaf in `value`; empty/root pointers, array `-`, noncanonical array
indices, nonexistent leaves, duplicate paths and ancestor/descendant paths
are rejected. Escapes are only `~0` and `~1`. Bindings replace leaves, never
merge or create application structure. Source is exactly `scheduled_at`,
`occurrence_id`, `trigger_id`, or `trigger_revision`; the first produces the
canonical UTC string, the next two strings, and the last a positive integer.
No expression evaluation, dispatch time or external fetch is accepted.
Validate the static structure with binding-aware placeholder types at apply,
then the complete materialized value against the task input schema before
preparation. Placeholder literal values do not need to pass the target type.
Files follow the task input contract; no separate trigger file syntax exists.

`RecurringPolicy` has `misfire = "skip" | "latest" | "all"` (default latest),
`misfire_grace_seconds` integer 0..315360000 (default 3600), and
`overlap = "skip" | "allow"` (default skip). `AtPolicy` has only grace;
explicit misfire or overlap on at is invalid. All/skip is valid: only the
first admission can proceed while a prior run remains nonterminal, and later
slots become explicit overlap skips. Enabled state is operational, separate
from immutable schedule/input revision identity.

## Time and bounded work

Cron has five ASCII whitespace-separated fields with case-insensitive JAN..DEC
and SUN..SAT names. Accept `*`, comma lists, ascending inclusive ranges and
positive steps on wildcard/range (`*/n`, `a-b/n`); reject bare `a/n`, wrapping
ranges, aliases, `?`, `L`, `W`, `#`, seconds/year fields. Bounds are minute
0..59, hour 0..23, DOM 1..31, month 1..12, DOW 0..7 (0 and 7 both Sunday).
Steps cannot exceed the field's cardinality. A field is unrestricted only
when its expression is exactly `*`; `*/1` is restricted even when its selected
set is complete. DOM and DOW combine with OR when both are restricted,
otherwise only the restricted one applies. Months always constrain matches.
Skip nonexistent local times; emit only fold 0 for repeated local timestamps,
including wildcard minute/hour schedules. Evaluate in the explicit IANA zone.

Next is strictly after its cursor; latest-due is at or before decision time
and at or after the first undecided slot. A single due slot dispatches under
all misfire policies when within inclusive grace. For two or more due slots,
skip discards all, latest coalesces all but the latest, all processes oldest
first. Expired slots are recorded, never admitted. Coverage can represent a
range without enumerating or claiming its count.

Interval anchor is first firing: supplied start must be >= registration time;
omitted start resolves once to registration time + interval. At likewise
allows equality at initial registration. Replacement selects strictly future
slots (> effective time), even if an otherwise valid supplied timestamp is
equal: an at replacement equal to effective time is rejected as having no
future slot. Unchanged reapply after firing remains a no-op. Changing input
or policy without changing the interval spec retains its resolved anchor.

Bounds below are host SDK settings, not accepted TOML fields. Every configured
value must be a positive integer <= its maximum, except grace above. A tick
stops at whichever work/time budget is reached first; unfinished work remains
at the first undecided cursor. Limits include skips as well as admissions.

| Setting | Default | Maximum | Unit |
| --- | ---: | ---: | --- |
| discovery_limit | 100 | 1000 | trigger snapshots/tick |
| decisions_per_tick | 100 | 1000 | occurrences or coverage spans |
| decisions_per_trigger | 10 | 100 | occurrences or coverage spans/tick |
| admissions_per_tick | 100 | 1000 | new runs |
| admissions_per_trigger | 10 | 100 | new runs/tick |
| candidate_evaluations | 10000 | 100000 | schedule candidates/tick |
| search_years | 8 | 400 | calendar years/calculation |
| search_candidates | 10000 | 100000 | candidates/calculation |
| tick_timeout_seconds | 5 | 60 | monotonic wall time |
| preparation_timeout_seconds | 2 | 30 | each preparation, also tick-bound |
| transaction_timeout_seconds | 2 | 30 | each transaction, also tick-bound |
| poll_interval_seconds | 1 | 60 | maximum sleep |
| admission_retry_attempts | 5 | 20 | consecutive failures per cursor |
| retry_base_seconds | 1 | 60 | deterministic exponential base |
| retry_max_seconds | 60 | 3600 | retry delay cap |
| stale_plan_retries | 2 | 10 | recalculations/tick |
| shutdown_timeout_seconds | 10 | 60 | stop/reconcile/close |
| diagnostic_bytes | 512 | 4096 | UTF-8 safe diagnostic |
| preview_count | 5 | 100 | occurrences |
| history_page_size | 50 | 200 | rows |
| orphan_grace_seconds | 86400 | 2592000 | minimum staging age |

Retry delay after failure n is min(max, base * 2**(n-1)); failure at the
attempt limit enters error. Persist the count and retry eligibility without
moving the cursor; a committed decision resets consecutive failures. Lock
contention/CAS conflicts are not failures. Orphan age alone never authorizes
deletion. Search exhaustion is an error distinct from proven impossibility;
Phase 1 must prove impossibility with calendar/dialect analysis or return
`trigger.search_budget_exhausted`. Calendar overflow is typed, not exhaustion.
Round-robin persistent discovery order `(last_processed_at, trigger_id)`
prevents a hot early trigger monopolizing batches. Successful decision, error
or bounded no-progress examination updates last_processed_at under CAS.

## Identity, state, API and codecs

`TriggerConfiguration` is the unregistered declarative loader result. Its
fields are schema_version (1), name, desired_enabled (boolean, default true),
task_ref (validated relative reference), schedule (`TriggerSpec`, allowing an
omitted interval anchor), input (static value plus bindings), and policy.
The loader maps TOML trigger.enabled to desired_enabled and task.ref to task_ref.
This type has no owner scope, allocated IDs, revision, generation, resolved task
identity or deployment identity. A trusted loader context supplies the source
base directory and permitted roots; they are not semantic configuration fields.

`TriggerApplyRequest` contains configuration: TriggerConfiguration and
expected_generation: int | None. None means create-if-absent; positive integers
are the existing-trigger CAS precondition. The client resolves task_ref using
its trusted loading context, validates the task/deployment and durably materializes
input before committing registration. On creation the store allocates the opaque
trigger ID, revision 1 and generation 1; its database decision time resolves the
implicit anchor. On replacement the store owns revision/generation increments.
Callers cannot supply these allocated identities through configuration.

`TriggerDefinition` is the registered immutable revision, whose fields are: schema_version, trigger_id, revision (>=1), name,
task_definition_id, execution_deployment_id, schedule, input, policy,
schedule_semantics_version. `TriggerState`: owner_scope_id, trigger_id,
revision, generation (>=1), status (active/paused/exhausted/error), next_at,
retry_after, failure_count, last_error_code, last_processed_at.
`TriggerInvocationContext`, defined in task/provenance.py, contains trigger_id,
trigger_revision, occurrence_id, scheduled_at, dispatched_at. Task code does
not import the trigger package. Context is immutable across retries/resumption.
Dispatch time is the authoritative admission decision time, not worker start.

Canonical JSON is UTF-8, sorted keys, compact separators, ensure_ascii=false,
no NaN, no insignificant fields, UTC strings with exactly six fractional
digits and Z. Negative zero floats normalize to 0.0. No Unicode normalization
of application values. Apply identity hashes the canonical semantic payload
with SHA-256: resolved task identity and deployment identity, declared schedule
(including whether anchor was omitted), input/bindings sorted by pointer, and
expanded policy defaults. It excludes name, owner, enabled, file path, resolved
implicit anchor, state and parser implementation version. Names cannot be
renamed in v1. Semantics version is 1 and is included; cron spelling is preserved
apart from uppercase names and collapsed whitespace (equivalent but differently
spelled expressions may create a revision). Resolved anchors are persisted in
the revision as execution data, separate from its declarative apply hash.

Occurrence UUID is UUIDv5 namespace `f279760b-a940-5ea5-a11d-7c96cd4fd690`,
name = canonical JSON array `[owner_scope_id, trigger_id, revision,
scheduled_at]`. It is independent of HMAC keys. Mandatory unique slot key is
(owner_scope_id, trigger_id, revision, scheduled_at); run_id has a unique
constraint when present. Task idempotency's window is this occurrence ID.
An admitted occurrence must have exactly one new run; every other disposition
must have no run. Dispositions: admitted, expired, skipped_misfire,
skipped_overlap, coalesced, superseded. Coverage spans allow the latter four
and expired, with first_at inclusive and until_at exclusive, first_at < until_at,
revision and nullable exact count. No spans for admissions. Under the trigger
lock, reject overlapping spans/rows and require both endpoints to delimit the
same schedule revision; until_at may be a non-slot effective-time boundary.
Null count means unavailable, never zero. History/deduplication is retained.

All effectful APIs are async. Host scope is bound to clients at construction.
`TriggerClient.validate(configuration: TriggerConfiguration) -> ValidationResult`,
`preview(configuration: TriggerConfiguration, *, reference_time, count=5)
-> TriggerPreview`,
`apply(request: TriggerApplyRequest) -> TriggerState`,
`inspect(name) -> TriggerState`, `list(*, cursor=None, limit=50) -> Page`,
`pause(name, *, expected_generation) -> TriggerState`,
`resume(name, *, expected_generation) -> TriggerState`, and
`occurrences/events(name, *, cursor=None, limit=50) -> Page` are the sole v1
management methods. The apply request
retains desired_enabled even though the registered revision excludes it. Applying
unchanged configuration with desired_enabled=true after pause resumes under CAS
without allocating another revision. None means create-if-absent only for apply; existing apply
requires the exact generation, even for no-op. Resume of exhausted stays
exhausted; error resume validates first, resets admission failures, increments
generation. No-op identical apply changes neither generation nor revision.
Any actual state/cursor mutation increments generation; semantic replacement
increments revision too. Pause/resume cannot cancel runs.

Replacement locks the row, closes the old revision, records old undecided due
slots through effective time as superseded, and selects strictly future new
slots. PostgreSQL cannot know future wall-clock commit time inside a transaction:
use one `clock_timestamp()` after obtaining the lock as its persisted effective
instant; it becomes visible only on commit. This is the precise meaning of
transaction-effective time, replacing the plan's loose "commit time" wording.
Old admitted work stays valid. Recovery first looks up committed slot identities
even after pause/revision advancement; only new work requires matching state.

`TriggerStore` handles CAS management/history; `TriggerAdmissionStore.admit(plan,
prepared) -> TriggerAdmissionResult` and `recover(occurrence_ids) -> RecoveryResult`
are typed atomic capabilities. Result outcomes are committed, not_committed,
unknown; unknown cannot release artifacts or authorize retry until fresh-connection
reconciliation. `TriggerScheduler.process_once() -> TriggerProcessResult` returns
admitted/skipped counts, decided ranges, conflicts, safe errors, remaining_work;
`serve()` owns its loop and shutdown. Generic external callbacks cannot submit SQL.

Replace `TaskClient.enqueue` with `submit(definition, *, request:
TaskSubmissionRequest) -> TaskSubmissionResult`. Request carries input_value,
files, metadata, available_at, queue selection and manual idempotency options;
owner scope is client-bound. A shared internal preparation service produces
`PreparedTaskSubmission` (definition/deployment identity, encrypted frozen input,
artifact ownership plan, provenance, queue and idempotency reservation). Queue
`submit_prepared(prepared, *, unit_of_work) -> TaskSubmissionResult` is a typed
internal participant; it never opens/commits another connection. Manual submit
opens the same unit of work itself. Remove enqueue_run and the superseded
submission DTOs; ordinary execution request/state/claim types remain where useful.

Persist envelopes `{format: <tag>, version: 1, payload: <closed object>}` for
trigger definition/state/occurrence/span/event, task submission and provenance.
Tags are `avalan.trigger.<record>`, `avalan.task.submission`,
`avalan.task.trigger_provenance`. Replace affected task request/context/container
and checkpoint task-provenance producers/consumers together; missing/unknown
versions fail explicitly, no legacy branch. Timestamps use the above codec;
enums use their listed strings, optionals use explicit null, integers reject
bool. Reconstruct frozen typed values and validate invariants on decode.
Opaque encrypted input stays inside existing task privacy envelopes; management
codecs never expose plaintext values/provider handles. Database schema head is
checked before writes; install the canonical migration in isolated schemas.
No automatic legacy reset, conversion or mixed-reader support is supplied.

Safe errors have code/path plus bounded non-sensitive message; no raw exception
or input. Closed initial codes: `trigger.invalid_config`,
`trigger.unsupported_version`, `trigger.invalid_binding`,
`trigger.invalid_schedule`, `trigger.unknown_timezone`,
`trigger.impossible_schedule`, `trigger.search_budget_exhausted`,
`trigger.datetime_overflow`, `trigger.past_schedule`,
`trigger.queue_required`, `trigger.capability_unavailable`,
`trigger.deployment_mismatch`, `trigger.artifact_not_durable`,
`trigger.provider_reference_expired`, `trigger.conflict`,
`trigger.store_incompatible`, `trigger.schema_mismatch`,
`trigger.admission_retryable`, `trigger.admission_exhausted`,
`trigger.commit_unknown`, `trigger.shutdown_timeout`. Existing task validation
issues remain nested safe diagnostics, not converted into successful validation.

## Execution identity and artifact ownership

Current task canonicalization hashes resolved schemas, provider instructions
and skill identity, but an execution ref string does not seal every agent/Flow
file or host capability. Do not claim it does. Scheduled activation additionally
requires a host-provided immutable `ExecutionDeployment` manifest, schema 1:
root-relative files and SHA-256 bytes, task canonical hash, resolved agent TOML,
referenced prompt/template files, Flow definition and referenced executable graph
files, nested agent/template references, skill manifest identity, runtime version,
container image digest if used, and a canonical allowlist of required tools and
runtime options. Manifest hash is execution_deployment_id. Resolve references
under configured roots; reject unresolved/dynamic includes or incomplete closure.
No secrets or mutable image tags belong in this identity. Model service weights,
external tools and data are outside the file-seal guarantee and are documented
as external dependencies. Flow support follows the configured task target runner:
CLI task registration installs FlowTaskTargetRunner with a strict resolver. Its
actual Flow contracts, graph/node registry, nested target capabilities and file
roots must validate; triggers cannot bypass or broaden those capabilities.

Apply validates the closure; workers verify it before each attempt and durable
resume, before provider/tool dispatch. Retain the immutable deployment for old
admitted runs after revision replacement. A mismatch fails the run safely; never
silently execute the current mutable checkout. Container transport verifies the
same digest inside its mounted immutable deployment. This is a new requirement
for Phase 2/5, not existing support; hosts unable to provide closure cannot activate.

At apply, local files become immutable durable artifacts under a staging owner;
the revision acquires ownership on commit. Trigger input may not rely on ephemeral
local paths. Provider references must either have durable validated lifetime
covering every scheduled use or be rejected (expiring references on unbounded
recurrence are invalid); validate again before admission. Paused revisions keep
ownership. Runs acquire independent references in the admission transaction.
Closing a revision releases its ownership only after all pending preparations
are reconciled and no undecided slot can need it. Run retention decrements only
that run's ownership. Deletion requires zero live owners, confirmed outcome,
staging age >= grace and a locked/rechecked tombstone before deleting bytes.
Unknown outcomes keep staging resources until recovery proves disposition.

## Integration audit at 38ae8eb5

| Boundary | Verified source / affected callers | Replacement or gate |
| --- | --- | --- |
| SDK preparation | task/client.py::TaskClient.enqueue | Shared submit/preparation; schemas, skill identity, privacy, file materialization, registration and idempotency stay mandatory |
| Transaction | task/queue.py::TaskQueue.enqueue_run, task/queues/pgsql.py::PgsqlTaskQueue.enqueue_run, pgsql.py::PgsqlUnitOfWork | Caller-owned participant, one transaction; queue-only enqueue is distinct worker transport and must be audited before removal |
| CLI | cli/commands/task.py queue branch (client.enqueue) | Migrate to submit; sibling trigger group and shared connection config in Phase 6 |
| Worker | task/worker.py::_queued input handling, task/context.py::TaskTargetContext, task/store.py::TaskExecutionRequest | Typed provenance through attempt/retry/resume; retain fencing |
| Persistence codecs | task/stores/pgsql.py::_request_to_payload/_request_from_payload, _context_to_payload/_context_from_payload; task/container.py | Versioned task submission and provenance; inspect interaction checkpoint embedding before replacement |
| Feature gates | task/feature_gate.py; task/validation.py; task/loader.py | JSON schema/task extra, PostgreSQL/worker extras and raw storage/remote URL restrictions remain enforced; FLOW_BACKED_TASKS metadata has no production callers and is not a global Flow gate |
| Flow target | task/targets/flow.py::FlowTaskTargetRunner.validate_definition, validate_flow_task_compatibility; cli/commands/task.py::_task_strict_flow_resolver and FlowTaskTargetRunner construction | Validate supported Flow contracts and strict graph/node capabilities; fixtures/flow.flow.toml passes the existing strict runner validator |
| Identity / lifetime | task/canonical.py, task/skills.py, task/input.py, task/artifact.py, task/retention.py | Deployment manifest and revision ownership are new; current task hash alone is insufficient |
| Tools and protocols | tool/a2a.py; server/mcp_tasks.py, server/routers/mcp.py, server/a2a/ | Separate protocol task abstractions; no TaskClient.enqueue caller found; no new management routes/tools in v1 |
| False positive | server/routers/responses.py::projection adapter enqueue | Stream projection, unrelated to queue submission; preserve |

Direct submission/queue callers in tests: task/client_test.py,
container_execution_test.py, full_e2e_matrix_test.py, queue_worker_e2e_test.py,
worker_test.py, queues/pgsql_protocol_test.py, stores/pgsql_queue_e2e_test.py,
stores/pgsql_queue_load_test.py; input/failure_matrix_task_e2e_test.py;
interaction/stores/conversation_atomic_pgsql_test.py and interaction_pgsql_e2e.py;
skill/skill_observability_phase12_test.py. Update these with Phase 2. Search
again when changing interfaces, including class imports and method references.
No `.enqueue(` submission example was found under docs at this baseline;
docs/task_file_delivery.md and docs/examples/tasks/ still require final API review.

Validation gates: focused contract tests now; Phase 1 parser/DST/import tests;
Phase 2+ transactional tests under tests/task/stores/pgsql_harness.py using
isolated schemas and fresh-process races. `make test-pgsql` installs dependencies
and uses scripts/task_pgsql_test_database.py --docker; inspect Docker/DSN availability
before running. No database behavior is proved by fixtures. Before every commit:
make lint, poetry run pytest --verbose -s, make test-coverage (100% src required).
When checksum-pinned task/queue/context sources change, run the applicable input,
conversation and PATCH verifiers (including verify_patch_types.py --through-phase
11 --include-planned), reseal only owned hashes, and never weaken gates. Phase 0
changes no src bytes or existing pinned manifest.

## Fixture contract

`fixtures/cases.json` identifies each TOML file, expected validity/error code,
and the validation boundary. These are executable specification inputs for
Phase 1, not evidence that a loader already rejects them. Phase 0 tests check
TOML parseability, complete inventory, policy combinations and existing task loading;
Phase 1 must run every case through its actual loader and task validation.
Reference time for registration cases is 2026-09-07T00:00:00.000000Z.

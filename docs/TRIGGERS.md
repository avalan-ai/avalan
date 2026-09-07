# Durable time triggers

Triggers admit a new queued task run for each scheduled occurrence. They do not
execute the task in the scheduler process. The mandatory owner/trigger/revision/
UTC-slot identity is independent of optional task HMAC idempotency. An admitted
run keeps its immutable deployment and trigger provenance through retries.

## Install and configure

```bash
pip install 'avalan[trigger,task-pgsql]'
export AVALAN_TASK_STORE_DSN='postgresql://user:password@host/database'
export AVALAN_TASK_STORE_SCHEMA='avalan_tasks'
export AVALAN_TASK_OWNER_SCOPE='my-service'
export AVALAN_TASK_ENCRYPTION_KEY_ID='service-key-v1'
export AVALAN_TASK_ENCRYPTION_KEY_B64='BASE64_OF_32_RANDOM_BYTES'
avalan task pgsql migrate head
```

Supply credentials through your service secret manager. Generate the AES-256 key
once with a cryptographically secure generator, retain it securely across service
restarts, and supply the same key ID and bytes to registration, scheduler, worker
and encrypted-artifact retention commands. Never generate a replacement key at
each startup. This host uses one configured key; keep the old key available for
all retained ciphertext. Key rotation requiring multiple simultaneous keys needs
an SDK key-provider host, not an undocumented CLI fallback.

`--store-dsn` and `--store-schema` override the corresponding `AVALAN_TASK_STORE_*`
variables. These are the single task/Flow/trigger/migration connection settings.
`--owner-scope` overrides `AVALAN_TASK_OWNER_SCOPE` for trigger commands. Owner
identity is trusted service configuration; trigger TOML and application input
cannot grant it. Use a dedicated database/schema and queue for each worker trust
boundary. Workers are trusted queue consumers, not per-owner authorization proxies.
Missing or incompatible schema fails before registration or byte writes. Run
migrations explicitly; the scheduler does not install or upgrade schema.

The trigger extra includes the cron parser, production AES-GCM and the lightweight
imports needed by the shipped native Agent/strict Flow host. It does not install
local inference engines, all vendor extras, a metrics SDK or PostgreSQL drivers;
`task-pgsql` supplies the last of these. SDK schedule/type imports remain separate
from CLI host composition. A native model/backend may need its own additional
extra and credentials. Existing task HMAC policies use
`AVALAN_TASK_HMAC_KEY_ID` and `AVALAN_TASK_HMAC_KEY_B64` on both scheduler and
worker; these are separate from the AES encryption key.

## Validate, preview and apply

The [examples](examples/triggers/README.md) use an actual strict Flow pass-through
node and require no model provider. Validate/preview read application files but
do not connect to PostgreSQL, retain deployment files, encrypt input or register
anything:

```bash
avalan trigger validate docs/examples/triggers/cron.trigger.toml --json
avalan trigger preview docs/examples/triggers/interval.trigger.toml \
  --reference-time 2026-09-07T12:00:00Z --count 5 --json
avalan trigger apply docs/examples/triggers/interval.trigger.toml \
  --deployment-root /var/lib/avalan/deployments --raw-storage-allowed --json
```

All trigger commands emit one structured JSON result without a console banner.
`--json` is an explicit spelling of that default. Names, safe diagnostic codes,
IDs and bounded history are output; input, key material and raw exceptions are
not. Preview includes UTC and local offset/timezone, policy and effective anchor.
An interval without a start uses an assumed preview anchor; registration chooses
its own anchor using the authoritative database time. Preview is not a promise
that registration will select those exact instants.

`--root` bounds trusted application references and defaults to the configuration
file directory. The CLI copies the verified task/agent/Flow files and reachable
templates into an immutable ID directory under `--deployment-root`. Keep that
private directory persistent and available at the same host path to schedulers
and workers; retain old ID directories after replacements. The host reloads the
manifest and verifies bytes/options/runtime on startup and each execution. A
changed runtime, deleted file, symlink escape or altered deployment fails closed.
Do not edit the retained directories. Keep compatible runtime versions available
for old admitted deployments; a new deployment does not authorize replacement
bytes for an old run. The catalog is not automatically garbage-collected.

Registration input is AES-GCM encrypted in PostgreSQL. Local static files are
materialized into the encrypted PostgreSQL byte backend before activation.
Physical bytes are shared safely across revision and run references. The explicit
`--raw-storage-allowed` flag authorizes encrypted raw retention, with default
`--retention-days 30` and `--max-artifact-bytes 16777216`. Task privacy rules still
apply; scheduled queued application input must remain recoverable. The examples
use `input="encrypt"`, one-day task raw retention and redacted output.

## Operate separate services

```bash
# Scheduler service: admits runs, never executes targets.
avalan trigger serve --deployment-root /var/lib/avalan/deployments \
  --raw-storage-allowed --metrics-file /var/lib/avalan/trigger.prom

# One bounded scheduler tick, suitable for an external supervisor.
avalan trigger serve --once --deployment-root /var/lib/avalan/deployments \
  --raw-storage-allowed --json

# Worker service invocation: consumes up to 100 available runs, then returns.
avalan task worker --limit 100 --deployment-root /var/lib/avalan/deployments \
  --raw-storage-allowed
```

Supervise/repeat the bounded worker invocation separately. Two schedulers can
share the database; locked admission, the mandatory unique occurrence and task
submission fences prevent duplicate new runs. Shutdown uses bounded owned-work
reconciliation. `shutdown_settled=false` or pending/unknown outcomes require
operator attention and retained resources, not an assumption of rollback.

Defaults are 100 discovered triggers, 100 decisions/admissions per tick, 10 per
trigger, 10,000 cumulative candidate evaluations, a one-second poll, five-second
tick and ten-second shutdown budget. Corresponding `serve` flags can lower or
raise these within the SDK's finite validated limits. Work exhaustion preserves
undecided cursor slots for later ticks. The database clock controls locked
admission decisions; waiting uses monotonic time. Cron uses five fields, literal
DOM/DOW OR semantics, explicit IANA zones, no nonexistent wall times and only the
first occurrence of an ambiguous wall time. Intervals are fixed anchored UTC
microsecond arithmetic. One-shot schedules complete or expire once.

## Inspect and control

```bash
avalan trigger list --limit 50 --cursor 0
avalan trigger inspect example-interval
avalan trigger occurrences example-interval --limit 50 --cursor 0
avalan trigger occurrences example-interval --coverage
avalan trigger events example-interval --event-file /var/log/avalan/events.jsonl
avalan trigger pause example-interval --expected-generation GENERATION
avalan trigger resume example-interval --expected-generation NEW_GENERATION
```

Read the current generation from inspect before a control or replacement. Stale
CAS returns `trigger.conflict`. Apply to an existing trigger also requires its
current `--expected-generation`. Declarative apply respects `enabled` (default
true): applying an enabled configuration to a paused trigger resumes it, whereas
changing a file on disk alone has no effect. Replacement retains admitted old
runs and marks superseded undecided coverage explicitly. Pause does not cancel
already admitted runs. Overlap skip checks all authoritative nonterminal runs,
including queued, retry-delayed, input-required and cancellation-requested runs
across revisions.

Inspect includes the current status/revision/cursor/retry/error plus five newest
occurrences and their observed task states. Occurrence and event pages are bounded
at 200, with opaque record version tags and offset cursors. Coverage spans are a
separate page; an unknown compressed count is not guessed. An admitted occurrence
means a run was queued, not that it succeeded. Use `task inspect`, task events,
output and usage for execution details.

Audit events commit in the same control/admission transaction. `events --event-file`
appends the selected committed page with stable event IDs; replay may duplicate
lines and consumers deduplicate by event ID. Sink failure is reported separately
and cannot roll back committed work. The optional Prometheus textfile contains
bounded disposition counters, conflicts, durable admission retry attempts, safe
errors, pending/health gauges and dispatch lag. Observed recovered decisions can
be counted again after restart; SQL history is authoritative. Dispatch lag is
admission decision time minus scheduled time, not worker queue wait or execution
duration. No IDs or input values become metric labels. A stalled observation
uses only the remaining tick wait budget and leaves explicit pending work. No
further tick overtakes it. Shutdown remains bounded but reports unsettled while
a sink or started file-writing thread remains active; resources close only after
that work settles.

## Retention and recovery

```bash
avalan task retention-sweep --encrypted-artifacts --raw-storage-allowed --limit 100
```

The same operator key unlocks the PostgreSQL artifact backend for this command.
Revision, staging and retained run references guard physical deletion. Paused
revisions keep their input. Scheduler cleanup uses `--orphan-grace-seconds 86400`
(default one day, maximum 30 days) as minimum age for settled unused staging;
unknown database/external-write outcomes never become safe merely through age.
Never remove retained deployment files or uncertain invocation staging while
admitted/paused/retrying/suspended work still needs them.

A lost acknowledgment is reconciled on a fresh READ COMMITTED connection after
the stable identity completion fence. Committed identity recovery precedes current
pause/revision/generation gates. A definite failed preparation reports
`not_committed` plus opaque recovery handles where resources remain uncertain.
Preserve those handles and durable state; do not blindly retry a manual submission
with newly generated identity or delete the referenced bytes.

## Supported host boundary and breaking changes

This CLI host supports native Agent and strict file-resolved Flow targets with
sealed files and no custom tool/skill/security injection. Native Agent files
must explicitly disable tool discovery with `enable = []` in the `[tool]` table. Unsupported supplied
worker settings, container hosts, unresolved closures and native continuation
service requirements are rejected explicitly. Ordinary task CLI runs keep their
existing tool/security settings. SDK verified native Agent/Flow continuation and
retained cold-loader support remain available to a host supplying the required
durable interaction/coordinator services; the CLI does not invent those services.

Scheduled containers require a trusted immutable image digest implementing the
invocation protocol, explicit rootful authority (false by default), and a backend
providing verified copied output bytes. The Phase5 Docker fixture's copied-byte
adapter is test-owned, not a shipped generic task image/backend. Default Docker
metadata alone is insufficient. Uncertain cleanup retains invocation staging
until the exact container is confirmed settled. No blanket backend certification
or new target engine is implied by this CLI.

Old `AVALAN_TASK_PGSQL_*`, task migration `--dsn`/`--schema`, `TaskClient.enqueue`,
`enqueue_run`, superseded submission DTOs and PG-private request/context codec
paths have no aliases. Use canonical `AVALAN_TASK_STORE_*`, `--store-*`,
`TaskClient.submit(definition, request=TaskSubmissionRequest(...))` and the closed
execution codecs. Queued payloads carry explicit versioned provenance; manual
requests carry null provenance. Upgrade callers, workers and schema together.
Unrelated memory CLI connection options are unchanged.

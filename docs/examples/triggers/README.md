# Runnable time-trigger examples

These configurations share `task.toml` and the strict `flow.toml` pass-through
node. No model provider or custom tool is needed. The application receives frozen
scheduled time, occurrence ID and revision bindings; task output is redacted.
Follow [operator setup](../../TRIGGERS.md) for PostgreSQL, a persistent private
catalog, and a persistent AES-256 key before apply/serve/worker commands.

| File | Behavior |
| --- | --- |
| `cron.trigger.toml` | Weekdays at 09:00 America/New_York; latest, overlap skip. |
| `interval.trigger.toml` | Anchored 60-second interval; latest, overlap skip. |
| `at.trigger.toml` | One firing at the explicit UTC timestamp; one-hour grace. |
| `catchup.trigger.toml` | All unexpired interval slots, bounded across ticks; overlap allow. |
| `paused.trigger.toml` | Initially disabled interval; skip policy, overlap skip. |

```bash
avalan trigger validate docs/examples/triggers/cron.trigger.toml
avalan trigger preview docs/examples/triggers/catchup.trigger.toml --count 5
avalan trigger apply docs/examples/triggers/interval.trigger.toml \
  --deployment-root /var/lib/avalan/deployments --raw-storage-allowed
avalan trigger serve --once --deployment-root /var/lib/avalan/deployments \
  --raw-storage-allowed
avalan task worker --once --deployment-root /var/lib/avalan/deployments \
  --raw-storage-allowed
```

The first interval slot is after registration, so an immediate `serve --once`
may correctly admit zero runs. Preview's assumed interval anchor is explicitly
unregistered. Change the one-shot example's distant `2099` timestamp to a future
aware timestamp for a near-term run. Never use a past timestamp to bypass grace.

Use `trigger inspect NAME` to obtain the current generation. Pause with
`--expected-generation`; a later declarative apply with `enabled=true` (or omitted)
and the new expected generation resumes it. Applying `paused.trigger.toml` keeps
it disabled until you change `enabled` or issue an explicit resume. Pausing does
not cancel runs already admitted. Catch-up and overlap policies affect future
admission decisions, not the identity/deployment of existing runs.

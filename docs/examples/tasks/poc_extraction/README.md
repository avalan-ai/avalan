# Invoice Extraction Task

This directory contains sanitized fixtures for extracting invoice line items
from one PDF. The native task sends the PDF directly to the hosted agent. The
image-flow task renders the PDF into ordered page images with built-in task
conversion, then sends only those generated image files to the same hosted
agent.

Both variants share the same agent instructions, user prompt, reasoning effort,
disabled tool selection, structured output format, and output schema. The only
provider envelope difference is the media block set: the native task sends one
`application/pdf` file block, while the image-flow task sends ordered
`image/png` image blocks.

Default tracked command from the repository root:

```bash
poetry run avalan task run docs/examples/tasks/poc_extraction/task.toml \
  --ephemeral \
  --pdf docs/examples/tasks/poc_extraction/sample.pdf \
  --json \
  --output extraction.json
```

Equivalent low-level file command:

```bash
poetry run avalan task run docs/examples/tasks/poc_extraction/task.toml \
  --ephemeral \
  --file input=docs/examples/tasks/poc_extraction/sample.pdf \
  --file-mime input=application/pdf \
  --json \
  --output extraction.json
```

Equivalent command after changing into this directory:

```bash
poetry run avalan task run task.toml --ephemeral --pdf sample.pdf --json --output extraction.json
```

Image-flow command after changing into this directory:

```bash
poetry run avalan task run image_flow_task.toml --ephemeral --pdf sample.pdf --json --output image.json
```

Equivalent image-flow command from the repository root:

```bash
poetry run avalan task run docs/examples/tasks/poc_extraction/image_flow_task.toml \
  --ephemeral \
  --pdf docs/examples/tasks/poc_extraction/sample.pdf \
  --json \
  --output image.json
```

Direct flow command after changing into this directory:

```bash
poetry run avalan flow run image_flow.toml --pdf sample.pdf --json --output image.json
```

CLI input files passed with `--pdf` and `--file` resolve from the caller's
current directory. TOML references such as agent, flow, and schema files resolve
from the TOML file that declares them. Output paths resolve from the caller's
current directory.

Validation checks expected for these fixtures:

* `task.toml` and `image_flow_task.toml` both accept one `application/pdf`
  input and reject pre-rendered `image/png` input at the task boundary.
* `image_flow.toml` uses a `file_convert` node with the `pdf_image` converter
  and an agent node configured with `files_input = "render_pages.files"` and
  `file_policy = "replace"`.
* The image-flow provider request must not include the original PDF block when
  replacement is configured.

The tracked sample is intentionally synthetic. It is suitable for validating
prompt composition, inline PDF delivery, reasoning settings, and structured
output shape without customer documents, network access, live services, or
provider credentials.

The image-flow path is semantic parity for this example, not a general
replacement for native PDF understanding. Rasterization drops text layers,
document structure, attachments, signatures, and hidden or non-rendered content.
Quality, latency, and cost depend on DPI, page count, converter behavior, and
provider vision limits.

Live smoke runs should be kept outside the default CI path and skipped unless
all of the following are true:

* A sanitized non-customer PDF is available.
* `AZURE_OPENAI_API_KEY` is set for the target deployment.
* The agent `engine.uri` names the intended deployment.
* The agent `engine.base_url` points at the intended Azure OpenAI `/openai/v1/`
  endpoint.
* The expected output is a JSON object with a non-empty `line_items` array.

Use an ignored local workspace for live inputs, outputs, and summaries. Keep
full extraction JSON and provider logs out of tracked files.

Ephemeral runs validate output shape only because usage records disappear when
the process exits:

```bash
poetry run avalan task run task.toml \
  --ephemeral \
  --pdf ./sample.pdf \
  --json \
  --output artifacts/native-output.json

poetry run avalan task run image_flow_task.toml \
  --ephemeral \
  --pdf ./sample.pdf \
  --json \
  --output artifacts/image-output.json
```

Durable CLI usage smoke requires a configured task store. Capture the run id
from the compact run summary, then inspect usage through the durable store:

```bash
poetry run avalan task pgsql migrate \
  --dsn "$AVALAN_TASK_STORE_DSN" \
  --schema "$AVALAN_TASK_STORE_SCHEMA" \
  head

poetry run avalan task pgsql check \
  --dsn "$AVALAN_TASK_STORE_DSN" \
  --schema "$AVALAN_TASK_STORE_SCHEMA"

poetry run avalan task run task.toml \
  --store-dsn "$AVALAN_TASK_STORE_DSN" \
  --store-schema "$AVALAN_TASK_STORE_SCHEMA" \
  --pdf ./sample.pdf \
  --output artifacts/native-output.json \
  | tee artifacts/native-run.txt

RUN_ID="$(awk '/^Task run completed:/ {print $4}' artifacts/native-run.txt)"

poetry run avalan task inspect "$RUN_ID" \
  --store-dsn "$AVALAN_TASK_STORE_DSN" \
  --store-schema "$AVALAN_TASK_STORE_SCHEMA" \
  > artifacts/native-inspect.json

poetry run avalan task usage "$RUN_ID" \
  --store-dsn "$AVALAN_TASK_STORE_DSN" \
  --store-schema "$AVALAN_TASK_STORE_SCHEMA" \
  --source exact \
  > artifacts/native-usage.txt

poetry run avalan task run image_flow_task.toml \
  --store-dsn "$AVALAN_TASK_STORE_DSN" \
  --store-schema "$AVALAN_TASK_STORE_SCHEMA" \
  --pdf ./sample.pdf \
  --output artifacts/image-output.json \
  | tee artifacts/image-run.txt

RUN_ID="$(awk '/^Task run completed:/ {print $4}' artifacts/image-run.txt)"

poetry run avalan task usage "$RUN_ID" \
  --store-dsn "$AVALAN_TASK_STORE_DSN" \
  --store-schema "$AVALAN_TASK_STORE_SCHEMA" \
  --source exact \
  > artifacts/image-usage.txt
```

Use `--output` without `--json` for durable usage smoke. `--json` prints only
the structured extraction object and suppresses the run summary line needed for
follow-up inspection commands.

SDK smoke can validate live usage without a durable store by inspecting the
same in-memory client that executed the task:

```python
from pathlib import Path

from avalan.task import (
    TaskClient,
    TaskDefinitionLoader,
    UsageTotals,
    usage_smoke_summary,
)
from avalan.task.stores import InMemoryTaskStore

definition = TaskDefinitionLoader().load(Path("task.toml"))
target = ...  # Configure the same agent target runner used by the CLI.
client = TaskClient(InMemoryTaskStore(), target=target)
result = await client.run(
    definition,
    input_value=TaskClient.local_file(
        "sample.pdf",
        mime_type="application/pdf",
    ),
)
inspection = await client.inspect(result.run.run_id)
totals = inspection.usage_totals or UsageTotals()
output = result.output if isinstance(result.output, dict) else {}
summary = usage_smoke_summary(
    task_variant="native_pdf",
    success=result.run.state.value == "succeeded",
    schema_valid=isinstance(result.output, dict),
    expected_output_match=bool(output.get("line_items")),
    totals=totals,
    required_counters=("input_tokens", "output_tokens", "total_tokens"),
)
```

For Azure GPT-5 reasoning validation, include `reasoning_tokens` in
`required_counters` when the provider response reports it. Cache counters
should be checked for presence and whether they are `missing`, `reported_zero`,
or `reported_positive`; do not fail a single live run only because a cache hit
is reported as zero.

Default CI uses fake-provider tests instead of network calls:

```bash
poetry run pytest --verbose -s \
  tests/task/direct_client_e2e_test.py \
  tests/task/full_e2e_matrix_test.py \
  tests/cli/task_test.py
```

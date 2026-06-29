# Task Examples

These task examples show common authoring patterns for Avalan task
definitions. They are designed to validate in CI without hosted model keys or
model downloads; actual execution still depends on the referenced agent and
runtime configuration.

## Valid Definitions

- [minimal_string_agent.task.toml](minimal_string_agent.task.toml) accepts one
  string and returns text.
- [structured_json.task.toml](structured_json.task.toml) accepts object input
  validated by JSON Schema and returns JSON.
- [file_document.task.toml](file_document.task.toml) accepts one local document
  file with conversion enabled.
- [file_array_comparison.task.toml](file_array_comparison.task.toml) accepts a
  bounded list of document files.
- [large_direct_file.task.toml](large_direct_file.task.toml) documents a direct
  file task with bounded local materialization, conversion, retrieval, and
  map-reduce fallback.
- [provider_reference_direct.task.toml](provider_reference_direct.task.toml)
  accepts provider file ids, hosted URLs, and object-store references supplied
  by CLI or SDK descriptors.
- [poc_extraction/task.toml](poc_extraction/task.toml) accepts one PDF and
  returns a structured invoice extraction object.
- [poc_extraction/flow_task.toml](poc_extraction/flow_task.toml) runs the same
  extraction through a native flow with an agent node.
- [pipeline_agent.task.toml](pipeline_agent.task.toml) runs an agent whose TOML
  explicitly opts into `shell.pipeline`.
- [pipeline_flow.task.toml](pipeline_flow.task.toml) runs a flow `tool` node
  backed by `shell.pipeline` when an SDK, host, or worker runtime provides a
  configured tool resolver and trusted shell settings, returning the formatted
  pipeline result text.
- [local_multimodal_media.task.toml](local_multimodal_media.task.toml) shows a
  local multimodal media contract without downloading a model in CI.
- [queued_file_task.task.toml](queued_file_task.task.toml) shows a queued file
  task with durable store requirements.
- [output_artifact.task.toml](output_artifact.task.toml) declares output
  artifact references.
- [sdk_definition.py](sdk_definition.py) builds the same logical definition as
  `structured_json.task.toml` from Python.
- [file_inputs_sdk.py](file_inputs_sdk.py) builds file task definitions and
  local/provider descriptors from Python.

Validate one definition with:

```bash
poetry run avalan task validate docs/examples/tasks/structured_json.task.toml \
  --input-json '{"question":"What changed?","priority":2}'
```

Run direct examples locally only when the referenced agent and provider are
configured:

```bash
poetry run avalan task run docs/examples/tasks/large_direct_file.task.toml \
  --ephemeral \
  --file document=docs/examples/playground/invoice.pdf \
  --file-mime document=application/pdf \
  --file-conversion document=text

poetry run avalan task run docs/examples/tasks/provider_reference_direct.task.toml \
  --ephemeral \
  --provider-file-id document=openai:file_abc123 \
  --file-mime document=application/pdf \
  --file-size document=2048

poetry run avalan task run docs/examples/tasks/provider_reference_direct.task.toml \
  --ephemeral \
  --hosted-url document=anthropic:https://files.example.test/report.pdf

poetry run avalan task run docs/examples/tasks/provider_reference_direct.task.toml \
  --ephemeral \
  --object-store-uri document=google:gs://bucket/report.pdf

poetry run avalan task run docs/examples/tasks/structured_json.task.toml \
  --ephemeral \
  --input-json '{"question":"What changed?","priority":2}' \
  --json \
  --output result.json

poetry run avalan task run docs/examples/tasks/poc_extraction/task.toml --ephemeral --pdf ./sample.pdf --json --output extraction.json

poetry run avalan task run docs/examples/tasks/poc_extraction/flow_task.toml --ephemeral --pdf ./sample.pdf --json --output extraction.json

poetry run avalan flow run docs/examples/tasks/poc_extraction/flow.toml --pdf ./sample.pdf --json --output extraction.json

poetry run avalan task run docs/examples/tasks/pipeline_agent.task.toml \
  --ephemeral \
  --input "Count shell.pipeline mentions under src and summarize the result"
```

The `pipeline_flow.task.toml` example is a flow-backed fixture for runtimes
that supply a `ToolManager` with `shell.pipeline` enabled. The standalone task
CLI uses the default flow validator for this target, which does not register
strict tool nodes, so CI covers it through focused loader/validator tests
rather than as a direct `task run` command:

```bash
poetry run pytest \
  tests/flow/validator_test.py::FlowValidatorTestCase::test_docs_shell_pipeline_flow_examples_validate_with_runtime \
  tests/task/task_loader_test.py::TaskDefinitionLoaderTest::test_docs_shell_pipeline_task_examples_load \
  -q
```

Queue examples require PostgreSQL task storage, a migrated schema, HMAC keys,
and an artifact root:

```bash
poetry run avalan task enqueue docs/examples/tasks/queued_file_task.task.toml \
  --store-dsn "$AVALAN_TASK_STORE_DSN" \
  --queue documents \
  --file documents=docs/examples/playground/invoice.pdf \
  --file-mime documents=application/pdf \
  --file-conversion documents=text \
  --wait

poetry run avalan task worker \
  --store-dsn "$AVALAN_TASK_STORE_DSN" \
  --queue default \
  --tool shell.pipeline \
  --tool-shell-allow-pipelines \
  --tool-shell-max-pipeline-stages 3
```

For local text models, use a file task with `file_conversions` and request a
conversion so the target receives text rather than a native file block. For
retrieval and map-reduce, use the same converted text path with file sizes or
context limits that exceed inline delivery; the planner will choose bounded
retrieval context first and map-reduce when retrieval cannot fit.

## Invalid Definitions

The files in [invalid](invalid) demonstrate safe diagnostics for path escape,
unsafe raw privacy, unsupported targets, and invalid schema declarations. They
are intentionally invalid and should not be used as runnable tasks.

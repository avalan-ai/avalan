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
- [output_artifact.task.toml](output_artifact.task.toml) declares output
  artifact references.
- [sdk_definition.py](sdk_definition.py) builds the same logical definition as
  `structured_json.task.toml` from Python.

Validate one definition with:

```bash
poetry run avalan task validate docs/examples/tasks/structured_json.task.toml \
  --input-json '{"question":"What changed?","priority":2}'
```

## Invalid Definitions

The files in [invalid](invalid) demonstrate safe diagnostics for path escape,
unsafe raw privacy, unsupported targets, and invalid schema declarations. They
are intentionally invalid and should not be used as runnable tasks.

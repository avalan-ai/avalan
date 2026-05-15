# DS4 Native Backend

Avalan can run DS4-supported DeepSeek V4 Flash GGUF files through the
native `ds4` backend. This backend is intentionally model-specific: it is
not a generic GGUF loader, and it should not be used for arbitrary GGUF
models.

Install the optional backend before running DS4 models:

```bash
python3 -m pip install -U "avalan[ds4]"
```

The production targets are macOS arm64 with Metal and Linux with CUDA. The
CPU backend exists for diagnostics and correctness checks only.

DS4 model files are opened directly from the local filesystem. Avalan does
not require `HF_TOKEN` and does not contact Hugging Face for DS4 runs.

## Basic Generation

Set a model path first. The file must be a DS4-supported DeepSeek V4 Flash
GGUF:

```bash
export DS4_MODEL=/path/to/ds4flash.gguf
```

Run greedy generation with an explicit native backend:

```bash
printf '%s\n' 'Write a short greeting.' \
  | avalan model run "ai://local/${DS4_MODEL}?backend=ds4" \
      --ds4-ctx 4096 \
      --ds4-native-backend metal \
      --max-new-tokens 64 \
      --temperature 0
```

Use `--ds4-native-backend cuda` on Linux CUDA builds. Use
`--ds4-native-backend cpu` only for diagnostics.

Relative paths are resolved from the current working directory. Absolute
paths use two slashes after `local`, because the first slash separates the
URI authority from the path and the second slash belongs to the filesystem
path:

```bash
avalan model run "ai://local/../pyds4/.local/ds4/ds4flash.gguf" --backend ds4
avalan model run "ai://local//Users/me/models/ds4flash.gguf" --backend ds4
```

Percent-encode path characters that are not URL-safe. An encoded absolute
path is also accepted:

```bash
avalan model run "ai://local/%2FUsers/me/DS4%20models/ds4flash.gguf" --backend ds4
```

The DS4-specific `model run` flags are:

- `--ds4-ctx`
- `--ds4-native-backend`
- `--ds4-mtp`
- `--ds4-mtp-draft`
- `--ds4-mtp-margin`
- `--ds4-warm-weights`
- `--ds4-quality`

## Disabled Reasoning

Disable Avalan reasoning parsing and request DS4's no-thinking mode with
`--no-reasoning`:

```bash
printf '%s\n' 'Answer in one sentence: what is distillation?' \
  | avalan model run \
      "ai://local/${DS4_MODEL}?backend=ds4&ds4_ctx=4096&ds4_native_backend=metal" \
      --no-reasoning \
      --max-new-tokens 64 \
      --temperature 0
```

Reasoning efforts map to DS4 thinking modes when reasoning remains enabled.
For example, `--reasoning-effort high` requests DS4 high thinking mode, and
`--reasoning-effort max` requests max thinking mode.

## URI Backend Config

DS4 backend settings can be provided in the `ai://` URI with `ds4_`
parameters:

```bash
printf '%s\n' 'List two practical GPU memory tips.' \
  | avalan model run \
      "ai://local/${DS4_MODEL}?backend=ds4&ds4_ctx=8192&ds4_native_backend=cuda&ds4_warm_weights=true&ds4_quality=true" \
      --max-new-tokens 96 \
      --temperature 0
```

Common URI keys are:

- `backend=ds4`
- `ds4_ctx`
- `ds4_native_backend`
- `ds4_mtp`
- `ds4_mtp_draft`
- `ds4_mtp_margin`
- `ds4_warm_weights`
- `ds4_quality`

Unknown `ds4_` URI parameters are rejected so configuration mistakes fail
early.

## Current Limitations

- DS4 only supports DS4-supported DeepSeek V4 Flash GGUF files in Avalan.
- Generic GGUF models are not supported by this backend.
- Native DS4 tool calls are not implemented yet. Do not use `--tool`,
  `--tools`, or tool-role histories with `--backend ds4`.
- CPU inference is a debug/reference path, not a production target.

## Integration Tests

Real-model DS4 tests are skipped unless a model path is configured:

```bash
AVALAN_DS4_MODEL=/path/to/ds4flash.gguf \
AVALAN_DS4_BACKEND=metal \
AVALAN_DS4_CTX=4096 \
poetry run pytest --verbose -s \
  tests/model/ds4_integration_test.py \
  tests/model/ds4_phase3_integration_test.py
```

`AVALAN_DS4_BACKEND` accepts `metal`, `cuda`, or `cpu`; use `cpu` only for
diagnostic runs. `AVALAN_DS4_CTX` defaults to `4096` when omitted.

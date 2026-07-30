# Avalan Shell Code Container

This image is a small `linux/arm64` shell-tool payload for code
exploration with Apple `container`. It includes the command binaries used
by Avalan's common read-only shell tools:

- `cat`
- `date`
- `file`
- `find`
- `git`
- `head`
- `jq`
- `kill`
- `ls`
- `lsof`
- `montage`
- `pgrep`
- `ps`
- `rg`
- `sed`
- `tail`
- `wc`
- `awk`

## Build With Apple Container

Start Apple `container` first. If it asks to install the recommended
default kernel, answer `Y`.

```sh
container system start
```

Build the image from the repository root:

```sh
container build \
  --platform linux/arm64 \
  --tag avalan-shell-code:latest \
  --file docs/containers/shell-code/Dockerfile \
  docs/containers/shell-code
```

Smoke test the image against the current checkout:

```sh
container run --rm \
  --platform linux/arm64 \
  --read-only \
  --network none \
  --no-dns \
  --mount type=bind,source="$(pwd)",target=/workspace,readonly \
  --workdir /workspace \
  avalan-shell-code:latest \
  rg -n "stream" src/avalan
```

Inspect the image and copy the `sha256:...` digest shown by Apple
`container`:

```sh
container image inspect avalan-shell-code:latest | jq -r '.[].configuration.descriptor.digest'
```

Use that digest with Avalan as:

```sh
--tool-container-image "avalan-shell-code@sha256:<digest>"
```

## Code Exploration Agent

Use explicit shell tools that match this image instead of `shell.*`:

```sh
echo "At a high level, how is token streaming handled in this codebase? Answer in two paragraphs." \
  | poetry run avalan agent run \
      --engine-uri 'ai://env:AZURE_OPENAI_API_KEY@openai/gpt-5?azure_api_version=preview' \
      --engine-base-url 'https://vdocintel-staging-openai.openai.azure.com/openai/v1/' \
      --tool shell.cat \
      --tool shell.date \
      --tool shell.file \
      --tool shell.find \
      --tool shell.git_log \
      --tool shell.git_status \
      --tool shell.head \
      --tool shell.jq \
      --tool shell.ls \
      --tool shell.nl \
      --tool shell.lsof \
      --tool shell.montage \
      --tool shell.pgrep \
      --tool shell.ps \
      --tool shell.rg \
      --tool shell.sed \
      --tool shell.tail \
      --tool shell.wc \
      --tool shell.awk \
      --tool-shell-executable-search-path /opt/homebrew/bin \
      --tool-shell-executable-search-path /opt/homebrew/opt/coreutils/libexec/gnubin \
      --tool-shell-executable-search-path /bin \
      --tool-shell-executable-search-path /usr/bin \
      --tool-shell-executable-search-path /usr/sbin \
      --tool-shell-executable-search-path /sbin \
      --tool-shell-backend container \
      --tool-shell-workspace-root . \
      --tool-shell-cwd . \
      --tool-container-backend apple-container \
      --tool-container-profile workspace-readonly \
      --tool-container-image 'avalan-shell-code@sha256:<digest>' \
      --tool-container-workspace-root . \
      --tool-container-pull-policy never \
      --tool-container-platform linux/arm64 \
      --tool-container-network-mode none \
      --tool-container-timeout-seconds 30 \
      --tool-shell-container-profile workspace-readonly \
      --tool-shell-container-required \
      --tool-shell-allow-media-tools \
      --tool-shell-allow-process-tools \
      --memory-recent \
      --run-max-new-tokens 8192 \
      --maximum-tool-cycles 64 \
      --name 'Tool' \
      --system 'You are a python code exploration expert named Leo that uses tools to answer.' \
      --theme basic
```

`--tool-shell-container-required` makes execution fail closed if Apple
`container`, the selected image, or the workspace mount is unavailable. Shell
tool execution does not fall back to the host when the container profile is
required.

`shell.date` is supplied by the image's `coreutils` package. Calls use only
optional `-u` and a fixed preset (`default`, `date`, `time`, `iso8601`, or
`unix`) or the separately validated `custom_format`. The custom form accepts
1-128 printable ASCII bytes, permits only the portable directive allowlist
documented in `docs/CLI.md`, and is mutually exclusive with non-default
presets. Avalan prepends the required `+`, so caller text cannot become an
option or date operand. `default` leaves off the format operand, while
`iso8601` emits `YYYY-MM-DDTHH:MM:SS+HHMM` or the corresponding negative
numeric offset; UTC is `+0000`. The reported clock and local timezone belong
to the container backend, not necessarily the host.

`shell.lsof` requires `--tool-shell-allow-process-tools` and inspects exactly
one PID. It returns only bounded numeric-descriptor metadata with canonical
type categories (`regular`, `directory`, `character`, `block`, `pipe`, `ipv4`,
`ipv6`, `unix_socket`, `socket`, `event`, or `other`) and protocol categories
(`tcp`, `udp`, `udplite`, `other`, or `-`). It does not expose filenames,
paths, command names, users, or network endpoints. Its row limit is applied
after bounded output capture and does not stop the underlying kernel scan. The
stdout-byte cap bounds retained subprocess capture and complete public rows;
it does not bound kernel scan duration or execution. The timeout is the
execution-duration bound. Process visibility is relative to the selected
backend: a host PID is not a container PID, and this one-shot container does
not preserve process identity between calls. The image installs the separate
Alpine `lsof` package. Some platform builds can use a per-user device cache;
keep the filesystem read-only and verify the behavior of any replacement
package.

The image includes the `kill` binary for completeness, but `shell.kill` uses a
local-only identity contract and fails closed under container execution.
One-shot container PID namespaces cannot safely carry process identity across
calls. The read-only exploration example does not expose it. Local operators
can enable it explicitly with both
`--tool-shell-allow-process-tools` and
`--tool-shell-allow-process-control`.

The image also installs ImageMagick, its Alpine JPEG codec subpackage, and a
default font package for `shell.montage`. Montage receives explicit image
paths rather than shell patterns or brace expressions, and returns one
bounded generated artifact. Each path selects its first image scene, embedded
labels and captions are cleared, ImageMagick resource limits are explicit,
and the returned raster signature and dimensions are checked. Container-mode
lowering selects the packaged `DejaVu-Sans` font; host and sandbox operators
can instead set
`--tool-shell-montage-font` for their ImageMagick installation. Enable the
tool with `--tool shell.montage` and
`--tool-shell-allow-media-tools`. A requested `output_filename` is a basename
for the returned artifact; it is not an arbitrary writable workspace path.

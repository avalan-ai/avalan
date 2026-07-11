# Avalan Shell Code Container

This image is a small `linux/arm64` shell-tool payload for code
exploration with Apple `container`. It includes the command binaries used
by Avalan's common read-only shell tools:

- `cat`
- `file`
- `find`
- `git`
- `head`
- `jq`
- `ls`
- `pgrep`
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
      --tool shell.file \
      --tool shell.find \
      --tool shell.git_log \
      --tool shell.git_status \
      --tool shell.head \
      --tool shell.jq \
      --tool shell.ls \
      --tool shell.nl \
      --tool shell.pgrep \
      --tool shell.rg \
      --tool shell.sed \
      --tool shell.tail \
      --tool shell.wc \
      --tool shell.awk \
      --tool-shell-executable-search-path /opt/homebrew/bin \
      --tool-shell-executable-search-path /opt/homebrew/opt/coreutils/libexec/gnubin \
      --tool-shell-executable-search-path /bin \
      --tool-shell-executable-search-path /usr/bin \
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

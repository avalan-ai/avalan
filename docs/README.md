**avalan**[^1] empowers developers and enterprises to
effortlessly build, orchestrate, and deploy intelligent
AI agents—locally or in the cloud—across millions of models through
an intuitive, unified SDK and CLI. With robust
multi-backend support ([transformers](https://github.com/huggingface/transformers),
[vLLM](https://github.com/vllm-project/vllm),
[mlx-lm](https://github.com/ml-explore/mlx-lm)), first-class
support of multiple AI protocols (MCP, A2A), plus native
integrations for OpenRouter, Ollama, OpenAI, DeepSeek, Gemini, and
beyond, avalan enables you to select the optimal  engine
tailored specifically to each use case.

Its versatile multi-modal architecture bridges NLP, vision, and
audio domains, allowing seamless integration and interaction among
diverse models within sophisticated workflows. Enhanced by built-in
memory management and state-of-the-art reasoning
capabilities—including ReACT tooling,
adaptive planning, and persistent long-term context—your agents
continuously learn, evolve, and intelligently respond to changing
environments.

avalan’s intuitive pipeline design supports advanced branching,
conditional filtering, and recursive flow-of-flows execution,
empowering you to create intricate, scalable AI workflows with
precision and ease. Comprehensive observability ensures complete
transparency through real-time metrics, detailed event tracing,
and statistical dashboards, facilitating deep insights,
optimization, and robust governance.

From solo developers prototyping innovative ideas locally to
enterprises deploying mission-critical AI systems across
distributed infrastructures, avalan provides flexibility,
visibility, and performance to confidently accelerate your
AI innovation journey.

# Development

## Releasing

You'll need the [github CLI](https://github.com/cli/cli)
for publishing versions. On MacOS, a simple `brew install gh` will do,
after which login with `gh auth login`.

Ensure you have the poetry-dynamic-versioning plugin:

```bash
poetry self add "poetry-dynamic-versioning[plugin]"
```

Patch new version (adjust to `minor` or `major` as appropriate):

```bash
poetry version patch
```

Commit the version patch:

```bash
git commit -m "Bumping version to vX.Y.Z"
```

Release version X.Y.Z:

```bash
git tag vX.Y.Z -m "Release vX.Y.Z"
```

Push:

```bash
git push origin main --follow-tags
```

Publish to PyPI with:

```bash
poetry publish --build
```

Add the release to Github:

```
gh release create vX.Y.Z \
  --title "vX.Y.Z" \
  --notes-file <(git log --format=%B -n1 vX.Y.Z)
```

## Running tests

If you want to run the tests, install the `tests` extra packages:

```bash
poetry install --extras test
```

You can run the tests with:

```bash
poetry run pytest --verbose
```

## Translations

If new translated strings are added (via `_()` and/or `_n()`), the gettext template file will need to be updated. Here's how you extract all `_()` and `_n()` references within the `src/` folder to `locale/avalan.pot`:

```bash
find src/avalan/. -name "*.py" | xargs xgettext \
    --language=Python \
    --keyword=_ \
    --keyword=_n \
    --package-name 'avalan' \
    --package-version `cat src/avalan/VERSION.txt` \
    --output=locale/avalan.pot
```

If you are translating to a new language (such as `es`), create the folder structure first:

```bash
mkdir -p locale/es/LC_MESSAGES
```

Update the existing `es` translation file with changes:

```bash
msgmerge --update locale/es/LC_MESSAGES/avalan.po locale/avalan.pot
```

If the `es` translation file does not exist, create it:

```bash
msginit --locale=es \
        --input=locale/avalan.pot \
        --output=locale/es/LC_MESSAGES/avalan.po
```

Edit the `locale/es/LC_MESSAGES/avalan.po` translation file filling in the needed `msgstr`. When you are done translating, compile it:

```bash
msgfmt --output-file=locale/es/LC_MESSAGES/avalan.mo \
       locale/es/LC_MESSAGES/avalan.po
```

If you are recording CLI usage and wish to share it in documentation, save
it as a 480p MOV file, say `recording.mov`, and then generate the palette
before conversion:

```bash
ffmpeg -i recording.mov \
    -vf "fps=2,scale=480:-1:flags=lanczos,palettegen" \
    /tmp/recording_palette.png
```

Now convert the MOV recording to GIF using the previously generated palette:

```bash
ffmpeg -i recording.mov \
    -i /tmp/recording_palette.png \
    -filter_complex "fps=2,scale=480:-1:flags=lanczos[x];[x][1:v]paletteuse" \
    docs/images/recording.gif
```


[^1]: Autonomous Virtually Assisted Language Agent Network

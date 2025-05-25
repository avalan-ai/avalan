<p align="center">
  <!-- Build status via GitHub Actions -->
  <img src="https://github.com/avalan-ai/avalan/actions/workflows/test.yml/badge.svg" alt="Tests" />

  <!-- PyPI version -->
  <img src="https://badge.fury.io/py/avalan.svg" alt="PyPI version" />

  <!-- Supported Python versions -->
  <img src="https://img.shields.io/pypi/pyversions/avalan.svg" alt="Python versions" />

  <!-- Total downloads -->
  <img src="https://img.shields.io/pypi/dm/avalan.svg" alt="Downloads" />

  <!-- Code coverage (Codecov) -->
  <img src="https://codecov.io/gh/avalan-ai/avalan/branch/main/graph/badge.svg" alt="Coverage" />

  <!-- License -->
  <img src="https://img.shields.io/pypi/l/avalan.svg" alt="License" />

  <!-- Latest commit -->
  <img src="https://img.shields.io/github/last-commit/avalan-ai/avalan.svg" alt="GitHub last commit" />
</p>

**avalan**[^1] is a framework that leverages [transformers](https://github.com/huggingface/transformers)
and [vLLM](https://github.com/vllm-project/vllm) to facilitate the
orchestration of AI agents as well as the execution, distillation, and
training of LMs, either directly via source code, or with its CLI.

Go through [the CLI documentation](#the-cli) to see what it can do, but if you
want to jump right in, run a model:

```bash
avalan model run meta-llama/Meta-Llama-3-8B-Instruct
```

![Example use of the CLI showing prompt based inference](https://avalan.ai/images/running_local_inference_example.gif)

Here's an example where we are getting detailed token generation information
using a particular model (check the GPU working at the bottom), and specifying
our prompt directly on the command line:

```bash
echo 'hello, who are you? answer in no less than 100 words' | \
    avalan model run deepseek-ai/deepseek-llm-7b-chat \
               --display-tokens \
               --display-pause 25
```

![Example use of the CLI showing token distributions](https://avalan.ai/images/running_token_distribution_example.gif)

Through the avalan microframework, you can easily integrate real time token
streaming with your own code, as [this example shows](https://github.com/avalan-ai/avalan/blob/main/docs/examples/text_generation.py):

```python
from asyncio import run
from avalan.model.entities import GenerationSettings
from avalan.model.nlp.text import TextGenerationModel

async def example() -> None:
    print("Loading model... ", end="", flush=True)
    with TextGenerationModel("meta-llama/Meta-Llama-3-8B-Instruct") as lm:
        print("DONE.", flush=True)

        system_prompt = """
            You are Leo Messi, the greatest football/soccer player of all
            times.
        """

        async for token in await lm(
            "Who are you?",
            system_prompt=system_prompt,
            settings=GenerationSettings(temperature=0.9, max_new_tokens=256)
        ):
            print(token, end="", flush=True)

if __name__ == "__main__":
    run(example())
```

Check the GPU hard at work towards the bottom:

![Running the local inference example](https://avalan.ai/images/running_local_inference_example_messi.gif)

Besides natural language processing, you can also work with other types of
models, such as those that handle vision, like the following
[image classification example](https://github.com/avalan-ai/avalan/blob/main/docs/examples/vision_image_classification.py):

```python
from asyncio import run
from avalan.model.vision.detection import ObjectDetectionModel
import os
import sys

async def example(path: str) -> None:
    print("Loading model... ", end="", flush=True)
    with ObjectDetectionModel("facebook/detr-resnet-50") as od:
        print(f"DONE. Running classification for {path}", flush=True)

        for entity in await od(path):
            print(entity, flush=True)

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv)==2 and os.path.isfile(sys.argv[1]) \
           else sys.exit(f"Usage: {sys.argv[0]} <valid_file_path>")
    run(example(path))
```

Looking for sequence to sequence models? Just as easy, like this [summarization
example shows](https://github.com/avalan-ai/avalan/blob/main/docs/examples/seq2seq_summarization.py):

```python
from asyncio import run
from avalan.model.entities import GenerationSettings
from avalan.model.nlp.sequence import SequenceToSequenceModel

async def example() -> None:
    print("Loading model... ", end="", flush=True)
    with SequenceToSequenceModel("facebook/bart-large-cnn") as s:
        print("DONE.", flush=True)

        text = """
            Andres Cuccittini, commonly known as Andy Cucci, is an Argentine
            professional footballer who plays as a forward for the Argentina
            national team. Regarded by many as the greatest footballer of all
            time, Cucci has achieved unparalleled success throughout his career.

            Born on July 25, 1988, in Ushuaia, Argentina, Cucci began playing
            football at a young age and joined the Boca Juniors youth
            academy.
            """

        summary = await s(text, GenerationSettings(num_beams=4, max_length=60))
        print(summary)

if __name__ == "__main__":
    run(example())
```

You can also perform translations, as [the following example shows](https://github.com/avalan-ai/avalan/blob/main/docs/examples/seq2seq_translation.py).
You'll need the `translation` extra installed for this to run:

```python
from asyncio import run
from avalan.model.entities import GenerationSettings
from avalan.model.nlp.sequence import TranslationModel

async def example() -> None:
    print("Loading model... ", end="", flush=True)
    with TranslationModel("facebook/mbart-large-50-many-to-many-mmt") as t:
        print("DONE.", flush=True)

        text = """
            Lionel Messi, commonly known as Leo Messi, is an Argentine
            professional footballer who plays as a forward for the Argentina
            national team. Regarded by many as the greatest footballer of all
            time, Messi has achieved unparalleled success throughout his career.
        """

        translation = await t(
            text,
            source_language="en_US",
            destination_language="es_XX",
            settings=GenerationSettings(num_beams=4, max_length=512)
        )

        print(" ".join([line.strip() for line in text.splitlines()]).strip())
        print("-" * 12)
        print(translation)

if __name__ == "__main__":
    run(example())
```

You can also create AI agents. Let's create one to handle gettext translations.
Create a file named [agent_gettext_translator.toml](https://github.com/avalan-ai/avalan/blob/main/docs/examples.agent_gettext_translator.toml)
with the following contents:

```toml
[agent]
role = """
You are an expert translator that specializes in translating gettext
translation files.
"""
task = """
Your task is to translate the given gettext template file,
from the original {{source_language}} to {{destination_language}}.
"""
instructions = """
The text to translate is marked with `msgid`, and it's quoted.
Your translation should be defined in `msgstr`.
"""
rules = [
    """
    Ensure you keep the gettext format intact, only altering
    the `msgstr` section.
    """,
    """
    Respond only with the translated file.
    """
]

[template]
source_language = "English"
destination_language = "Spanish"

[engine]
uri = "meta-llama/Meta-Llama-3-8B-Instruct"

[run]
use_cache = true
max_new_tokens = 1024
skip_special_tokens = true
```

You can now run your agent. Let's give it a gettext translation template file,
have our agent translate it for us, and show a visual difference of what the
agent changed:

```bash
icdiff locale/avalan.pot <(
    cat locale/avalan.pot |
        avalan agent run docs/examples/agent_gettext_translator.toml --quiet
)
```

![diff showing what the AI translator agent modified](https://avalan.ai/images/agent_translator_diff.png)

There are more agent, NLP, multimodal, audio, and vision examples in the
[docs/examples](https://github.com/avalan-ai/avalan/blob/main/docs/examples)
folder.

# Table of contents

* [Install](#install)
* [The CLI](#the-cli)
    - [agent](#agent)
        * [run](#agent-run)
    - [cache](#cache)
        * [delete](#cache-delete)
        * [download](#cache-download)
        * [list](#cache-list)
    - [flow](#flow)
        * [run](#flow-run)
    - [memory](#memory)
        * [embeddings](#memory-embeddings)
        * [search](#memory-search)
        * [document](#memory-document)
            - [index](#memory-document-index)
    - [model](#model)
        * [display](#model-display)
        * [install](#model-install)
        * [run](#model-run)
            - [Quiet mode](#quiet_mode)
            - [Attention implementation](#attention-implementation)
            - [Stopping patterns and token limitation](#stopping-patterns-and-token-limitation)
            - [Reasoning support](#reasoning-support)
            - [Displaying generation details](#displaying-generation-details)
                * [Showing generation performance](#showing-generation-performance)
                * [Probability distributions](#probability-distributions)
        * [search](#model-search)
        * [uninstall](#model-uninstall)
    - [tokenizer](#tokenizer)
        * [Adding tokens and special tokens](#adding-tokens-and-special-tokens)
        * [Saving and loading modified tokenizers](#saving-and-loading-modified-tokenizers)
* [Development](#development)
	- [Building](#building)
    - [Running tests](#running-tests)
    - [Translations](#translations)
    - [TODO](#todo)

# Install

Create your virtual environment and install packages:

```bash
poetry lock && poetry install
```

> [!TIP]
> At time of this writing, while Python 3.12 is stable and available
> in Homebrew, sentenpiece, a package added by the extra `translation`,
> requires Python 3.11, so you may want to force the python version when
> creating the virtual environment: `python-3.11 -m venv .venv/`

> [!TIP]
> If you will be using avalan with a device other than `cuda`, or wish to
> use `--low-cpu-mem-usage` you'll need the CPU packages installed, so run
> `poetry install --extras 'cpu'` You can also specify multiple extras to install,
> for example with:
>
> ```bash
> poetry install --extras 'agent audio cpu memory secrets server test translation vision'
> ```
>
> Or you can install all extras at once with:
>
> ```bash
> poetry install --extras all
> ```

> [!TIP]
> If you are going to be using transformer loading classes that haven't yet
> made it into a transformers package released version, install transformers
> development edition:
> `poetry install git+https://github.com/huggingface/transformers --no-cache`

Depending on your architecture, you may need to add pytorch's index:

```bash
poetry install --extra-index-url https://download.pytorch.org/whl/nightly/cpu
```

> [!TIP]
> At time of this writing, while Python 3.12 is stable and available
> in Homebrew, sentenpiece, a package added by the extra `translation`,
> requires Python 3.11, so you may want to force the python version when
> creating the virtual environment: `python-3.11 -m venv .venv/`

> [!TIP]
> On MacOS, sentencepiece may have issues while installing. If so,
> ensure Xcode CLI is installed, and install needed Homebrew packages
> with:
>
> `xcode-select --install`
> `brew install cmake pkg-config protobuf sentencepiece`

# The CLI

The CLI offers the following commands, some of them with multiple subcommands:

* [agent](#agent): Run and manage AI agents.
* [cache](#cache): Manage the local cache for model data, and download models.
* [flow](#flow): Execute flows describing multiple processing steps.
* [memory](#memory): Generate embeddings, search them and index documents.
* [model](#model): Search for models, install and manage them, show
their information, and run them.
* [tokenizer](#tokenizer): Manage tokenizers and save them to filesystem.

If you want to list all available commands and global options, run:

```bash
avalan --help
```

If you want help on a specific command, add `--help` to the command, for
example:

```bash
avalan model --help
```

Some commands, like `model`, contain subcommands of their own, which are listed
when showing the help for the command. You can learn about a subcommand (like
`run` for `model`) set of options with:

```bash
avalan model run --help
```

Global options may affect more than one command. For example, to change the
output language from the default english to spanish, add the `locale` option,
specifying `es` as the locale:

```bash
avalan model run meta-llama/Meta-Llama-3-8B-Instruct --locale es
```

![Running the CLI in spanish](https://avalan.ai/images/spanish_translation.png)

You'll need your Huggingface access token exported as `HF_ACCESS_TOKEN`.

> [!TIP]
> If you are on an Apple silicon chip, run the
> [configure_mlx.sh](https://github.com/avalan-ai/avalan/blob/main/scripts/configure_mlx.sh)
> script, created by [@AlexCheema](https://github.com/AlexCheema), which
> empirically reduces the time to first token and the tokens per second ratio.

## agent

### agent run

### agent init

Generate a TOML template for a new agent. Missing values will be
requested interactively:

```bash
avalan agent init --name "Leo Messi" --engine-uri microsoft/Phi-4-mini-instruct
```

## cache

To run models locally you'll need to cache their data on a filesystem. A
default location of `$HOME/.cache/huggingface/hub` will be assumed, unless
the `--cache-dir` global option is utilized.

### cache delete

You can delete all cached data for a model:

```bash
avalan cache delete --model 'qingy2024/UwU-7B-Instruct'
```

![Deleting a model](https://avalan.ai/images/cache_delete_model.png)

Or you can specify which model revisions to delete:

```bash
avalan cache delete --model 'google/owlvit-base-patch16' \
                    --revision '10e842' \
                    --revision '4b420d'
```

![Deleting all revisions in a model](https://avalan.ai/images/cache_delete_revisions.png)

### cache download

You can pre-emptively download all the needed files to run a really small
model to your local cache:

```bash
avalan cache download --model 'hf-internal-testing/tiny-random-bert'
```

![Downloading a tiny model to cache](https://avalan.ai/images/cache_download.gif)

### cache list

You can inspect the state of your cached models with:

```bash
avalan cache list
```

![Inspecting cached models](https://avalan.ai/images/cache_list.png)

The cache list is sorted by size on disk, starting with the largest models. In
our case, we see our cached models are occupying a total of 436.4 GB.

Let's inspect the cache contents of the `Qwen/Qwen2.5-7B-Instruct` model we
have installed, which has two revisions, using the option `--model` (you can
specify multiple models by adding more `--model` options):

```bash
avalan cache list --model 'Qwen/Qwen2.5-7B-Instruct'
```

![Showing cached model details](https://avalan.ai/images/cache_list_details.png)

> [!NOTE]
> When the same file appears in multiple revisions of a model, that does
> not mean the file is stored multiple times. If a file hasn't changed
> across revisions, a symlink is used, to only keep one version of the file.

## flow

### flow run

Run a flow definition. Provide the path to the flow description file:

```bash
avalan flow run docs/examples/my_flow.toml
```

## memory

### memory embeddings

Generate embeddings from text. You can compare the generated embeddings or
search across them:

```bash
avalan memory embeddings --model microsoft/Phi-4-mini-instruct \
                         --compare 'hello there' \
                         --search 'hola mundo' \
                         --search-k 3
```

Use `--partition` to split long inputs into windows when indexing and
`--display-partitions` to show a summary of the generated partitions.

### memory search

Look for memories stored in PostgreSQL using vector search:

```bash
avalan memory search --dsn postgresql://user:pass@localhost/db \
                    --participant 123e4567-e89b-12d3-a456-426614174000 \
                    --namespace chat \
                    --function l2_distance \
                    --limit 5
```

### memory document index

Add a document to the memory index so its contents become searchable:

```bash
avalan memory document index README.md \
                           --dsn postgresql://user:pass@localhost/db \
                           --participant 123e4567-e89b-12d3-a456-426614174000 \
                           --namespace docs \
                           --partitioner code \
                           --language python \
                           --encoding utf-8
```
Use `--identifier` to override the default identifier (the source path or URL).

## model

### model display

You can show detailed model information (such as architectures, vocabulary
size, hidden and attention layers, special tokens, etc) if you load the model:

```bash
avalan model display --load deepseek-ai/DeepSeek-R1-Distill-Qwen-14B
```

![Looking for models that match search criteria](https://avalan.ai/images/running_show_example.png)

### model install

You can install any of the +1.4 million models available:

```bash
avalan model install microsoft/Phi-4-mini-instruct
```

### model run

You can run a model by entering your prompt at the, well, prompt:

```bash
avalan model run meta-llama/Meta-Llama-3-8B-Instruct
```

You can also specify your prompt by piping it, on this case to a gated repo
(which is why we also `--login`):

```bash
echo 'explain LLM distillation in no more than three paragraphs' |
    avalan model run meta-llama/Meta-Llama-3-8B-Instruct --login
```

#### Quiet mode

If you want to prompt a model and get nothing but its response, try `--quiet`
mode. It will only stream generated tokens directly to output, without any
added statistics or styling:

```bash
echo 'Who is Leo Messi?' |
    avalan model run meta-llama/Meta-Llama-3-8B-Instruct --quiet
```

![Quiet mode](https://avalan.ai/images/running_quiet_mode.gif)

#### Attention implementation

When running a model, by default the best available attention implementation
is utilized. If you'd like to change it, use the `--attention` option,
specifying one of the available implementations: `eager`, `flash_attention_2`
(you'll need CUDA and the [flash-attn](https://pypi.org/project/flash-attn/)
package installed), `sdpa`, and `flex_attention` (only for CUDA):

```bash
echo 'hello, who are you? answer in no less than 100 words' |
    avalan model run deepseek-ai/deepseek-llm-7b-chat --attention sdpa
```

#### Stopping patterns and token limitation

There are multiple ways to stop the inference process. You can choose to limit
the amount of tokens generated with `--max-new-tokens`:

```bash
echo 'Who is Leo Messi?' | \
    avalan model run meta-llama/Meta-Llama-3-8B-Instruct --max-new-tokens 10
```

![Limiting number of generated tokens](https://avalan.ai/images/running_generation_max_new_tokens.png)

You can also stop the token generation when one (or one of many) expression
is found with `--stop-on-keyword` (use as many as needed):

```bash
echo 'Who is Leo Messi?' | \
    avalan model run meta-llama/Meta-Llama-3-8B-Instruct
                     --stop_on_keyword 'Argentina' \
                     --stop_on_keyword 'Barcelona' \
                     --stop_on_keyword 'footballer'
```

![Stopping generation when certain keywords are found](https://avalan.ai/images/running_generation_stop_on_keyword.png)

#### Reasoning support

If you run a model with reasoning support, you'll see the model reasoning
preceeding its response:

```bash
echo 'explain LLM distillation in no more than three paragraphs' |
    avalan model run deepseek-ai/DeepSeek-R1-Distill-Qwen-14B
```

![Reasoning section for models that support it](https://avalan.ai/images/running_local_inference_with_reasoning.png)

#### Displaying generation details

To get details on the tokens generated by the chosen model, use the
`--display-tokens` option, optionally setting it to the number of tokens with
details to display at a time, for example `--display-tokens 20`.
If the option is present but no value provided, a default of `15` tokens will
be used.

```bash
echo 'hello, who are you? answer in no less than 100 words' | \
    avalan model run deepseek-ai/deepseek-llm-7b-chat --display-tokens
```

> [!IMPORTANT]
> When the option `--display-tokens` is used, inference tokens are displayed
> only after the model has finished producing all tokens, unlike the default
> real token streaming behavior when the option is not present.

When displaying generation details, tokens may (hopefully) advance too rapidly to
follow. You can add a delay between tokens with `--display-pause`. If no value
specified, a default of `500` milliseconds will be used. Following, we are
introducing a much lower `25` milliseconds delay between tokens:

```bash
echo 'hello, who are you? answer in no less than 100 words' | \
    avalan model run deepseek-ai/deepseek-llm-7b-chat \
               --display-tokens \
               --display-pause 25
```

##### Showing generation performance

While the CLI is displaying the generated tokens, you will see some statistics
at the bottom right side:

* `token count`: the total number of tokens that have been generated.
* `ttft`: time to first token, the time it took for the model to generate
the first token.
* `ttnt`: time to Nth token, the time it took for the model to generate the
Nth token (defaulting to 256.)
* `t/s`: tokens per second, on average, how many tokens the model generates
in a second.

You can choose another value for `ttnt`. For example, by setting
`---display-time-to-n-token` to `100` we can learn how long it takes the model
to produce the 100th token:

```bash
echo 'hello, who are you? answer in no less than 100 words' | \
    avalan model run deepseek-ai/deepseek-llm-7b-chat \
                     --display-time-to-n-token 100
```

![Displaying time to 100th token](https://avalan.ai/images/running_local_inference_speed.png)

We can see it took `deepseek-llm-7b-chat` a total of `4.61 seconds` until
generating the 100th token.

##### Probability distributions

If you are interested in seeing the token generation progress, including
details such as token alternatives per generation step with different
distributions, do:

```bash
echo 'hello, who are you? answer in no less than 100 words' | \
    avalan model run deepseek-ai/deepseek-llm-7b-chat \
                     --max-new-tokens 300 \
                     --temperature 0.9 \
                     --do-sample \
                     --display-tokens 15 \
                     --display-probabilities \
                     --display-probabilities-maximum 0.8 \
                     --display-probabilities-sample-minimum 0.1 \
                     --display-pause
```

![Example use of the CLI showing token distributions](https://avalan.ai/images/running_token_distribution_example.gif)

### model search

Let's search for up to two models matching a query (`deepseek r1`) and a
filter (`transformers`), ensuring we are previously logged in to the hub:

```bash
avalan model search 'deepseek r1' \
					--login \
                    --filter 'transformers' \
                    --limit 2
```

![Looking for models that match search criteria](https://avalan.ai/images/running_search_example.png)

### model uninstall

You can uninstall an install model:

```bash
avalan model uninstall microsoft/Phi-4-mini-instruct
```

## tokenizer

If you want to see how a tokenizer deals with text, you can have the CLI
ask for the text to tokenize, or provide it via standard input:

```bash
echo 'Leo Messi is the GOAT' |
    avalan tokenizer --tokenizer 'deepseek-ai/deepseek-llm-7b-chat'
```

![Tokenization of text](https://avalan.ai/images/running_tokenization_simple_example.png)

### Adding tokens and special tokens

When viewing token displays, you may have noticed some of the token boxes
are colored differently. Two kinds of token that are always going to be colored
are the added token, and the special token, a small subset of tokens the
tokenizer treates differently.

To see this in action, we'll add a token ourselves: `<avalan_special_token>`.
Let's first see how the tokenizer deails with our token when it has no
knowledge of it:

```bash
echo 'is <avalan_special_token> a special token?' | \
    avalan tokenizer --tokenizer 'deepseek-ai/deepseek-llm-7b-chat'
```

We see the tokenizer split it as: `<｜begin▁of▁sentence｜>`, `is`, `<`, `aval`,
`an`, `_`, `special`, `_`, `token`, `>`, `a`, `special`, `token`, `?`,
`<｜end▁of▁sentence｜>`.

Now let's run the same, but also add our token to the tokenizer with
`--token` (you can add multiple by adding more arguments,
like: `--token my_token_1 --token my_token_2`):

```bash
echo 'is <avalan_special_token> a special token?' | \
    avalan tokenizer --tokenizer 'deepseek-ai/deepseek-llm-7b-chat' \
                     --token '<avalan_special_token>'
```

![Tokenization of an added token unknown to the tokenizer](https://avalan.ai/images/running_tokenization_example.png)

This time, the tokenizer splits it as: `<｜begin▁of▁sentence｜>`, `is`,
`<avalan_special_token>`, `a`, `special`, `token`, `?"`,
`<｜end▁of▁sentence｜>`, so our added token is a mere token for the tokenizer
now, versus previously using 8 tokens for it, a whooping 87.5% in savings :]

### Saving and loading modified tokenizers

If you want to persist your tokenizer modifications, use the `--save` option:

```bash
avalan tokenizer --tokenizer 'deepseek-ai/deepseek-llm-7b-chat' \
                 --token '<avalan_special_token>' \
                 --save './my_custom_tokenizer'
```

![Saving a modified tokenizer](https://avalan.ai/images/running_tokenization_saving_example.png)

Load your modified tokenizer, and see it in action:

```bash
echo 'is <avalan_special_token> a special token?' | \
    avalan tokenizer --tokenizer './my_custom_tokenizer'
```

![Loading a modified tokenizer](https://avalan.ai/images/running_tokenization_loading_example.png)

# Development

If you're going to work on this package, install it in editable mode so changes
are reflected instantly:

```bash
poetry install
```

## Building

You can build the package with:

```bash
python -m build
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

If new translated strings are added (via `_()` and/or `_n()`), the gettext
template file will need to be updated. Here's how you extract all `_()` and
`_n()`  references within the `src/` folder to `locale/messages.pot`:

```bash
find src/avalan/. -name "*.py" | xargs xgettext \
    --language=Python \
    --keyword=_ \
    --keyword=_n \
    --package-name 'avalan' \
    --package-version `cat src/avalan/VERSION.txt` \
    --output=locale/avalan.pot
```

If you are translating to a new language (such as `es`), create the folder
structure first:

```bash
mkdir -p locale/es/LC_MESSAGES
```

Update the existing `es` translation file with changes:

```bash
msgmerge --update locale/es/LC_MESSAGES/avalan.po locale/avalan.pot
```

If the `es` translation file didn't exist, create it:

```bash
msginit --locale=es \
        --input=locale/avalan.pot \
        --output=locale/es/LC_MESSAGES/avalan.po
```

Edit the `locale/es/LC_MESSAGES/avalan.po` translation file filling in the
needed `msgstr`. When you are done translating, compile it:

```bash
msgfmt --output-file=locale/es/LC_MESSAGES/avalan.mo \
       locale/es/LC_MESSAGES/avalan.po
```

## Documentation

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


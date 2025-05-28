<h1 align="center">avalan</h1>
<h3 align="center">The multi-backend, multi-modal framework for effortless AI agent development, orchestration, and deployment</h3>

<p align="center">
  <img src="https://github.com/avalan-ai/avalan/actions/workflows/test.yml/badge.svg" alt="Tests" />
  <img src="https://codecov.io/gh/avalan-ai/avalan/branch/main/graph/badge.svg" alt="Code coverage" />
  <img src="https://img.shields.io/github/last-commit/avalan-ai/avalan.svg" alt="Last commit" />
  <img src="https://img.shields.io/github/v/release/avalan-ai/avalan?label=Release" alt="Release" />
  <img src="https://img.shields.io/pypi/l/avalan.svg" alt="License" />
</p>

avalan empowers developers and enterprises to easily build, orchestrate, and deploy intelligent AI agents—locally or in the cloud—across millions of models via a unified SDK and CLI, featuring multi-backend support ([transformers](https://github.com/huggingface/transformers), [vLLM](https://github.com/vllm-project/vllm), [mlx-lm](https://github.com/ml-explore/mlx-lm)), multi-modal integration (NLP, vision, audio), and native adapters for platforms like OpenRouter, Ollama, OpenAI, DeepSeek, and Gemini. Enhanced by sophisticated memory management, advanced reasoning (including ReACT tooling and adaptive planning), and intuitive pipelines with branching, filtering, and recursive workflows, avalan ensures agents continuously learn and adapt. Comprehensive observability through real-time metrics, event tracing, and statistical dashboards provides deep insights and robust governance, making avalan ideal for everything from individual experimentation to enterprise-scale AI deployments.

# Quick Look

Check out [the CLI documentation](docs/CLI.md) to see what
it can do, but if you want to jump right in, run a model:

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

# Install

Create your virtual environment and install packages:

```bash
poetry install avalan
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
> poetry install avalan --extras 'agent audio cpu memory secrets server test translation vision'
> ```
>
> Or you can install all extras at once with:
>
> ```bash
> poetry install avalan --extras all
> ```

> [!TIP]
> If you are going to be using transformer loading classes that haven't yet
> made it into a transformers package released version, install transformers
> development edition:
> `poetry install git+https://github.com/huggingface/transformers --no-cache`

> [!TIP]
> On MacOS, sentencepiece may have issues while installing. If so,
> ensure Xcode CLI is installed, and install needed Homebrew packages
> with:
>
> `xcode-select --install`
> `brew install cmake pkg-config protobuf sentencepiece`


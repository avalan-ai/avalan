[Skip to main content](#content)

Switch to mobile version

Warning
Some features may not work without JavaScript. Please try enabling it if you encounter
problems.

2025 Python Packaging Survey is now live!
[Take the survey now](https://anaconda.surveymonkey.com/r/py-package-2025)

[![PyPI](/static/images/logo-small.8998e9d1.svg)](/)

Search PyPI

Search

* [Help](/help/)
* [Docs](https://docs.pypi.org/)
* [Sponsors](/sponsors/)
* [Log in](/account/login/)
* [Register](/account/register/)

Menu

* [Help](/help/)
* [Sponsors](/sponsors/)
* [Log in](/account/login/)
* [Register](/account/register/)

Search PyPI

Search

# avalan 1.0.10

pip install avalan

Copy PIP instructions

[Latest version](/project/avalan/)

Released: Jun 3, 2025

Multi-backend, multi-modal framework for seamless AI agent development, orchestration,
and deployment

### Navigation

* [Project description](#description)
* [Release history](#history)
* [Download files](#files)

### Verified details

*These details have been [verified by
PyPI](https://docs.pypi.org/project_metadata/#verified-details)*

###### Maintainers

[![Avatar for mariano from
gravatar.com](https://pypi-camo.freetls.fastly.net/b8d9495485e89470e52c9f7bf27c89e45c40d0
fb/68747470733a2f2f7365637572652e67726176617461722e636f6d2f6176617461722f3965636566353461
6132316534623162626130363736336334316634656432323f73697a653d3530 "Avatar for mariano from
gravatar.com")

mariano](/user/mariano/)

### Unverified details

*These details have **not** been verified by PyPI*

###### Project links

* [Homepage](https://avalan.ai)
* [Bug Tracker](https://github.com/avalan-ai/avalan/issues)
* [Documentation](https://github.com/avalan-ai/avalan#readme)
* [Repository](https://github.com/avalan-ai/avalan)

###### Meta

* **License:** MIT License (MIT)
* **Author:** The Avalan Team
* **Requires:** Python >=3.11.11

###### Classifiers

* **Development Status**
  + [4 - Beta](/search/?c=Development+Status+%3A%3A+4+-+Beta)
* **Intended Audience**
  + [Developers](/search/?c=Intended+Audience+%3A%3A+Developers)
  + [Education](/search/?c=Intended+Audience+%3A%3A+Education)
  + [Science/Research](/search/?c=Intended+Audience+%3A%3A+Science%2FResearch)
* **License**
  + [OSI Approved :: MIT
License](/search/?c=License+%3A%3A+OSI+Approved+%3A%3A+MIT+License)
* **Operating System**
  + [OS Independent](/search/?c=Operating+System+%3A%3A+OS+Independent)
* **Programming Language**
  + [Python :: 3.11](/search/?c=Programming+Language+%3A%3A+Python+%3A%3A+3.11)
* **Topic**
  + [Scientific/Engineering :: Artificial
Intelligence](/search/?c=Topic+%3A%3A+Scientific%2FEngineering+%3A%3A+Artificial+Intellig
ence)
  + [Software Development ::
Libraries](/search/?c=Topic+%3A%3A+Software+Development+%3A%3A+Libraries)

[![Sponsored: Python Software
Foundation](https://ethicalads.blob.core.windows.net/media/images/2025/06/Indeed_Logo_RGB
_q0kmZNH_ouxC9qZ.png)](https://server.ethicalads.io/proxy/click/3348/019745cb-cf7a-74f0-b
d69-c9adf71d83d6/)

Indeed is a Contributing sponsor of the Python Software Foundation.

*[PSF Sponsor](https://www.python.org/psf/sponsorship/?ref=ethicalads-placement) ·
[Served ethically](https://www.ethicalads.io/sponsorship-platform/?ref=psf)*

[Report project as malware](https://pypi.org/project/avalan/submit-malware-report/)

* [Project description](#description)
* [Project details](#data)
* [Release history](#history)
* [Download files](#files)

## Project description

# avalan

### The multi-backend, multi-modal framework for effortless AI agent development,
orchestration, and deployment

![Tests](https://pypi-camo.freetls.fastly.net/fb712b667ebe65c619403180c809c6589c10ff93/68
747470733a2f2f6769746875622e636f6d2f6176616c616e2d61692f6176616c616e2f616374696f6e732f776
f726b666c6f77732f746573742e796d6c2f62616467652e737667)
![Code
coverage](https://pypi-camo.freetls.fastly.net/639d20cd9e87f3c52930df16cdd041fe9d9a0744/6
8747470733a2f2f636f6465636f762e696f2f67682f6176616c616e2d61692f6176616c616e2f6272616e6368
2f6d61696e2f67726170682f62616467652e737667)
![Last
commit](https://pypi-camo.freetls.fastly.net/dfe2cb0e3987205a13d0084a24d75c1235a585b9/687
47470733a2f2f696d672e736869656c64732e696f2f6769746875622f6c6173742d636f6d6d69742f6176616c
616e2d61692f6176616c616e2e737667)
![Release](https://pypi-camo.freetls.fastly.net/e31b6ac3ebc49f6993cf0e8c51535555d27b87af/
68747470733a2f2f696d672e736869656c64732e696f2f6769746875622f762f72656c656173652f6176616c6
16e2d61692f6176616c616e3f6c6162656c3d52656c65617365)
![License](https://pypi-camo.freetls.fastly.net/4296065f001ddd7404a97c8e07a45afa53b85756/
68747470733a2f2f696d672e736869656c64732e696f2f707970692f6c2f6176616c616e2e737667)

avalan empowers developers and enterprises to easily build, orchestrate, and deploy
intelligent AI agents—locally or in the cloud—across millions of models via a unified SDK
and CLI, featuring multi-backend support
([transformers](https://github.com/huggingface/transformers),
[vLLM](https://github.com/vllm-project/vllm),
[mlx-lm](https://github.com/ml-explore/mlx-lm)), multi-modal integration (NLP, vision,
audio), and native adapters for platforms like OpenRouter, Ollama, OpenAI, LiteLLM, DeepSeek, and
Gemini. Enhanced by sophisticated memory management, advanced reasoning (including ReACT
tooling and adaptive planning), and intuitive pipelines with branching, filtering, and
recursive workflows, avalan ensures agents continuously learn and adapt. Comprehensive
observability through real-time metrics, event tracing, and statistical dashboards
provides deep insights and robust governance, making avalan ideal for everything from
individual experimentation to enterprise-scale AI deployments.

# Quick Look

Check out [the CLI documentation](docs/CLI.md) to see what it can do, but if you want to
jump right in, you can run any locally installed model and tweak sampling settings like
`--temperature`, `--top-p`, and `--top-k`. In this example, we prompt the model as
"Aurora" and limit the response to 100 new tokens:

```
echo 'Who are you, and who is Leo Messi?' \
  | avalan model run "meta-llama/Meta-Llama-3-8B-Instruct" \
      --system "You are Aurora, a helpful assistant" \
      --max-new-tokens 100 \
      --temperature .1 \
      --top-p .9 \
      --top-k 20

```

Just as easily as you can run local models, you can use vendors. Simply swap in a
vendor-backed [engine URI](docs/ai_uri.md) to run on an external API. For instance, to
hit OpenAI's GPT-4o endpoint with the same sampling parameters:

```
echo 'Who are you, and who is Leo Messi?' \
  | avalan model run "ai://$OPENAI_API_KEY@openai/gpt-4o" \
      --system "You are Aurora, a helpful assistant" \
      --max-new-tokens 100 \
      --temperature .1 \
      --top-p .9 \
      --top-k 20

```

Avalan makes it trivial to spin up a chat-based agent that can invoke external tools
(e.g., a calculator). Below is an example using a locally installed 8B-parameter LLM,
enabling recent memory, and loading a calculator tool. The agent starts with a math
question and then keeps the conversation open for follow-up questions:

```
echo "What is (4 + 6) and then that result times 5, divided by 2?" \
  | avalan agent run \
      --engine-uri "NousResearch/Hermes-3-Llama-3.1-8B" \
      --tool "calculator" \
      --memory-recent \
      --run-max-new-tokens 1024 \
      --name "Tool" \
      --role "You are a helpful assistant named Tool, that can resolve user requests
using tools." \
      --stats \
      --conversation

```

Check the GPU hard at work towards the bottom:

![Example use of an ephemeral tool agent with
memory](https://pypi-camo.freetls.fastly.net/8f62276017c007340b93a15dc049b9ca4fd674f9/687
47470733a2f2f6176616c616e2e61692f696d616765732f6167656e745f657068656d6572616c5f746f6f6c2e
676966)

Through the avalan microframework, you can easily integrate real time token
streaming with your own code, as [this example
shows](https://github.com/avalan-ai/avalan/blob/main/docs/examples/text_generation.py):

```
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

Besides natural language processing, you can also work with other types of
models, such as those that handle vision, like the following
[image classification
example](https://github.com/avalan-ai/avalan/blob/main/docs/examples/vision_image_classif
ication.py):

```
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
example
shows](https://github.com/avalan-ai/avalan/blob/main/docs/examples/seq2seq_summarization.
py):

```
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

You can also perform translations, as [the following example
shows](https://github.com/avalan-ai/avalan/blob/main/docs/examples/seq2seq_translation.py
).
You'll need the `translation` extra installed for this to run:

```
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
Create a file named
[agent\_gettext\_translator.toml](https://github.com/avalan-ai/avalan/blob/main/docs/exam
ples.agent_gettext_translator.toml)
with the following contents:

```
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

```
icdiff locale/avalan.pot <(
    cat locale/avalan.pot |
        avalan agent run docs/examples/agent_gettext_translator.toml --quiet
)

```

![diff showing what the AI translator agent
modified](https://pypi-camo.freetls.fastly.net/dcae6e2ed66638ad6c08b874e6beae36a321b04d/6
8747470733a2f2f6176616c616e2e61692f696d616765732f6167656e745f7472616e736c61746f725f646966
662e706e67)

There are more agent, NLP, multimodal, audio, and vision examples in the
[docs/examples](https://github.com/avalan-ai/avalan/blob/main/docs/examples)
folder.

# Install

Create your virtual environment and install packages:

```
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
> ```
> poetry install avalan --extras 'agent audio cpu memory secrets server test translation
vision'
>
> ```
>
> Or you can install all extras at once with:
>
> ```
> poetry install avalan --extras all
>
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

## Project details

### Verified details

*These details have been [verified by
PyPI](https://docs.pypi.org/project_metadata/#verified-details)*

###### Maintainers

[![Avatar for mariano from
gravatar.com](https://pypi-camo.freetls.fastly.net/b8d9495485e89470e52c9f7bf27c89e45c40d0
fb/68747470733a2f2f7365637572652e67726176617461722e636f6d2f6176617461722f3965636566353461
6132316534623162626130363736336334316634656432323f73697a653d3530 "Avatar for mariano from
gravatar.com")

mariano](/user/mariano/)

### Unverified details

*These details have **not** been verified by PyPI*

###### Project links

* [Homepage](https://avalan.ai)
* [Bug Tracker](https://github.com/avalan-ai/avalan/issues)
* [Documentation](https://github.com/avalan-ai/avalan#readme)
* [Repository](https://github.com/avalan-ai/avalan)

###### Meta

* **License:** MIT License (MIT)
* **Author:** The Avalan Team
* **Requires:** Python >=3.11.11

###### Classifiers

* **Development Status**
  + [4 - Beta](/search/?c=Development+Status+%3A%3A+4+-+Beta)
* **Intended Audience**
  + [Developers](/search/?c=Intended+Audience+%3A%3A+Developers)
  + [Education](/search/?c=Intended+Audience+%3A%3A+Education)
  + [Science/Research](/search/?c=Intended+Audience+%3A%3A+Science%2FResearch)
* **License**
  + [OSI Approved :: MIT
License](/search/?c=License+%3A%3A+OSI+Approved+%3A%3A+MIT+License)
* **Operating System**
  + [OS Independent](/search/?c=Operating+System+%3A%3A+OS+Independent)
* **Programming Language**
  + [Python :: 3.11](/search/?c=Programming+Language+%3A%3A+Python+%3A%3A+3.11)
* **Topic**
  + [Scientific/Engineering :: Artificial
Intelligence](/search/?c=Topic+%3A%3A+Scientific%2FEngineering+%3A%3A+Artificial+Intellig
ence)
  + [Software Development ::
Libraries](/search/?c=Topic+%3A%3A+Software+Development+%3A%3A+Libraries)

## Release history [Release notifications](/help/#project-release-notifications) | [RSS
feed](/rss/project/avalan/releases.xml)

This version

![](https://pypi.org/static/images/blue-cube.572a5bfb.svg)

[1.0.10

Jun 3, 2025](/project/avalan/1.0.10/)

![](https://pypi.org/static/images/white-cube.2351a86c.svg)

[1.0.9

Jun 3, 2025](/project/avalan/1.0.9/)

![](https://pypi.org/static/images/white-cube.2351a86c.svg)

[1.0.8

Jun 2, 2025](/project/avalan/1.0.8/)

![](https://pypi.org/static/images/white-cube.2351a86c.svg)

[1.0.7

May 30, 2025](/project/avalan/1.0.7/)

![](https://pypi.org/static/images/white-cube.2351a86c.svg)

[1.0.6

May 28, 2025](/project/avalan/1.0.6/)

![](https://pypi.org/static/images/white-cube.2351a86c.svg)

[1.0.5

May 28, 2025](/project/avalan/1.0.5/)

![](https://pypi.org/static/images/white-cube.2351a86c.svg)

[1.0.4

May 25, 2025](/project/avalan/1.0.4/)

![](https://pypi.org/static/images/white-cube.2351a86c.svg)

[1.0.3

May 22, 2025](/project/avalan/1.0.3/)

## Download files

Download the file for your platform. If you're not sure which to choose, learn more about
[installing packages](https://packaging.python.org/tutorials/installing-packages/
"External link").

### Source Distribution

[avalan-1.0.10.tar.gz](https://files.pythonhosted.org/packages/65/02/7ba1fdfd8f5a6ea974ca
2ad248befb2bcbf4598a6d940b7277e26fca5feb/avalan-1.0.10.tar.gz)
(95.7 kB
[view details](#avalan-1.0.10.tar.gz))

Uploaded Jun 3, 2025
`Source`

### Built Distribution

[avalan-1.0.10-py3-none-any.whl](https://files.pythonhosted.org/packages/b7/d8/421ad02bb3
ab65aa7334a163b4da6b0707ef80699d1f23f74417e8369f47/avalan-1.0.10-py3-none-any.whl)
(131.1 kB
[view details](#avalan-1.0.10-py3-none-any.whl))

Uploaded Jun 3, 2025
`Python 3`

## File details

Details for the file `avalan-1.0.10.tar.gz`.

### File metadata

* Download URL:
[avalan-1.0.10.tar.gz](https://files.pythonhosted.org/packages/65/02/7ba1fdfd8f5a6ea974ca
2ad248befb2bcbf4598a6d940b7277e26fca5feb/avalan-1.0.10.tar.gz)
* Upload date: Jun 3, 2025
* Size: 95.7 kB
* Tags: Source
* Uploaded using Trusted Publishing? No
* Uploaded via: poetry/2.1.2 CPython/3.13.3 Darwin/24.5.0

### File hashes

Hashes for avalan-1.0.10.tar.gz

| Algorithm | Hash digest |  |
| --- | --- | --- |
| SHA256 | `266d87c3b65dec99f16a62fb4b1b4d398d961c4f2dff16d0c3f3ba2cf1d34638` | Copy |
| MD5 | `98c6085f1d54782d95338d8291c38ca9` | Copy |
| BLAKE2b-256 | `65027ba1fdfd8f5a6ea974ca2ad248befb2bcbf4598a6d940b7277e26fca5feb` | Copy
|

[See more details on using hashes
here.](https://pip.pypa.io/en/stable/topics/secure-installs/#hash-checking-mode "External
link")

## File details

Details for the file `avalan-1.0.10-py3-none-any.whl`.

### File metadata

* Download URL:
[avalan-1.0.10-py3-none-any.whl](https://files.pythonhosted.org/packages/b7/d8/421ad02bb3
ab65aa7334a163b4da6b0707ef80699d1f23f74417e8369f47/avalan-1.0.10-py3-none-any.whl)
* Upload date: Jun 3, 2025
* Size: 131.1 kB
* Tags: Python 3
* Uploaded using Trusted Publishing? No
* Uploaded via: poetry/2.1.2 CPython/3.13.3 Darwin/24.5.0

### File hashes

Hashes for avalan-1.0.10-py3-none-any.whl

| Algorithm | Hash digest |  |
| --- | --- | --- |
| SHA256 | `0e2ddeb4a55d28c83672142a6c40c9390ca3024304e5de52b3b17b0e885b4e91` | Copy |
| MD5 | `6e92deb206967408a69c56c81252b6f7` | Copy |
| BLAKE2b-256 | `b7d8421ad02bb3ab65aa7334a163b4da6b0707ef80699d1f23f74417e8369f47` | Copy
|

[See more details on using hashes
here.](https://pip.pypa.io/en/stable/topics/secure-installs/#hash-checking-mode "External
link")

![](/static/images/white-cube.2351a86c.svg)

## Help

* [Installing packages](https://packaging.python.org/tutorials/installing-packages/
"External link")
* [Uploading packages](https://packaging.python.org/tutorials/packaging-projects/
"External link")
* [User guide](https://packaging.python.org/ "External link")
* [Project name retention](https://www.python.org/dev/peps/pep-0541/ "External link")
* [FAQs](/help/)

## About PyPI

* [PyPI Blog](https://blog.pypi.org "External link")
* [Infrastructure dashboard](https://dtdg.co/pypi "External link")
* [Statistics](/stats/)
* [Logos & trademarks](/trademarks/)
* [Our sponsors](/sponsors/)

## Contributing to PyPI

* [Bugs and feedback](/help/#feedback)
* [Contribute on GitHub](https://github.com/pypi/warehouse "External link")
* [Translate PyPI](https://hosted.weblate.org/projects/pypa/warehouse/ "External link")
* [Sponsor PyPI](/sponsors/)
* [Development credits](https://github.com/pypi/warehouse/graphs/contributors "External
link")

## Using PyPI

* [Terms of Service](https://policies.python.org/pypi.org/Terms-of-Service/ "External
link")
* [Report security issue](/security/)
* [Code of conduct](https://policies.python.org/python.org/code-of-conduct/ "External
link")
* [Privacy Notice](https://policies.python.org/pypi.org/Privacy-Notice/ "External link")
* [Acceptable Use Policy](https://policies.python.org/pypi.org/Acceptable-Use-Policy/
"External link")

---

Status:[All Systems Operational](https://status.python.org/ "External link")

Developed and maintained by the Python community, for the Python community.
[Donate today!](https://donate.pypi.org)

"PyPI", "Python Package Index", and the blocks logos are registered
[trademarks](/trademarks/) of the [Python Software
Foundation](https://www.python.org/psf-landing).

© 2025 [Python Software Foundation](https://www.python.org/psf-landing/ "External link")
[Site map](/sitemap/)

Switch to desktop version

* English
* español
* français
* 日本語
* português (Brasil)
* українська
* Ελληνικά
* Deutsch
* 中文 (简体)
* 中文 (繁體)
* русский
* עברית
* Esperanto
* 한국어

Supported by

[![](https://pypi-camo.freetls.fastly.net/ed7074cadad1a06f56bc520ad9bd3e00d0704c5b/687474
70733a2f2f73746f726167652e676f6f676c65617069732e636f6d2f707970692d6173736574732f73706f6e7
36f726c6f676f732f6177732d77686974652d6c6f676f2d7443615473387a432e706e67)
AWS

Cloud computing and Security Sponsor](https://aws.amazon.com/)
[![](https://pypi-camo.freetls.fastly.net/8855f7c063a3bdb5b0ce8d91bfc50cf851cc5c51/687474
70733a2f2f73746f726167652e676f6f676c65617069732e636f6d2f707970692d6173736574732f73706f6e7
36f726c6f676f732f64617461646f672d77686974652d6c6f676f2d6668644c4e666c6f2e706e67)
Datadog

Monitoring](https://www.datadoghq.com/)
[![](https://pypi-camo.freetls.fastly.net/df6fe8829cbff2d7f668d98571df1fd011f36192/687474
70733a2f2f73746f726167652e676f6f676c65617069732e636f6d2f707970692d6173736574732f73706f6e7
36f726c6f676f732f666173746c792d77686974652d6c6f676f2d65684d3077735f6f2e706e67)
Fastly

CDN](https://www.fastly.com/)
[![](https://pypi-camo.freetls.fastly.net/420cc8cf360bac879e24c923b2f50ba7d1314fb0/687474
70733a2f2f73746f726167652e676f6f676c65617069732e636f6d2f707970692d6173736574732f73706f6e7
36f726c6f676f732f676f6f676c652d77686974652d6c6f676f2d616734424e3774332e706e67)
Google

Download Analytics](https://careers.google.com/)
[![](https://pypi-camo.freetls.fastly.net/d01053c02f3a626b73ffcb06b96367fdbbf9e230/687474
70733a2f2f73746f726167652e676f6f676c65617069732e636f6d2f707970692d6173736574732f73706f6e7
36f726c6f676f732f70696e67646f6d2d77686974652d6c6f676f2d67355831547546362e706e67)
Pingdom

Monitoring](https://www.pingdom.com/)
[![](https://pypi-camo.freetls.fastly.net/67af7117035e2345bacb5a82e9aa8b5b3e70701d/687474
70733a2f2f73746f726167652e676f6f676c65617069732e636f6d2f707970692d6173736574732f73706f6e7
36f726c6f676f732f73656e7472792d77686974652d6c6f676f2d4a2d6b64742d706e2e706e67)
Sentry

Error logging](https://getsentry.com/for/python)
[![](https://pypi-camo.freetls.fastly.net/b611884ff90435a0575dbab7d9b0d3e60f136466/687474
70733a2f2f73746f726167652e676f6f676c65617069732e636f6d2f707970692d6173736574732f73706f6e7
36f726c6f676f732f737461747573706167652d77686974652d6c6f676f2d5467476c6a4a2d502e706e67)
StatusPage

Status page](https://statuspage.io)


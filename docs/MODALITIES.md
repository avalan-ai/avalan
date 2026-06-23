# Modalities

The following examples show each modality in action. Use the table of contents below to jump to the
task you need:

* 🎧 [**Audio**](#audio): Turn audio into text or produce speech for accessibility and media.
  - 🦻 [Audio classification](#audio-classification): Label an audio based on sentiment.
  - 🗣️ [Speech recognition](#speech-recognition): Convert spoken audio to text.
  - 🔊 [Text to speech](#text-to-speech): Synthesize natural speech from text.
  - 🎵 [Audio generation](#audio-generation): Generate or continue audio from prompts.
* 📝 [**Text**](#text): Understand, transform, and generate text across core NLP tasks.
  - ❓ [Question answering](#question-answering): Extract precise answers from context.
  - 🏷️ [Sequence classification](#sequence-classification): Classify text into labels.
  - 🔁 [Sequence to sequence](#sequence-to-sequence): Rewrite or transform text-to-text.
  - ✍️ [Text generation](#text-generation): Produce completions and long-form responses.
  - 🧩 [Token classification](#token-classification): Label entities and token-level tags.
  - 🌍 [Translation](#translation): Translate text between languages.
* 👁️ [**Vision**](#vision): Analyze images and create visual outputs from prompts.
  - 🖼️ [Encoder decoder](#encoder-decoder): Caption images with encoder-decoder models.
  - 🧪 [Image classification](#image-classification): Predict image classes and scores.
  - 📝 [Image to text](#image-to-text): Describe image content in natural language.
  - 💬 [Image text to text](#image-text-to-text): Answer questions grounded in images.
  - 🎯 [Object detection](#object-detection): Detect and localize objects.
  - 🧱 [Semantic segmentation](#semantic-segmentation): Label each pixel by class.
  - 🕺 [Text to animation](#text-to-animation): Create animated clips from prompts.
  - 🎨 [Text to image](#text-to-image): Generate images from text prompts.
  - 🎬 [Text to video](#text-to-video): Generate short videos from prompts.

## Audio

### Audio classification

Determine the sentiment (neutral, happy, angry, sad) of a given audio file:

```sh
avalan model run "superb/hubert-base-superb-er" \
    --modality audio_classification \
    --path docs/examples/playground/oprah.wav \
    --audio-sampling-rate 16000
```

Output:

```text
┏━━━━━━━┳━━━━━━━┓
┃ Label ┃ Score ┃
┡━━━━━━━╇━━━━━━━┩
│ ang   │ 0.49  │
├───────┼───────┤
│ hap   │ 0.45  │
├───────┼───────┤
│ neu   │ 0.04  │
├───────┼───────┤
│ sad   │ 0.02  │
└───────┴───────┘
```

Python:

```python
from avalan.model.audio.classification import AudioClassificationModel

with AudioClassificationModel("superb/hubert-base-superb-er") as model:
    labels = await model("oprah.wav", sampling_rate=16000)
    print(labels)
```
For a runnable script, see [docs/examples/audio_classification.py](examples/audio_classification.py).

### Speech recognition

Transcribe speech from an audio file:

```sh
avalan model run "facebook/wav2vec2-base-960h" \
    --modality audio_speech_recognition \
    --path docs/examples/playground/oprah.wav \
    --audio-sampling-rate 16000
```

The output is the transcript of the provided audio:

```text
AND THEN I GREW UP AND HAD THE ESTEEMED HONOUR OF MEETING HER AND WASN'T
THAT A SURPRISE HERE WAS THIS PETITE ALMOST DELICATE LADY WHO WAS THE
PERSONIFICATION OF GRACE AND GOODNESS
```

Python:

```python
from avalan.model.audio.speech_recognition import SpeechRecognitionModel

with SpeechRecognitionModel("facebook/wav2vec2-base-960h") as model:
    output = await model("oprah.wav", sampling_rate=16000)
    print(output)
```
For a runnable script, see [docs/examples/audio_speech_recognition.py](examples/audio_speech_recognition.py).

### Text to speech

Generate speech in Oprah's voice from a text prompt. The example uses an 18-second clip from her [eulogy for Rosa Parks](https://www.americanrhetoric.com/speeches/oprahwinfreyonrosaparks.htm) as a reference:

```sh
echo "[S1] Leo Messi is the greatest football player of all times." | \
    avalan model run "nari-labs/Dia-1.6B-0626" \
            --modality audio_text_to_speech \
            --path example.wav \
            --audio-reference-path docs/examples/playground/oprah.wav \
            --audio-reference-text "[S1] And then I grew up and had the esteemed honor of meeting her. And wasn't that a surprise. Here was this petite, almost delicate lady who was the personification of grace and goodness."
```

Python:

```python
from avalan.model.audio.speech import TextToSpeechModel

with TextToSpeechModel("nari-labs/Dia-1.6B-0626") as model:
    await model(
        "[S1] Leo Messi is the greatest football player of all times.",
        "example.wav",
        reference_path="docs/examples/playground/oprah.wav",
        reference_text=(
            "[S1] And then I grew up and had the esteemed honor of meeting her. "
            "And wasn't that a surprise. Here was this petite, almost delicate "
            "lady who was the personification of grace and goodness."
        ),
    )
```
For a runnable script, see [docs/examples/audio_text_to_speech.py](examples/audio_text_to_speech.py).

### Audio generation

Create a short melody from a text prompt:

```sh
echo "A funky riff about Leo Messi." |
    avalan model run "facebook/musicgen-small" \
        --modality audio_generation \
        --max-new-tokens 1024 \
        --path melody.wav
```

Python:

```python
from avalan.model.audio.generation import AudioGenerationModel

with AudioGenerationModel("facebook/musicgen-small") as model:
    await model("A funky riff about Leo Messi.", "melody.wav", max_new_tokens=1024)
```
For a runnable script, see [docs/examples/audio_generation.py](examples/audio_generation.py).

## Text

### Question answering

Answer a question based on context using a question answering model:

```sh
echo "What sport does Leo play?" \
    | avalan model run "deepset/roberta-base-squad2" \
        --modality "text_question_answering" \
        --text-context "Lionel Messi, known as Leo Messi, is an Argentine professional footballer widely regarded as one of the greatest football players of all time."
```

Output:

```text
football
```

Python:

```python
from avalan.model.nlp.question import QuestionAnsweringModel

with QuestionAnsweringModel("deepset/roberta-base-squad2") as model:
    answer = await model(
        "What sport does Leo play?",
        context="Lionel Messi, known as Leo Messi, is an Argentine professional footballer widely regarded as one of the greatest football players of all time."
    )
    print(answer)
```
For a runnable script, see [docs/examples/question_answering.py](examples/question_answering.py).

### Sequence classification

Classify the sentiment of short text:

```sh
echo "We love Leo Messi." \
    | avalan model run "distilbert-base-uncased-finetuned-sst-2-english" \
        --modality "text_sequence_classification"
```

The result is positive as expected:

```text
POSITIVE
```

Python:

```python
from avalan.model.nlp.sequence import SequenceClassificationModel

with SequenceClassificationModel("distilbert-base-uncased-finetuned-sst-2-english") as model:
    output = await model("We love Leo Messi.")
    print(output)
```
For a runnable script, see [docs/examples/sequence_classification.py](examples/sequence_classification.py).

### Sequence to sequence

Summarize text using a sequence-to-sequence model:

```sh
echo "
    Andres Cuccittini, commonly known as Andy Cucci, is an Argentine
    professional footballer who plays as a forward for the Argentina
    national team. Regarded by many as the greatest footballer of all
    time, Cucci has achieved unparalleled success throughout his career.

    Born on July 25, 1988, in Ushuaia, Argentina, Cucci began playing
    football at a young age and joined the Boca Juniors youth
    academy.
" | avalan model run "facebook/bart-large-cnn" \
        --modality "text_sequence_to_sequence"
```

The summary:

```text
Andres Cuccittini, commonly known as Andy Cucci, is an Argentine professional
footballer. He plays as a forward for the Argentina national team. Cucci began
playing football at the age of 19 in his native Ushuaia.
```

Python:

```python
from avalan.model.nlp.sequence import SequenceToSequenceModel

with SequenceToSequenceModel("facebook/bart-large-cnn") as model:
    output = await model("""
    Andres Cuccittini, commonly known as Andy Cucci, is an Argentine
    professional footballer who plays as a forward for the Argentina
    national team. Regarded by many as the greatest footballer of all
    time, Cucci has achieved unparalleled success throughout his career.

    Born on July 25, 1988, in Ushuaia, Argentina, Cucci began playing
    football at a young age and joined the Boca Juniors youth
    academy.
    """)
    print(output)
```
For a runnable script, see [docs/examples/seq2seq_summarization.py](examples/seq2seq_summarization.py).

### Text generation

Run a local model and control sampling with `--temperature`, `--top-p`, and `--top-k`. The example instructs the assistant to act as "Aurora" and limits the output to 100 tokens:

```sh
echo "Who are you, and who is Leo Messi?" \
    | avalan model run "meta-llama/Meta-Llama-3-8B-Instruct" \
        --system "You are Aurora, a helpful assistant" \
        --max-new-tokens 100 \
        --temperature .1 \
        --top-p .9 \
        --top-k 20 \
        --backend mlx
```

Python:

```python
import asyncio

from avalan.entities import GenerationSettings
from avalan.model.nlp.text.generation import TextGenerationModel
from avalan.model.stream import CanonicalStreamItem, StreamItemKind

async def main() -> None:
    with TextGenerationModel("meta-llama/Meta-Llama-3-8B-Instruct") as model:
        async for item in await model(
            "Who are you, and who is Leo Messi?",
            system_prompt="You are Aurora, a helpful assistant",
            settings=GenerationSettings(
                max_new_tokens=100,
                temperature=0.1,
                top_p=0.9,
                top_k=20,
                use_async_generator=True,
            )
        ):
            assert isinstance(item, CanonicalStreamItem)
            if (
                item.kind is StreamItemKind.ANSWER_DELTA
                and item.text_delta is not None
            ):
                print(item.text_delta, end="", flush=True)

asyncio.run(main())
```

Vendor APIs use the same interface. Swap in a vendor [engine URI](docs/ai_uri.md) to call an external service. The example below uses OpenAI's GPT-4o with the same parameters:

```sh
echo "Who are you, and who is Leo Messi?" \
    | avalan model run "ai://env:OPENAI_API_KEY@openai/gpt-4o" \
        --system "You are Aurora, a helpful assistant" \
        --max-new-tokens 100 \
        --temperature .1 \
        --top-p .9 \
        --top-k 20
```

Python:

```python
import asyncio
from os import environ

from avalan.entities import GenerationSettings, TransformerEngineSettings
from avalan.model.nlp.text.vendor.openai import OpenAIModel
from avalan.model.stream import CanonicalStreamItem, StreamItemKind

async def main() -> None:
    api_key = environ["OPENAI_API_KEY"]
    settings = TransformerEngineSettings(access_token=api_key)

    with OpenAIModel("gpt-4o", settings) as model:
        async for item in await model(
            "Who are you, and who is Leo Messi?",
            system_prompt="You are Aurora, a helpful assistant",
            settings=GenerationSettings(
                max_new_tokens=100,
                temperature=0.1,
                top_p=0.9,
                top_k=20,
                use_async_generator=True,
            )
        ):
            assert isinstance(item, CanonicalStreamItem)
            if (
                item.kind is StreamItemKind.ANSWER_DELTA
                and item.text_delta is not None
            ):
                print(item.text_delta, end="", flush=True)

asyncio.run(main())
```
For a runnable script, see [docs/examples/text_generation.py](examples/text_generation.py).

Amazon Bedrock models use the same workflow. With your AWS credentials
configured (for example with `AWS_PROFILE` or environment variables),
you can target any Bedrock region via `--base-url`:

```sh
echo "Summarize the latest AWS re:Invent keynote in three bullet points." \
      | avalan model run "ai://bedrock/us.amazon.nova-lite-v1:0" \
          --base-url "us-east-1" \
          --max-new-tokens 256 \
          --temperature .7
```

Example output:

```
- **Hybrid and Multicloud**: AWS expanded its hybrid and multicloud capabilities with new services to help customers seamlessly connect their on-premises environments with AWS, and manage workloads across multiple clouds.

- **Security and Compliance**: AWS announced new security and compliance features to help customers meet their regulatory requirements and protect their data, including new services for data encryption, identity management, and threat detection.

These highlights capture some of the major themes and announcements from the keynote, but there were many more details and product updates as well. Hopefully this summary gives you a good overview! Let me know if you have any other
```

> [!TIP]
> Some Bedrock models are only available through geo-prefixed IDs in a
> given source region, such as `us.anthropic.claude-sonnet-4-6` for US
> Anthropic routing. Those profile IDs change over time, so you can
> inspect currently active options with
> `aws bedrock list-inference-profiles --region us-east-1`. Anthropic
> models can also require submitting the Bedrock use-case details form for
> your account before inference is allowed.

### Token classification

Classify tokens with labels for Named Entity Recognition (NER) or
Part-of-Speech (POS):

```sh
echo "
    Lionel Messi, commonly known as Leo Messi, is an Argentine
    professional footballer widely regarded as one of the
    greatest football players of all time.
" | avalan model run "dslim/bert-base-NER" \
    --modality text_token_classification \
    --text-labeled-only
```

Output:

```text
┏━━━━━━━━━━━┳━━━━━━━━┓
┃ Token     ┃ Label  ┃
┡━━━━━━━━━━━╇━━━━━━━━┩
│ Lionel    │ B-PER  │
├───────────┼────────┤
│ Me        │ I-PER  │
├───────────┼────────┤
│ ##ssi     │ I-PER  │
├───────────┼────────┤
│ Leo       │ B-PER  │
├───────────┼────────┤
│ Argentine │ B-MISC │
└───────────┴────────┘
```

Python:

```python
from avalan.model.nlp.token import TokenClassificationModel

with TokenClassificationModel("dslim/bert-base-NER") as model:
    labels = await model(
        "Lionel Messi, commonly known as Leo Messi, is an Argentine professional footballer widely regarded as one of the greatest football players of all time.",
        labeled_only=True
    )
    print(labels)
```
For a runnable script, see [docs/examples/token_classification.py](examples/token_classification.py).

### Translation

Translate text between languages with a sequence-to-sequence model:

```sh
echo "
    Lionel Messi, commonly known as Leo Messi, is an Argentine
    professional footballer who plays as a forward for the Argentina
    national team. Regarded by many as the greatest footballer of all
    time, Messi has achieved unparalleled success throughout his career.
" | avalan model run "facebook/mbart-large-50-many-to-many-mmt" \
        --modality "text_translation" \
        --text-from-lang "en_US" \
        --text-to-lang "es_XX" \
        --text-num-beams 4 \
        --text-max-length 512
```

Output:

```text
Lionel Messi, conocido comúnmente como Leo Messi, es un futbolista argentino
profesional que representa a la Argentina en el equipo nacional de Argentina.
Considerado por muchos como el mejor futbolista de todos los tiempos, Messi ha
conseguido un éxito sin precedentes en toda su carrera.
```

Python:

```python
from avalan.entities import GenerationSettings
from avalan.model.nlp.sequence import TranslationModel

with TranslationModel("facebook/mbart-large-50-many-to-many-mmt") as model:
    output = await model(
        "Lionel Messi, commonly known as Leo Messi, is an Argentine professional footballer who plays as a forward for the Argentina national team. Regarded by many as the greatest footballer of all time, Messi has achieved unparalleled success throughout his career.",
        source_language="en_US",
        destination_language="es_XX",
        settings=GenerationSettings(
            num_beams=4,
            max_length=512
        )
    )
    print(output)
```
For a runnable script, see [docs/examples/seq2seq_translation.py](examples/seq2seq_translation.py).

## Vision

### Encoder decoder

Answer questions to extract information from an image, without using OCR.

```sh
echo "<s_docvqa><s_question>
    What is the FACTURA Number?
</s_question><s_answer>" | \
    avalan model run "naver-clova-ix/donut-base-finetuned-docvqa" \
        --modality vision_encoder_decoder \
        --path docs/examples/playground/invoice-factura.png
```

Output:

```
<s_docvqa>
<s_question> What is the FACTURA Number?</s_question>
<s_answer> 0012-00187506</s_answer>
</s>
```

Python:

```python
from avalan.model.vision.decoder import VisionEncoderDecoderModel

with VisionEncoderDecoderModel("naver-clova-ix/donut-base-finetuned-docvqa") as model:
    answer = await model(
        "docs/examples/playground/invoice-factura.png",
        prompt="<s_docvqa><s_question>What is the FACTURA Number?</s_question><s_answer>"
    )
    print(answer)
```
For a runnable script, see [docs/examples/vision_encoder_decoder.py](examples/vision_encoder_decoder.py).

### Image classification

Classify an image, such as determining whether it is a hot dog, or not a hot dog 🤓:

```sh
avalan model run "microsoft/resnet-50" \
    --modality vision_image_classification \
    --path docs/examples/playground/cat.jpg
```

The model identifies the image:

```text
┏━━━━━━━━━━━━━━━━━━┓
┃ Label            ┃
┡━━━━━━━━━━━━━━━━━━┩
│ tabby, tabby cat │
└──────────────────┘
```

Python:

```python
from avalan.model.vision.image import ImageClassificationModel

with ImageClassificationModel("microsoft/resnet-50") as model:
    output = await model("docs/examples/playground/cat.jpg")
    print(output)
```
For a runnable script, see [docs/examples/vision_image_classification.py](examples/vision_image_classification.py).

### Image to text

Generate a caption for an image:

```sh
avalan model run "salesforce/blip-image-captioning-base" \
    --modality vision_image_to_text \
    --path docs/examples/playground/Example_Image_1.jpg
```

Example output:

```text
a sign for a gas station on the side of a building [SEP]
```

Python:

```python
from avalan.model.vision.image import ImageToTextModel

with ImageToTextModel("salesforce/blip-image-captioning-base") as model:
    caption = await model("docs/examples/playground/Example_Image_1.jpg")
    print(caption)
```
For a runnable script, see [docs/examples/vision_image_to_text.py](examples/vision_image_to_text.py).

### Image text to text

Provide an image and an instruction to an `image-text-to-text` model:

```sh
echo "Transcribe the text on this image, keeping format" | \
    avalan model run "ai://local/google/gemma-3-12b-it" \
        --modality vision_image_text_to_text \
        --path docs/examples/playground/typewritten_partial_sheet.jpg \
        --vision-width 512 \
        --max-new-tokens 1024
```

The transcription (truncated for brevity):

```text
**INTRODUCCIÓN**

Guillermo de Ockham (según se utiliza la grafía latina o la inglesa) es tan
célebre como mal conocido. Su doctrina suele merecer las más diversas
interpretaciones, y su biografía adolece tremendas oscuridades.

Aún más, y como dice un renombrado autor, el estudio de su pensamiento "parece,
por la falta de buenas ediciones de sus obras, una consecuencia del ‘anatema’
que, durante siglos, ha pesado sobre el incipor del nominalismo" (1).
```

Python:

```python
from avalan.entities import GenerationSettings
from avalan.model.vision.image import ImageTextToTextModel

with ImageTextToTextModel("google/gemma-3-12b-it") as model:
    output = await model(
        "docs/examples/playground/typewritten_partial_sheet.jpg",
        "Transcribe the text on this image, keeping format",
        settings=GenerationSettings(max_new_tokens=1024),
        width=512
    )
    print(output)
```
For a runnable script, see [docs/examples/vision_ocr.py](examples/vision_ocr.py).

### Object detection

Detect objects in an image and list them with accuracy scores:

```sh
avalan model run "facebook/detr-resnet-50" \
    --modality vision_object_detection \
    --path docs/examples/playground/kitchen.jpg \
    --vision-threshold 0.3
```

Results are sorted by accuracy and include bounding boxes:

```text
┏━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Label        ┃ Score ┃ Box                              ┃
┡━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ refrigerator │  1.00 │ 855.28, 377.27, 1035.67, 679.42  │
├──────────────┼───────┼──────────────────────────────────┤
│ oven         │  1.00 │ 411.62, 570.92, 651.66, 872.05   │
├──────────────┼───────┼──────────────────────────────────┤
│ potted plant │  0.99 │ 1345.95, 498.15, 1430.21, 603.84 │
├──────────────┼───────┼──────────────────────────────────┤
│ sink         │  0.96 │ 1077.43, 631.51, 1367.12, 703.23 │
├──────────────┼───────┼──────────────────────────────────┤
│ potted plant │  0.94 │ 179.69, 557.44, 317.14, 629.77   │
├──────────────┼───────┼──────────────────────────────────┤
│ vase         │  0.83 │ 1357.88, 562.67, 1399.38, 616.44 │
├──────────────┼───────┼──────────────────────────────────┤
│ handbag      │  0.72 │ 287.08, 544.47, 332.73, 602.24   │
├──────────────┼───────┼──────────────────────────────────┤
│ sink         │  0.68 │ 1079.68, 627.04, 1495.40, 714.07 │
├──────────────┼───────┼──────────────────────────────────┤
│ bird         │  0.38 │ 628.57, 536.31, 666.62, 574.39   │
├──────────────┼───────┼──────────────────────────────────┤
│ sink         │  0.35 │ 1077.98, 629.29, 1497.90, 723.95 │
├──────────────┼───────┼──────────────────────────────────┤
│ spoon        │  0.31 │ 646.69, 505.31, 673.04, 543.10   │
└──────────────┴───────┴──────────────────────────────────┘
```

Python:

```python
from avalan.model.vision.detection import ObjectDetectionModel

with ObjectDetectionModel("facebook/detr-resnet-50") as model:
    labels = await model("docs/examples/playground/kitchen.jpg", threshold=0.3)
    print(labels)
```
For a runnable script, see [docs/examples/vision_object_detection.py](examples/vision_object_detection.py).

### Semantic segmentation

Classify each pixel using a semantic segmentation model:

```sh
avalan model run "nvidia/segformer-b0-finetuned-ade-512-512" \
    --modality vision_semantic_segmentation \
    --path docs/examples/playground/kitchen.jpg
```

The output lists each annotation:

```text
┏━━━━━━━━━━━━━━━━━━┓
┃ Label            ┃
┡━━━━━━━━━━━━━━━━━━┩
│ wall             │
├──────────────────┤
│ floor            │
├──────────────────┤
│ ceiling          │
├──────────────────┤
│ windowpane       │
├──────────────────┤
│ cabinet          │
├──────────────────┤
│ door             │
├──────────────────┤
│ plant            │
├──────────────────┤
│ rug              │
├──────────────────┤
│ lamp             │
├──────────────────┤
│ chest of drawers │
├──────────────────┤
│ sink             │
├──────────────────┤
│ refrigerator     │
├──────────────────┤
│ flower           │
├──────────────────┤
│ stove            │
├──────────────────┤
│ kitchen island   │
├──────────────────┤
│ light            │
├──────────────────┤
│ chandelier       │
├──────────────────┤
│ oven             │
├──────────────────┤
│ microwave        │
├──────────────────┤
│ dishwasher       │
├──────────────────┤
│ hood             │
├──────────────────┤
│ vase             │
├──────────────────┤
│ fan              │
└──────────────────┘
```

Python:

```python
from avalan.model.vision.segmentation import SemanticSegmentationModel

with SemanticSegmentationModel("nvidia/segformer-b0-finetuned-ade-512-512") as model:
    labels = await model("docs/examples/playground/kitchen.jpg")
    print(labels)
```
For a runnable script, see [docs/examples/vision_semantic_segmentation.py](examples/vision_semantic_segmentation.py).

### Text to animation

Create an animation from a prompt using a base model for styling:

```sh
echo 'A tabby cat slowly walking' | \
    avalan model run "ByteDance/AnimateDiff-Lightning" \
        --modality vision_text_to_animation \
        --base-model "stablediffusionapi/mistoonanime-v30" \
        --checkpoint "animatediff_lightning_4step_diffusers.safetensors" \
        --weight "fp16" \
        --path example_cat_walking.gif \
        --vision-beta-schedule "linear" \
        --vision-guidance-scale 1.0 \
        --vision-steps 4 \
        --vision-timestep-spacing "trailing"
```

And here's the generated anime inspired animation of a walking cat:

![An anime cat slowly walking](https://avalan.ai/images/github/vision_text_to_animation_generated.webp)

Python:

```python
from avalan.entities import EngineSettings
from avalan.model.vision.diffusion import TextToAnimationModel

with TextToAnimationModel("ByteDance/AnimateDiff-Lightning", settings=EngineSettings(base_model_id="stablediffusionapi/mistoonanime-v30", checkpoint="animatediff_lightning_4step_diffusers.safetensors", weight_type="fp16")) as model:
    await model(
        "A tabby cat slowly walking",
        "example_cat_walking.gif",
        beta_schedule="linear",
        guidance_scale=1.0,
        steps=4,
        timestep_spacing="trailing"
    )
```
For a runnable script, see [docs/examples/vision_text_to_animation.py](examples/vision_text_to_animation.py).

### Text to image

Create an image from a text prompt:

```sh
echo 'Leo Messi petting a purring tubby cat' | \
    avalan model run "stabilityai/stable-diffusion-xl-base-1.0" \
        --modality vision_text_to_image \
        --refiner-model "stabilityai/stable-diffusion-xl-refiner-1.0" \
        --weight "fp16" \
        --path example_messi_petting_cat.jpg \
        --vision-color-model RGB \
        --vision-image-format JPEG \
        --vision-high-noise-frac 0.8 \
        --vision-steps 150
```

Here is the generated image of Leo Messi petting a cute cat:

![Leo Messi petting a cute cat](https://avalan.ai/images/github/vision_text_to_image_generated.webp)

Python:

```python
from avalan.entities import TransformerEngineSettings
from avalan.model.vision.diffusion import TextToImageModel

with TextToImageModel("stabilityai/stable-diffusion-xl-base-1.0", settings=TransformerEngineSettings(refiner_model_id="stabilityai/stable-diffusion-xl-refiner-1.0", weight_type="fp16")) as model:
    await model(
        "Leo Messi petting a purring tubby cat",
        "example_messi_petting_cat.jpg",
        color_model="RGB",
        image_format="JPEG",
        high_noise_frac=0.8,
        n_steps=150
    )
```
For a runnable script, see [docs/examples/vision_text_to_image.py](examples/vision_text_to_image.py).

### Text to video

Create an MP4 video from a prompt, using a negative prompt for guardrails and an image as reference:

```sh
echo 'A cute little penguin takes out a book and starts reading it' | \
    avalan model run "Lightricks/LTX-Video-0.9.7-dev" \
        --modality vision_text_to_video \
        --upsampler-model "Lightricks/ltxv-spatial-upscaler-0.9.7" \
        --weight "fp16" \
        --vision-steps 30 \
        --vision-negative-prompt "worst quality, inconsistent motion, blurry, jittery, distorted" \
        --vision-inference-steps 10 \
        --vision-reference-path docs/examples/playground/penguin.png \
        --vision-width 832 \
        --vision-height 480 \
        --vision-frames 96 \
        --vision-fps 24 \
        --vision-decode-timestep 0.05 \
        --vision-denoise-strength 0.4 \
        --path example_text_to_video.mp4
```

And here's the generated video:

![A penguin opening a book](https://avalan.ai/images/github/vision_text_to_video_generated.webp)

Python:

```python
from avalan.entities import EngineSettings
from avalan.model.vision.diffusion import TextToVideoModel

with TextToVideoModel("Lightricks/LTX-Video-0.9.7-dev", settings=EngineSettings(upsampler_model_id="Lightricks/ltxv-spatial-upscaler-0.9.7", weight_type="fp16")) as model:
    await model(
        "A cute little penguin takes out a book and starts reading it",
        "worst quality, inconsistent motion, blurry, jittery, distorted",
        "docs/examples/playground/penguin.png",
        "example_text_to_video.mp4",
        steps=30,
        inference_steps=10,
        width=832,
        height=480,
        frames=96,
        fps=24,
        decode_timestep=0.05,
        denoise_strength=0.4
    )
```
For a runnable script, see [docs/examples/vision_text_to_video.py](examples/vision_text_to_video.py).

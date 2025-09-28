# Examples

This directory showcases configuration files and small programs that demonstrate Avalan's agent framework and model wrappers. Each entry links to the source and notes tunable parameters for experimentation.

## Agent configurations
- [agent_tool.toml](agent_tool.toml) – Agent with a calculator tool. Modify `engine.uri` to use another model, extend the `tool.enable` list, or change `run.max_new_tokens` to control response length.
- [agent_gettext_translator.toml](agent_gettext_translator.toml) – Template-driven gettext translator. Adjust `source_language`, `destination_language`, or switch `engine.uri` to a different model.
- [agent_messi.toml](agent_messi.toml) – Persona of Leo Messi with recent and PostgreSQL-backed memory. Change `engine.uri`, memory connection strings, or `run.max_new_tokens` for longer answers.
- [agent_nagini.toml](agent_nagini.toml) – Python-focused assistant emitting code between `<llm-code>` tags. Tweak `engine.uri`, `weight_type`, or `stop_strings` to fit your needs.
- [agent_sequel.toml](agent_sequel.toml) – SQL specialist returning code inside `<llm-code>` tags. Update `engine.uri`, `weight_type`, or alter `stop_strings` for different output markers.
- [deploy_agent_tool.toml](deploy_agent_tool.toml) – Sample deployment recipe for `agent_tool.toml`. Edit `port`, AWS `zone`, or instance sizes to match your infrastructure.

## Agent clients
- [openai_client_agent_messi.py](openai_client_agent_messi.py) – Queries a locally served `agent_messi` through the OpenAI client. Change the message content, `model`, or `base_url`.
- [openai_client_agent_tool.py](openai_client_agent_tool.py) – Similar client hitting a tool‑enabled agent. Modify the prompt, `model`, or server `base_url`.

## Text generation
- [text_generation.py](text_generation.py) – Streams tokens from a Llama 3 model. Adjust `model` id, `temperature`, or `max_new_tokens`.
- [text_generation_openai.py](text_generation_openai.py) – Uses OpenAI models. Set `OPENAI_API_KEY`, then tweak `model`, `temperature`, or `max_new_tokens`.
- [text_generation_anthropic.py](text_generation_anthropic.py) – Anthropic API example requiring `ANTHROPIC_API_KEY`. Change `model` or generation settings.
- [text_generation_google.py](text_generation_google.py) – Gemini via `GOOGLE_API_KEY`. Modify `model`, `temperature`, or token limits.
- [text_generation_groq.py](text_generation_groq.py) – Groq endpoint needing `GROQ_API_KEY`. Adjust `model` or `GenerationSettings`.
- [text_generation_mlxlm.py](text_generation_mlxlm.py) – Runs an MLX‑LM model locally. Swap `model` ids or tune `temperature` and `max_new_tokens`.
- [text_generation_ollama.py](text_generation_ollama.py) – Uses an Ollama server; ensure one is running. Switch `model` or generation parameters.
- [text_generation_vllm.py](text_generation_vllm.py) – Example with vLLM. Change `model` id or `GenerationSettings` values.

## NLP tasks
- [sequence_classification.py](sequence_classification.py) – Sentiment analysis. Replace the sample `text` or choose another `model`.
- [token_classification.py](token_classification.py) – Named entity recognition. Edit the `text` or disable `labeled_only` to see all tokens.
- [question_answering.py](question_answering.py) – Question answering over a context. Toggle `add_knowledge`, `ask_knowledge`, or `ask_professions` to vary the question and context.
- [seq2seq_summarization.py](seq2seq_summarization.py) – Summarizes long passages. Adjust `num_beams`, `max_length`, or the input `text`.
- [seq2seq_translation.py](seq2seq_translation.py) – Translation using mBART. Modify `source_language`, `destination_language`, or beam search parameters.
- [autocomplete.py](autocomplete.py) – Builds a token‑distance matrix for autocompletion. Change the dataset and output filenames passed on the command line.

## Audio
- [audio_classification.py](audio_classification.py) – Classifies audio files. Provide a different audio path or switch `AudioClassificationModel` ids.
- [audio_generation.py](audio_generation.py) – Generates music from text. Vary the prompt, output file, or `max_new_tokens`.
- [audio_speech_recognition.py](audio_speech_recognition.py) – Transcribes speech; ensure 16 kHz audio. Adjust `sampling_rate` or choose another model id.
- [audio_text_to_speech.py](audio_text_to_speech.py) – Text‑to‑speech with optional voice cloning. Change `reference_path`, `reference_text`, or `max_new_tokens` for longer clips.

## Vision
- [vision_image_classification.py](vision_image_classification.py) – Image classification. Provide different image paths or model ids.
- [vision_image_to_text.py](vision_image_to_text.py) – Generates captions for images. Swap the model or input image.
- [vision_object_detection.py](vision_object_detection.py) – Detects objects in images. Change the model id or test image.
- [vision_semantic_segmentation.py](vision_semantic_segmentation.py) – Returns segmentation labels. Modify the model or input path.
- [vision_encoder_decoder.py](vision_encoder_decoder.py) – OCR and document Q&A. Pass an image path and optional question; change `model_id` logic as needed.
- [vision_ocr.py](vision_ocr.py) – General OCR via `ImageTextToTextModel`. Alter `max_new_tokens` or select another model.
- [vision_text_to_image.py](vision_text_to_image.py) – Diffusion‑based image generation. Adjust the prompt, `refiner_model_id`, or `weight_type`.
- [vision_text_to_animation.py](vision_text_to_animation.py) – Creates short animations. Change `steps`, `base_model_id`, or checkpoint settings.
- [vision_text_to_video.py](vision_text_to_video.py) – Generates videos from text and a reference image. Modify the `prompt`, `negative_prompt`, `reference_path`, or `upsampler_model_id`.

## Miscellaneous
- [pypi_avalan_source.md](pypi_avalan_source.md) – Snapshot of the avalan project page on PyPI for reference.

These examples can be run with `poetry run python <script>` unless noted otherwise. Many require setting API keys via environment variables or passing file paths on the command line.

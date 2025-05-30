from ..entities import (
    AttentionImplementation,
    EngineUri,
    TextGenerationLoaderClass,
    TransformerEngineSettings,
    Vendor,
    WeightType,
)
from ..model.hubs.huggingface import HuggingfaceHub
from ..model.nlp.sentence import SentenceTransformerModel
from ..model.nlp.text.generation import TextGenerationModel
from ..secrets import KeyringSecrets
from contextlib import ContextDecorator, ExitStack
from logging import Logger
from typing import Any, get_args
from urllib.parse import urlparse, parse_qsl


class ModelManager(ContextDecorator):
    _hub: HuggingfaceHub
    _stack: ExitStack
    _logger: Logger
    _secrets: KeyringSecrets

    def __init__(
        self,
        hub: HuggingfaceHub,
        logger: Logger,
        secrets: KeyringSecrets | None = None,
    ):
        self._hub, self._logger = hub, logger
        self._stack = ExitStack()
        self._secrets = secrets or KeyringSecrets()

    def __enter__(self):
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: Any | None,
    ):
        return self._stack.__exit__(exc_type, exc_value, traceback)

    def get_engine_settings(
        self,
        engine_uri,
        settings: dict | None = None,
        is_sentence_transformer: bool | None = None,
    ) -> TransformerEngineSettings:
        engine_settings_args = settings or {}

        if not is_sentence_transformer and not engine_uri.is_local:
            token = None
            if engine_uri.password and engine_uri.user:
                if engine_uri.user == "secret":
                    token = self._secrets.read(engine_uri.password)
                else:
                    token = None
            elif engine_uri.user:
                token = engine_uri.user

            if token:
                engine_settings_args.update(access_token=token)

        engine_settings = TransformerEngineSettings(**engine_settings_args)
        return engine_settings

    def load(
        self,
        engine_uri: EngineUri,
        *args,
        attention: AttentionImplementation | None = None,
        device: str | None = None,
        disable_loading_progress_bar: bool = False,
        is_sentence_transformer: bool | None = None,
        loader_class: TextGenerationLoaderClass | None = "auto",
        low_cpu_mem_usage: bool = False,
        quiet: bool = False,
        revision: str | None = None,
        special_tokens: list[str] | None = None,
        tokenizer: str | None = None,
        tokens: list[str] | None = None,
        trust_remote_code: bool | None = None,
        weight_type: WeightType = "auto",
    ) -> SentenceTransformerModel | TextGenerationModel:
        engine_settings_args = dict(
            cache_dir=self._hub.cache_dir,
            device=device,
            disable_loading_progress_bar=quiet or disable_loading_progress_bar,
            low_cpu_mem_usage=low_cpu_mem_usage,
            loader_class=loader_class,
            revision=revision,
            special_tokens=special_tokens or None,
            tokenizer_name_or_path=tokenizer,
            tokens=tokens or None,
            weight_type=weight_type,
        )
        if not is_sentence_transformer:
            engine_settings_args.update(
                attention=attention or None,
                trust_remote_code=trust_remote_code or None,
            )

        engine_settings = self.get_engine_settings(
            engine_uri,
            engine_settings_args,
            is_sentence_transformer=is_sentence_transformer,
        )
        return self.load_engine(
            engine_uri, engine_settings, is_sentence_transformer
        )

    def load_engine(
        self,
        engine_uri: EngineUri,
        engine_settings: TransformerEngineSettings,
        is_sentence_transformer: bool | None = None,
    ) -> SentenceTransformerModel | TextGenerationModel:
        model_load_args = dict(
            model_id=engine_uri.model_id,
            settings=engine_settings,
            logger=self._logger,
        )

        # Load local model, or lazy-import per vendor
        if is_sentence_transformer or engine_uri.is_local:
            model = (
                SentenceTransformerModel(**model_load_args)
                if is_sentence_transformer
                else TextGenerationModel(**model_load_args)
            )
        elif engine_uri.vendor == "openai":
            from ..model.nlp.text.vendor.openai import OpenAIModel

            model = OpenAIModel(**model_load_args)
        elif engine_uri.vendor == "openrouter":
            from ..model.nlp.text.vendor.openrouter import OpenRouterModel

            model = OpenRouterModel(**model_load_args)
        elif engine_uri.vendor == "anyscale":
            from ..model.nlp.text.vendor.anyscale import AnyScaleModel

            model = AnyScaleModel(**model_load_args)
        elif engine_uri.vendor == "together":
            from ..model.nlp.text.vendor.together import TogetherModel

            model = TogetherModel(**model_load_args)
        elif engine_uri.vendor == "deepseek":
            from ..model.nlp.text.vendor.deepseek import DeepSeekModel

            model = DeepSeekModel(**model_load_args)
        elif engine_uri.vendor == "deepinfra":
            from ..model.nlp.text.vendor.deepinfra import DeepInfraModel

            model = DeepInfraModel(**model_load_args)
        elif engine_uri.vendor == "groq":
            from ..model.nlp.text.vendor.groq import GroqModel

            model = GroqModel(**model_load_args)
        elif engine_uri.vendor == "ollama":
            from ..model.nlp.text.vendor.ollama import OllamaModel

            model = OllamaModel(**model_load_args)
        elif engine_uri.vendor == "huggingface":
            from ..model.nlp.text.vendor.huggingface import HuggingfaceModel

            model = HuggingfaceModel(**model_load_args)
        elif engine_uri.vendor == "hyperbolic":
            from ..model.nlp.text.vendor.hyperbolic import HyperbolicModel

            model = HyperbolicModel(**model_load_args)
        elif engine_uri.vendor == "litellm":
            from ..model.nlp.text.vendor.litellm import LiteLLMModel

            model = LiteLLMModel(**model_load_args)

        self._stack.enter_context(model)
        return model

    @staticmethod
    def parse_uri(uri: str) -> EngineUri:
        parsed = urlparse(uri)
        if not parsed.scheme:
            uri = f"ai://{uri}"
            parsed = urlparse(uri)

        if parsed.scheme != "ai":
            raise ValueError(f"Invalid scheme {parsed.scheme!r}, expected 'ai'")

        vendor = parsed.hostname
        if not vendor or vendor not in get_args(Vendor) or vendor == "local":
            vendor = None
        use_host = bool(vendor)
        path_prefixed = parsed.path.startswith("/")
        params = dict(parse_qsl(parsed.query))

        # urlparse() normalizes hostname to lowercase, so keep original case
        authority = parsed.netloc.rsplit("@", 1)[-1]
        hostname = authority.split(":", 1)[0]

        model_id = (
            hostname + ("/" if path_prefixed else "")
            if not vendor and hostname != "local"
            else ""
        ) + (parsed.path[1:] if path_prefixed else parsed.path)
        engine_uri = EngineUri(
            vendor=vendor,
            host=hostname if use_host else None,
            port=(parsed.port or None) if use_host else None,
            user=parsed.username or None,
            password=parsed.password or None,
            model_id=model_id,
            params=params,
        )
        return engine_uri

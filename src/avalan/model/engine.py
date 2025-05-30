from abc import ABC, abstractmethod
from ..entities import (
    EngineSettings,
    Input,
    ModelConfig,
    SentenceTransformerModelConfig,
    TokenizerConfig,
)
from ..model import (
    EngineResponse,
    ModelAlreadyLoadedException,
    TextGenerationVendor,
    TokenizerAlreadyLoadedException,
    TokenizerNotSupportedException,
)
from contextlib import ExitStack
from importlib.util import find_spec
from logging import ERROR, Logger, getLogger
from torch import cuda
from torch.backends import mps
from transformers import logging as transformers_logging
from transformers.utils.logging import disable_progress_bar, enable_progress_bar
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)
from typing import Any, Type, Union


class Engine(ABC):
    _device: str
    _logger: Logger
    _model_id: str
    _settings: EngineSettings
    _transformers_logging_logger: Logger
    _transformers_logging_level: int
    _loaded_model: bool = False
    _loaded_tokenizer: bool = False
    _tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] | None = (
        None
    )
    _model: Union[PreTrainedModel, TextGenerationVendor] | None = None
    _config: Union[ModelConfig, SentenceTransformerModelConfig] | None = None
    _tokenizer_config: TokenizerConfig | None = None
    _exit_stack: ExitStack = ExitStack()

    def __init__(
        self,
        model_id: str,
        settings: EngineSettings | None = None,
        logger: Logger | None = None,
    ):
        assert model_id
        self._logger = logger if logger else getLogger()
        self._model_id = model_id
        self._settings = settings if settings else EngineSettings()
        self._device = (
            self._settings.device
            if self._settings.device
            else Engine.get_default_device()
        )
        self._transformers_logging_logger = (
            transformers_logging.get_logger()
            if self._settings.change_transformers_logging_level
            else None
        )
        self._transformers_logging_level = (
            self._transformers_logging_logger.level
            if self._settings.change_transformers_logging_level
            else None
        )

        auto_load_tokenizer = (
            self.uses_tokenizer and self._settings.auto_load_tokenizer
        )
        if self._settings.auto_load_model or auto_load_tokenizer:
            self._load(
                load_tokenizer=auto_load_tokenizer,
                tokenizer_name_or_path=self._settings.tokenizer_name_or_path
                if auto_load_tokenizer
                else None,
            )

    @property
    def uses_tokenizer(self) -> bool:
        return False

    @property
    def config(
        self,
    ) -> Union[ModelConfig, SentenceTransformerModelConfig] | None:
        return self._config

    @property
    def model(self) -> PreTrainedModel | None:
        return self._model

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def tokenizer_config(self) -> TokenizerConfig | None:
        return self._tokenizer_config

    @property
    def tokenizer(
        self,
    ) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast] | None:
        return self._tokenizer

    @abstractmethod
    async def __call__(self, input: Input, **kwargs) -> EngineResponse:
        raise NotImplementedError()

    @abstractmethod
    def _load_model(self) -> Union[PreTrainedModel, TextGenerationVendor]:
        raise NotImplementedError()

    def _load_tokenizer_with_tokens(
        self, tokenizer_name_or_path: str | None, use_fast: bool = True
    ) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
        raise (
            TokenizerNotSupportedException()
            if not self.uses_tokenizer()
            else NotImplementedError()
        )

    def __enter__(self):
        _l = self._log
        if (
            self._transformers_logging_logger
            and self._transformers_logging_level != ERROR
        ):
            transformers_logging.set_verbosity_error()
            _l(
                "Changed transformers logging level from "
                f"{self._transformers_logging_level} to "
                f"{self._transformers_logging_logger.level}"
            )
        return self

    def __exit__(
        self,
        exc_type: Type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: Any | None,
    ):
        _l = self._log
        if (
            self._transformers_logging_logger
            and self._transformers_logging_level
            and self._transformers_logging_level != ERROR
        ):
            self._transformers_logging_logger.setLevel(
                self._transformers_logging_level
            )
            _l(
                f"Restored transformers logging level to "
                f"{self._transformers_logging_level}"
            )
        return self._exit_stack.__exit__(exc_type, exc_value, traceback)

    def _load(
        self, *args, load_tokenizer: bool, tokenizer_name_or_path: str | None
    ) -> None:
        if (
            self._settings.auto_load_model
            and hasattr(self, "_loaded_model")
            and self._loaded_model
        ):
            raise ModelAlreadyLoadedException()
        elif (
            load_tokenizer
            and hasattr(self, "_loaded_tokenizer")
            and self._loaded_tokenizer
        ):
            raise TokenizerAlreadyLoadedException()

        _l = self._log

        if self._settings.disable_loading_progress_bar:
            disable_progress_bar()

        if load_tokenizer:
            _l(f"Loading tokenizer {tokenizer_name_or_path or self._model_id}")
            self._tokenizer = self._load_tokenizer_with_tokens(
                tokenizer_name_or_path or self._model_id, use_fast=True
            )
            assert isinstance(
                self._tokenizer, PreTrainedTokenizer
            ) or isinstance(self._tokenizer, PreTrainedTokenizerFast), (
                "Unexpected pretrained tokenizer type: "
                + f"{type(self._tokenizer)}"
            )
            _l(f"Loaded tokenizer {self._tokenizer.name_or_path}")

            self._loaded_tokenizer = True

        is_sentence_transformer = False
        if self._settings.auto_load_model:
            _l(
                f"Loading pretrained model {self._model_id} from "
                f"cache {self._settings.cache_dir}"
            )
            self._model = self._load_model()

            is_mlx = False
            is_sentence_transformer = False

            if not isinstance(self._model, PreTrainedModel) and not isinstance(
                self._model, TextGenerationVendor
            ):
                if find_spec("mlx"):
                    from mlx.nn import Module

                    is_mlx = isinstance(self._model, Module)

                if find_spec("sentence_transformers"):
                    from sentence_transformers import SentenceTransformer

                    is_sentence_transformer = isinstance(
                        self._model, SentenceTransformer
                    )

            assert (
                isinstance(self._model, PreTrainedModel)
                or isinstance(self._model, TextGenerationVendor)
                or is_mlx
                or is_sentence_transformer
            ), f"Unexpected pretrained model type: {type(self._model)}"

            _l(
                f"Loaded pretrained model {self._model_id} from "
                f"cache {self._settings.cache_dir}"
            )

            if self._settings.enable_eval:
                _l(f"Setting model {self._model_id} in eval mode")
                self._model.eval()

            if self._tokenizer and (
                self._settings.tokens or self._settings.special_tokens
            ):
                total_tokens = len(self._tokenizer)
                _l(
                    f"Resizing embedding matrix to {total_tokens} tokens "
                    f"for model {self._model_id}"
                )
                self._model.resize_token_embeddings(total_tokens)
                _l(
                    f"Resized embedding matrix to {total_tokens} tokens "
                    f"for model {self._model_id}"
                )

            self._loaded_model = True

        if self._model and not self._config:
            config: (
                Union[ModelConfig, SentenceTransformerModelConfig] | None
            ) = None
            mc = (
                self._model.config
                if hasattr(self._model, "config")
                else self._model[0].auto_model.config
                if is_sentence_transformer
                else None
            )

            if mc:
                config = ModelConfig(
                    architectures=mc.architectures,
                    attribute_map=mc.attribute_map,
                    bos_token_id=mc.bos_token_id,
                    bos_token=self._tokenizer.decode(mc.bos_token_id)
                    if self._tokenizer and mc.bos_token_id
                    else None,
                    decoder_start_token_id=mc.decoder_start_token_id,
                    eos_token_id=mc.eos_token_id,
                    eos_token=self._tokenizer.decode(mc.eos_token_id)
                    if self._tokenizer and mc.eos_token_id
                    else None,
                    finetuning_task=mc.finetuning_task,
                    hidden_size=mc.hidden_size
                    if hasattr(mc, "hidden_size")
                    else None,
                    hidden_sizes=mc.hidden_sizes
                    if hasattr(mc, "hidden_sizes")
                    else None,
                    keys_to_ignore_at_inference=mc.keys_to_ignore_at_inference
                    if hasattr(mc, "keys_to_ignore_at_inference")
                    else None,
                    loss_type=mc.loss_type
                    if hasattr(mc, "loss_type")
                    else None,
                    max_position_embeddings=mc.max_position_embeddings
                    if hasattr(mc, "max_position_embeddings")
                    else None,
                    model_type=mc.model_type,
                    num_attention_heads=mc.num_attention_heads
                    if hasattr(mc, "num_attention_heads")
                    else None,
                    num_hidden_layers=mc.num_hidden_layers
                    if hasattr(mc, "num_hidden_layers")
                    else None,
                    num_labels=mc.num_labels,
                    output_attentions=mc.output_attentions,
                    output_hidden_states=mc.output_hidden_states,
                    pad_token_id=mc.pad_token_id,
                    pad_token=self._tokenizer.decode(mc.pad_token_id)
                    if self._tokenizer and mc.pad_token_id
                    else None,
                    prefix=mc.prefix,
                    sep_token_id=mc.sep_token_id,
                    sep_token=self._tokenizer.decode(mc.sep_token_id)
                    if self._tokenizer and mc.sep_token_id
                    else None,
                    state_size=len(self._model.state_dict().keys())
                    if self._model.state_dict
                    else 0,
                    task_specific_params=mc.task_specific_params,
                    torch_dtype=str(mc.torch_dtype),
                    vocab_size=mc.vocab_size
                    if hasattr(mc, "vocab_size")
                    else None,
                    tokenizer_class=mc.tokenizer_class,
                )

            if is_sentence_transformer and config:
                config = SentenceTransformerModelConfig(
                    backend=self._model.backend,
                    similarity_function=self._model.similarity_fn_name,
                    truncate_dimension=self._model.truncate_dim,
                    transformer_model_config=config,
                )

            self._config = config

        if self._tokenizer and not self._tokenizer_config:
            self._tokenizer_config = TokenizerConfig(
                name_or_path=self._tokenizer.name_or_path,
                tokens=self._settings.tokens,
                special_tokens=self._tokenizer.all_special_tokens,
                tokenizer_model_max_length=self._tokenizer.model_max_length,
                fast=isinstance(self._tokenizer, PreTrainedTokenizerFast),
            )

        if self._settings.disable_loading_progress_bar:
            enable_progress_bar()

    @staticmethod
    def get_default_device() -> str:
        return (
            "cuda"
            if cuda.is_available()
            else "mps"
            if mps.is_available()
            else "cpu"
        )

    def _log(self, message: str) -> None:
        self._logger.debug(f"<{self._model_id}> {message}")

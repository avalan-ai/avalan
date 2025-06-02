from abc import ABC, abstractmethod
from ..entities import (
    Input,
    Token,
    TransformerEngineSettings,
)
from ..model.engine import Engine
from logging import Logger
from tokenizers import AddedToken
from torch import Tensor
from transformers.tokenization_utils_base import BatchEncoding
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)
from typing import Literal


class TransformerModel(Engine, ABC):
    @property
    def supports_sample_generation(self) -> bool:
        return False

    @property
    def supports_token_streaming(self) -> bool:
        return False

    @property
    def uses_tokenizer(self) -> bool:
        return True

    @abstractmethod
    def _tokenize_input(
        self,
        input: Input,
        context: str | None = None,
        tensor_format: Literal["pt"] = "pt",
        **kwargs,
    ) -> dict[str, Tensor] | BatchEncoding | Tensor:
        raise NotImplementedError()

    def __init__(
        self,
        model_id: str,
        settings: TransformerEngineSettings | None = None,
        logger: Logger | None = None,
    ) -> None:
        super().__init__(
            model_id, settings or TransformerEngineSettings(), logger
        )

    def save_tokenizer(self, path: str) -> list[str]:
        assert path and self._tokenizer
        _l = self._log
        _l(f"Saving tokenizer {self._tokenizer.name_or_path} to {path}")
        paths = self._tokenizer.save_pretrained(path)
        _l(f"Saved tokenizer {self._tokenizer.name_or_path} to {path}: {paths}")
        return list(paths)

    def tokenize(
        self, text: str, tokenizer_name_or_path: str | None = None
    ) -> list[Token]:
        _l = self._log
        if not hasattr(self, "_loaded_tokenizer") or not self._loaded_tokenizer:
            self.load(
                load_model=False,
                load_tokenizer=True,
                tokenizer_name_or_path=tokenizer_name_or_path,
            )

        _l(f'Tokenizing text "{text}"')
        token_ids = self._tokenizer.encode(text, add_special_tokens=True)
        _l(f'Tokenized text "{text}" into {len(token_ids)} tokens')

        return [
            Token(
                id=token_id,
                token=self._tokenizer.decode(
                    token_id, skip_special_tokens=False
                ),
                probability=1,
            )
            for i, token_id in enumerate(token_ids)
        ]

    def input_token_count(
        self, input: Input, system_prompt: str | None = None
    ) -> int:
        _l = self._log
        assert self._tokenizer, (
            f"Model {self._model} can't be executed "
            + "without a tokenizer loaded first"
        )
        inputs = self._tokenize_input(input, system_prompt, context=None)
        return (
            len(inputs["input_ids"][0])
            if inputs and "input_ids" in inputs
            else 0
        )

    def _load_tokenizer(
        self, tokenizer_name_or_path: str | None, use_fast: bool
    ) -> PreTrainedTokenizer | PreTrainedTokenizerFast:
        return AutoTokenizer.from_pretrained(
            tokenizer_name_or_path or self._model_id, use_fast=use_fast
        )

    def _load_tokenizer_with_tokens(
        self, tokenizer_name_or_path: str | None, use_fast: bool = True
    ) -> PreTrainedTokenizer | PreTrainedTokenizerFast:
        _l = self._log
        tokenizer = self._load_tokenizer(tokenizer_name_or_path, use_fast)
        if self._settings.tokens:
            _l(
                f"Adding {len(self._tokens)} tokens to tokenizer "
                f"{tokenizer.name_or_path}: {self._settings.tokens}"
            )
            added_tokens = tokenizer.add_tokens(self._settings.tokens)
            _l(
                f"Added {added_tokens} tokens to tokenizer "
                f"{tokenizer.name_or_path}"
            )

        if self._settings.special_tokens:
            _l(
                f"Adding {len(self._settings.special_tokens)} special tokens "
                f"to tokenizer {tokenizer.name_or_path}: "
                f"{self._settings.special_tokens}"
            )
            added_tokens = tokenizer.add_special_tokens(
                {
                    "additional_special_tokens": [
                        AddedToken(
                            token,
                            # Defines whether this token should strip all potential
                            # whitespaces on its left side
                            lstrip=False,
                            # Defines whether this token should match against the
                            # normalized version of the input text
                            normalized=False,
                            # Defines whether this token should strip all potential
                            # whitespaces on its right side
                            rstrip=False,
                            # Defines whether this token should only match single
                            # words
                            single_word=False,
                            # Defines whether this token should be skipped when
                            # decoding
                            special=False,
                        )
                        for token in self._settings.special_tokens
                    ]
                }
            )
            _l(
                f"Added {added_tokens} special tokens to tokenizer "
                f"{tokenizer.name_or_path}"
            )

        return tokenizer

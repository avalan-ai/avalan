from ...compat import override
from ...filters import Partitioner
from ...entities import TextPartition
from ...model.nlp.sentence import SentenceTransformerModel
from logging import Logger
from re import split
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from typing import Callable


class TextPartitioner(Partitioner):
    _PARAGRAPH_PATTERN = r"(?:\r\n|\r|\n){2,}"
    _model: SentenceTransformerModel
    _tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast
    _logger: Logger
    _max_tokens: int
    _window_size: int
    _overlap_size: int

    def __init__(
        self,
        model: SentenceTransformerModel,
        logger: Logger,
        *args,
        max_tokens: int,
        overlap_size: int,
        window_size: int,
    ):
        assert model and model.tokenizer
        self._model = model
        self._tokenizer = model.tokenizer
        self._logger = logger
        self.configure(
            max_tokens=max_tokens,
            overlap_size=overlap_size,
            window_size=window_size,
        )

    def configure(
        self,
        *args,
        max_tokens: int,
        overlap_size: int,
        window_size: int,
    ) -> None:
        assert (
            max_tokens
            and window_size
            and overlap_size
            and max_tokens > 0
            and window_size > 0
            and overlap_size > 0
            and window_size < max_tokens
            and overlap_size < max_tokens
            and window_size > overlap_size
        )
        self._max_tokens = max_tokens
        self._window_size = window_size
        self._overlap_size = overlap_size

    @override
    @property
    def sentence_model(self) -> Callable:
        return self._model

    @override
    async def __call__(
        self,
        text: str,
    ) -> list[TextPartition]:
        assert text
        partitions: list[TextPartition] = []

        for paragraph in split(self._PARAGRAPH_PATTERN, text):
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            token_ids = self._tokenizer.encode(
                paragraph,
                add_special_tokens=False,
            )
            length = len(token_ids)
            if length <= self._max_tokens:
                embeddings = await self._model(paragraph)
                partitions.append(
                    TextPartition(
                        data=paragraph,
                        embeddings=embeddings,
                        total_tokens=length,
                    )
                )
            else:
                step = self._window_size - self._overlap_size

                for start in range(0, length, step):
                    window_token_ids = token_ids[
                        start : start + self._window_size
                    ]
                    section_length = len(window_token_ids)
                    section = self._tokenizer.decode(
                        window_token_ids, skip_special_tokens=True
                    )
                    embeddings = await self._model(section)
                    partitions.append(
                        TextPartition(
                            data=section,
                            embeddings=embeddings,
                            total_tokens=section_length,
                        )
                    )

        return partitions

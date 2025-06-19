from avalan.memory.partitioner.text import TextPartitioner, TextPartition
from avalan.model.nlp.sentence import SentenceTransformerModel
from logging import Logger
from numpy import arange, ndarray
from numpy.testing import assert_array_equal

from unittest import IsolatedAsyncioTestCase, main
from unittest.mock import AsyncMock, MagicMock, call, _Call


class TextPartitionerTestCase(IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.sources: list[
            tuple[
                str,
                int,
                int,
                int,
                str,
                list[tuple[int, int, int, int | None, int, str]],
            ]
        ] = [
            (
                "sentence-transformers/all-MiniLM-L6-v2",
                60,
                30,
                15,
                (
                    "Lionel Messi, often hailed as one of the greatest"
                    " footballers of all time, has captivated audiences around"
                    " the globe with his extraordinary talent and humble"
                    " personality. Born on June 24, 1987, in Rosario,"
                    " Argentina, Messi began his football journey at an early"
                    " age. Diagnosed with a growth hormone deficiency, his"
                    " potential was initially overshadowed by health concerns."
                    " Nevertheless, his remarkable talent soon became evident,"
                    " and at 13, he moved to Spain to join FC Barcelona, the"
                    " club that shaped him into a global phenomenon."
                ),
                [
                    (
                        0,
                        184,
                        0,
                        30,
                        30,
                        (
                            "Lionel Messi, often hailed as one of the greatest"
                            " footballers of all time, has captivated"
                            " audiences around the globe with his"
                            " extraordinary talent and humble personality."
                            " Born on June 24, "
                        ),
                    ),
                    (
                        90,
                        277,
                        15,
                        45,
                        30,
                        (
                            "audiences around the globe with his extraordinary"
                            " talent and humble personality. Born on June 24,"
                            " 1987, in Rosario, Argentina, Messi began his"
                            " football journey at an early age. Diagnosed"
                            " with "
                        ),
                    ),
                    (
                        188,
                        390,
                        30,
                        60,
                        30,
                        (
                            "1987, in Rosario, Argentina, Messi began his"
                            " football journey at an early age. Diagnosed with"
                            " a growth hormone deficiency, his potential was"
                            " initially overshadowed by health concerns."
                            " Nevertheless, his remarkable "
                        ),
                    ),
                    (
                        282,
                        469,
                        45,
                        75,
                        30,
                        (
                            "a growth hormone deficiency, his potential was"
                            " initially overshadowed by health concerns."
                            " Nevertheless, his remarkable talent soon became"
                            " evident, and at 13, he moved to Spain to join FC"
                            " Barcelona, "
                        ),
                    ),
                    (
                        401,
                        518,
                        60,
                        90,
                        24,
                        (
                            "talent soon became evident, and at 13, he moved"
                            " to Spain to join FC Barcelona, the club that"
                            " shaped him into a global phenomenon."
                        ),
                    ),
                    (
                        480,
                        518,
                        75,
                        None,
                        9,
                        "the club that shaped him into a global phenomenon.",
                    ),
                ],
            )
        ]

    async def test_partition(self):
        logger_mock = MagicMock(spec=Logger)
        for (
            model_id,
            max_tokens,
            window_size,
            overlap_size,
            input_string,
            expected_chunks,
        ) in self.sources:
            input_string = input_string.strip()
            input_string_length = len(input_string)
            # Naive tokenization for testing, where each token ID is actually
            # the index of the given word within the original string. Allows
            # for quick token ID decoding
            token_ids = [
                i
                for i, c in enumerate(input_string)
                if c != " " and (i == 0 or input_string[i - 1] == " ")
            ]
            assert token_ids

            with self.subTest():
                model_mock = AsyncMock(spec=SentenceTransformerModel)
                model_mock.tokenizer = MagicMock()
                model_mock.tokenizer.encode.side_effect = [token_ids]

                decode_side_effect: list[str] = []
                call_side_effect: list[ndarray] = []
                for start, finish, _, _, _, _ in expected_chunks:
                    decode_side_effect.append(
                        input_string[
                            start : next(
                                (
                                    token_ids[i + 1]
                                    for i, p in enumerate(token_ids)
                                    if p == finish
                                ),
                                input_string_length,
                            )
                        ],
                    )
                    call_side_effect.append(arange(start, finish))

                model_mock.tokenizer.decode.side_effect = decode_side_effect
                model_mock.side_effect = call_side_effect

                partitioner = TextPartitioner(
                    model_mock,
                    logger=logger_mock,
                    max_tokens=max_tokens,
                    window_size=window_size,
                    overlap_size=overlap_size,
                )
                self.assertIsInstance(partitioner, TextPartitioner)
                logger_mock.assert_not_called()

                partitions = await partitioner(input_string)

                self.assertEqual(model_mock.tokenizer.encode.call_count, 1)
                model_mock.tokenizer.encode.assert_has_calls(
                    [call(input_string, add_special_tokens=False)]
                )

                self.assertEqual(
                    model_mock.tokenizer.decode.call_count,
                    len(expected_chunks),
                )
                self.assertEqual(model_mock.call_count, len(expected_chunks))

                expected_calls: list[_Call] = [
                    call.tokenizer.__bool__(),
                    call.tokenizer.encode(
                        input_string, add_special_tokens=False
                    ),
                ]

                for _, _, beg, end, _, chunk_string in expected_chunks:
                    expected_calls.append(
                        call.tokenizer.decode(
                            token_ids[beg:end] if end else token_ids[beg:],
                            skip_special_tokens=True,
                        )
                    )
                    expected_calls.append(call(chunk_string))

                model_mock.assert_has_calls(expected_calls)

                self.assertEqual(len(partitions), len(expected_chunks))

                i = 0
                for (
                    start,
                    finish,
                    beg,
                    end,
                    token_count,
                    chunk_string,
                ) in expected_chunks:
                    partition = partitions[i]
                    self.assertIsInstance(partition, TextPartition)
                    self.assertEqual(partition.data, chunk_string)
                    self.assertEqual(partition.total_tokens, token_count)
                    assert_array_equal(
                        partition.embeddings, arange(start, finish)
                    )
                    i = i + 1


class TextPartitionerPropertyTestCase(IsolatedAsyncioTestCase):
    def test_sentence_model_property(self):
        model_mock = AsyncMock(spec=SentenceTransformerModel)
        model_mock.tokenizer = MagicMock()
        logger_mock = MagicMock(spec=Logger)

        partitioner = TextPartitioner(
            model_mock,
            logger=logger_mock,
            max_tokens=5,
            overlap_size=1,
            window_size=2,
        )

        self.assertIs(partitioner.sentence_model, model_mock)


class TextPartitionerShortTextTestCase(IsolatedAsyncioTestCase):
    async def test_call_with_empty_lines_short_text(self):
        model_mock = AsyncMock(spec=SentenceTransformerModel)
        model_mock.tokenizer = MagicMock()
        model_mock.tokenizer.encode.side_effect = [[1, 2], [3, 4, 5]]
        model_mock.side_effect = [arange(2), arange(3, 6)]

        logger_mock = MagicMock(spec=Logger)
        partitioner = TextPartitioner(
            model_mock,
            logger=logger_mock,
            max_tokens=10,
            overlap_size=1,
            window_size=5,
        )

        text = "alpha beta\n\n\ngamma delta"
        partitions = await partitioner(text)

        self.assertEqual(len(partitions), 2)

        model_mock.tokenizer.encode.assert_has_calls(
            [
                call("alpha beta", add_special_tokens=False),
                call("gamma delta", add_special_tokens=False),
            ]
        )
        model_mock.assert_has_awaits([call("alpha beta"), call("gamma delta")])
        model_mock.tokenizer.decode.assert_not_called()

        part1, part2 = partitions
        self.assertEqual(part1.data, "alpha beta")
        self.assertEqual(part1.total_tokens, 2)
        assert_array_equal(part1.embeddings, arange(2))

        self.assertEqual(part2.data, "gamma delta")
        self.assertEqual(part2.total_tokens, 3)
        assert_array_equal(part2.embeddings, arange(3, 6))

    async def test_call_with_leading_trailing_newlines(self):
        model_mock = AsyncMock(spec=SentenceTransformerModel)
        model_mock.tokenizer = MagicMock()
        model_mock.tokenizer.encode.return_value = [1, 2]
        model_mock.return_value = arange(2)

        logger_mock = MagicMock(spec=Logger)
        partitioner = TextPartitioner(
            model_mock,
            logger=logger_mock,
            max_tokens=10,
            overlap_size=1,
            window_size=5,
        )

        text = "\n\nalpha beta\n\n"
        partitions = await partitioner(text)

        self.assertEqual(len(partitions), 1)
        self.assertEqual(partitions[0].data, "alpha beta")
        model_mock.assert_awaited_once_with("alpha beta")


if __name__ == "__main__":
    main()

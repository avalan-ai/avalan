from argparse import Namespace
from asyncio import to_thread
from ...cli import get_input
from ...cli.commands import get_model_settings
from ...entities import DistanceType, Modality, SearchMatch, Similarity
from ...cli.commands.model import model_display
from ...memory.partitioner.text import TextPartitioner, TextPartition
from ...memory.partitioner.code import CodePartitioner
from ...memory.permanent import MemoryType
from ...memory.permanent.pgsql.raw import PgsqlRawMemory
from uuid import UUID
from ...model.hubs.huggingface import HuggingfaceHub
from ...model.manager import ModelManager
from faiss import IndexFlatL2
from httpx import AsyncClient, Response
from pathlib import Path
from urllib.parse import urlparse
from io import BytesIO
from logging import Logger
from markitdown import MarkItDown, DocumentConverterResult
from numpy import abs, corrcoef, dot, sum, vstack
from numpy.linalg import norm
from rich.console import Console
from rich.theme import Theme


async def memory_document_index(
    args: Namespace,
    console: Console,
    theme: Theme,
    hub: HuggingfaceHub,
    logger: Logger,
) -> None:
    assert args.model and args.source and args.partition_max_tokens
    assert args.partition_overlap and args.partition_window
    assert args.dsn and args.participant and args.namespace

    def transform(html: bytes) -> DocumentConverterResult:
        return MarkItDown().convert_stream(BytesIO(html))

    _, _i = theme._, theme.icons
    model_id = args.model
    source = args.source
    participant_id = UUID(args.participant)
    namespace = args.namespace
    dsn = args.dsn
    identifier = args.identifier or None
    display_partitions = (
        args.display_partitions if not args.no_display_partitions else None
    )

    with ModelManager(hub, logger) as manager:
        engine_uri = manager.parse_uri(args.model)
        model_settings = get_model_settings(
            args, hub, logger, engine_uri, modality=Modality.EMBEDDING
        )

        with manager.load(**model_settings) as stm:
            logger.debug("Loaded model %s", stm.config.__repr__())

            model_display(
                args, console, theme, hub, logger, model=stm, summary=True
            )

            contents: str | None = None

            is_url = urlparse(source).scheme in ("http", "https")
            if is_url:
                async with AsyncClient() as client:
                    response: Response = await client.get(source)
                    response.raise_for_status()
                    result = await to_thread(transform, response.content)
                    contents = result.text_content
            else:
                contents = Path(source).read_text(encoding=args.encoding)

            if not identifier:
                identifier = source if is_url else str(Path(source).resolve())

            if is_url:
                memory_type = MemoryType.URL
            else:
                memory_type = (
                    MemoryType.CODE
                    if args.partitioner == "code"
                    else MemoryType.FILE
                )

            if is_url or args.partitioner == "text":
                partitioner = TextPartitioner(
                    stm,
                    logger,
                    max_tokens=args.partition_max_tokens,
                    window_size=args.partition_window,
                    overlap_size=args.partition_overlap,
                )
                partitions = await partitioner(contents)
            else:
                code_partitioner = CodePartitioner(logger)
                code_partitions, _ = await to_thread(
                    code_partitioner.partition,
                    args.language or "python",
                    contents,
                    args.encoding,
                    args.partition_max_tokens,
                )
                partitions: list[TextPartition] = []
                for cp in code_partitions:
                    embeddings = await stm(cp.data)
                    tokens = stm.token_count(cp.data)
                    partitions.append(
                        TextPartition(
                            data=cp.data,
                            embeddings=embeddings,
                            total_tokens=tokens,
                        )
                    )

            memory_store = await PgsqlRawMemory.create_instance(
                dsn=dsn, logger=logger
            )
            await memory_store.append_with_partitions(
                namespace,
                participant_id,
                memory_type=memory_type,
                data=contents,
                identifier=identifier,
                partitions=partitions,
                symbols={},
                model_id=model_id,
            )

            if display_partitions:
                console.print(
                    theme.memory_partitions(
                        partitions,
                        display_partitions=display_partitions,
                    )
                )


async def memory_embeddings(
    args: Namespace,
    console: Console,
    theme: Theme,
    hub: HuggingfaceHub,
    logger: Logger,
) -> None:
    assert args.model
    _, _i = theme._, theme.icons
    model_id = args.model
    display_partitions = (
        args.display_partitions if not args.no_display_partitions else None
    )
    compare_strings = args.compare or None
    searches = args.search or None
    search_k = args.search_k or 1
    sort_by: DistanceType = args.sort or DistanceType.L2

    sort_key = {
        DistanceType.COSINE: lambda s: s.cosine_distance,
        DistanceType.DOT: lambda s: s.inner_product,
        DistanceType.L1: lambda s: s.l1_distance,
        DistanceType.L2: lambda s: s.l2_distance,
        DistanceType.PEARSON: lambda s: s.pearson,
    }[sort_by]
    reverse_sort = sort_by in (DistanceType.COSINE, DistanceType.PEARSON)

    model_settings = get_model_settings(
        args, hub, logger, model_id, modality=Modality.EMBEDDING
    )
    with ModelManager(hub, logger) as manager:
        with manager.load(**model_settings) as stm:
            logger.debug("Loaded model %s", stm.config.__repr__())

            model_display(
                args, console, theme, hub, logger, model=stm, summary=True
            )

            input_string = get_input(
                console,
                _i["user_input"] + " ",
                echo_stdin=not args.no_repl,
                is_quiet=args.quiet,
            )
            if not input_string:
                return

            partitioner = (
                TextPartitioner(
                    stm,
                    logger,
                    max_tokens=args.partition_max_tokens,
                    window_size=args.partition_window,
                    overlap_size=args.partition_overlap,
                )
                if args.partition
                else None
            )

            logger.debug(
                f'Looking to embed string "{input_string}" with {model_id}'
            )

            input_strings = (
                [input_string, *compare_strings]
                if compare_strings
                else input_string
            )

            embeddings = await stm(input_strings)

            input_string_embeddings = (
                embeddings[0] if compare_strings else embeddings
            )
            total_tokens = stm.token_count(input_string)

            # Subject string
            if partitioner and display_partitions:
                partitions = await partitioner(input_string)

                console.print(
                    theme.memory_partitions(
                        partitions, display_partitions=display_partitions
                    )
                )
            else:
                console.print(
                    theme.memory_embeddings(
                        input_string,
                        input_string_embeddings,
                        total_tokens=total_tokens,
                        minv=input_string_embeddings.min().item(),
                        maxv=input_string_embeddings.max().item(),
                        meanv=input_string_embeddings.mean().item(),
                        stdv=input_string_embeddings.std().item(),
                        normv=norm(input_string_embeddings).item(),
                    )
                )

            # Comparisons
            if compare_strings:
                joined = '", "'.join(compare_strings)
                logger.debug(
                    f'Calculating similarities between "{input_string}" and '
                    f'["{joined}"]'
                )
                embeddings = embeddings[1:]
                comparisons = dict(zip(compare_strings, embeddings))
                # Calculate similarities
                similarities: dict[str, Similarity] = {}
                for compare_string, compare_embeddings in comparisons.items():
                    dot_product = dot(
                        input_string_embeddings, compare_embeddings
                    )
                    cosine_distance_denom = norm(
                        input_string_embeddings
                    ) * norm(compare_embeddings)
                    cosine_distance = (
                        (dot_product / cosine_distance_denom).item()
                        if cosine_distance_denom != 0
                        else 1.0
                    )
                    inner_product = -1 * dot_product
                    l1_distance = sum(
                        abs(input_string_embeddings - compare_embeddings)
                    ).item()
                    l2_distance = norm(
                        input_string_embeddings - compare_embeddings
                    ).item()
                    pearson = corrcoef(
                        input_string_embeddings, compare_embeddings
                    )[0, 1].item()
                    similarities[compare_string] = Similarity(
                        cosine_distance=cosine_distance,
                        inner_product=inner_product,
                        l1_distance=l1_distance,
                        l2_distance=l2_distance,
                        pearson=pearson,
                    )

                similarities = dict(
                    sorted(
                        similarities.items(),
                        key=lambda item: sort_key(item[1]),
                        reverse=reverse_sort,
                    )
                )

                joined = '", "'.join(compare_strings)
                logger.debug(
                    f'Similarities between "{input_string}" and ["{joined}"]: '
                    + similarities.__repr__()
                )

                # Closest match
                most_similar = next(iter(similarities))

                console.print(
                    theme.memory_embeddings_comparison(
                        similarities, most_similar
                    )
                )

            if searches:
                knowledge_partitions = (
                    await partitioner(input_string) if partitioner else None
                )

                if knowledge_partitions and display_partitions:
                    console.print(
                        theme.memory_partitions(
                            knowledge_partitions,
                            display_partitions=display_partitions,
                        )
                    )

                index = IndexFlatL2(input_string_embeddings.shape[0])

                if partitioner:
                    knowledge_stack = vstack(
                        [kp.embeddings for kp in knowledge_partitions]
                    ).astype("float32", copy=False)
                    index.add(knowledge_stack)
                else:
                    index.add(
                        input_string_embeddings.reshape(1, -1).astype(
                            "float32", copy=False
                        )
                    )

                search_embeddings = await stm(searches)
                search_stack = vstack(search_embeddings).astype(
                    "float32", copy=False
                )
                distances, ids = index.search(search_stack, search_k)
                matches: list[tuple[int, int, float]] = [
                    (q_id, kn_id, float(dist))
                    for q_id, (dist_row, id_row) in enumerate(
                        zip(distances, ids)
                    )
                    for dist, kn_id in zip(dist_row, id_row)
                ]
                # smallest distance first
                matches.sort(key=lambda t: t[2])

                search_matches: list[SearchMatch] = []
                for q_id, kn_id, l2_distance in matches:
                    search_query = searches[q_id]
                    knowledge_chunk = (
                        knowledge_partitions[kn_id].data
                        if knowledge_partitions
                        else input_string if kn_id == 0 else None
                    )
                    if not knowledge_chunk:
                        continue
                    search_match = SearchMatch(
                        query=search_query,
                        match=knowledge_chunk,
                        l2_distance=l2_distance,
                    )
                    search_matches.append(search_match)

                console.print(theme.memory_embeddings_search(search_matches))


async def memory_search(
    args: Namespace,
    console: Console,
    theme: Theme,
    hub: HuggingfaceHub,
    logger: Logger,
) -> None:
    assert args.model and args.dsn and args.participant and args.namespace
    assert args.function

    _, _i = theme._, theme.icons
    model_id = args.model
    participant_id = UUID(args.participant)
    namespace = args.namespace
    dsn = args.dsn
    limit = args.limit

    input_string = get_input(
        console,
        _i["user_input"] + " ",
        echo_stdin=not args.no_repl,
        is_quiet=args.quiet,
    )
    if not input_string:
        return

    model_settings = get_model_settings(
        args, hub, logger, model_id, modality=Modality.EMBEDDING
    )

    with ModelManager(hub, logger) as manager:
        with manager.load(**model_settings) as stm:
            logger.debug("Loaded model %s", stm.config.__repr__())

            model_display(
                args, console, theme, hub, logger, model=stm, summary=True
            )

            partitioner = TextPartitioner(
                stm,
                logger,
                max_tokens=args.partition_max_tokens,
                window_size=args.partition_window,
                overlap_size=args.partition_overlap,
            )
            search_partitions = await partitioner(input_string)

            memory_store = await PgsqlRawMemory.create_instance(
                dsn=dsn, logger=logger
            )
            memories = await memory_store.search_memories(
                search_partitions=search_partitions,
                participant_id=participant_id,
                namespace=namespace,
                function=args.function,
                limit=limit,
            )

            console.print(
                theme.memory_search_matches(
                    participant_id, namespace, memories
                )
            )

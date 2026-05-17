from ....backends.ds4_native import Backend as Ds4NativeBackend
from ....backends.ds4_native import (
    Ds4ApiVersionError,
    Ds4BackendUnavailable,
    Ds4ContextError,
    Ds4GenerationError,
    Ds4InvalidModel,
    Ds4LoadError,
    EngineOptions,
    SamplingOptions,
    ThinkMode,
    import_compatible_binding,
)
from ....backends.ds4_native import Engine as Ds4Engine
from ....backends.ds4_native.availability import binding_capabilities
from ....backends.ds4_native.metadata import DS4_BINDING_IMPORT_NAME
from ....entities import (
    GenerationSettings,
    Input,
    Message,
    MessageContent,
    MessageContentFile,
    MessageContentImage,
    MessageContentText,
    MessageRole,
    MessageToolCall,
    ProbabilityDistribution,
    ReasoningEffort,
    Token,
    TokenDetail,
    ToolCallToken,
    TransformerEngineSettings,
)
from ....model.response.text import TextGenerationResponse
from ....tool.dsml import DsmlParseResult, DsmlPromptMessage, DsmlTools
from ....tool.manager import ToolManager
from .generation import TextGenerationModel

import asyncio
import hashlib
import importlib
import json
from asyncio import CancelledError
from collections.abc import (
    AsyncGenerator,
    Awaitable,
    Callable,
    Coroutine,
    Iterator,
)
from dataclasses import dataclass, replace
from inspect import Parameter, signature
from logging import Logger, getLogger
from math import exp
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import Any, TypeVar, cast

_CPU_WARNING = (
    "DS4 CPU backend is debug/reference only and is not recommended for "
    "production inference."
)
_DEFAULT_NATIVE_BACKEND = Ds4NativeBackend.METAL
_DEFAULT_CONTEXT_SIZE = 4096
_DEFAULT_MAX_NEW_TOKENS = 20
_DEVELOPER_PROMPT_PREFIX = "Developer instructions:"
_SYSTEM_DEVELOPER_SEPARATOR = f"\n\n{_DEVELOPER_PROMPT_PREFIX}\n"
_UNSUPPORTED_TOOLS_MESSAGE = (
    "DS4 native backend does not yet support Avalan tool schemas or tool "
    "messages. Tool-call rendering is planned for a later DS4 phase."
)
_UNSUPPORTED_GENERATION_MESSAGE = (
    "DS4 native backend does not support {feature}. Use greedy or sampling "
    "single-sequence generation."
)
_CONTEXT_ERROR_MARKERS = (
    "context size",
    "ctx_size",
    "prompt exceeds context",
)
_WORKER_JOIN_TIMEOUT_SECONDS = 2.0
_BYTES_PER_MIB = 1024 * 1024
_DS4_TOOL_REPLAY_MAX_ENTRIES = 10000
_DSML_TOOL_START_MARKERS = (
    "\n\n<｜DSML｜tool_calls>",
    "\n<｜DSML｜tool_calls>",
    DsmlTools.TOOL_CALLS_START,
    "\n\n<DSML｜tool_calls>",
    "\n<DSML｜tool_calls>",
    "<DSML｜tool_calls>",
    "\n\n<tool_calls>",
    "\n<tool_calls>",
    "<tool_calls>",
)
_TOKEN_SCORE_MODE_VALUES = {
    "TOKEN_LOGPROB": "token_logprob",
    "TOKEN_LOGPROB_AND_TOP_LOGPROBS": "token_logprob_and_top_logprobs",
}

_T = TypeVar("_T")


def _capability_bool(capabilities: object | None, name: str) -> bool | None:
    if capabilities is None:
        return None
    value = getattr(capabilities, name, None)
    if value is None:
        return None
    if not isinstance(value, bool):
        raise Ds4LoadError(f"DS4 capability {name!r} must be a boolean.")
    return value


@dataclass(frozen=True, slots=True)
class _Ds4GenerationPlan:
    max_new_tokens: int
    sampling_options: SamplingOptions
    stop_strings: tuple[str, ...]
    use_sampling: bool
    emit_token_details: bool = False
    parse_dsml_tools: bool = False
    probability_distribution: ProbabilityDistribution = "log_softmax"
    top_logprobs: int = 0


_Ds4PromptMessage = DsmlPromptMessage


@dataclass(frozen=True, slots=True)
class _Ds4TokenProbability:
    token_id: int
    probability: float


@dataclass(frozen=True, slots=True)
class _Ds4Step:
    token_id: int
    is_eos: bool
    token_bytes: bytes | None
    token_detail: TokenDetail | None


@dataclass(frozen=True, slots=True)
class _Ds4DiskCacheConfig:
    directory: Path | None
    budget_bytes: int

    @property
    def enabled(self) -> bool:
        return self.directory is not None and self.budget_bytes > 0


@dataclass(frozen=True, slots=True)
class _Ds4CacheEntryPath:
    key: str
    metadata_path: Path
    payload_path: Path
    token_digest: str


class _Ds4DiskKvCache:
    """Adapt Avalan's async DS4 worker to pyds4's generic cache helper."""

    def __init__(
        self,
        directory: Path,
        budget_bytes: int,
        logger: Logger,
        namespace: str,
        backend: str = _DEFAULT_NATIVE_BACKEND.value,
    ) -> None:
        self._budget_bytes = budget_bytes
        self._directory = directory
        self._logger = logger
        self._namespace = namespace
        self._cache: Any = self._cache_type()(
            directory,
            namespace,
            backend=backend,
        )

    async def restore(
        self,
        session: object,
        prompt_tokens: list[int],
        ctx_size: int,
    ) -> bool:
        """Restore a cached DS4 session payload when it matches."""
        if not callable(getattr(session, "load_payload", None)):
            return False

        result = await self._cache.arestore(
            session,
            prompt_tokens,
            ctx_size,
            sync_on_miss=False,
        )
        if result.status == "hit":
            if result.warning:
                self._logger.warning(
                    "DS4 disk KV cache metadata update failed: %s",
                    result.warning,
                )
            return True
        if result.error:
            self._logger.warning(
                "DS4 disk KV cache restore failed; using live session: %s",
                result.error,
            )
        return False

    async def store(
        self,
        session: object,
        prompt_tokens: list[int],
        ctx_size: int,
    ) -> None:
        """Persist a DS4 session payload for a token-prefix key."""
        if not callable(getattr(session, "save_payload", None)):
            return

        result = await self._cache.astore(
            session,
            prompt_tokens,
            ctx_size,
            size_budget_bytes=self._budget_bytes,
        )
        if result.status == "stored":
            return
        if result.error:
            self._logger.warning(
                "DS4 disk KV cache payload save failed; "
                "continuing without cache: %s",
                result.error,
            )

    def _entry_path(
        self, prompt_tokens: list[int], ctx_size: int
    ) -> _Ds4CacheEntryPath:
        entry = self._cache.entry_for(prompt_tokens, ctx_size)
        return _Ds4CacheEntryPath(
            key=entry.key,
            metadata_path=entry.metadata_path,
            payload_path=entry.payload_path,
            token_digest=entry.token_sha256,
        )

    @staticmethod
    def _cache_type() -> Any:
        module = importlib.import_module("pyds4.kv_cache")
        cache_type = getattr(module, "Ds4DiskKvCache", None)
        if not callable(cache_type):
            raise Ds4LoadError(
                "DS4 binding does not expose pyds4.kv_cache.Ds4DiskKvCache."
            )
        return cast(type[object], cache_type)


class _StopStringBuffer:
    """Buffer recent text so stop strings spanning tokens are suppressed."""

    def __init__(self, stop_strings: tuple[str, ...]) -> None:
        self._pending = ""
        self._stopped = False
        self._stop_strings = stop_strings
        self._keep = (
            max(len(stop_string) for stop_string in stop_strings) - 1
            if stop_strings
            else 0
        )

    @property
    def stopped(self) -> bool:
        return self._stopped

    def push(self, text: str) -> Iterator[str]:
        if not self._stop_strings:
            if text:
                yield text
            return

        self._pending += text
        stop_index = self._stop_index()
        if stop_index is not None:
            text_before_stop = self._pending[:stop_index]
            self._pending = ""
            self._stopped = True
            if text_before_stop:
                yield text_before_stop
            return

        emit_length = len(self._pending) - self._keep
        if emit_length > 0:
            chunk = self._pending[:emit_length]
            self._pending = self._pending[emit_length:]
            if chunk:
                yield chunk

    def flush(self) -> Iterator[str]:
        if self._pending and not self._stopped:
            yield self._pending
        self._pending = ""

    def _stop_index(self) -> int | None:
        first_index: int | None = None
        for stop_string in self._stop_strings:
            index = self._pending.find(stop_string)
            if index >= 0 and (first_index is None or index < first_index):
                first_index = index
        return first_index


class Ds4Worker:
    """Own Avalan's async DS4 runtime state."""

    def __init__(
        self,
        options: EngineOptions,
        ctx_size: int,
        logger: Logger,
        disk_cache_config: _Ds4DiskCacheConfig | None = None,
    ) -> None:
        self._binding: object | None = None
        self._capabilities: object | None = None
        self._closed = True
        self._ctx_size = ctx_size
        self._engine: object | None = None
        self._logger = logger
        self._options = options
        self._disk_cache = self._build_disk_cache(
            disk_cache_config, options, logger
        )
        self._reset_tasks: set[asyncio.Task[BaseException | None]] = set()
        self._session: object | None = None
        self._tool_dsml_replay: dict[str, str] = {}

    @property
    def closed(self) -> bool:
        """Return whether the DS4 worker has been closed."""
        return self._closed

    @property
    def is_alive(self) -> bool:
        """Return whether the async DS4 engine is open."""
        return not self._closed and self._engine is not None

    def start(self) -> None:
        """Open the pyds4 async engine and create the reusable session."""
        if not self._closed:
            return

        self._closed = False
        try:
            self._run_sync(self._open)
        except BaseException:
            self._closed = True
            self._run_sync(self._close_resources)
            raise

    def close(self) -> None:
        """Close the async session and native engine."""
        if self._closed:
            return

        self._run_sync(self.aclose)

    def exact_dsml_for_tool_calls(
        self, calls: tuple[MessageToolCall, ...]
    ) -> str | None:
        """Return remembered sampled DSML when all call IDs match."""
        if not calls:
            return None

        replay: str | None = None
        for call in calls:
            if not call.id:
                return None
            candidate = self._tool_dsml_replay.get(call.id)
            if candidate is None:
                return None
            if replay is None:
                replay = candidate
            elif replay != candidate:
                return None
        return replay

    async def aclose(self) -> None:
        """Close the async session and native engine."""
        if self._closed and self._engine is None and self._session is None:
            return

        self._closed = True
        await self._close_resources()

    def render_prompt_tokens(
        self,
        system_content: str | None,
        messages: list[_Ds4PromptMessage],
        think_mode: ThinkMode,
    ) -> list[int]:
        """Render DS4 chat prompt tokens for legacy synchronous callers."""
        return self._run_sync(
            lambda: self.render_prompt_tokens_async(
                system_content, messages, think_mode
            )
        )

    async def render_prompt_tokens_async(
        self,
        system_content: str | None,
        messages: list[_Ds4PromptMessage],
        think_mode: ThinkMode,
    ) -> list[int]:
        """Render DS4 chat prompt tokens through pyds4's async engine."""
        engine = self._require_engine()
        effective_think_mode = self._effective_think_mode(think_mode)

        if len(messages) == 1 and messages[0].role is MessageRole.USER:
            result = await self._call_async(
                engine,
                "encode_chat_prompt",
                system_content,
                messages[0].content,
                effective_think_mode,
            )
        else:
            result = await self._call_async(engine, "chat_begin")
            tokens = self._token_list(result)
            if system_content is not None:
                await self._call_async(
                    engine,
                    "chat_append_message",
                    tokens,
                    "system",
                    system_content,
                )
            for message in messages:
                await self._call_async(
                    engine,
                    "chat_append_message",
                    tokens,
                    message.role.value,
                    message.content,
                )
            await self._call_async(
                engine,
                "chat_append_assistant_prefix",
                tokens,
                effective_think_mode,
            )
            result = tokens

        return self._token_list(result)

    async def tokenize_rendered_chat_async(self, text: str) -> list[int]:
        """Tokenize a DS4 server-rendered chat prompt."""
        result = await self._call_async(
            self._require_engine(), "tokenize_rendered_chat", text
        )
        return self._token_list(result)

    async def stream(
        self,
        prompt_tokens: list[int],
        generation_plan: _Ds4GenerationPlan,
    ) -> AsyncGenerator[str | TokenDetail | ToolCallToken, None]:
        """Yield DS4 output chunks from the pyds4 async facade."""
        if self._closed:
            raise Ds4LoadError("DS4 worker is closed.")

        recovery_snapshot: bytes | None = None
        try:
            session = self._require_session()
            restored = await self._restore_disk_cache(session, prompt_tokens)
            if not restored:
                await self._call_async(session, "sync", list(prompt_tokens))
                await self._store_disk_cache(session, prompt_tokens)
            recovery_snapshot = await self._save_snapshot(session)
            async for chunk in self._generate_chunks(session, generation_plan):
                yield chunk
        except (CancelledError, GeneratorExit):
            self._schedule_reset(invalidate=True, snapshot=recovery_snapshot)
            raise
        except BaseException as error:
            reset_error = await self._reset_session(
                invalidate=True, snapshot=recovery_snapshot
            )
            if reset_error is not None:
                error.add_note(
                    "DS4 session reset after generation failure failed: "
                    f"{reset_error}"
                )
            raise

    def generate_string(
        self,
        prompt_tokens: list[int],
        generation_plan: _Ds4GenerationPlan,
    ) -> str:
        """Return complete DS4 output text from the shared generation core."""
        return self._run_sync(
            lambda: self.generate_string_async(prompt_tokens, generation_plan)
        )

    async def generate_string_async(
        self,
        prompt_tokens: list[int],
        generation_plan: _Ds4GenerationPlan,
    ) -> str:
        """Return complete DS4 output text from the async generation core."""
        return "".join(
            [
                (
                    chunk.token
                    if isinstance(chunk, (TokenDetail, ToolCallToken))
                    else chunk
                )
                async for chunk in self.stream(prompt_tokens, generation_plan)
            ]
        )

    async def _open(self) -> None:
        self._binding = import_compatible_binding(
            backend=self._options.backend.value
        )
        self._capabilities = binding_capabilities(self._binding)
        if _capability_bool(self._capabilities, "payloads") is False:
            self._disk_cache = None
        async_engine_type = getattr(self._binding, "AsyncEngine", None)
        if not callable(async_engine_type):
            raise Ds4LoadError(
                "DS4 binding does not expose AsyncEngine. Install a pyds4 "
                "build with async facade support."
            )

        native_options = Ds4Engine._native_engine_options(
            self._binding, self._options
        )
        open_method = getattr(async_engine_type, "open", None)
        with Ds4Engine._native_stderr_context(
            Ds4Engine._native_log_context_enabled(self._binding, self._options)
        ):
            if callable(open_method):
                try:
                    self._engine = await open_method(native_options)
                except (
                    Ds4BackendUnavailable,
                    Ds4InvalidModel,
                    Ds4LoadError,
                ):
                    raise
                except Exception as error:
                    Ds4Engine._raise_mapped_open_error(error)
            else:
                try:
                    self._engine = async_engine_type(native_options)
                except (
                    Ds4BackendUnavailable,
                    Ds4InvalidModel,
                    Ds4LoadError,
                ):
                    raise
                except Exception as error:
                    Ds4Engine._raise_mapped_open_error(error)
                enter = getattr(self._engine, "__aenter__", None)
                if callable(enter):
                    await enter()

        self._session = await self._call_async(
            self._require_engine(), "create_session", self._ctx_size
        )

    async def _generate_chunks(
        self, session: object, generation_plan: _Ds4GenerationPlan
    ) -> AsyncGenerator[str | TokenDetail | ToolCallToken, None]:
        if generation_plan.parse_dsml_tools:
            async for chunk in self._generate_dsml_tool_chunks(
                session, generation_plan
            ):
                yield chunk
            return

        async for text_chunk in self._generate_text_chunks(
            session, generation_plan
        ):
            yield text_chunk

    async def _generate_dsml_tool_chunks(
        self, session: object, generation_plan: _Ds4GenerationPlan
    ) -> AsyncGenerator[str | ToolCallToken, None]:
        buffered: list[str] = []
        dsml_start: int | None = None
        argument_emitted_until = 0
        content_emitted_until = 0

        async for chunk in self._generate_text_chunks(
            session, generation_plan
        ):
            text_chunk = (
                chunk.token if isinstance(chunk, TokenDetail) else chunk
            )
            buffered.append(text_chunk)
            text = "".join(buffered)

            if dsml_start is None:
                start_span = DsmlTools.tool_call_start_span(text)
                if start_span is None:
                    safe_end = len(text) - self._dsml_start_suffix_length(text)
                    if content_emitted_until < safe_end:
                        yield text[content_emitted_until:safe_end]
                        content_emitted_until = safe_end
                    continue
                dsml_start = start_span[0]
                content_end = len(text[:dsml_start].rstrip())
                if content_emitted_until < content_end:
                    yield text[content_emitted_until:content_end]
                    content_emitted_until = content_end

            raw_dsml = text[dsml_start:]
            deltas, argument_emitted_until = DsmlTools.stream_argument_deltas(
                raw_dsml, argument_emitted_until
            )
            for delta in deltas:
                yield ToolCallToken(token=delta)

        text = "".join(buffered)
        parsed = DsmlTools.parse_generated_message(text)
        if parsed is None:
            raise Ds4GenerationError("DS4 generated malformed DSML.")
        self._remember_dsml_tool_replay(parsed)
        if dsml_start is None and content_emitted_until < len(text):
            yield text[content_emitted_until:]
        for call in parsed.calls:
            yield ToolCallToken(token="", call=call)

    @staticmethod
    def _dsml_start_suffix_length(text: str) -> int:
        """Return unsafe suffix length that may become a DSML start marker."""
        max_length = min(
            len(text), max(len(marker) for marker in _DSML_TOOL_START_MARKERS)
        )
        for length in range(max_length, 0, -1):
            suffix = text[-length:]
            if any(
                marker.startswith(suffix)
                for marker in _DSML_TOOL_START_MARKERS
            ):
                return length
        return 0

    def _remember_dsml_tool_replay(self, parsed: DsmlParseResult) -> None:
        if not parsed.raw_dsml or not parsed.calls:
            return
        for call in parsed.calls:
            self._tool_dsml_replay[str(call.id)] = parsed.raw_dsml
            while len(self._tool_dsml_replay) > _DS4_TOOL_REPLAY_MAX_ENTRIES:
                oldest = next(iter(self._tool_dsml_replay))
                del self._tool_dsml_replay[oldest]

    async def _generate_text_chunks(
        self, session: object, generation_plan: _Ds4GenerationPlan
    ) -> AsyncGenerator[str | TokenDetail, None]:
        engine = self._require_engine()
        stop_buffer = _StopStringBuffer(generation_plan.stop_strings)

        for step_index in range(generation_plan.max_new_tokens):
            step = (
                await self._detailed_generation_step(
                    engine, session, generation_plan, step_index
                )
                if generation_plan.emit_token_details
                else await self._plain_generation_step(
                    session, generation_plan
                )
            )
            if bool(getattr(step, "is_eos", False)):
                break

            text = await self._token_text(engine, step)
            for chunk in stop_buffer.push(text):
                detail = getattr(step, "token_detail", None)
                if isinstance(detail, TokenDetail) and chunk == detail.token:
                    yield detail
                else:
                    yield chunk
            if stop_buffer.stopped:
                break

        for chunk in stop_buffer.flush():
            yield chunk

    async def _plain_generation_step(
        self, session: object, generation_plan: _Ds4GenerationPlan
    ) -> object:
        return await self._call_async(
            session,
            "next_token",
            (
                self._native_sampling_options(generation_plan.sampling_options)
                if generation_plan.use_sampling
                else None
            ),
            decode=True,
        )

    async def _detailed_generation_step(
        self,
        engine: object,
        session: object,
        generation_plan: _Ds4GenerationPlan,
        step_index: int,
    ) -> object:
        self._require_logprob_support(
            session, generation_plan.top_logprobs, self._capabilities
        )
        score_options = self._generation_score_options(generation_plan)
        if score_options is not None and self._method_supports_keyword(
            session, "next_token", "scores"
        ):
            step = await self._call_async(
                session,
                "next_token",
                (
                    self._native_sampling_options(
                        generation_plan.sampling_options
                    )
                    if generation_plan.use_sampling
                    else None
                ),
                decode=True,
                scores=score_options,
            )
            return await self._token_detail_step_from_generation_step(
                engine, step, generation_plan, step_index
            )

        token_id = await self._select_token(session, generation_plan)
        eos_token_id = await self._eos_token_id(engine)
        is_eos = token_id == eos_token_id
        if is_eos:
            return _Ds4Step(token_id, True, None, None)

        alternatives = await self._token_probabilities(
            engine, session, generation_plan.top_logprobs
        )
        chosen_probability = await self._chosen_token_probability(
            session, token_id, alternatives
        )
        await self._call_async(session, "eval", token_id)
        token_text = await self._call_async(engine, "token_text", token_id)
        token_bytes = self._bytes_value(token_text, "token_text")
        text = token_bytes.decode("utf-8", errors="replace")
        detail = TokenDetail(
            id=token_id,
            token=text,
            probability=chosen_probability,
            probability_distribution=generation_plan.probability_distribution,
            step=step_index,
            tokens=(
                [
                    Token(
                        id=alternative.token_id,
                        token=await self._token_string(
                            engine, alternative.token_id
                        ),
                        probability=alternative.probability,
                    )
                    for alternative in alternatives
                ]
                if alternatives
                else None
            ),
        )
        return _Ds4Step(token_id, False, token_bytes, detail)

    def _generation_score_options(
        self, generation_plan: _Ds4GenerationPlan
    ) -> object | None:
        binding = self._require_binding()
        score_options_type = getattr(binding, "GenerationScoreOptions", None)
        score_mode_type = getattr(binding, "TokenScoreMode", None)
        if not callable(score_options_type) or score_mode_type is None:
            return None

        mode_name = (
            "TOKEN_LOGPROB_AND_TOP_LOGPROBS"
            if generation_plan.top_logprobs > 0
            else "TOKEN_LOGPROB"
        )
        mode = getattr(score_mode_type, mode_name, None)
        if mode is None and callable(score_mode_type):
            try:
                mode = score_mode_type(_TOKEN_SCORE_MODE_VALUES[mode_name])
            except (TypeError, ValueError):
                return None
        if mode is None:
            return None

        try:
            return cast(
                object,
                score_options_type(
                    mode=mode,
                    top_k=generation_plan.top_logprobs,
                ),
            )
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _method_supports_keyword(
        target: object, name: str, keyword: str
    ) -> bool:
        method = getattr(target, name, None)
        if not callable(method):
            return False
        try:
            parameters = signature(method).parameters
        except (TypeError, ValueError):
            return True
        return keyword in parameters or any(
            parameter.kind is Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )

    async def _token_detail_step_from_generation_step(
        self,
        engine: object,
        step: object,
        generation_plan: _Ds4GenerationPlan,
        step_index: int,
    ) -> _Ds4Step:
        token_id = self._token_id(
            getattr(step, "token_id", None), "next_token"
        )
        if bool(getattr(step, "is_eos", False)):
            return _Ds4Step(token_id, True, None, None)

        token_bytes = getattr(step, "token_bytes", None)
        if token_bytes is None:
            token_bytes = await self._call_async(
                engine, "token_text", token_id
            )
        token_bytes = self._bytes_value(token_bytes, "token_text")
        text = token_bytes.decode("utf-8", errors="replace")

        alternatives = [
            self._token_probability(score, "top_logprobs")
            for score in getattr(step, "top_logprobs", ()) or ()
        ]
        token_logprob = getattr(step, "token_logprob", None)
        chosen_probability = (
            self._probability_from_logprob(token_logprob, "token_logprob")
            if token_logprob is not None
            else await self._chosen_token_probability(
                object(), token_id, alternatives
            )
        )
        detail = TokenDetail(
            id=token_id,
            token=text,
            probability=chosen_probability,
            probability_distribution=generation_plan.probability_distribution,
            step=step_index,
            tokens=(
                [
                    Token(
                        id=alternative.token_id,
                        token=await self._token_string(
                            engine, alternative.token_id
                        ),
                        probability=alternative.probability,
                    )
                    for alternative in alternatives
                ]
                if alternatives
                else None
            ),
        )
        return _Ds4Step(token_id, False, token_bytes, detail)

    async def _select_token(
        self, session: object, generation_plan: _Ds4GenerationPlan
    ) -> int:
        if generation_plan.use_sampling:
            value = await self._call_async(
                session,
                "sample",
                self._native_sampling_options(
                    generation_plan.sampling_options
                ),
            )
        else:
            value = await self._call_async(session, "argmax")
        operation = "sample" if generation_plan.use_sampling else "argmax"
        return self._token_id(value, operation)

    @staticmethod
    def _require_logprob_support(
        session: object,
        top_k: int,
        capabilities: object | None = None,
    ) -> None:
        logprobs = _capability_bool(capabilities, "logprobs")
        top_logprobs = _capability_bool(capabilities, "top_logprobs")
        if logprobs is False or (
            logprobs is None
            and not callable(getattr(session, "token_logprob", None))
        ):
            raise NotImplementedError(
                "DS4 native backend does not support token logprobs. Install a"
                " pyds4 build with token_logprob support."
            )
        if top_k > 0 and (
            top_logprobs is False
            or (
                top_logprobs is None
                and not callable(getattr(session, "top_logprobs", None))
            )
        ):
            raise NotImplementedError(
                "DS4 native backend does not support token logprobs. "
                "Install a pyds4 build with top_logprobs support."
            )

    async def _eos_token_id(self, engine: object) -> int:
        value = getattr(engine, "eos_token_id", None)
        if callable(value):
            value = value()
        if isinstance(value, Awaitable):
            value = await value
        return self._token_id(value, "eos_token_id")

    async def _token_probabilities(
        self, engine: object, session: object, top_k: int
    ) -> list[_Ds4TokenProbability]:
        _ = engine
        if top_k == 0:
            return []
        top_logprobs = _capability_bool(self._capabilities, "top_logprobs")
        if top_logprobs is False or (
            top_logprobs is None
            and not callable(getattr(session, "top_logprobs", None))
        ):
            raise NotImplementedError(
                "DS4 native backend does not support token logprobs. "
                "Install a pyds4 build with top_logprobs support."
            )

        value = await self._call_async(session, "top_logprobs", top_k)
        if not isinstance(value, (list, tuple)):
            raise Ds4GenerationError(
                "DS4 top_logprobs must return a list of token scores."
            )
        probabilities = [
            self._token_probability(score, "top_logprobs") for score in value
        ]
        return probabilities

    async def _chosen_token_probability(
        self,
        session: object,
        token_id: int,
        alternatives: list[_Ds4TokenProbability],
    ) -> float | None:
        token_logprob = getattr(session, "token_logprob", None)
        if callable(token_logprob):
            value = await self._call_async(session, "token_logprob", token_id)
            return self._probability_from_logprob(value, "token_logprob")

        for alternative in alternatives:
            if alternative.token_id == token_id:
                return alternative.probability

        if alternatives:
            return None
        raise NotImplementedError(
            "DS4 native backend does not support token logprobs. Install a "
            "pyds4 build with token_logprob support."
        )

    async def _token_string(self, engine: object, token_id: int) -> str:
        value = await self._call_async(engine, "token_text", token_id)
        return self._bytes_value(value, "token_text").decode(
            "utf-8", errors="replace"
        )

    @classmethod
    def _token_probability(
        cls, value: object, operation: str
    ) -> _Ds4TokenProbability:
        token_id: object
        logprob: object
        probability: object
        if isinstance(value, dict):
            token_id = value.get("token_id", value.get("id"))
            probability = value.get("probability")
            logprob = value.get("logprob")
        elif isinstance(value, (list, tuple)):
            if len(value) >= 3:
                token_id = value[0]
                probability = None
                logprob = value[2]
            elif len(value) >= 2:
                token_id = value[0]
                probability = None
                logprob = value[1]
            else:
                token_id = None
                probability = None
                logprob = None
        else:
            token_id = getattr(value, "token_id", getattr(value, "id", None))
            probability = getattr(value, "probability", None)
            logprob = getattr(value, "logprob", None)

        return _Ds4TokenProbability(
            cls._token_id(token_id, operation),
            (
                cls._probability(probability, operation)
                if probability is not None
                else cls._probability_from_logprob(logprob, operation)
            ),
        )

    @staticmethod
    def _token_id(value: object, operation: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise Ds4GenerationError(
                f"DS4 {operation} must return a token id."
            )
        return value

    @staticmethod
    def _probability(value: object, operation: str) -> float:
        if (
            isinstance(value, bool)
            or not isinstance(value, (float, int))
            or not 0.0 <= float(value) <= 1.0
        ):
            raise Ds4GenerationError(
                f"DS4 {operation} must return probabilities."
            )
        return float(value)

    @classmethod
    def _probability_from_logprob(cls, value: object, operation: str) -> float:
        if isinstance(value, bool) or not isinstance(value, (float, int)):
            raise Ds4GenerationError(
                f"DS4 {operation} must return log probabilities."
            )
        return cls._probability(exp(float(value)), operation)

    async def _token_text(self, engine: object, step: object) -> str:
        token_bytes = getattr(step, "token_bytes", None)
        if token_bytes is None:
            token_id = getattr(step, "token_id", None)
            if isinstance(token_id, bool) or not isinstance(token_id, int):
                raise Ds4GenerationError(
                    "DS4 generation step must include a token id."
                )
            token_bytes = await self._call_async(
                engine, "token_text", token_id
            )

        if isinstance(token_bytes, (bytearray, memoryview)):
            token_bytes = bytes(token_bytes)
        elif not isinstance(token_bytes, bytes):
            raise Ds4GenerationError("DS4 token text must be bytes.")
        return token_bytes.decode("utf-8", errors="replace")

    async def _reset_session(
        self, *, invalidate: bool, snapshot: bytes | None = None
    ) -> BaseException | None:
        old_session = self._session
        if old_session is not None and snapshot is not None:
            try:
                if await self._load_snapshot(old_session, snapshot):
                    return None
            except BaseException as error:
                self._logger.warning(
                    "DS4 snapshot recovery failed; rebuilding session: %s",
                    error,
                )

        try:
            self._session = None
            if old_session is not None:
                try:
                    if invalidate:
                        await self._call_async(old_session, "invalidate")
                finally:
                    await self._close_async_resource(old_session)

            if not self._closed:
                self._session = await self._call_async(
                    self._require_engine(), "create_session", self._ctx_size
                )
        except BaseException as error:
            return error
        return None

    def _schedule_reset(
        self, *, invalidate: bool, snapshot: bytes | None = None
    ) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return

        task = loop.create_task(
            self._reset_session(invalidate=invalidate, snapshot=snapshot)
        )
        self._reset_tasks.add(task)
        task.add_done_callback(self._reset_tasks.discard)

    async def _save_snapshot(self, session: object) -> bytes | None:
        if _capability_bool(self._capabilities, "snapshots") is False:
            return None
        if not callable(getattr(session, "save_snapshot", None)):
            return None
        value = await self._call_async(session, "save_snapshot")
        return self._snapshot_bytes(value, "save_snapshot")

    async def _load_snapshot(self, session: object, snapshot: bytes) -> bool:
        if _capability_bool(self._capabilities, "snapshots") is False:
            return False
        if not callable(getattr(session, "load_snapshot", None)):
            return False
        await self._call_async(session, "load_snapshot", snapshot)
        return True

    async def _restore_disk_cache(
        self, session: object, prompt_tokens: list[int]
    ) -> bool:
        if self._disk_cache is None:
            return False
        return await self._disk_cache.restore(
            session, prompt_tokens, self._ctx_size
        )

    async def _store_disk_cache(
        self, session: object, prompt_tokens: list[int]
    ) -> None:
        if self._disk_cache is None:
            return
        await self._disk_cache.store(session, prompt_tokens, self._ctx_size)

    async def _close_resources(self) -> None:
        current_loop = asyncio.get_running_loop()
        reset_tasks = tuple(
            task
            for task in self._reset_tasks
            if not task.done() and task.get_loop() is current_loop
        )
        if reset_tasks:
            await asyncio.gather(*reset_tasks, return_exceptions=True)

        errors: list[BaseException] = []
        session = self._session
        self._session = None
        if session is not None:
            try:
                await self._close_async_resource(session)
            except BaseException as error:
                errors.append(error)

        engine = self._engine
        self._engine = None
        if engine is not None:
            try:
                await self._close_async_resource(engine)
            except BaseException as error:
                errors.append(error)

        if errors:
            self._logger.warning("DS4 async cleanup failed: %s", errors[0])

    async def _close_async_resource(self, resource: object) -> None:
        close = getattr(resource, "aclose", None)
        if callable(close):
            await close()
            return
        close = getattr(resource, "close", None)
        if callable(close):
            result = close()
            if isinstance(result, Awaitable):
                await result

    def _effective_think_mode(self, think_mode: ThinkMode) -> object:
        binding = self._require_binding()
        native_mode = Ds4Engine._native_enum_value(
            binding, "ThinkMode", ThinkMode(think_mode)
        )
        helper = getattr(binding, "think_mode_for_context", None)
        if not callable(helper):
            return native_mode
        return helper(native_mode, self._ctx_size)

    def _native_sampling_options(self, options: SamplingOptions) -> object:
        binding = self._require_binding()
        native_options_type = getattr(binding, "SamplingOptions", None)
        if not callable(native_options_type):
            return options
        return native_options_type(
            temperature=options.temperature,
            top_k=options.top_k,
            top_p=options.top_p,
            min_p=options.min_p,
            seed=options.seed,
        )

    @staticmethod
    async def _call_async(
        target: object, name: str, *args: object, **kwargs: object
    ) -> object:
        method = getattr(target, name, None)
        if not callable(method):
            raise Ds4LoadError(f"DS4 async object does not expose {name}.")
        try:
            result = method(*args, **kwargs)
            if isinstance(result, Awaitable):
                return await result
            return result
        except (Ds4ContextError, Ds4GenerationError, Ds4LoadError):
            raise
        except Exception as error:
            message = str(error) or type(error).__name__
            if Ds4Worker._is_native_context_error(error, message):
                raise Ds4ContextError(
                    f"DS4 {name} failed: {message}"
                ) from error
            raise Ds4GenerationError(
                f"DS4 {name} failed: {message}"
            ) from error

    @staticmethod
    def _is_native_context_error(error: Exception, message: str) -> bool:
        if type(error).__name__ == "Ds4ContextError":
            return True
        lowered = message.lower()
        return any(marker in lowered for marker in _CONTEXT_ERROR_MARKERS)

    @staticmethod
    def _run_sync(factory: Callable[[], Coroutine[object, object, _T]]) -> _T:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(factory())

        response_queue: Queue[tuple[bool, object]] = Queue(maxsize=1)

        def run_in_thread() -> None:
            try:
                response_queue.put((True, asyncio.run(factory())))
            except BaseException as error:
                response_queue.put((False, error))

        thread = Thread(
            target=run_in_thread,
            name="ds4-async-compat",
            daemon=True,
        )
        thread.start()
        ok, result = response_queue.get()
        thread.join(_WORKER_JOIN_TIMEOUT_SECONDS)
        if ok:
            return cast(_T, result)
        raise cast(BaseException, result)

    @staticmethod
    def _token_list(value: object) -> list[int]:
        if isinstance(value, tuple):
            value = list(value)
        if isinstance(value, list):
            tokens: list[int] = []
            for token_id in value:
                if isinstance(token_id, bool) or not isinstance(token_id, int):
                    break
                tokens.append(token_id)
            else:
                return tokens
        raise Ds4GenerationError("DS4 prompt rendering must return token IDs.")

    @staticmethod
    def _snapshot_bytes(value: object, operation: str) -> bytes:
        return Ds4Worker._bytes_value(value, operation)

    @staticmethod
    def _bytes_value(value: object, operation: str) -> bytes:
        if isinstance(value, bytes):
            return value
        if isinstance(value, (bytearray, memoryview)):
            return bytes(value)
        raise Ds4GenerationError(f"DS4 {operation} must return bytes.")

    @staticmethod
    def _build_disk_cache(
        config: _Ds4DiskCacheConfig | None,
        options: EngineOptions,
        logger: Logger,
    ) -> _Ds4DiskKvCache | None:
        if config is None or not config.enabled or config.directory is None:
            return None

        namespace = hashlib.sha256(
            json.dumps(
                {
                    "backend": options.backend.value,
                    "model_path": str(Path(options.model_path).expanduser()),
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        return _Ds4DiskKvCache(
            config.directory,
            config.budget_bytes,
            logger,
            namespace,
            backend=options.backend.value,
        )

    def _require_binding(self) -> object:
        if self._binding is None:
            raise Ds4LoadError("DS4 binding is not loaded.")
        return self._binding

    def _require_engine(self) -> object:
        if self._engine is None:
            raise Ds4LoadError("DS4 async engine is not loaded.")
        return self._engine

    def _require_session(self) -> object:
        if self._session is None:
            raise Ds4GenerationError("DS4 async session is not loaded.")
        return self._session


class Ds4Model(TextGenerationModel):
    """Load DS4-supported GGUF files through the native DS4 backend."""

    @classmethod
    def is_available(cls) -> bool:
        """Return whether the DS4 binding can be imported safely."""
        try:
            import_compatible_binding()
        except (Ds4ApiVersionError, Ds4BackendUnavailable):
            return False
        return True

    @property
    def uses_tokenizer(self) -> bool:
        return False

    @property
    def supports_sample_generation(self) -> bool:
        return True

    @property
    def supports_token_streaming(self) -> bool:
        return True

    def __init__(
        self,
        model_id: str,
        settings: TransformerEngineSettings | None = None,
        logger: Logger = getLogger(__name__),
    ) -> None:
        settings = settings or TransformerEngineSettings()
        settings = replace(
            settings,
            auto_load_tokenizer=False,
            enable_eval=False,
        )
        super().__init__(model_id, settings, logger)

    def _load_model(self) -> Ds4Worker:
        assert self._model_id, "A DS4 model path is required."
        settings = self._ds4_settings()
        config = settings.backend_config or {}
        model_path = self._validated_file_path(
            self._model_id, "DS4 model path"
        )
        mtp_path = self._optional_file_path(
            config.get("mtp_path"), "DS4 MTP path"
        )
        directional_steering_attn = self._float_config(
            config, "directional_steering_attn", 0.0
        )
        directional_steering_ffn = self._float_config(
            config, "directional_steering_ffn", 0.0
        )
        directional_steering_file = self._steering_file_path(
            config,
            directional_steering_attn,
            directional_steering_ffn,
        )
        native_backend = self._native_backend(config)
        if native_backend is Ds4NativeBackend.CPU:
            self._logger.warning(_CPU_WARNING)

        options = EngineOptions(
            model_path=model_path,
            backend=native_backend,
            mtp_path=mtp_path,
            n_threads=self._non_negative_int_config(config, "n_threads", 0),
            mtp_draft_tokens=self._non_negative_int_config(
                config, "mtp_draft_tokens", 0
            ),
            mtp_margin=self._non_negative_float_config(
                config, "mtp_margin", 0.0
            ),
            directional_steering_file=directional_steering_file,
            directional_steering_attn=directional_steering_attn,
            directional_steering_ffn=directional_steering_ffn,
            warm_weights=self._bool_config(config, "warm_weights", False),
            quality=self._bool_config(config, "quality", False),
            native_log=self._bool_config(config, "native_log", False),
        )
        worker = Ds4Worker(
            options,
            self._context_size(),
            self._logger,
            self._disk_cache_config(config),
        )
        worker.start()
        return worker

    def _accepts_loaded_model(self, model: object) -> bool:
        if isinstance(model, (Ds4Engine, Ds4Worker)):
            return True

        try:
            binding = import_compatible_binding()
        except (Ds4ApiVersionError, Ds4BackendUnavailable):
            binding = None
        if binding is not None:
            native_engine_type = getattr(binding, "Engine", None)
            if isinstance(native_engine_type, type) and isinstance(
                model, native_engine_type
            ):
                return True
            native_async_engine_type = getattr(binding, "AsyncEngine", None)
            if isinstance(native_async_engine_type, type) and isinstance(
                model, native_async_engine_type
            ):
                return True

        module_name = type(model).__module__
        return (
            module_name == DS4_BINDING_IMPORT_NAME
            or module_name.startswith(f"{DS4_BINDING_IMPORT_NAME}.")
        )

    def close(self) -> None:
        """Close DS4 worker resources."""
        model = self._model
        if isinstance(model, Ds4Worker):
            model.close()
        elif isinstance(model, Ds4Engine):
            model.close()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object | None,
    ) -> bool:
        self.close()
        return bool(super().__exit__(exc_type, exc_value, traceback))

    async def __call__(
        self,
        input: Input,
        system_prompt: str | None = None,
        developer_prompt: str | None = None,
        settings: GenerationSettings | None = None,
        *,
        manual_sampling: bool | int = False,
        pick: int | None = None,
        skip_special_tokens: bool = False,
        tool: ToolManager | None = None,
    ) -> TextGenerationResponse:
        _ = skip_special_tokens
        generation_settings = settings or GenerationSettings()
        parse_dsml_tools = self._uses_dsml_tools(input, tool)
        prompt_tokens = await self._render_prompt_tokens_async(
            input,
            system_prompt,
            developer_prompt,
            generation_settings,
            tool=tool,
        )
        generation_plan = self._generation_plan(
            generation_settings,
            len(prompt_tokens),
            manual_sampling=manual_sampling,
            parse_dsml_tools=parse_dsml_tools,
            pick=pick,
        )
        generation_settings = replace(
            generation_settings, do_sample=generation_plan.use_sampling
        )

        return TextGenerationResponse(
            self._generation_stream,
            inputs=prompt_tokens,
            logger=self._logger,
            generation_settings=generation_settings,
            generation_plan=generation_plan,
            settings=generation_settings,
            use_async_generator=True,
            bos_token=None,
        )

    def input_token_count(
        self,
        input: Input,
        system_prompt: str | None = None,
        developer_prompt: str | None = None,
    ) -> int:
        """Return the DS4-rendered prompt token count."""
        return len(
            self._render_prompt_tokens(
                input,
                system_prompt,
                developer_prompt,
                GenerationSettings(),
                tool=None,
            )
        )

    def _ds4_settings(self) -> TransformerEngineSettings:
        settings = self._settings
        assert isinstance(settings, TransformerEngineSettings)
        return settings

    def _ds4_worker(self) -> Ds4Worker:
        worker = self._model
        if isinstance(worker, Ds4Worker):
            return worker
        raise Ds4LoadError("DS4 worker is not loaded.")

    def _context_size(self) -> int:
        config = self._ds4_settings().backend_config or {}
        value = config.get("ctx_size", _DEFAULT_CONTEXT_SIZE)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(
                "DS4 backend config 'ctx_size' must be a positive integer."
            )
        return value

    def _disk_cache_config(
        self, config: dict[str, object]
    ) -> _Ds4DiskCacheConfig:
        directory_value = config.get("kv_disk_dir")
        budget_mb = self._int_config(config, "kv_disk_space_mb", 0)
        if directory_value is None:
            return _Ds4DiskCacheConfig(None, 0)
        if not isinstance(directory_value, str) or not directory_value:
            raise ValueError(
                "DS4 backend config 'kv_disk_dir' must be a non-empty string."
            )
        return _Ds4DiskCacheConfig(
            Path(directory_value).expanduser(),
            budget_mb * _BYTES_PER_MIB,
        )

    def _render_prompt_tokens(
        self,
        input: Input,
        system_prompt: str | None,
        developer_prompt: str | None,
        settings: GenerationSettings,
        *,
        tool: ToolManager | None,
    ) -> list[int]:
        worker = self._ds4_worker()
        system_content, messages = self._ds4_prompt_messages(
            input, system_prompt, developer_prompt
        )
        tool_schemas = self._tool_schemas(tool)
        if tool_schemas is not None or self._messages_include_tools(messages):
            rendered = DsmlTools.render_prompt(
                system_content,
                messages,
                tool_schemas,
                self._think_mode(settings),
                worker.exact_dsml_for_tool_calls,
            )
            return self._run_tokenize_rendered_chat(worker, rendered)
        return worker.render_prompt_tokens(
            system_content, messages, self._think_mode(settings)
        )

    async def _render_prompt_tokens_async(
        self,
        input: Input,
        system_prompt: str | None,
        developer_prompt: str | None,
        settings: GenerationSettings,
        *,
        tool: ToolManager | None,
    ) -> list[int]:
        worker = self._ds4_worker()
        system_content, messages = self._ds4_prompt_messages(
            input, system_prompt, developer_prompt
        )
        tool_schemas = self._tool_schemas(tool)
        if tool_schemas is not None or self._messages_include_tools(messages):
            rendered = DsmlTools.render_prompt(
                system_content,
                messages,
                tool_schemas,
                self._think_mode(settings),
                worker.exact_dsml_for_tool_calls,
            )
            return await worker.tokenize_rendered_chat_async(rendered)
        return await worker.render_prompt_tokens_async(
            system_content, messages, self._think_mode(settings)
        )

    def _ds4_prompt_messages(
        self,
        input: Input,
        system_prompt: str | None,
        developer_prompt: str | None,
    ) -> tuple[str | None, list[_Ds4PromptMessage]]:
        try:
            raw_messages = self._messages(input, None, None, None)
        except AssertionError as error:
            raise ValueError(input) from error

        system_parts: list[str] = []
        developer_parts: list[str] = []
        if system_prompt:
            system_parts.append(system_prompt)
        if developer_prompt:
            developer_parts.append(developer_prompt)
        messages: list[_Ds4PromptMessage] = []
        for message in raw_messages:
            role = self._message_role(message)
            content = self._message_text(message.content)
            if role is MessageRole.DEVELOPER:
                if content:
                    developer_parts.append(content)
                continue
            if role is MessageRole.SYSTEM:
                if content:
                    system_parts.append(content)
                continue
            if role in {
                MessageRole.USER,
                MessageRole.ASSISTANT,
                MessageRole.TOOL,
            }:
                messages.append(
                    _Ds4PromptMessage(
                        role=role,
                        content=content,
                        reasoning=message.thinking,
                        tool_calls=tuple(message.tool_calls or ()),
                    )
                )
                continue
            raise ValueError(f"Unsupported DS4 message role {role!r}.")

        if not messages or not any(
            message.content.strip() for message in messages
        ):
            raise ValueError("DS4 prompt must include non-empty text input.")

        return (
            self._merge_system_parts(system_parts, developer_parts),
            messages,
        )

    @staticmethod
    def _merge_system_parts(
        system_parts: list[str], developer_parts: list[str]
    ) -> str | None:
        system_content = "\n\n".join(system_parts) if system_parts else None
        developer_content = (
            "\n\n".join(developer_parts) if developer_parts else None
        )
        if system_content and developer_content:
            return (
                f"{system_content}{_SYSTEM_DEVELOPER_SEPARATOR}"
                f"{developer_content}"
            )
        if developer_content:
            return f"{_DEVELOPER_PROMPT_PREFIX}\n{developer_content}"
        return system_content

    @staticmethod
    def _message_role(message: Message) -> MessageRole:
        try:
            return MessageRole(message.role)
        except ValueError as error:
            raise ValueError(
                f"Unsupported DS4 message role {message.role!r}."
            ) from error

    @staticmethod
    def _has_tool_payload(message: Message) -> bool:
        return bool(
            message.tool_calls
            or message.tool_call_result
            or message.tool_call_error
        )

    @staticmethod
    def _messages_include_tools(messages: list[_Ds4PromptMessage]) -> bool:
        return any(
            message.role is MessageRole.TOOL or bool(message.tool_calls)
            for message in messages
        )

    @classmethod
    def _tool_schemas(cls, tool: ToolManager | None) -> str | None:
        if tool is None or tool.is_empty:
            return None
        return DsmlTools.render_tool_schemas(tool.json_schemas())

    def _uses_dsml_tools(self, input: Input, tool: ToolManager | None) -> bool:
        if tool is not None and not tool.is_empty:
            return True
        try:
            raw_messages = self._messages(input, None, None, None)
        except AssertionError:
            return False
        return any(
            self._has_tool_payload(message)
            or self._message_role(message) is MessageRole.TOOL
            for message in raw_messages
        )

    @staticmethod
    def _run_tokenize_rendered_chat(
        worker: Ds4Worker, rendered: str
    ) -> list[int]:
        return Ds4Worker._run_sync(
            lambda: worker.tokenize_rendered_chat_async(rendered)
        )

    @classmethod
    def _message_text(
        cls, content: str | MessageContent | list[MessageContent] | None
    ) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, MessageContentText):
            return content.text
        if isinstance(content, (MessageContentFile, MessageContentImage)):
            return ""
        if isinstance(content, list):
            return "\n".join(
                cls._message_text(item)
                for item in content
                if isinstance(item, MessageContentText)
            )
        return str(content)

    @staticmethod
    def _think_mode(settings: GenerationSettings) -> ThinkMode:
        if not settings.reasoning.enabled:
            return ThinkMode.NONE

        effort = settings.reasoning.effort
        if effort is None:
            return ThinkMode.NONE

        effort_value = (
            effort.value if isinstance(effort, ReasoningEffort) else effort
        )
        if effort_value == ReasoningEffort.NONE.value:
            return ThinkMode.NONE
        if effort_value == ReasoningEffort.MAX.value:
            return ThinkMode.MAX
        return ThinkMode.HIGH

    async def _generation_stream(
        self,
        inputs: list[int],
        generation_plan: _Ds4GenerationPlan,
        settings: GenerationSettings | None = None,
    ) -> AsyncGenerator[str | TokenDetail | ToolCallToken, None]:
        async for chunk in self._ds4_worker().stream(inputs, generation_plan):
            yield chunk

    def _generation_string(
        self,
        inputs: list[int],
        generation_plan: _Ds4GenerationPlan,
        settings: GenerationSettings | None = None,
    ) -> str:
        return self._ds4_worker().generate_string(inputs, generation_plan)

    def _generation_plan(
        self,
        settings: GenerationSettings,
        prompt_length: int,
        *,
        manual_sampling: bool | int = False,
        parse_dsml_tools: bool = False,
        pick: int | None = None,
    ) -> _Ds4GenerationPlan:
        self._validate_generation_features(settings)
        sampling_options = self._sampling_options(settings)
        use_sampling = bool(
            settings.do_sample and sampling_options.temperature > 0.0
        )
        top_logprobs = self._top_logprobs(manual_sampling, pick)
        return _Ds4GenerationPlan(
            emit_token_details=bool(manual_sampling),
            max_new_tokens=self._max_new_tokens(settings, prompt_length),
            parse_dsml_tools=parse_dsml_tools,
            probability_distribution="log_softmax",
            sampling_options=sampling_options,
            stop_strings=self._stop_strings(settings.stop_strings),
            top_logprobs=top_logprobs,
            use_sampling=use_sampling,
        )

    @staticmethod
    def _top_logprobs(manual_sampling: bool | int, pick: int | None) -> int:
        if not manual_sampling:
            return 0
        if pick is not None:
            if isinstance(pick, bool) or not isinstance(pick, int) or pick < 0:
                raise ValueError("DS4 top_logprobs must be non-negative.")
            return pick
        if isinstance(manual_sampling, bool):
            return 0
        if (
            isinstance(manual_sampling, int)
            and not isinstance(manual_sampling, bool)
            and manual_sampling >= 0
        ):
            return manual_sampling
        raise ValueError("DS4 top_logprobs must be non-negative.")

    @classmethod
    def _max_new_tokens(
        cls, settings: GenerationSettings, prompt_length: int
    ) -> int:
        if settings.max_new_tokens is not None:
            return cls._non_negative_int(
                "max_new_tokens", settings.max_new_tokens
            )
        if settings.max_length is not None:
            max_length = cls._non_negative_int(
                "max_length", settings.max_length
            )
            return max(max_length - prompt_length, 0)
        return _DEFAULT_MAX_NEW_TOKENS

    def _sampling_options(
        self, settings: GenerationSettings
    ) -> SamplingOptions:
        defaults = SamplingOptions()
        return SamplingOptions(
            temperature=self._non_negative_float(
                "temperature", settings.temperature, defaults.temperature
            ),
            top_k=self._non_negative_int(
                "top_k", settings.top_k, defaults.top_k
            ),
            top_p=self._probability("top_p", settings.top_p, defaults.top_p),
            min_p=self._probability("min_p", settings.min_p, defaults.min_p),
            seed=self._seed(),
        )

    def _seed(self) -> int | None:
        config = self._ds4_settings().backend_config or {}
        value = config.get("seed")
        if value is None:
            return None
        return self._non_negative_int("seed", value)

    @staticmethod
    def _stop_strings(value: str | list[str] | None) -> tuple[str, ...]:
        if value is None:
            return ()
        stop_strings = (value,) if isinstance(value, str) else tuple(value)
        if not stop_strings:
            return ()
        if not all(isinstance(item, str) and item for item in stop_strings):
            raise ValueError(
                "DS4 stop_strings must be a string or a list of non-empty "
                "strings."
            )
        return stop_strings

    @staticmethod
    def _validate_generation_features(settings: GenerationSettings) -> None:
        if settings.num_beams not in (None, 1):
            raise NotImplementedError(
                _UNSUPPORTED_GENERATION_MESSAGE.format(feature="beam search")
            )
        if settings.num_beam_groups not in (None, 1):
            raise NotImplementedError(
                _UNSUPPORTED_GENERATION_MESSAGE.format(feature="beam groups")
            )
        if settings.num_return_sequences not in (None, 1):
            raise NotImplementedError(
                _UNSUPPORTED_GENERATION_MESSAGE.format(
                    feature="multiple return sequences"
                )
            )

    @staticmethod
    def _non_negative_int(
        key: str, value: object, default: int | None = None
    ) -> int:
        if value is None and default is not None:
            return default
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"DS4 {key} must be a non-negative integer.")
        return value

    @staticmethod
    def _non_negative_float(key: str, value: object, default: float) -> float:
        if value is None:
            return default
        if (
            isinstance(value, bool)
            or not isinstance(value, (float, int))
            or float(value) < 0.0
        ):
            raise ValueError(f"DS4 {key} must be non-negative.")
        return float(value)

    @staticmethod
    def _probability(key: str, value: object, default: float) -> float:
        if value is None:
            return default
        if (
            isinstance(value, bool)
            or not isinstance(value, (float, int))
            or not 0.0 <= float(value) <= 1.0
        ):
            raise ValueError(f"DS4 {key} must be between 0.0 and 1.0.")
        return float(value)

    def _native_backend(self, config: dict[str, object]) -> Ds4NativeBackend:
        requested = config.get("native_backend")
        if requested is None or requested == "auto":
            if self._device.startswith("cuda"):
                return Ds4NativeBackend.CUDA
            if self._device == "mps":
                return Ds4NativeBackend.METAL
            return _DEFAULT_NATIVE_BACKEND
        if not isinstance(requested, str):
            raise Ds4BackendUnavailable("DS4 native_backend must be a string.")
        try:
            return Ds4NativeBackend(requested)
        except ValueError as error:
            supported = ", ".join(
                backend.value for backend in Ds4NativeBackend
            )
            raise Ds4BackendUnavailable(
                f"Unsupported DS4 native backend {requested!r}. "
                f"Supported native backends: {supported}."
            ) from error

    @staticmethod
    def _validated_file_path(path: str, label: str) -> str:
        file_path = Path(path).expanduser()
        if not file_path.exists():
            raise Ds4InvalidModel(f"{label} does not exist: {file_path}.")
        if file_path.is_dir():
            raise Ds4InvalidModel(
                f"{label} must be a file, got directory: {file_path}."
            )
        if not file_path.is_file():
            raise Ds4InvalidModel(
                f"{label} must be a regular file: {file_path}."
            )
        return str(file_path)

    @classmethod
    def _optional_file_path(cls, value: object, label: str) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str) or not value:
            raise Ds4InvalidModel(f"{label} must be a non-empty path.")
        return cls._validated_file_path(value, label)

    @classmethod
    def _steering_file_path(
        cls,
        config: dict[str, object],
        directional_steering_attn: float,
        directional_steering_ffn: float,
    ) -> str | None:
        path = cls._optional_file_path(
            config.get("directional_steering_file"),
            "DS4 directional steering file",
        )
        if path is None and (
            directional_steering_attn != 0.0 or directional_steering_ffn != 0.0
        ):
            raise Ds4InvalidModel(
                "DS4 directional_steering_file is required when directional "
                "steering coefficients are non-zero."
            )
        return path

    @staticmethod
    def _int_config(config: dict[str, object], key: str, default: int) -> int:
        value = config.get(key, default)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"DS4 backend config {key!r} must be an integer.")
        return value

    @classmethod
    def _non_negative_int_config(
        cls, config: dict[str, object], key: str, default: int
    ) -> int:
        value = cls._int_config(config, key, default)
        if value < 0:
            raise ValueError(
                f"DS4 backend config {key!r} must be non-negative."
            )
        return value

    @staticmethod
    def _float_config(
        config: dict[str, object], key: str, default: float
    ) -> float:
        value = config.get(key, default)
        if isinstance(value, bool) or not isinstance(value, (float, int)):
            raise ValueError(f"DS4 backend config {key!r} must be a number.")
        return float(value)

    @classmethod
    def _non_negative_float_config(
        cls, config: dict[str, object], key: str, default: float
    ) -> float:
        value = cls._float_config(config, key, default)
        if value < 0.0:
            raise ValueError(
                f"DS4 backend config {key!r} must be non-negative."
            )
        return value

    @staticmethod
    def _bool_config(
        config: dict[str, object], key: str, default: bool
    ) -> bool:
        value = config.get(key, default)
        if not isinstance(value, bool):
            raise ValueError(f"DS4 backend config {key!r} must be a boolean.")
        return value

    @staticmethod
    def _optional_string_config(
        config: dict[str, object], key: str
    ) -> str | None:
        value = config.get(key)
        if value is None:
            return None
        if not isinstance(value, str) or not value:
            raise ValueError(
                f"DS4 backend config {key!r} must be a non-empty string."
            )
        return value

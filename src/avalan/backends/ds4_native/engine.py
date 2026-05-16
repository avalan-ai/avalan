from .availability import import_compatible_binding
from .errors import (
    Ds4BackendUnavailable,
    Ds4ContextError,
    Ds4GenerationError,
    Ds4InvalidModel,
    Ds4LoadError,
)
from .metadata import DS4_BINDING_IMPORT_NAME, DS4_SUPPORTED_NATIVE_BACKENDS
from .types import Backend, EngineOptions, SamplingOptions, ThinkMode

from collections.abc import Callable
from dataclasses import replace
from enum import StrEnum
from pathlib import Path
from threading import RLock
from typing import NoReturn, Protocol, cast

_SUPPORTED_CHAT_ROLES = frozenset(("assistant", "system", "user"))
_MAX_PROBABILITY = 1.0


class _SessionHandle(Protocol):
    def close(self) -> None:
        """Close the native session handle."""


class Engine:
    """Own a validated DS4 native engine through the pyds4 binding."""

    def __init__(
        self,
        options: EngineOptions,
        *,
        module_name: str = DS4_BINDING_IMPORT_NAME,
    ) -> None:
        self._binding: object | None = None
        self._closed = True
        self._native: object | None = None
        self._options = self._normalize_options(options)
        self._session_lock = RLock()
        self._sessions: list[_SessionHandle] = []

        self._options = self._validate_model_path(self._options)
        self._binding = import_compatible_binding(
            module_name, backend=self._options.backend.value
        )

        native_options = self._native_engine_options(
            self._binding, self._options
        )
        native_engine_type = getattr(self._binding, "Engine", None)
        if not callable(native_engine_type):
            raise Ds4LoadError(
                f"DS4 binding {module_name!r} does not expose Engine."
            )

        try:
            # The blocking native open is delegated to pyds4.Engine; the
            # compiled binding is responsible for releasing the GIL there.
            self._native = native_engine_type(native_options)
        except (
            Ds4BackendUnavailable,
            Ds4InvalidModel,
            Ds4LoadError,
        ):
            raise
        except Exception as error:
            self._raise_mapped_open_error(error)
        self._closed = False

    @property
    def native(self) -> object:
        """Return the wrapped pyds4 engine object."""
        return self._require_open()

    @property
    def options(self) -> EngineOptions:
        """Return the normalized engine options."""
        return self._options

    @property
    def closed(self) -> bool:
        """Return whether the engine has been closed."""
        return self._closed

    @property
    def routed_quant_bits(self) -> int:
        """Return the routed quantization bits reported by DS4."""
        return self._metadata_int("routed_quant_bits")

    @property
    def has_mtp(self) -> bool:
        """Return whether the loaded DS4 engine has MTP state."""
        return self._metadata_bool("has_mtp")

    @property
    def mtp_draft_tokens(self) -> int:
        """Return the DS4 MTP draft-token count."""
        return self._metadata_int("mtp_draft_tokens")

    @property
    def eos_token_id(self) -> int:
        """Return the DS4 EOS token id."""
        return self._metadata_int("eos_token_id")

    def token_text(self, token_id: int) -> bytes:
        """Return the UTF-8 byte sequence for a DS4 token id."""
        assert isinstance(token_id, int), "A DS4 token id must be an integer."
        value = self._call_generation("token_text", token_id)
        if isinstance(value, bytes):
            return value
        if isinstance(value, (bytearray, memoryview)):
            return bytes(value)
        raise Ds4GenerationError("DS4 token_text must return bytes.")

    def tokenize_text(self, text: str) -> list[int]:
        """Tokenize raw text with DS4's native tokenizer."""
        assert isinstance(text, str), "DS4 tokenization input must be text."
        return self._token_list(
            self._call_generation("tokenize_text", text),
            "tokenize_text",
        )

    def tokenize_rendered_chat(self, text: str) -> list[int]:
        """Tokenize rendered DS4 chat text."""
        assert isinstance(text, str), "DS4 chat input must be text."
        return self._token_list(
            self._call_generation("tokenize_rendered_chat", text),
            "tokenize_rendered_chat",
        )

    def chat_begin(self) -> list[int]:
        """Return initial DS4 chat prompt tokens."""
        return self._token_list(
            self._call_generation("chat_begin"), "chat_begin"
        )

    def chat_append_message(
        self,
        tokens: list[int],
        role: str,
        content: str,
    ) -> None:
        """Append a supported DS4 chat message to a token buffer."""
        self._validate_token_buffer(tokens)
        if role not in _SUPPORTED_CHAT_ROLES:
            supported = ", ".join(sorted(_SUPPORTED_CHAT_ROLES))
            raise ValueError(
                f"Unsupported DS4 chat role {role!r}. "
                f"Supported roles: {supported}."
            )
        assert isinstance(content, str), "DS4 chat content must be text."

        result = self._call_generation(
            "chat_append_message", tokens, role, content
        )
        self._replace_tokens_if_returned(tokens, result, "chat_append_message")

    def chat_append_assistant_prefix(
        self, tokens: list[int], think_mode: ThinkMode
    ) -> None:
        """Append DS4's assistant prefix for a thinking mode."""
        self._validate_token_buffer(tokens)
        mode = self._coerce_think_mode(think_mode)
        result = self._call_generation(
            "chat_append_assistant_prefix",
            tokens,
            self._native_enum_value(self._binding_module(), "ThinkMode", mode),
        )
        self._replace_tokens_if_returned(
            tokens, result, "chat_append_assistant_prefix"
        )

    def encode_chat_prompt(
        self,
        system: str | None,
        prompt: str,
        think_mode: ThinkMode,
    ) -> list[int]:
        """Encode a single-turn DS4 chat prompt."""
        assert system is None or isinstance(
            system, str
        ), "DS4 system prompt must be text or None."
        assert isinstance(prompt, str), "DS4 user prompt must be text."
        mode = self._coerce_think_mode(think_mode)
        return self._token_list(
            self._call_generation(
                "encode_chat_prompt",
                system,
                prompt,
                self._native_enum_value(
                    self._binding_module(), "ThinkMode", mode
                ),
            ),
            "encode_chat_prompt",
        )

    def think_mode_for_context(
        self, think_mode: ThinkMode, ctx_size: int
    ) -> ThinkMode:
        """Return DS4's effective thinking mode for a context size."""
        self._require_open()
        assert ctx_size > 0, "A DS4 context size must be positive."
        mode = self._coerce_think_mode(think_mode)
        helper = getattr(
            self._binding_module(), "think_mode_for_context", None
        )
        if not callable(helper):
            raise Ds4GenerationError(
                "DS4 binding does not expose think_mode_for_context."
            )
        native_mode = self._native_enum_value(
            self._binding_module(), "ThinkMode", mode
        )
        try:
            value = cast(Callable[[object, int], object], helper)(
                native_mode, ctx_size
            )
        except Exception as error:
            self._raise_mapped_generation_error(
                error, "think_mode_for_context"
            )
        return self._think_mode_from_native(value)

    def create_session(self, ctx_size: int) -> "Session":
        """Create a DS4 session for a positive context size."""
        self._validate_context_size(ctx_size)
        native = self._require_open()
        try:
            native_session = self._create_native_session(native, ctx_size)
        except (Ds4ContextError, Ds4LoadError):
            raise
        except Exception as error:
            self._raise_mapped_context_error(error, "session_create")
        if native_session is None:
            raise Ds4ContextError("DS4 session creation returned no session.")

        session = Session(self, native_session)
        self._track_session(session)
        return session

    def close(self) -> None:
        """Close live sessions and the wrapped native engine."""
        if self._closed:
            return

        sessions = self._sessions
        self._sessions = []
        for session in reversed(sessions):
            session.close()

        native = self._native
        self._native = None
        self._closed = True
        close = getattr(native, "close", None)
        if callable(close):
            cast(Callable[[], None], close)()

    def __enter__(self) -> "Engine":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: object | None,
    ) -> None:
        self.close()

    def _track_session(self, session: _SessionHandle) -> None:
        self._require_open()
        self._sessions.append(session)

    def _untrack_session(self, session: _SessionHandle) -> None:
        self._sessions = [
            candidate
            for candidate in self._sessions
            if candidate is not session
        ]

    def _require_open(self) -> object:
        native = self._native
        if self._closed or native is None:
            raise Ds4LoadError("DS4 engine is closed.")
        return native

    def _binding_module(self) -> object:
        self._require_open()
        binding = self._binding
        if binding is None:
            raise Ds4LoadError("DS4 binding is unavailable.")
        return binding

    def _generation_method(self, name: str) -> Callable[..., object]:
        method = getattr(self._require_open(), name, None)
        if not callable(method):
            raise Ds4LoadError(f"DS4 engine does not expose {name}.")
        return cast(Callable[..., object], method)

    def _call_generation(self, name: str, *args: object) -> object:
        try:
            return self._generation_method(name)(*args)
        except (Ds4GenerationError, Ds4LoadError):
            raise
        except Exception as error:
            self._raise_mapped_generation_error(error, name)

    @staticmethod
    def _raise_mapped_generation_error(
        error: Exception, operation: str
    ) -> NoReturn:
        message = str(error) or type(error).__name__
        raise Ds4GenerationError(
            f"DS4 {operation} failed: {message}"
        ) from error

    def _metadata_value(self, name: str) -> object:
        native = self._require_open()
        value = getattr(native, name, None)
        if callable(value):
            value = cast(Callable[[], object], value)()
        if value is None:
            raise Ds4LoadError(f"DS4 engine metadata {name!r} is unavailable.")
        return value

    def _metadata_int(self, name: str) -> int:
        value = self._metadata_value(name)
        if isinstance(value, bool) or not isinstance(value, int):
            raise Ds4LoadError(
                f"DS4 engine metadata {name!r} must be an integer."
            )
        return value

    def _metadata_bool(self, name: str) -> bool:
        value = self._metadata_value(name)
        if not isinstance(value, bool):
            raise Ds4LoadError(
                f"DS4 engine metadata {name!r} must be a boolean."
            )
        return value

    @staticmethod
    def _coerce_think_mode(think_mode: ThinkMode) -> ThinkMode:
        try:
            return ThinkMode(think_mode)
        except ValueError as error:
            supported = ", ".join(mode.value for mode in ThinkMode)
            raise ValueError(
                f"Unsupported DS4 thinking mode {think_mode!r}. "
                f"Supported modes: {supported}."
            ) from error

    @staticmethod
    def _is_token_id(value: object) -> bool:
        return isinstance(value, int) and not isinstance(value, bool)

    @classmethod
    def _token_list(cls, value: object, operation: str) -> list[int]:
        if isinstance(value, tuple):
            value = list(value)
        if isinstance(value, list) and all(
            cls._is_token_id(token_id) for token_id in value
        ):
            return value
        raise Ds4GenerationError(
            f"DS4 {operation} must return a list of token ids."
        )

    @classmethod
    def _validate_token_buffer(cls, tokens: list[int]) -> None:
        assert isinstance(tokens, list), "DS4 token buffer must be a list."
        if not all(cls._is_token_id(token_id) for token_id in tokens):
            raise ValueError(
                "DS4 token buffers must contain integer token IDs."
            )

    @classmethod
    def _replace_tokens_if_returned(
        cls, tokens: list[int], value: object, operation: str
    ) -> None:
        if value is None:
            cls._validate_token_buffer(tokens)
            return
        tokens[:] = cls._token_list(value, operation)

    @staticmethod
    def _think_mode_from_native(value: object) -> ThinkMode:
        if isinstance(value, ThinkMode):
            return value
        if isinstance(value, StrEnum):
            value = value.value
        if not isinstance(value, str):
            raise Ds4GenerationError(
                f"DS4 returned unsupported thinking mode {value!r}."
            )
        try:
            return ThinkMode(value)
        except ValueError as error:
            raise Ds4GenerationError(
                f"DS4 returned unsupported thinking mode {value!r}."
            ) from error

    @staticmethod
    def _normalize_options(options: EngineOptions) -> EngineOptions:
        try:
            backend = Backend(options.backend)
        except ValueError as error:
            supported = ", ".join(DS4_SUPPORTED_NATIVE_BACKENDS)
            raise Ds4BackendUnavailable(
                f"Unsupported DS4 native backend {options.backend!r}. "
                f"Supported native backends: {supported}."
            ) from error
        if backend is options.backend:
            return options
        return replace(options, backend=backend)

    @staticmethod
    def _validate_model_path(options: EngineOptions) -> EngineOptions:
        if not options.model_path:
            raise Ds4InvalidModel("A DS4 model path is required.")

        model_path = Path(options.model_path).expanduser()
        if not model_path.exists():
            raise Ds4InvalidModel(
                f"DS4 model path does not exist: {model_path}."
            )
        if model_path.is_dir():
            raise Ds4InvalidModel(
                f"DS4 model path must be a file, got directory: {model_path}."
            )
        if not model_path.is_file():
            raise Ds4InvalidModel(
                f"DS4 model path must be a regular file: {model_path}."
            )
        return replace(options, model_path=str(model_path))

    @staticmethod
    def _native_engine_options(
        binding: object, options: EngineOptions
    ) -> object:
        native_options_type = getattr(binding, "EngineOptions", None)
        if not callable(native_options_type):
            return options

        return native_options_type(
            model_path=options.model_path,
            backend=Engine._native_enum_value(
                binding, "Backend", options.backend
            ),
            mtp_path=options.mtp_path,
            n_threads=options.n_threads,
            mtp_draft_tokens=options.mtp_draft_tokens,
            mtp_margin=options.mtp_margin,
            directional_steering_file=options.directional_steering_file,
            directional_steering_attn=options.directional_steering_attn,
            directional_steering_ffn=options.directional_steering_ffn,
            warm_weights=options.warm_weights,
            quality=options.quality,
        )

    @staticmethod
    def _native_enum_value(
        binding: object, enum_name: str, value: StrEnum
    ) -> object:
        enum_type = getattr(binding, enum_name, None)
        if enum_type is None:
            return value.value
        try:
            return enum_type(value.value)
        except (TypeError, ValueError):
            member = getattr(enum_type, value.name, None)
            return member if member is not None else value.value

    @staticmethod
    def _validate_context_size(ctx_size: int) -> None:
        if (
            isinstance(ctx_size, bool)
            or not isinstance(ctx_size, int)
            or ctx_size <= 0
        ):
            raise Ds4ContextError(
                "A DS4 context size must be a positive integer."
            )

    def _create_native_session(self, native: object, ctx_size: int) -> object:
        creator = getattr(native, "create_session", None)
        if callable(creator):
            return cast(Callable[[int], object], creator)(ctx_size)

        session_type = getattr(self._binding_module(), "Session", None)
        if callable(session_type):
            return cast(Callable[[object, int], object], session_type)(
                native, ctx_size
            )

        raise Ds4LoadError("DS4 engine does not expose create_session.")

    @staticmethod
    def _raise_mapped_context_error(
        error: Exception, operation: str
    ) -> NoReturn:
        message = str(error) or type(error).__name__
        if type(error).__name__ == "Ds4ContextError":
            raise Ds4ContextError(message) from error
        raise Ds4ContextError(f"DS4 {operation} failed: {message}") from error

    @staticmethod
    def _raise_mapped_open_error(error: Exception) -> None:
        message = str(error) or type(error).__name__
        error_name = type(error).__name__
        if error_name == "Ds4BackendUnavailable":
            raise Ds4BackendUnavailable(message) from error
        if error_name == "Ds4InvalidModel":
            raise Ds4InvalidModel(message) from error
        raise Ds4LoadError(message) from error


class Session:
    """Own a DS4 native session through the pyds4 binding."""

    def __init__(self, engine: Engine, native: object) -> None:
        self._closed = False
        self._engine = engine
        self._lock = engine._session_lock
        self._native: object | None = native

    @property
    def closed(self) -> bool:
        """Return whether the session has been closed."""
        return self._closed

    @property
    def native(self) -> object:
        """Return the wrapped pyds4 session object."""
        return self._require_open()

    @property
    def pos(self) -> int:
        """Return the current DS4 session position."""
        return self._session_int("pos")

    @property
    def ctx(self) -> int:
        """Return the DS4 session context size."""
        return self._session_int("ctx")

    @property
    def tokens(self) -> tuple[int, ...]:
        """Return the session token history."""
        value = self._session_value("tokens")
        if isinstance(value, tuple):
            tokens = value
        elif isinstance(value, list):
            tokens = tuple(value)
        else:
            raise Ds4GenerationError(
                "DS4 session tokens must be a list or tuple of token ids."
            )
        if all(Engine._is_token_id(token_id) for token_id in tokens):
            return tokens
        raise Ds4GenerationError(
            "DS4 session tokens must contain integer token IDs."
        )

    def sync(self, prompt_tokens: list[int]) -> None:
        """Synchronize the session with rendered prompt tokens."""
        Engine._validate_token_buffer(prompt_tokens)
        # The compiled pyds4 binding owns GIL release for this native call.
        self._call_generation("sync", prompt_tokens)

    def eval(self, token_id: int) -> None:
        """Evaluate one token and advance the DS4 session."""
        self._validate_token_id(token_id)
        # DS4 eval advances session state; pyds4 owns GIL release here.
        self._call_generation("eval", token_id)

    def argmax(self) -> int:
        """Return the greedy next-token candidate without consuming it."""
        return self._token_result(self._call_generation("argmax"), "argmax")

    def argmax_excluding(self, token_id: int) -> int:
        """Return the greedy token while excluding a token id."""
        self._validate_token_id(token_id)
        return self._token_result(
            self._call_generation("argmax_excluding", token_id),
            "argmax_excluding",
        )

    def sample(self, options: SamplingOptions) -> int:
        """Sample the next-token candidate without consuming it."""
        normalized = self._normalize_sampling_options(options)
        native_options = self._native_sampling_options(
            self._engine._binding_module(), normalized
        )
        return self._token_result(
            self._call_generation("sample", native_options),
            "sample",
        )

    def top_logprobs(self, top_k: int) -> tuple[tuple[int, float], ...]:
        """Return the top next-token log probabilities."""
        self._validate_top_k(top_k)
        return self._top_logprobs_result(
            self._call_generation("top_logprobs", top_k),
            "top_logprobs",
        )

    def token_logprob(self, token_id: int) -> float:
        """Return the next-token log probability for one token."""
        self._validate_token_id(token_id)
        return self._logprob_result(
            self._call_generation("token_logprob", token_id),
            "token_logprob",
        )

    def rewind(self, pos: int) -> None:
        """Rewind the DS4 session to a previous non-negative position."""
        if isinstance(pos, bool) or not isinstance(pos, int) or pos < 0:
            raise Ds4ContextError(
                "A DS4 rewind position must be a non-negative integer."
            )
        self._call_generation("rewind", pos)

    def invalidate(self) -> None:
        """Invalidate DS4 session state through the native binding."""
        self._call_generation("invalidate")

    def save_snapshot(self) -> bytes:
        """Return an in-memory snapshot of the current DS4 session."""
        return self._snapshot_bytes(
            self._call_generation("save_snapshot"), "save_snapshot"
        )

    def load_snapshot(self, snapshot: bytes) -> None:
        """Restore a DS4 session from an in-memory snapshot."""
        assert isinstance(snapshot, bytes), "A DS4 snapshot must be bytes."
        self._call_generation("load_snapshot", snapshot)

    def close(self) -> None:
        """Close the wrapped native session."""
        with self._lock:
            if self._closed:
                return

            native = self._native
            self._native = None
            self._closed = True
            try:
                close = getattr(native, "close", None)
                if callable(close):
                    cast(Callable[[], None], close)()
            finally:
                self._engine._untrack_session(self)

    def __enter__(self) -> "Session":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: object | None,
    ) -> None:
        self.close()

    def _require_open(self) -> object:
        native = self._native
        if self._closed or native is None:
            raise Ds4GenerationError("DS4 session is closed.")
        self._engine._require_open()
        return native

    def _session_method(self, name: str) -> Callable[..., object]:
        method = getattr(self._require_open(), name, None)
        if not callable(method):
            raise Ds4LoadError(f"DS4 session does not expose {name}.")
        return cast(Callable[..., object], method)

    def _call_generation(self, name: str, *args: object) -> object:
        with self._lock:
            try:
                return self._session_method(name)(*args)
            except (Ds4ContextError, Ds4GenerationError, Ds4LoadError):
                raise
            except Exception as error:
                self._raise_mapped_generation_error(error, name)

    def _session_value(self, name: str) -> object:
        with self._lock:
            native = self._require_open()
            try:
                value = getattr(native, name, None)
                if callable(value):
                    value = cast(Callable[[], object], value)()
            except Exception as error:
                self._raise_mapped_generation_error(error, name)
        if value is None:
            raise Ds4LoadError(f"DS4 session value {name!r} is unavailable.")
        return value

    def _session_int(self, name: str) -> int:
        value = self._session_value(name)
        if isinstance(value, bool) or not isinstance(value, int):
            raise Ds4GenerationError(
                f"DS4 session value {name!r} must be an integer."
            )
        return value

    @staticmethod
    def _validate_token_id(token_id: int) -> None:
        assert Engine._is_token_id(
            token_id
        ), "A DS4 token id must be an integer."

    @staticmethod
    def _token_result(value: object, operation: str) -> int:
        if Engine._is_token_id(value):
            return cast(int, value)
        raise Ds4GenerationError(f"DS4 {operation} must return one token id.")

    @staticmethod
    def _validate_top_k(top_k: int) -> None:
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 0:
            raise ValueError("DS4 top_k must be a non-negative integer.")

    @classmethod
    def _top_logprobs_result(
        cls, value: object, operation: str
    ) -> tuple[tuple[int, float], ...]:
        if not isinstance(value, (list, tuple)):
            raise Ds4GenerationError(
                f"DS4 {operation} must return token log probabilities."
            )

        return tuple(cls._token_score(score, operation) for score in value)

    @classmethod
    def _token_score(cls, value: object, operation: str) -> tuple[int, float]:
        token_id: object
        logprob: object
        if isinstance(value, dict):
            token_id = value.get("token_id", value.get("id"))
            logprob = value.get("logprob")
        elif isinstance(value, (list, tuple)):
            if len(value) >= 3:
                token_id = value[0]
                logprob = value[2]
            elif len(value) >= 2:
                token_id = value[0]
                logprob = value[1]
            else:
                token_id = None
                logprob = None
        else:
            token_id = getattr(value, "token_id", getattr(value, "id", None))
            logprob = getattr(value, "logprob", None)

        if Engine._is_token_id(token_id):
            return cast(int, token_id), cls._logprob_result(logprob, operation)
        raise Ds4GenerationError(
            f"DS4 {operation} must return token ids and log probabilities."
        )

    @staticmethod
    def _logprob_result(value: object, operation: str) -> float:
        if isinstance(value, bool) or not isinstance(value, (float, int)):
            raise Ds4GenerationError(
                f"DS4 {operation} must return a numeric log probability."
            )
        return float(value)

    @staticmethod
    def _snapshot_bytes(value: object, operation: str) -> bytes:
        if isinstance(value, bytes):
            return value
        if isinstance(value, (bytearray, memoryview)):
            return bytes(value)
        raise Ds4GenerationError(f"DS4 {operation} must return bytes.")

    @staticmethod
    def _is_probability(value: object) -> bool:
        return (
            not isinstance(value, bool)
            and isinstance(value, (float, int))
            and 0.0 <= float(value) <= _MAX_PROBABILITY
        )

    @staticmethod
    def _normalize_sampling_options(
        options: SamplingOptions,
    ) -> SamplingOptions:
        assert isinstance(
            options, SamplingOptions
        ), "DS4 sampling options are required."
        if (
            isinstance(options.top_k, bool)
            or not isinstance(options.top_k, int)
            or options.top_k < 0
        ):
            raise ValueError("DS4 top_k must be a non-negative integer.")
        if not Session._is_probability(options.top_p):
            raise ValueError("DS4 top_p must be between 0.0 and 1.0.")
        if not Session._is_probability(options.min_p):
            raise ValueError("DS4 min_p must be between 0.0 and 1.0.")
        if (
            isinstance(options.temperature, bool)
            or not isinstance(options.temperature, (float, int))
            or float(options.temperature) < 0.0
        ):
            raise ValueError("DS4 temperature must be non-negative.")
        if options.seed is not None and (
            isinstance(options.seed, bool) or not isinstance(options.seed, int)
        ):
            raise ValueError("DS4 seed must be an integer or None.")

        return replace(
            options,
            temperature=float(options.temperature),
            top_p=float(options.top_p),
            min_p=float(options.min_p),
        )

    @staticmethod
    def _native_sampling_options(
        binding: object, options: SamplingOptions
    ) -> object:
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
    def _raise_mapped_generation_error(
        error: Exception, operation: str
    ) -> NoReturn:
        message = str(error) or type(error).__name__
        raise Ds4GenerationError(
            f"DS4 {operation} failed: {message}"
        ) from error

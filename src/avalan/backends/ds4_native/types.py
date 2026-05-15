from dataclasses import dataclass
from enum import StrEnum


class Backend(StrEnum):
    """Name the native DS4 execution backend."""

    METAL = "metal"
    CUDA = "cuda"
    CPU = "cpu"


class ThinkMode(StrEnum):
    """Name the DS4 reasoning prompt mode."""

    NONE = "none"
    HIGH = "high"
    MAX = "max"


@dataclass(frozen=True, slots=True)
class EngineOptions:
    """Store native DS4 engine-open options."""

    model_path: str
    backend: Backend = Backend.METAL
    mtp_path: str | None = None
    n_threads: int = 0
    mtp_draft_tokens: int = 0
    mtp_margin: float = 0.0
    directional_steering_file: str | None = None
    directional_steering_attn: float = 0.0
    directional_steering_ffn: float = 0.0
    warm_weights: bool = False
    quality: bool = False


@dataclass(frozen=True, slots=True)
class SamplingOptions:
    """Store DS4 token sampling options."""

    temperature: float = 0.0
    top_k: int = 0
    top_p: float = 1.0
    min_p: float = 0.0
    seed: int | None = None

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from collections.abc import Callable
from numpy import ndarray
from torch import dtype, Tensor
from typing import Literal
from uuid import UUID

AttentionImplementation = Literal[
    "eager", "flash_attention_2", "flex_attention", "sdpa"
]

ProbabilityDistribution = Literal[
    "entmax", "gumbel_softmax", "log_softmax", "sparsemax", "softmax"
]

SimilarityFunction = Literal["cosine", "dot", "euclidean", "manhattan"]

ImageTextGenerationLoaderClass = Literal["gemma3", "qwen2"]

TextGenerationLoaderClass = Literal["auto", "gemma3", "mistral3"]

ToolValue = bool | float | int | str | None

Vendor = Literal[
    "anthropic",
    "anyscale",
    "deepinfra",
    "deepseek",
    "google",
    "groq",
    "huggingface",
    "hyperbolic",
    "local",
    "openai",
    "openrouter",
    "ollama",
    "litellm",
    "together",
]

WeightType = Literal[
    "auto",
    "bool",
    "bf16",
    "f16",
    "f32",
    "f64",
    "i8",
    "i16",
    "i32",
    "i64",
    "ui8",
]


class DistanceType(StrEnum):
    COSINE = "cosine"
    DOT = "dot"
    L1 = "l1"
    L2 = "l2"
    PEARSON = "pearson"


class MessageRole(StrEnum):
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"
    USER = "user"


class Modality(StrEnum):
    AUDIO_SPEECH_RECOGNITION = "audio_speech_recognition"
    AUDIO_TEXT_TO_SPEECH = "audio_text_to_speech"
    EMBEDDING = "embedding"
    TEXT_GENERATION = "text_generation"
    VISION_OBJECT_DETECTION = "vision_object_detection"
    VISION_IMAGE_CLASSIFICATION = "vision_image_classification"
    VISION_IMAGE_TO_TEXT = "vision_image_to_text"
    VISION_ENCODER_DECODER = "vision_encoder_decoder"
    VISION_SEMANTIC_SEGMENTATION = "vision_semantic_segmentation"


class ParallelStrategy(StrEnum):
    AUTO = "auto"
    COLWISE = "colwise"
    ROWWISE = "rowwise"
    COLWISE_REP = "colwise_rep"
    ROWWISE_REP = "rowwise_rep"
    LOCAL_COLWISE = "local_colwise"
    LOCAL_ROWWISE = "local_rowwise"
    LOCAL = "local"
    GATHER = "gather"
    LOCAL_PACKED_ROWWISE = "local_packed_rowwise"
    SEQUENCE_PARALLEL = "sequence_parallel"
    REPLICATE = "replicate"


class ToolFormat(StrEnum):
    JSON = "json"
    REACT = "react"
    BRACKET = "bracket"
    OPENAI = "openai"


@dataclass(frozen=True, kw_only=True)
class EngineSettings:
    auto_load_model: bool = True
    auto_load_tokenizer: bool = True
    cache_dir: str | None = None
    change_transformers_logging_level = True
    device: str | None = None
    disable_loading_progress_bar: bool = True
    enable_eval: bool = True
    parallel: ParallelStrategy | dict[str, ParallelStrategy] | None = None
    trust_remote_code: bool = False
    tokenizer_name_or_path: str | None = None


@dataclass(kw_only=True, frozen=True)
class EngineUri:
    host: str | None
    port: int | None
    user: str | None
    password: str | None
    vendor: Vendor | None
    model_id: str | None
    params: dict[str, str | int | float | bool]

    @property
    def is_local(self) -> bool:
        return not self.vendor or self.vendor == "local"


@dataclass(frozen=True, kw_only=True)
class GenerationSettings:
    # Generation length ------------------------------------------------------
    # The minimum numbers of tokens to generate, ignoring the number of tokens
    # in the prompt.
    min_new_tokens: int | None = None
    # The maximum numbers of tokens to generate, ignoring the number of tokens
    # in the prompt
    max_new_tokens: int | None = None
    # The minimum length of the sequence to be generated. Corresponds to the
    # length of the input prompt + min_new_tokens. Its effect is overridden
    # by min_new_tokens, if also set.
    min_length: int | None = 0
    # The maximum length the generated tokens can have. Corresponds to the
    # length of the input prompt + max_new_tokens. Its effect is overridden
    # by max_new_tokens, if also set.
    max_length: int | None = 20
    # Controls the stopping condition for beam-based methods, like
    # beam-search. It accepts the following values: True, where the
    # generation stops as soon as there are num_beams complete candidates;
    # False, where an heuristic is applied and the generation stops when is
    # it very unlikely to find better candidates; "never", where the beam
    # search procedure only stops when there cannot be better candidates
    # (canonical beam search algorithm)
    early_stopping: str | bool | None = False
    # The maximum amount of time you allow the computation to run for in
    # seconds. generation will still finish the current pass after allocated
    # time has been passed.
    max_time: float | None = None
    # A string or a list of strings that should terminate generation if the
    # model outputs them.
    stop_strings: str | list[str] | None = None

    # Generation strategy ----------------------------------------------------
    # Whether or not to use sampling. Use greedy decoding otherwise
    do_sample: bool = False
    # Number of beams for beam search. 1 means no beam search.
    num_beams: int | None = 1
    # Number of groups to divide num_beams into in order to ensure diversity
    # among different groups of beams
    num_beam_groups: int | None = 1
    # The values balance the model confidence and the degeneration penalty in
    # contrastive search decoding
    penalty_alpha: float | None = None

    # Cache ------------------------------------------------------------------
    # Whether or not the model should use the past last key/values attentions
    # (if applicable to the model) to speed up decoding
    use_cache: bool = True

    # Generation output variables --------------------------------------------
    # The number of independently computed returned sequences for each element
    # in the batch.
    num_return_sequences: int | None = 1
    # Whether or not to return the attentions tensors of all attention layers.
    # See attentions under returned tensors for more details.
    output_attentions: bool | None = False
    # Whether or not to return the hidden states of all layers. See
    # hidden_states under returned tensors for more details.
    output_hidden_states: bool | None = False
    # Whether or not to return the prediction scores. See scores under
    # returned tensors for more details.
    output_scores: bool | None = False
    # Whether or not to return the unprocessed prediction logit scores. See
    # logits under returned tensors for more details.
    output_logits: bool | None = None
    # Whether or not to return a ModelOutput, as opposed to returning
    # exclusively the generated sequence. This flag must be set to True to
    # return the generation cache (when use_cache is True) or optional
    # outputs (see flags starting with output_)
    return_dict_in_generate: bool | None = False

    # Output logits manipulation ---------------------------------------------
    # The value used to module the next token probabilities
    temperature: float | None = 1.0
    # The number of highest probability vocabulary tokens to keep for
    # top-k-filtering
    top_k: int | None = 50
    # If set to float < 1, only the smallest set of most probable tokens with
    # probabilities that add up to top_p or higher are kept for generation
    top_p: float | None = 1.0
    # Minimum token probability, which will be scaled by the probability of
    # the most likely token. It must be a value between 0 and 1. Typical
    # values are in the 0.01-0.2 range, comparably selective as setting
    # top_p in the 0.99-0.8 range (use the opposite of normal top_p values)
    min_p: float | None = None
    # The parameter for repetition penalty. 1.0 means no penalty
    repetition_penalty: float | None = 1.0
    # This value is subtracted from a beam’s score if it generates a token
    # same as any beam from other group at a particular time. Note that
    # diversity_penalty is only effective if group beam search is enabled
    diversity_penalty: float | None = 0.0
    # The id of the token to force as the first generated token after the
    # decoder_start_token_id. Useful for multilingual models like mBART
    # where the first generated token needs to be the target language token.
    forced_bos_token_id: int | None = None
    # The id of the token to force as the last generated token when 3
    # max_length is reached. Optionally, use a list to set multiple
    # end-of-sequence tokens
    forced_eos_token_id: int | list[int] | None = None

    # Special token usage in generation --------------------------------------
    # The id of the padding token.
    pad_token_id: int | None = None
    # The id of the beginning-of-sequence token
    bos_token_id: int | None = None
    # The id of the end-of-sequence token. Optionally, use a list to set
    # multiple end-of-sequence tokens
    eos_token_id: int | list[int] | None = None

    # Assistant generation ---------------------------------------------------
    # The number of tokens to be output as candidate tokens.
    prompt_lookup_num_tokens: int | None = None

    # Inference settings -----------------------------------------------------
    # Gradient calculation (set to true when torch.backwards() is called)
    enable_gradient_calculation: bool = False
    # Use async generator (token streaming)
    use_async_generator: bool = True
    # Optional attention mask passed directly to ``generate``
    attention_mask: Tensor | None = None
    # Parameters passed to tokenizer.apply_chat_template
    chat_template_settings: dict[str, object] | None = field(
        default_factory=lambda: {
            "add_generation_prompt": True,
            "tokenize": True,
            "add_special_tokens": True,
            "return_dict": True,
        }
    )


@dataclass(frozen=True, kw_only=True)
class GenericProxyConfig:
    scheme: str
    host: str
    port: int
    username: str | None = None
    password: str | None = None

    def to_dict(self) -> dict[str, str]:
        credentials = (
            f"{self.username}:{self.password}@"
            if self.username and self.password
            else ""
        )
        url = f"{self.scheme}://{credentials}{self.host}:{self.port}"
        return {"http": url, "https": url}


@dataclass(frozen=True, kw_only=True)
class HubCacheFile:
    name: str
    path: str
    size_on_disk: int
    last_accessed: datetime
    last_modified: datetime


@dataclass(frozen=True, kw_only=True)
class HubCache:
    model_id: str
    path: str
    size_on_disk: int
    revisions: list[str]
    files: dict[str, list[HubCacheFile]]
    total_files: int
    total_revisions: int


@dataclass(frozen=True, kw_only=True)
class HubCacheDeletion:
    model_id: str
    revisions: list[str]
    deletable_size_on_disk: float
    deletable_blobs: list[str]
    deletable_refs: list[str]
    deletable_repos: list[str]
    deletable_snapshots: list[str]


@dataclass(frozen=True, kw_only=True)
class ImageEntity:
    label: str
    score: float | None = None
    box: list[float] | None = None


@dataclass(frozen=True, kw_only=True)
class Message:
    role: MessageRole
    content: str
    name: str | None = None
    arguments: dict | None = None


Input = str | list[str] | Message | list[Message]


@dataclass(frozen=True, kw_only=True)
class EngineMessage:
    agent_id: UUID
    model_id: str
    message: Message

    @property
    def is_from_agent(self) -> bool:
        return self.message.role == MessageRole.ASSISTANT


@dataclass(frozen=True, kw_only=True)
class EngineMessageScored(EngineMessage):
    score: float


@dataclass(frozen=True, kw_only=True)
class Model:
    id: str
    parameters: int | None
    parameter_types: list[str] | None
    inference: str | None
    library_name: str | None
    license: str | None
    pipeline_tag: str | None
    tags: list[str]
    architectures: list[str] | None
    model_type: str | None
    auto_model: str | None
    processor: str | None
    gated: Literal["auto", "manual", False] | None
    private: bool
    disabled: bool | None
    last_downloads: int
    downloads: int
    likes: int
    ranking: int | None
    author: str
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, kw_only=True)
class ModelConfig:
    # Model architectures that can be used with the model pretrained weights
    architectures: list[str] | None
    # A dict that maps model specific attribute names to the standardized
    # naming of attributes
    attribute_map: dict[str, str]
    # The id of the beginning-of-stream token
    bos_token_id: str | None
    bos_token: str | None
    # If an encoder-decoder model starts decoding with a different token
    # than bos, the id of that token
    decoder_start_token_id: int | None
    # The id of the end-of-stream token
    eos_token_id: int | None
    eos_token: str | None
    # Name of the task used to fine-tune the model. This can be used when
    # converting from an original (TensorFlow or PyTorch) checkpoint
    finetuning_task: str | None
    # The hidden size of the model
    hidden_size: int | None
    # The hidden sizes of the model (ResNet)
    hidden_sizes: list[int] | None
    # A list of keys to ignore by default when looking at dictionary outputs
    # of the model during inference
    keys_to_ignore_at_inference: list[str]
    # The type of loss that the model should use
    loss_type: str | None
    # Maximum input sequence length
    max_position_embeddings: int | None
    # An identifier for the model type
    model_type: str
    # The number of attention heads used in the multi-head attention layers
    # of the model
    num_attention_heads: int | None
    # The number of blocks in the model
    num_hidden_layers: int | None
    # Number of labels to use in the last layer added to the model, typically
    # for a classification task
    num_labels: int | None
    # Whether or not the model should returns all attentions
    output_attentions: bool
    # Whether or not the model should return all hidden-states
    output_hidden_states: bool
    # The id of the padding token
    pad_token_id: int | None
    pad_token: str | None
    # A specific prompt that should be added at the beginning of each text
    # before calling the model
    prefix: str | None
    # The id of the separation token
    sep_token_id: int | None
    sep_token: str | None
    state_size: int
    # Additional keyword arguments to store for the current task
    task_specific_params: dict[str, any] | None
    # The dtype of the weight. Since the config object is stored in plain
    # text, this attribute contains just the floating type string without the
    # torch
    torch_dtype: dtype
    # The number of tokens in the vocabulary, which is also the first
    # dimension of the embeddings matrix
    vocab_size: int | None
    # The name of the associated tokenizer class to use (if none is set, will
    # use the tokenizer associated to the model by default)
    tokenizer_class: str | None


@dataclass(frozen=True, kw_only=True)
class OrchestratorSettings:
    agent_id: UUID
    orchestrator_type: str | None
    agent_config: dict
    uri: str
    engine_config: dict
    call_options: dict | None
    template_vars: dict | None
    memory_permanent_message: str | None
    permanent_memory: dict[str, str] | None
    memory_recent: bool
    sentence_model_id: str
    sentence_model_engine_config: dict | None
    sentence_model_max_tokens: int
    sentence_model_overlap_size: int
    sentence_model_window_size: int
    json_config: dict | None
    tools: list[str]
    log_events: bool


@dataclass(frozen=True, kw_only=True)
class SearchMatch:
    query: str
    match: str
    l2_distance: float


@dataclass(frozen=True, kw_only=True)
class SentenceTransformerModelConfig:
    backend: Literal["torch", "onnx", "openvino"]
    similarity_function: SimilarityFunction | None
    truncate_dimension: int | None
    transformer_model_config: ModelConfig


@dataclass(frozen=True, kw_only=True)
class Similarity:
    cosine_distance: float
    inner_product: float
    l1_distance: float
    l2_distance: float
    pearson: float


@dataclass(frozen=True, kw_only=True)
class TokenizerConfig:
    name_or_path: str
    tokens: list[str] | None
    special_tokens: list[str] | None
    # Maximum sequence length the tokenizer supports
    tokenizer_model_max_length: int
    # Wether tokenizer is a Rust-based tokenizer
    fast: bool = False


@dataclass(frozen=True, kw_only=True)
class Token:
    id: Tensor | int
    token: str
    probability: float | None = None


@dataclass(frozen=True, kw_only=True)
class TokenDetail(Token):
    step: int | None = None
    probability_distribution: ProbabilityDistribution | None = None
    tokens: list[Token] | None = None


@dataclass(frozen=True, kw_only=True)
class ToolCall:
    id: UUID
    name: str
    arguments: dict[str, ToolValue] | None = None


@dataclass(frozen=True, kw_only=True)
class ToolCallContext:
    agent_id: UUID | None = None
    input: Input | None = None
    participant_id: UUID | None = None
    session_id: UUID | None = None
    calls: list[ToolCall] | None = None


@dataclass(frozen=True, kw_only=True)
class ToolCallResult(ToolCall):
    id: UUID
    call: ToolCall
    result: ToolValue | None = None


@dataclass(frozen=True, kw_only=True)
class ToolManagerSettings:
    eos_token: str | None = None
    tool_format: ToolFormat | None = None
    avoid_repetition: bool = False
    maximum_depth: int | None = None
    filters: (
        list[
            Callable[
                [ToolCall, ToolCallContext],
                tuple[ToolCall, ToolCallContext] | None,
            ]
        ]
        | None
    ) = None
    transformers: (
        list[
            Callable[
                [ToolCall, ToolCallContext, ToolValue | None], ToolValue | None
            ]
        ]
        | None
    ) = None


@dataclass(frozen=True, kw_only=True)
class QuantizationSettings:
    load_in_4bit: bool
    bnb_4bit_quant_type: Literal["nf4"]
    bnb_4bit_use_double_quant: bool
    bnb_4bit_compute_dtype: type


@dataclass(frozen=True, kw_only=True)
class TextPartition:
    data: str
    total_tokens: int
    embeddings: ndarray


@dataclass(frozen=True, kw_only=True)
class TransformerEngineSettings(EngineSettings):
    access_token: str | None = None
    attention: AttentionImplementation | None = None
    base_url: str | None = None
    loader_class: TextGenerationLoaderClass | None = "auto"
    local_files_only: bool = False
    low_cpu_mem_usage: bool = False
    quantization: QuantizationSettings | None = None
    revision: str | None = None
    special_tokens: list[str] | None = None
    state_dict: dict[str, Tensor] = None
    tokens: list[str] | None = None
    weight_type: WeightType = "auto"


@dataclass(frozen=True, kw_only=True)
class User:
    name: str
    full_name: str | None = None
    access_token_name: str | None = None


@dataclass(frozen=True, kw_only=True)
class WebshareProxyConfig:
    host: str
    port: int
    username: str
    password: str
    scheme: str = "http"

    def to_generic(self) -> GenericProxyConfig:
        return GenericProxyConfig(
            scheme=self.scheme,
            host=self.host,
            port=self.port,
            username=self.username,
            password=self.password,
        )

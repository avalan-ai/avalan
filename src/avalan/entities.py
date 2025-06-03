from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from torch import dtype, Tensor
from typing import Literal, Optional, Union
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

ToolValue = Union[bool, float, int, str, None]


class MessageRole(StrEnum):
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"
    USER = "user"


class ToolFormat(StrEnum):
    JSON = "json"
    REACT = "react"
    BRACKET = "bracket"
    OPENAI = "openai"


class DistanceType(StrEnum):
    COSINE = "cosine"
    DOT = "dot"
    L1 = "l1"
    L2 = "l2"
    PEARSON = "pearson"


@dataclass(frozen=True, kw_only=True)
class EngineSettings:
    auto_load_model: bool = True
    auto_load_tokenizer: bool = True
    cache_dir: Optional[str] = None
    change_transformers_logging_level = True
    device: Optional[str] = None
    disable_loading_progress_bar: bool = True
    enable_eval: bool = True
    trust_remote_code: bool = False
    tokenizer_name_or_path: Optional[str] = None


@dataclass(kw_only=True, frozen=True)
class EngineUri:
    host: Optional[str]
    port: Optional[int]
    user: Optional[str]
    password: Optional[str]
    vendor: Optional[Vendor]
    model_id: Optional[str]
    params: dict[str, Union[str, int, float, bool]]

    @property
    def is_local(self) -> bool:
        return not self.vendor or self.vendor == "local"


@dataclass(frozen=True, kw_only=True)
class GenerationSettings:
    # Generation length ------------------------------------------------------
    # The minimum numbers of tokens to generate, ignoring the number of tokens in the prompt.
    min_new_tokens: Optional[int] = None
    # The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt
    max_new_tokens: Optional[int] = None
    # The minimum length of the sequence to be generated. Corresponds to the length of the input prompt + min_new_tokens. Its effect is overridden by min_new_tokens, if also set.
    min_length: Optional[int] = 0
    # The maximum length the generated tokens can have. Corresponds to the length of the input prompt + max_new_tokens. Its effect is overridden by max_new_tokens, if also set.
    max_length: Optional[int] = 20
    # Controls the stopping condition for beam-based methods, like beam-search. It accepts the following values: True, where the generation stops as soon as there are num_beams complete candidates; False, where an heuristic is applied and the generation stops when is it very unlikely to find better candidates; "never", where the beam search procedure only stops when there cannot be better candidates (canonical beam search algorithm)
    early_stopping: Optional[Union[str, bool]] = False
    # The maximum amount of time you allow the computation to run for in seconds. generation will still finish the current pass after allocated time has been passed.
    max_time: Optional[float] = None
    # A string or a list of strings that should terminate generation if the model outputs them.
    stop_strings: Optional[Union[str, list[str]]] = None

    # Generation strategy ----------------------------------------------------
    # Whether or not to use sampling. Use greedy decoding otherwise
    do_sample: bool = False
    # Number of beams for beam search. 1 means no beam search.
    num_beams: Optional[int] = 1
    # Number of groups to divide num_beams into in order to ensure diversity among different groups of beams
    num_beam_groups: Optional[int] = 1
    # The values balance the model confidence and the degeneration penalty in contrastive search decoding
    penalty_alpha: Optional[float] = None

    # Cache ------------------------------------------------------------------
    # Whether or not the model should use the past last key/values attentions (if applicable to the model) to speed up decoding
    use_cache: bool = True

    # Generation output variables --------------------------------------------
    # The number of independently computed returned sequences for each element in the batch.
    num_return_sequences: Optional[int] = 1
    # Whether or not to return the attentions tensors of all attention layers. See attentions under returned tensors for more details.
    output_attentions: Optional[bool] = False
    # Whether or not to return the hidden states of all layers. See hidden_states under returned tensors for more details.
    output_hidden_states: Optional[bool] = False
    # Whether or not to return the prediction scores. See scores under returned tensors for more details.
    output_scores: Optional[bool] = False
    # Whether or not to return the unprocessed prediction logit scores. See logits under returned tensors for more details.
    output_logits: Optional[bool] = None
    # Whether or not to return a ModelOutput, as opposed to returning exclusively the generated sequence. This flag must be set to True to return the generation cache (when use_cache is True) or optional outputs (see flags starting with output_)
    return_dict_in_generate: Optional[bool] = False

    # Output logits manipulation ---------------------------------------------
    # The value used to module the next token probabilities
    temperature: Optional[float] = 1.0
    # The number of highest probability vocabulary tokens to keep for top-k-filtering
    top_k: Optional[int] = 50
    # If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation
    top_p: Optional[float] = 1.0
    # Minimum token probability, which will be scaled by the probability of the most likely token. It must be a value between 0 and 1. Typical values are in the 0.01-0.2 range, comparably selective as setting top_p in the 0.99-0.8 range (use the opposite of normal top_p values)
    min_p: Optional[float] = None
    # The parameter for repetition penalty. 1.0 means no penalty
    repetition_penalty: Optional[float] = 1.0
    # This value is subtracted from a beam’s score if it generates a token same as any beam from other group at a particular time. Note that diversity_penalty is only effective if group beam search is enabled
    diversity_penalty: Optional[float] = 0.0
    # The id of the token to force as the first generated token after the decoder_start_token_id. Useful for multilingual models like mBART where the first generated token needs to be the target language token.
    forced_bos_token_id: Optional[int] = None
    # The id of the token to force as the last generated token when max_length is reached. Optionally, use a list to set multiple end-of-sequence tokens
    forced_eos_token_id: Optional[Union[int, list[int]]] = None

    # Special token usage in generation --------------------------------------
    # The id of the padding token.
    pad_token_id: Optional[int] = None
    # The id of the beginning-of-sequence token
    bos_token_id: Optional[int] = None
    # The id of the end-of-sequence token. Optionally, use a list to set multiple end-of-sequence tokens
    eos_token_id: Optional[Union[int, list[int]]] = None

    # Assistant generation ---------------------------------------------------
    # The number of tokens to be output as candidate tokens.
    prompt_lookup_num_tokens: Optional[int] = None

    # Inference settings -----------------------------------------------------
    # Gradient calculation (set to true when torch.backwards() is called)
    enable_gradient_calculation: bool = False
    # Use async generator (token streaming)
    use_async_generator: bool = True


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
    score: Optional[float] = None
    box: Optional[list[float]] = None


@dataclass(frozen=True, kw_only=True)
class Message:
    role: MessageRole
    content: str
    name: Optional[str] = None
    arguments: Optional[dict] = None


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
    parameters: Optional[int]
    parameter_types: Optional[list[str]]
    inference: Optional[str]
    library_name: Optional[str]
    license: Optional[str]
    pipeline_tag: Optional[str]
    tags: list[str]
    architectures: Optional[list[str]]
    model_type: Optional[str]
    auto_model: Optional[str]
    processor: Optional[str]
    gated: Optional[Literal["auto", "manual", False]]
    private: bool
    disabled: Optional[bool]
    last_downloads: int
    downloads: int
    likes: int
    ranking: Optional[int]
    author: str
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, kw_only=True)
class ModelConfig:
    # Model architectures that can be used with the model pretrained weights
    architectures: Optional[list[str]]
    # A dict that maps model specific attribute names to the standardized naming of attributes
    attribute_map: dict[str, str]
    # The id of the beginning-of-stream token
    bos_token_id: Optional[str]
    bos_token: Optional[str]
    # If an encoder-decoder model starts decoding with a different token than bos, the id of that token
    decoder_start_token_id: Optional[int]
    # The id of the end-of-stream token
    eos_token_id: Optional[int]
    eos_token: Optional[str]
    # Name of the task used to fine-tune the model. This can be used when converting from an original (TensorFlow or PyTorch) checkpoint
    finetuning_task: Optional[str]
    # The hidden size of the model
    hidden_size: Optional[int]
    # The hidden sizes of the model (ResNet)
    hidden_sizes: Optional[list[int]]
    # A list of keys to ignore by default when looking at dictionary outputs of the model during inference
    keys_to_ignore_at_inference: list[str]
    # The type of loss that the model should use
    loss_type: Optional[str]
    # Maximum input sequence length
    max_position_embeddings: Optional[int]
    # An identifier for the model type
    model_type: str
    # The number of attention heads used in the multi-head attention layers of the model
    num_attention_heads: Optional[int]
    # The number of blocks in the model
    num_hidden_layers: Optional[int]
    # Number of labels to use in the last layer added to the model, typically for a classification task
    num_labels: Optional[int]
    # Whether or not the model should returns all attentions
    output_attentions: bool
    # Whether or not the model should return all hidden-states
    output_hidden_states: bool
    # The id of the padding token
    pad_token_id: Optional[int]
    pad_token: Optional[str]
    # A specific prompt that should be added at the beginning of each text before calling the model
    prefix: Optional[str]
    # The id of the separation token
    sep_token_id: Optional[int]
    sep_token: Optional[str]
    state_size: int
    # Additional keyword arguments to store for the current task
    task_specific_params: Optional[dict[str, any]]
    # The dtype of the weight. Since the config object is stored in plain text, this attribute contains just the floating type string without the torcha
    torch_dtype: dtype
    # The number of tokens in the vocabulary, which is also the first dimension of the embeddings matrix
    vocab_size: Optional[int]
    # The name of the associated tokenizer class to use (if none is set, will use the tokenizer associated to the model by default)
    tokenizer_class: Optional[str]


@dataclass(frozen=True, kw_only=True)
class SearchMatch:
    query: str
    match: str
    l2_distance: float


@dataclass(frozen=True, kw_only=True)
class SentenceTransformerModelConfig:
    backend: Literal["torch", "onnx", "openvino"]
    similarity_function: Optional[SimilarityFunction]
    truncate_dimension: Optional[int]
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
    tokens: Optional[list[str]]
    special_tokens: Optional[list[str]]
    # Maximum sequence length the tokenizer supports
    tokenizer_model_max_length: int
    # Wether tokenizer is a Rust-based tokenizer
    fast: bool = False


@dataclass(frozen=True, kw_only=True)
class Token:
    id: Union[Tensor, int]
    token: str
    probability: Optional[float] = None


@dataclass(frozen=True, kw_only=True)
class TokenDetail(Token):
    step: Optional[int] = None
    probability_distribution: Optional[ProbabilityDistribution] = None
    tokens: Optional[list[Token]] = None


@dataclass(frozen=True, kw_only=True)
class ToolCall:
    id: UUID
    name: str
    arguments: Optional[dict[str, ToolValue]] = None


@dataclass(frozen=True, kw_only=True)
class ToolCallResult(ToolCall):
    id: UUID
    call: ToolCall
    result: Optional[ToolValue] = None


@dataclass(frozen=True, kw_only=True)
class QuantizationSettings:
    load_in_4bit: bool
    bnb_4bit_quant_type: Literal["nf4"]
    bnb_4bit_use_double_quant: bool
    bnb_4bit_compute_dtype: type


@dataclass(frozen=True, kw_only=True)
class TransformerEngineSettings(EngineSettings):
    access_token: Optional[str] = None
    attention: Optional[AttentionImplementation] = None
    base_url: Optional[str] = None
    loader_class: Optional[TextGenerationLoaderClass] = "auto"
    local_files_only: bool = False
    low_cpu_mem_usage: bool = False
    quantization: Optional[QuantizationSettings] = None
    revision: Optional[str] = None
    special_tokens: Optional[list[str]] = None
    state_dict: dict[str, Tensor] = None
    tokens: Optional[list[str]] = None
    weight_type: WeightType = "auto"


@dataclass(frozen=True, kw_only=True)
class User:
    name: str
    full_name: Optional[str] = None
    access_token_name: Optional[str] = None


Input = Union[str, list[str], Message, list[Message]]

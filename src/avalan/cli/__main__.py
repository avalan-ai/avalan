from argparse import ArgumentParser, Namespace, _SubParsersAction
import sys
from asyncio import run as run_in_loop
from asyncio.exceptions import CancelledError
from torch.cuda import device_count, is_available, set_device
from torch.distributed import destroy_process_group
from .. import license, name, site, version
from ..cli import CommandAbortException, has_input
from ..cli.commands.agent import (
    agent_message_search,
    agent_run,
    agent_serve,
    agent_init,
)
from ..cli.commands.cache import cache_delete, cache_download, cache_list
from ..cli.commands.memory import (
    memory_document_index,
    memory_embeddings,
    memory_search,
)
from ..cli.commands.model import (
    model_display,
    model_install,
    model_run,
    model_search,
    model_uninstall,
)
from ..cli.commands.tokenizer import tokenize
from ..cli.commands.deploy import deploy_run
from ..cli.theme.fancy import FancyTheme
from ..entities import (
    AttentionImplementation,
    DistanceType,
    Modality,
    ParallelStrategy,
    TextGenerationLoaderClass,
    User,
    WeightType,
)
from ..memory.permanent import VectorFunction
from ..model.hubs.huggingface import HuggingfaceHub
from ..model.manager import ModelManager
from ..model.transformer import TransformerModel
from ..agent.loader import OrchestratorLoader
from ..utils import logger_replace
import gettext
from gettext import translation
from importlib.util import find_spec
from locale import getlocale
from logging import basicConfig, DEBUG, Filter, getLogger, INFO, Logger, LogRecord, WARNING
from os import getenv, environ
from subprocess import run
from os.path import join
from pathlib import Path
from rich.console import Console
from rich.logging import RichHandler
from rich.prompt import Confirm, Prompt
from rich.theme import Theme
from transformers.utils import logging as hf_logging
from typing import get_args, Optional, get_origin, get_args as get_type_args
from dataclasses import fields
from ..tool.browser import BrowserToolSettings
from uuid import uuid4
from warnings import filterwarnings


class CLI:
    _REFRESH_RATE = 4

    def __init__(self, logger: Logger):
        self._name = name()
        self._site = site()
        self._version = version()
        self._license = license()
        self._logger = logger

        cache_dir = HuggingfaceHub.DEFAULT_CACHE_DIR
        default_locale, _ = getlocale()
        default_locales_path = join(
            Path(__file__).resolve().parents[3], "locale"
        )
        default_device = TransformerModel.get_default_device()
        self._parser = CLI._create_parser(
            default_device, cache_dir, default_locales_path, default_locale
        )

    @staticmethod
    def _default_parallel_count() -> int:
        return device_count() if is_available() else 1

    @staticmethod
    def _create_parser(
        default_device: str,
        cache_dir: str,
        default_locales_path: str,
        default_locale: str,
    ):
        global_parser = ArgumentParser(add_help=False)
        global_parser.add_argument(
            "--cache-dir",
            default=cache_dir,
            type=str,
            help=(
                f"Path to huggingface cache hub (defaults to {cache_dir}, "
                "can also be specified with $HF_HUB_CACHE)"
            ),
        )
        global_parser.add_argument(
            "--device",
            type=str,
            required=False,
            default=default_device,
            help="Device to use (cpu, cuda, mps). Defaults to "
            + default_device,
        )
        global_parser.add_argument(
            "--parallel",
            type=str,
            choices=[p.value for p in ParallelStrategy],
            help="Tensor parallelism strategy to use",
        )
        global_parser.add_argument(
            "--parallel-count",
            type=int,
            default=CLI._default_parallel_count(),
            help=(
                "Number of processes to launch when --parallel is used "
                "(defaults to the number of available GPUs)"
            ),
        )
        global_parser.add_argument(
            "--disable-loading-progress-bar",
            action="store_true",
            default=False,
            help=(
                "If specified, the shard loading progress bar "
                "will not be shown"
            ),
        )
        global_parser.add_argument(
            "--hf-token",
            type=str,
            default=getenv("HF_TOKEN"),
            help="Your Huggingface access token",
        )
        global_parser.add_argument(
            "--locale",
            type=str,
            default=default_locale,
            help=f"Language to use (defaults to {default_locale})",
        )
        global_parser.add_argument(
            "--loader-class",
            type=str,
            default="auto",
            choices=get_args(TextGenerationLoaderClass),
            help='Loader class to use (defaults to "auto")',
        )
        global_parser.add_argument(
            "--locales",
            type=str,
            default=default_locales_path,
            help=f"Path to locale files (defaults to {default_locales_path})",
        )
        global_parser.add_argument(
            "--low-cpu-mem-usage",
            action="store_true",
            default=False,
            help=(
                "If specified, loads the model using ~1x model size CPU memory"
            ),
        )
        global_parser.add_argument(
            "--login",
            action="store_true",
            help="Login to main hub (huggingface)",
        )
        global_parser.add_argument(
            "--no-repl",
            action="store_true",
            help="Don't echo input coming from stdin",
        )
        global_parser.add_argument(
            "--quiet",
            "-q",
            default=False,
            action="store_true",
            help=(
                "If specified, no welcome screen and only model output is "
                "displayed in model run (sets "
            )
            + ", ".join(
                [
                    "--disable-loading-progress-bar",
                    "--skip-hub-access-check",
                    "--skip-special-tokens",
                ]
            )
            + " automatically)",
        )
        global_parser.add_argument(
            "--record",
            action="store_true",
            default=False,
            help=(
                "If specified, the current console output will be regularly "
                "saved to SVG files."
            ),
        )
        global_parser.add_argument(
            "--revision",
            type=str,
            help="Model revision to use",
        )
        global_parser.add_argument(
            "--skip-hub-access-check",
            action="store_true",
            default=False,
            help="If specified, skip hub model access check",
        )
        global_parser.add_argument(
            "--verbose", "-v", action="count", help="Set verbosity"
        )
        global_parser.add_argument(
            "--version",
            action="store_true",
            help="Display this program's version, and exit",
        )

        global_parser.add_argument(
            "--weight-type",
            type=str,
            choices=get_args(WeightType),
            help="Weight type to use (defaults to best available)",
        )

        parser = ArgumentParser(
            description="Avalan CLI", parents=[global_parser]
        )

        command_parsers = parser.add_subparsers(dest="command")

        # Memory options shared by commands: memory embeddings, memory document
        memory_partitions_parser = ArgumentParser(add_help=False)
        memory_partitions_display_group = (
            memory_partitions_parser.add_mutually_exclusive_group()
        )
        memory_partitions_display_group.add_argument(
            "--no-display-partitions",
            action="store_true",
            default=False,
            help="If specified, don't display memory partitions",
        )
        memory_partitions_display_group.add_argument(
            "--display-partitions",
            default=6,
            type=int,
            help="Display up to this many partitions, if more summarize",
        )
        memory_partitions_parser.add_argument(
            "--partition",
            action="store_true",
            default=False,
            help="If specified, partition string",
        )
        memory_partitions_parser.add_argument(
            "--partition-max-tokens",
            default=500,
            type=int,
            help="Maximum number of tokens to allow on each partition",
        )
        memory_partitions_parser.add_argument(
            "--partition-overlap",
            default=125,
            type=int,
            help=(
                "How many tokens can potentially overlap in "
                "different partitions"
            ),
        )
        memory_partitions_parser.add_argument(
            "--partition-window",
            default=250,
            type=int,
            help="Number of tokens per window when partitioning",
        )

        # Model options shared by commands: memory embeddings, model
        model_options_parser = ArgumentParser(add_help=False)
        model_options_parser.add_argument(
            "model",
            type=str,
            help="Model to use",
        )
        model_options_parser.add_argument(
            "--base-url",
            type=str,
            help=(
                "If specified and model is a vendor model that supports it,"
                "load model using the given base URL"
            ),
        )
        model_options_parser.add_argument(
            "--load",
            action="store_true",
            help="If specified, load model and show more information",
        )
        model_options_parser.add_argument(
            "--special-token",
            type=str,
            action="append",
            help=(
                "Special token to add to tokenizer, only when model is loaded"
            ),
        )
        model_options_parser.add_argument(
            "--token",
            type=str,
            action="append",
            help="Token to add to tokenizer, only when model is loaded",
        )
        model_options_parser.add_argument(
            "--tokenizer",
            type=str,
            help=(
                "Path to tokenizer to use instead of model's default, only "
                "if model is loaded"
            ),
        )

        # Inference options shared by commands: agent run, model run
        model_inference_display_parser = ArgumentParser(add_help=False)
        model_inference_display_parser.add_argument(
            "--display-events",
            action="store_true",
            help=(
                "If --display-events is specified and there's an orchestrator"
                " / agent involved, show the events panel."
            ),
        )
        model_inference_display_parser.add_argument(
            "--display-pause",
            type=int,
            nargs="?",
            const=500,  # 500 is the default if argument present but no value
            default=None,
            help=(
                "Pause (in ms.) when cycling through selected tokens as "
                "defined by --display-probabilities"
            ),
        )
        model_inference_display_parser.add_argument(
            "--display-probabilities",
            action="store_true",
            help=(
                "If --display-tokens specified, show also the token "
                "probability distribution"
            ),
        )
        model_inference_display_parser.add_argument(
            "--display-probabilities-maximum",
            type=float,
            default=0.8,
            help=(
                "When --display-probabilities is used, select tokens which "
                "logit probability is no higher than this value. "
                "Defaults to 0.8"
            ),
        )
        model_inference_display_parser.add_argument(
            "--display-probabilities-sample-minimum",
            type=float,
            default=0.1,
            help=(
                "When --display-probabilities is used, select tokens that "
                "have alternate tokens with a logit probability at least or "
                "higher than this value. Defaults to 0.1"
            ),
        )
        model_inference_display_parser.add_argument(
            "--display-time-to-n-token",
            type=int,
            nargs="?",
            const=256,  # 256 is the default if argument present but no value
            default=None,
            help=(
                "Display the time it takes to reach the given Nth token "
                "(defaults to 256)"
            ),
        )
        model_inference_display_parser.add_argument(
            "--display-tokens",
            type=int,
            nargs="?",
            const=15,  # 15 is the default if argument present but no value
            default=None,
            help="How many tokens with full information to display at a time",
        )
        model_inference_display_parser.add_argument(
            "--display-tools",
            action="store_true",
            help=(
                "If --display-events is specified and there's an orchestrator"
                " / agent involved, show the events panel."
            ),
        )
        model_inference_display_parser.add_argument(
            "--display-tools-events",
            type=int,
            default=2,
            help="How many tool events to show on tool call panel",
        )

        # Agent command
        agent_parser = command_parsers.add_parser(
            name="agent",
            description="Manage AI agents",
            parents=[global_parser],
        )
        agent_command_parsers = agent_parser.add_subparsers(
            dest="agent_command"
        )

        agent_message_parser = agent_command_parsers.add_parser(
            name="message",
            description="Manage AI agent messages",
            parents=[global_parser],
        )
        agent_message_command_parsers = agent_message_parser.add_subparsers(
            dest="agent_message_command"
        )

        agent_message_search_parser = agent_message_command_parsers.add_parser(
            name="search",
            description="Search within an agent's message memory",
            parents=[global_parser],
        )
        agent_message_search_parser.add_argument(
            "specifications_file",
            type=str,
            nargs="?",
            help="File that holds the agent specifications",
        )
        agent_message_search_parser.add_argument(
            "--function",
            type=VectorFunction,
            choices=list(VectorFunction),
            required=True,
            default=VectorFunction.L2_DISTANCE,
            help="Vector function to use for searching",
        )
        agent_message_search_parser.add_argument(
            "--id", type=str, required=True
        )
        agent_message_search_parser.add_argument(
            "--limit",
            type=int,
            help="If specified, load up to these many recent messages",
        )
        agent_message_search_parser.add_argument(
            "--participant",
            type=str,
            required=True,
            help="Search messages with given participant",
        )
        agent_message_search_parser.add_argument(
            "--session",
            type=str,
            required=True,
            help="Search within the given session",
        )
        CLI._add_agent_settings_arguments(agent_message_search_parser)
        CLI._add_tool_settings_arguments(
            agent_message_search_parser,
            prefix="browser",
            settings_cls=BrowserToolSettings,
        )

        agent_run_parser = agent_command_parsers.add_parser(
            name="run",
            description="Run an AI agent",
            parents=[global_parser, model_inference_display_parser],
        )
        agent_run_parser.add_argument(
            "specifications_file",
            type=str,
            nargs="?",
            help="File that holds the agent specifications",
        )
        agent_run_parser.add_argument(
            "--conversation",
            action="store_true",
            default=False,
            help="Activate conversation mode with the agent",
        )
        agent_run_parser.add_argument(
            "--watch",
            action="store_true",
            default=False,
            help=(
                "Reload agent when the specification file changes "
                "(only with --conversation)"
            ),
        )
        agent_run_parser.add_argument(
            "--id", type=str, help="Use given ID as the agent ID"
        )
        agent_session_group = agent_run_parser.add_mutually_exclusive_group()
        agent_session_group.add_argument(
            "--no-session",
            action="store_true",
            default=False,
            help=(
                "If specified, don't use sessions in persistent message memory"
            ),
        )
        agent_session_group.add_argument(
            "--session",
            type=str,
            help="Continue the conversation on the given session",
        )

        agent_run_parser.add_argument(
            "--skip-load-recent-messages",
            default=False,
            action="store_true",
            help="If specified, skips loading recent messages",
        )
        agent_run_parser.add_argument(
            "--load-recent-messages-limit",
            type=int,
            help="If specified, load up to these many recent messages",
        )
        agent_run_parser.add_argument(
            "--participant",
            default=uuid4(),
            help=(
                "If specified, this is the participant ID interacting with "
                "the agent"
            ),
        )
        agent_run_parser.add_argument(
            "--stats",
            action="store_true",
            default=False,
            help="Show token generation statistics for agent output",
        )
        agent_run_parser.add_argument(
            "--sync",
            dest="use_sync_generator",
            action="store_true",
            default=False,
            help="Don't use an async generator (token streaming)",
        )
        agent_run_parser.add_argument(
            "--tty",
            default="/dev/tty",
            help=(
                "TTY stream (only applicable if combining --conversation "
                "with input piping)"
            ),
        )
        agent_run_parser.add_argument(
            "--tools-confirm",
            action="store_true",
            help="Confirm tool calls before execution",
        )

        CLI._add_agent_settings_arguments(agent_run_parser)
        CLI._add_tool_settings_arguments(
            agent_run_parser,
            prefix="browser",
            settings_cls=BrowserToolSettings,
        )

        agent_serve_parser = agent_command_parsers.add_parser(
            name="serve",
            description="Serve an AI agent as an API endpoint",
            parents=[global_parser],
        )
        agent_serve_parser.add_argument(
            "specifications_file",
            type=str,
            nargs="?",
            help="File that holds the agent specifications",
        )
        agent_serve_parser.add_argument(
            "--host",
            default="127.0.0.1",
            type=str,
            help="Host (defaults to 127.0.0.1)",
        )
        agent_serve_parser.add_argument(
            "--port",
            default=9001,
            type=int,
            help="Port (defaults to 9001, HAL 9000+1)",
        )
        agent_serve_parser.add_argument(
            "--prefix-mcp",
            default="/mcp",
            type=str,
            help="URL prefix for MCP endpoints (defaults to /mcp)",
        )
        agent_serve_parser.add_argument(
            "--prefix-openai",
            default="/v1",
            type=str,
            help="URL prefix fir OpenAI endpoints (defaults to /v1)",
        )
        agent_serve_parser.add_argument(
            "--reload",
            action="store_true",
            default=False,
            help="Hot reload on code changes",
        )

        CLI._add_agent_settings_arguments(agent_serve_parser)
        CLI._add_tool_settings_arguments(
            agent_serve_parser,
            prefix="browser",
            settings_cls=BrowserToolSettings,
        )

        agent_init_parser = agent_command_parsers.add_parser(
            name="init",
            description="Create an agent definition",
            parents=[global_parser],
        )
        CLI._add_agent_settings_arguments(agent_init_parser)

        # Cache command
        cache_parser = command_parsers.add_parser(
            name="cache",
            description="Manage models cache",
            parents=[global_parser],
        )
        cache_command_parsers = cache_parser.add_subparsers(
            dest="cache_command"
        )
        cache_delete_parser = cache_command_parsers.add_parser(
            name="delete",
            description="Delete cached model data",
            parents=[global_parser],
        )
        cache_delete_parser.add_argument(
            "--delete",
            action="store_true",
            help=(
                "Actually delete. If not provided, a dry run is performed "
                "and data that would be deleted is shown, yet not deleted"
            ),
        )
        cache_delete_parser.add_argument(
            "--model",
            "-m",
            type=str,
            required=True,
            help="Model to delete",
        )
        cache_delete_parser.add_argument(
            "--delete-revision",
            type=str,
            action="append",
            help="Revision to delete",
        )
        cache_download_parser = cache_command_parsers.add_parser(
            name="download",
            description="Download model data to cache",
            parents=[global_parser],
        )
        cache_download_parser.add_argument(
            "--model",
            "-m",
            type=str,
            required=True,
            help="Model to load",
        )
        cache_list_parser = cache_command_parsers.add_parser(
            name="list",
            description="List cache contents",
            parents=[global_parser],
        )
        cache_list_parser.add_argument(
            "--model",
            type=str,
            action="append",
            help="Models to show content details on",
        )
        cache_list_parser.add_argument(
            "--summary",
            action="store_true",
            help=(
                "If specified, when showing one or more models show only "
                "summary"
            ),
        )

        # Deploy command
        deploy_parser = command_parsers.add_parser(
            name="deploy",
            description="Manage AI deployments",
            parents=[global_parser],
        )
        deploy_command_parsers = deploy_parser.add_subparsers(
            dest="deploy_command"
        )
        deploy_run_parser = deploy_command_parsers.add_parser(
            name="run",
            description="Perform a deployment",
            parents=[global_parser],
        )
        deploy_run_parser.add_argument(
            "deployment",
            type=str,
            help="Deployment to run",
        )

        # Flow command
        flow_parser = command_parsers.add_parser(
            name="flow", description="Manage AI flows", parents=[global_parser]
        )
        flow_command_parsers = flow_parser.add_subparsers(dest="flow_command")
        flow_run_parser = flow_command_parsers.add_parser(
            name="run", description="Run a given flow", parents=[global_parser]
        )
        flow_run_parser.add_argument(
            "flow",
            type=str,
            help="Flow to run",
        )

        # Memory command
        memory_parser = command_parsers.add_parser(
            name="memory", description="Manage memory", parents=[global_parser]
        )
        memory_command_parsers = memory_parser.add_subparsers(
            dest="memory_command"
        )
        memory_embeddings_parser = memory_command_parsers.add_parser(
            name="embeddings",
            description="Obtain and manipulate embeddings",
            parents=[
                global_parser,
                model_options_parser,
                memory_partitions_parser,
            ],
        )
        memory_embeddings_parser.add_argument(
            "--compare",
            type=str,
            action="append",
            help="If specified, compare embeddings with this string",
        )
        memory_embeddings_parser.add_argument(
            "--search",
            type=str,
            action="append",
            help="If specified, search across embeddings for this string",
        )
        memory_embeddings_parser.add_argument(
            "--search-k",
            default=1,
            type=int,
            help="How many nearest neighbors to obtain with search",
        )
        memory_embeddings_parser.add_argument(
            "--sort",
            type=DistanceType,
            choices=list(DistanceType),
            default=DistanceType.L2,
            help="Sort comparison results using the given similarity measure",
        )

        memory_search_parser = memory_command_parsers.add_parser(
            name="search",
            description="Search memories",
            parents=[
                global_parser,
                model_options_parser,
                memory_partitions_parser,
            ],
        )
        memory_search_parser.add_argument(
            "--dsn",
            type=str,
            required=True,
            help="PostgreSQL DSN for searching",
        )
        memory_search_parser.add_argument(
            "--participant",
            type=str,
            required=True,
            help="Participant ID to search",
        )
        memory_search_parser.add_argument(
            "--namespace", type=str, required=True, help="Namespace to search"
        )
        memory_search_parser.add_argument(
            "--function",
            type=VectorFunction,
            choices=list(VectorFunction),
            required=True,
            default=VectorFunction.L2_DISTANCE,
            help="Vector function to use for searching",
        )
        memory_search_parser.add_argument(
            "--limit", type=int, help="Return up to this many memories"
        )
        memory_doc_parser = memory_command_parsers.add_parser(
            name="document",
            description="Manage memory indexed documents",
        )
        memory_doc_command_parsers = memory_doc_parser.add_subparsers(
            dest="memory_document_command"
        )
        memory_doc_index_parser = memory_doc_command_parsers.add_parser(
            name="index",
            description="Add a document to the memory index",
            parents=[
                global_parser,
                model_options_parser,
                memory_partitions_parser,
            ],
        )
        memory_doc_index_parser.add_argument(
            "source",
            type=str,
            help="Source to index (an URL or a file path)",
        )
        memory_doc_index_parser.add_argument(
            "--partitioner",
            choices=["text", "code"],
            default="text",
            help="Partitioner to use when indexing a file",
        )
        memory_doc_index_parser.add_argument(
            "--language",
            type=str,
            help="Programming language for the code partitioner",
        )
        memory_doc_index_parser.add_argument(
            "--encoding",
            type=str,
            default="utf-8",
            help="File encoding used when reading a local file",
        )
        memory_doc_index_parser.add_argument(
            "--identifier",
            type=str,
            help="Identifier for the memory entry (defaults to the source)",
        )
        memory_doc_index_parser.add_argument(
            "--dsn",
            type=str,
            required=True,
            help="PostgreSQL DSN for storing the document",
        )
        memory_doc_index_parser.add_argument(
            "--participant",
            type=str,
            required=True,
            help="Participant ID for the memory entry",
        )
        memory_doc_index_parser.add_argument(
            "--namespace",
            type=str,
            required=True,
            help="Namespace for the memory entry",
        )

        # Model command
        model_parser = command_parsers.add_parser(
            name="model",
            description=(
                "Manage a model, showing details, loading or downloading it"
            ),
        )
        model_command_parsers = model_parser.add_subparsers(
            dest="model_command"
        )

        model_display_parser = model_command_parsers.add_parser(
            name="display",
            description="Show information about a model",
            parents=[global_parser, model_options_parser],
        )
        model_display_parser.add_argument(
            "--sentence-transformer",
            help="Load the model as a SentenceTransformer model",
            default=False,
            action="store_true",
        )
        model_display_parser.add_argument(
            "--summary",
            default=False,
            action="store_true",
        )

        model_command_parsers.add_parser(
            name="install",
            description="Install a model",
            parents=[global_parser, model_options_parser],
        )
        model_run_parser = model_command_parsers.add_parser(
            name="run",
            description="Run a model",
            parents=[
                global_parser,
                model_options_parser,
                model_inference_display_parser,
            ],
        )
        model_run_parser.add_argument(
            "--attention",
            type=str,
            choices=get_args(AttentionImplementation),
            help=(
                "Attention implementation to use (defaults to best available)"
            ),
        )
        model_run_parser.add_argument(
            "--audio-path",
            type=str,
            help=(
                "Path where to store generated audio. "
                "Only applicable to audio modalities."
            ),
        )
        model_run_parser.add_argument(
            "--audio-reference-path",
            type=str,
            help=(
                "Path to existing audio file to use for voice cloning. "
                "Only applicable to audio modalities."
            ),
        )
        model_run_parser.add_argument(
            "--audio-reference-text",
            type=str,
            help=(
                "Text transcript of the reference audio given in "
                "--audio-reference-path. "
                "Only applicable to audio modalities."
            ),
        )
        model_run_parser.add_argument(
            "--audio-sampling-rate",
            default=44_100,
            type=int,
            help=(
                "Sampling rate to use for audio generation. "
                "Only applicable to audio modalities."
            ),
        )
        model_run_parser.add_argument(
            "--do-sample",
            default=True,
            action="store_true",
            help=(
                "Tell if the token generation process should be "
                "deterministic or stochastic. When enabled, it's stochastic "
                "and uses probability distribution."
            ),
        )
        model_run_parser.add_argument(
            "--enable-gradient-calculation",
            default=False,
            action="store_true",
            help="Enable gradient calculation.",
        )
        model_run_parser.add_argument(
            "--use-cache",
            default=False,
            action="store_true",
            help=(
                "Past key values are used to speed up decoding if applicable "
                "to model."
            ),
        )
        model_run_parser.add_argument(
            "--max-new-tokens",
            default=10 * 1024,
            type=int,
            help="Maximum number of tokens to generate",
        )
        model_run_parser.add_argument(
            "--modality",
            default=Modality.TEXT_GENERATION,
            type=str,
            choices=[m.value for m in Modality],
        )
        model_run_parser.add_argument(
            "--min-p",
            type=float,
            help=(
                "Minimum token probability, which will be scaled by the "
                "probability of the most likely token [0, 1]"
            ),
        )
        model_run_parser.add_argument(
            "--repetition-penalty",
            default=1.0,
            type=float,
            help=(
                "Exponential penalty on sequences not in the original input. "
                "Defaults to 1.0, which means no penalty."
            ),
        )
        model_run_parser.add_argument(
            "--skip-special-tokens",
            default=False,
            action="store_true",
            help="If specified, skip special tokens when decoding",
        )
        model_run_parser.add_argument(
            "--system",
            type=str,
            help="Use this as system prompt",
        )
        model_run_parser.add_argument(
            "--start-thinking",
            default=False,
            action="store_true",
            help="If specified, assume model response starts with reasoning",
        )
        model_run_parser.add_argument(
            "--stop_on_keyword",
            type=str,
            action="append",
            help="Stop token generation when this keyword is found",
        )
        model_run_parser.add_argument(
            "--temperature",
            default=0.7,
            type=float,
            help="Temperature [0, 1]",
        )
        model_run_parser.add_argument(
            "--top-k",
            type=int,
            help=(
                "Number of highest probability vocabulary tokens to keep for "
                "top-k-filtering."
            ),
        )
        model_run_parser.add_argument(
            "--top-p",
            type=float,
            help=(
                "If set to < 1, only the smallest set of most probable "
                "tokens with probabilities that add up to top_p or higher "
                "are kept for generation."
            ),
        )
        model_run_parser.add_argument(
            "--trust-remote-code",
            action="store_true",
        )
        model_search_parser = model_command_parsers.add_parser(
            name="search",
            description="Search for models",
            parents=[global_parser],
        )
        model_search_parser.add_argument(
            "--search",
            type=str,
            required=False,
            help="Search for models matching given expression",
        )
        model_search_parser.add_argument(
            "--filter",
            type=str,
            help="Filter models on this (e.g: text-classification)",
        )
        model_search_parser.add_argument(
            "--limit",
            default=10,
            type=int,
            help="Maximum number of models to return",
        )
        model_uninstall_parser = model_command_parsers.add_parser(
            name="uninstall",
            description="Uninstall a model",
            parents=[global_parser, model_options_parser],
        )
        model_uninstall_parser.add_argument(
            "--delete",
            action="store_true",
            help=(
                "Actually delete. If not provided, a dry run is performed "
                "and data that would be deleted is shown, yet not deleted"
            ),
        )

        # Tokenizer command
        tokenizer_parser = command_parsers.add_parser(
            name="tokenizer",
            description=(
                "Manage tokenizers, loading, modifying and saving them"
            ),
            parents=[global_parser],
        )
        tokenizer_parser.add_argument(
            "--tokenizer",
            "-t",
            type=str,
            required=True,
            help="Tokenizer to load",
        )
        tokenizer_parser.add_argument(
            "--save",
            type=str,
            help=(
                "Save tokenizer (useful if modified via --special-token or "
                "--token) to given path, only if model is loaded"
            ),
        )
        tokenizer_parser.add_argument(
            "--special-token",
            type=str,
            action="append",
            help="Special token to add to tokenizer",
        )
        tokenizer_parser.add_argument(
            "--token",
            type=str,
            action="append",
            help="Token to add to tokenizer",
        )

        # Train command
        train_parser = command_parsers.add_parser(
            name="train", description="Training", parents=[global_parser]
        )
        train_command_parsers = train_parser.add_subparsers(
            dest="train_command"
        )
        train_run_parser = train_command_parsers.add_parser(
            name="run",
            description="Run a given training",
            parents=[global_parser],
        )
        train_run_parser.add_argument(
            "training",
            type=str,
            help="Training to run",
        )

        parser.add_argument(
            "--help-full",
            action="store_true",
            help="Show help for all commands and subcommands",
        )

        return parser

    @staticmethod
    def _extract_chat_template_settings(
        argv: list[str],
    ) -> tuple[list[str], dict[str, bool]]:
        options: dict[str, bool] = {}
        new_argv: list[str] = []
        for arg in argv:
            if arg.startswith("--run-chat-"):
                key = arg[len("--run-chat-") :].replace("-", "_")
                options[key] = True
            else:
                new_argv.append(arg)
        return new_argv, options

    @staticmethod
    def _add_agent_settings_arguments(
        parser: ArgumentParser,
    ) -> ArgumentParser:
        group = parser.add_argument_group("inline agent settings")
        group.add_argument("--engine-uri", type=str, help="Agent engine URI")
        group.add_argument("--name", type=str, help="Agent name")
        group.add_argument("--role", type=str, help="Agent role")
        group.add_argument("--task", type=str, help="Agent task")
        group.add_argument(
            "--instructions", type=str, help="Agent instructions"
        )
        group.add_argument(
            "--memory-recent",
            dest="memory_recent",
            action="store_true",
            default=None,
        )
        group.add_argument(
            "--no-memory-recent", dest="memory_recent", action="store_false"
        )
        group.add_argument(
            "--memory-permanent-message",
            type=str,
            help="Permanent message memory DSN",
        )
        group.add_argument(
            "--memory-permanent",
            action="append",
            dest="memory_permanent",
            help="Permanent memory definition namespace@dsn",
        )
        group.add_argument(
            "--memory-engine-model-id",
            type=str,
            default=OrchestratorLoader.DEFAULT_SENTENCE_MODEL_ID,
            help="Sentence transformer model for memory",
        )
        group.add_argument(
            "--memory-engine-max-tokens",
            type=int,
            default=OrchestratorLoader.DEFAULT_SENTENCE_MODEL_MAX_TOKENS,
            help="Maximum tokens for memory sentence transformer",
        )
        group.add_argument(
            "--memory-engine-overlap",
            type=int,
            default=OrchestratorLoader.DEFAULT_SENTENCE_MODEL_OVERLAP_SIZE,
            help="Overlap size for memory sentence transformer",
        )
        group.add_argument(
            "--memory-engine-window",
            type=int,
            default=OrchestratorLoader.DEFAULT_SENTENCE_MODEL_WINDOW_SIZE,
            help="Window size for memory sentence transformer",
        )
        group.add_argument(
            "--run-max-new-tokens",
            type=int,
            help="Maximum count of tokens on output",
            default=None,
        )
        group.add_argument(
            "--run-skip-special-tokens",
            action="store_true",
            default=False,
            help="Skip special tokens on output",
        )
        group.add_argument(
            "--run-temperature",
            default=0.7,
            type=float,
            help="Temperature [0, 1]",
        )
        group.add_argument(
            "--run-top-k",
            type=int,
            help=(
                "Number of highest probability vocabulary tokens to keep for "
                "top-k-filtering."
            ),
        )
        group.add_argument(
            "--run-top-p",
            type=float,
            help=(
                "If set to < 1, only the smallest set of most probable "
                "tokens with probabilities that add up to top_p or higher "
                "are kept for generation."
            ),
        )
        group.add_argument(
            "--tool", type=str, action="append", help="Enable tool"
        )
        return group

    @staticmethod
    def _add_tool_settings_arguments(
        parser: ArgumentParser, *, prefix: str, settings_cls: type
    ) -> ArgumentParser:
        group = parser.add_argument_group(f"{prefix} tool settings")

        for field in fields(settings_cls):
            option = f"--tool-{prefix}-{field.name.replace('_', '-')}"
            dest = f"tool_{prefix}_{field.name}"

            ftype = field.type
            origin = get_origin(ftype)
            args = get_type_args(ftype)
            if origin is not None:
                if origin is list or origin is tuple:
                    ftype = args[0]
                elif origin is Optional or type(None) in args:
                    ftype = next((a for a in args if a is not type(None)), str)
                elif origin.__name__ == "Literal":
                    ftype = type(args[0])

            if ftype is bool or isinstance(field.default, bool):
                group.add_argument(
                    option, dest=dest, action="store_true", default=None
                )
            elif ftype is int or isinstance(field.default, int):
                group.add_argument(option, dest=dest, type=int, default=None)
            elif ftype is float or isinstance(field.default, float):
                group.add_argument(option, dest=dest, type=float, default=None)
            else:
                group.add_argument(option, dest=dest, type=str, default=None)

        return group

    @staticmethod
    def _needs_hf_token(args: Namespace) -> bool:
        command = args.command
        if command == "model" and (args.model_command or "display") == "run":
            engine_uri = ModelManager.parse_uri(args.model)
            return engine_uri.is_local
        if command == "agent" and (
            (args.agent_command or "run") in {"run", "serve"}
        ):
            engine = getattr(args, "engine_uri", None)
            if engine:
                engine_uri = ModelManager.parse_uri(engine)
                return engine_uri.is_local
        return True

    async def __call__(self) -> None:
        argv, chat_opts = self._extract_chat_template_settings(sys.argv[1:])
        args = self._parser.parse_args(argv)

        if args.parallel and not args.quiet:
            args.quiet = True

        if args.parallel and "LOCAL_RANK" not in environ:
            cmd = [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--nproc-per-node",
                str(args.parallel_count),
                "-m",
                "avalan.cli",
            ] + argv
            run(cmd, check=True)
            return

        if args.parallel and "LOCAL_RANK" in environ:
            rank = int(environ["LOCAL_RANK"])
            if args.device.startswith("cuda") and ":" not in args.device:
                args.device = f"cuda:{rank}"
                set_device(rank)

        if args.version:
            print(f"{self._name} {self._version}")
            return

        for key, value in chat_opts.items():
            setattr(args, f"run_chat_{key}", value)

        current_locale, _ = getlocale()
        try:
            translator = translation(
                self._name, localedir=args.locales, languages=[args.locale]
            )
        except FileNotFoundError:
            translator = gettext

        assert self._logger is not None and isinstance(self._logger, Logger)
        theme = FancyTheme(translator.gettext, translator.ngettext)
        _ = theme._
        console = Console(
            theme=Theme(styles=theme.get_styles()), record=args.record
        )

        if args.help_full:
            return self._help(console, self._parser)

        access_token = args.hf_token
        requires_token = self._needs_hf_token(args)

        if requires_token:
            if not access_token:
                prompt_kwargs = {}
                if has_input(console):
                    try:
                        prompt_kwargs["stream"] = open("/dev/tty")
                    except OSError:
                        pass
                access_token = Prompt.ask(
                    theme.ask_access_token(), **prompt_kwargs
                )
            assert access_token
        else:
            access_token = access_token or "anonymous"

        hub = HuggingfaceHub(access_token, args.cache_dir, self._logger)

        try:
            await self._main(args, theme, console, hub)
        except (CancelledError, KeyboardInterrupt, CommandAbortException):
            if not args.quiet:
                console.print(theme.bye())
        if args.parallel and "LOCAL_RANK" in environ:
            try:
                destroy_process_group()
            except AssertionError:
                # Process group might be dead already
                pass

    def _help(
        self, console: Console, parser: ArgumentParser, path: list[str] = []
    ) -> None:
        prog = parser.prog
        is_root_command = not path
        console.print(
            ("#" if is_root_command else "#" * (len(path) + 1)) + f" {prog}"
        )
        console.print("")
        console.print("```")
        console.print(parser.format_help().strip())
        console.print("```")
        console.print("")
        for action in parser._actions:
            if isinstance(action, _SubParsersAction):
                for name, subparser in action.choices.items():
                    self._help(console, subparser, path + [name])

    async def _main(
        self,
        args: Namespace,
        theme: Theme,
        console: Console,
        hub: HuggingfaceHub,
        suggest_login: bool = False,
    ) -> None:
        user: User | None = None
        _ = theme._

        verbosity = args.verbose or 0
        log_level = (
            DEBUG if verbosity >= 2 else INFO if verbosity >= 1 else WARNING
        )
        previous_log_level = self._logger.getEffectiveLevel()

        self._logger.setLevel(log_level)

        if find_spec("sentence_transformers"):
            logger_replace(self._logger, ["sentence_transformers"])

        if find_spec("httpx"):
            logger_replace(self._logger, ["httpx"])

        filterwarnings(
            "ignore",
            message=r".*`do_sample` is set to `False`. "
            r"However, `temperature` is set.*",
        )

        class _MaskSpecEmbedFilter(Filter):
            def filter(self, record: LogRecord) -> bool:
                return "wav2vec2.masked_spec_embed" not in record.getMessage()

        hf_logger = hf_logging.get_logger("transformers.modeling_utils")
        hf_logger.addFilter(_MaskSpecEmbedFilter())

        filterwarnings(
            "ignore",
            message=r".*Some weights of Wav2Vec2ForCTC were not initialized.*",
        )

        suggest_login = suggest_login and not has_input(console)
        if args.login or (
            suggest_login
            and Confirm.ask(theme.ask_login_to_hub(), default=False)
        ):
            with console.status(
                theme.logging_in(hub.domain),
                spinner=(theme.get_spinner("connecting")),
                refresh_per_second=self._REFRESH_RATE,
            ):
                hub.login()
                user = hub.user()

        if not args.quiet:
            console.print(
                theme.welcome(
                    self._site.geturl(),
                    self._name,
                    self._version,
                    self._license,
                    user,
                )
            )

        match args.command:
            case "agent":
                subcommand = args.agent_command or "run"
                match subcommand:
                    case "message":
                        innercommand = args.agent_message_command or "search"
                        match innercommand:
                            case "search":
                                await agent_message_search(
                                    args,
                                    console,
                                    theme,
                                    hub,
                                    self._logger,
                                    refresh_per_second=self._REFRESH_RATE,
                                )
                    case "run":
                        await agent_run(
                            args,
                            console,
                            theme,
                            hub,
                            self._logger,
                            refresh_per_second=self._REFRESH_RATE,
                        )
                    case "serve":
                        await agent_serve(
                            args,
                            hub,
                            self._logger,
                            self._name,
                            str(self._version),
                        )
                    case "init":
                        await agent_init(args, console, theme)
            case "cache":
                subcommand = args.cache_command or "list"
                match subcommand:
                    case "delete":
                        cache_delete(args, console, theme, hub)
                    case "download":
                        cache_download(args, console, theme, hub)
                    case "list":
                        cache_list(args, console, theme, hub)
            case "memory":
                subcommand = args.memory_command or "embeddings"
                match subcommand:
                    case "document":
                        innercommand = args.memory_document_command or "index"
                        match innercommand:
                            case "index":
                                await memory_document_index(
                                    args, console, theme, hub, self._logger
                                )
                    case "search":
                        await memory_search(
                            args, console, theme, hub, self._logger
                        )
                    case "embeddings":
                        await memory_embeddings(
                            args, console, theme, hub, self._logger
                        )
            case "model":
                subcommand = args.model_command or "display"
                match subcommand:
                    case "display":
                        model_display(args, console, theme, hub, self._logger)
                    case "install":
                        model_install(args, console, theme, hub)
                    case "run":
                        await model_run(
                            args,
                            console,
                            theme,
                            hub,
                            self._REFRESH_RATE,
                            self._logger,
                        )
                    case "search":
                        await model_search(
                            args, console, theme, hub, self._REFRESH_RATE
                        )
                    case "uninstall":
                        model_uninstall(args, console, theme, hub)
            case "deploy":
                subcommand = args.deploy_command or "run"
                match subcommand:
                    case "run":
                        await deploy_run(args, self._logger)

            case "tokenizer":
                await tokenize(args, console, theme, hub, self._logger)

        self._logger.setLevel(previous_log_level)


def main() -> None:
    basicConfig(
        level=INFO,
        format="%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[RichHandler(rich_tracebacks=True)],
    )
    logger = getLogger(name())

    cli = CLI(logger)
    run_in_loop(cli())


if __name__ == "__main__":
    main()  # pragma: no cover

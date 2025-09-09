from ..agent.orchestrator import Orchestrator
from ..agent.orchestrator.orchestrators.default import DefaultOrchestrator
from ..agent.orchestrator.orchestrators.json import JsonOrchestrator, Property
from ..entities import (
    EngineUri,
    OrchestratorSettings,
    TransformerEngineSettings,
    ToolFormat,
    ToolManagerSettings,
)
from ..event.manager import EventManager
from ..memory.manager import MemoryManager
from ..memory.partitioner.text import TextPartitioner
from ..memory.permanent.pgsql.raw import PgsqlRawMemory
from ..model.hubs.huggingface import HuggingfaceHub
from ..model.manager import ModelManager
from ..model.nlp.sentence import SentenceTransformerModel
from ..tool.browser import BrowserToolSet, BrowserToolSettings
from ..tool.code import CodeToolSet
from ..tool.context import ToolSettingsContext
from ..tool.database import DatabaseToolSet, DatabaseToolSettings
from ..tool.manager import ToolManager
from ..tool.math import MathToolSet
from ..tool.memory import MemoryToolSet
from contextlib import AsyncExitStack
from logging import Logger, DEBUG, INFO
from os import access, R_OK
from os.path import exists
from tomllib import load
from typing import Any, Callable
from uuid import UUID, uuid4


class OrchestratorLoader:
    DEFAULT_SENTENCE_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
    DEFAULT_SENTENCE_MODEL_MAX_TOKENS = 500
    DEFAULT_SENTENCE_MODEL_OVERLAP_SIZE = 125
    DEFAULT_SENTENCE_MODEL_WINDOW_SIZE = 250

    _hub: HuggingfaceHub
    _logger: Logger
    _participant_id: UUID
    _stack: AsyncExitStack

    def __init__(
        self,
        *,
        hub: HuggingfaceHub,
        logger: Logger,
        participant_id: UUID,
        stack: AsyncExitStack,
    ) -> None:
        self._hub = hub
        self._logger = logger
        self._participant_id = participant_id
        self._stack = stack

    @property
    def hub(self) -> HuggingfaceHub:
        return self._hub

    @property
    def participant_id(self) -> UUID:
        return self._participant_id

    async def from_file(
        self,
        path: str,
        *,
        agent_id: UUID | None,
        disable_memory: bool = False,
        uri: str | None = None,
        tool_settings: ToolSettingsContext | None = None,
    ) -> Orchestrator:
        _l = self._log_wrapper(self._logger)

        if not exists(path):
            raise FileNotFoundError(path)
        elif not access(path, R_OK):
            raise PermissionError(path)

        _l("Loading agent from %s", path, is_debug=False)

        with open(path, "rb") as file:
            config = load(file)

            # Validate settings

            assert "agent" in config, "No agent section in configuration"
            assert (
                "engine" in config
            ), "No engine section defined in configuration"
            assert (
                "uri" in config["engine"]
            ), "No uri defined in engine section of configuration"

            agent_config = config["agent"]
            assert not (
                "user" in agent_config and "user_template" in agent_config
            ), "user and user_template are mutually exclusive"

            assert (
                "engine" in config
            ), "No engine section defined in configuration"
            assert (
                "uri" in config["engine"]
            ), "No uri defined in engine section of configuration"

            uri = uri or config["engine"]["uri"]
            engine_config = config["engine"]
            enable_tools = (
                engine_config["tools"] if "tools" in engine_config else None
            )
            engine_config.pop("uri", None)
            engine_config.pop("tools", None)
            orchestrator_type = (
                config["agent"]["type"] if "type" in config["agent"] else None
            )
            agent_id = (
                agent_id
                if agent_id
                else (
                    config["agent"]["id"]
                    if "id" in config["agent"]
                    else uuid4()
                )
            )

            assert orchestrator_type is None or orchestrator_type in [
                "json"
            ], (
                f"Unknown type {config['agent']['type']} in agent section "
                + "of configuration"
            )

            call_options = config["run"] if "run" in config else None
            if call_options and "chat" in call_options:
                call_options["chat_settings"] = call_options.pop("chat")
            template_vars = (
                config["template"] if "template" in config else None
            )

            # Memory configuration

            memory_options = (
                config["memory"]
                if "memory" in config and not disable_memory
                else None
            )

            memory_permanent_message = (
                memory_options["permanent_message"]
                if memory_options and "permanent_message" in memory_options
                else None
            )

            memory_permanent: dict[str, str] | None = None
            if memory_options and "permanent" in memory_options:
                memory_permanent_option = memory_options["permanent"]
                assert isinstance(
                    memory_permanent_option, dict
                ), "Permanent memory should be a mapping"
                memory_permanent = {
                    str(ns): str(dsn)
                    for ns, dsn in memory_permanent_option.items()
                }
            memory_recent = (
                memory_options["recent"]
                if memory_options and "recent" in memory_options
                else False
            )
            assert isinstance(
                memory_recent, bool
            ), "Recent message memory can only be set or unset"

            sentence_model_id = (
                config["memory.engine"]["model_id"]
                if "memory.engine" in config
                and "model_id" in config["memory.engine"]
                else OrchestratorLoader.DEFAULT_SENTENCE_MODEL_ID
            )
            sentence_model_engine_config = (
                config["memory.engine"] if "memory.engine" in config else None
            )
            sentence_model_max_tokens = (
                config["memory.engine"]["max_tokens"]
                if sentence_model_engine_config
                and "max_tokens" in sentence_model_engine_config
                else OrchestratorLoader.DEFAULT_SENTENCE_MODEL_MAX_TOKENS
            )
            sentence_model_overlap_size = (
                config["memory.engine"]["overlap_size"]
                if sentence_model_engine_config
                and "overlap_size" in sentence_model_engine_config
                else OrchestratorLoader.DEFAULT_SENTENCE_MODEL_OVERLAP_SIZE
            )
            sentence_model_window_size = (
                config["memory.engine"]["window_size"]
                if sentence_model_engine_config
                and "window_size" in sentence_model_engine_config
                else OrchestratorLoader.DEFAULT_SENTENCE_MODEL_WINDOW_SIZE
            )

            if sentence_model_engine_config:
                sentence_model_engine_config.pop("model_id", None)
                sentence_model_engine_config.pop("max_tokens", None)
                sentence_model_engine_config.pop("overlap_size", None)
                sentence_model_engine_config.pop("window_size", None)

            settings = OrchestratorSettings(
                agent_id=agent_id,
                orchestrator_type=orchestrator_type,
                agent_config=agent_config,
                uri=uri,
                engine_config=engine_config,
                tools=enable_tools,
                call_options=call_options,
                template_vars=template_vars,
                memory_permanent_message=memory_permanent_message,
                permanent_memory=memory_permanent,
                memory_recent=memory_recent,
                sentence_model_id=sentence_model_id,
                sentence_model_engine_config=sentence_model_engine_config,
                sentence_model_max_tokens=sentence_model_max_tokens,
                sentence_model_overlap_size=sentence_model_overlap_size,
                sentence_model_window_size=sentence_model_window_size,
                json_config=(
                    config.get("json") if isinstance(config, dict) else None
                ),
                log_events=True,
            )

            tool_section = config.get("tool", {})
            browser_config = tool_section.get("browser", {}).get("open")
            if not browser_config and "browser" in tool_section:
                browser_config = tool_section["browser"]
            browser_settings = None
            if browser_config:
                if "debug_source" in browser_config and isinstance(
                    browser_config["debug_source"], str
                ):
                    browser_config["debug_source"] = open(
                        browser_config["debug_source"]
                    )
                browser_settings = BrowserToolSettings(**browser_config)

            database_settings = None
            database_config = tool_section.get("database")
            if database_config:
                database_settings = DatabaseToolSettings(**database_config)

            if tool_settings:
                browser_settings = tool_settings.browser or browser_settings
                database_settings = tool_settings.database or database_settings
                extra = tool_settings.extra
            else:
                extra = None

            tool_settings = ToolSettingsContext(
                browser=browser_settings,
                database=database_settings,
                extra=extra,
            )

            tool_format = None
            tool_format_str = tool_section.get("format")
            if tool_format_str:
                tool_format = ToolFormat(tool_format_str)

            _l("Loaded agent from %s", path, is_debug=False)

            return await self.from_settings(
                settings, tool_settings=tool_settings, tool_format=tool_format
            )

    async def from_settings(
        self,
        settings: OrchestratorSettings,
        *,
        tool_settings: ToolSettingsContext | None = None,
        tool_format: ToolFormat | None = None,
    ) -> Orchestrator:
        _l = self._log_wrapper(self._logger)

        _l("Loading agent from settings", is_debug=False)

        sentence_model_engine_settings = (
            TransformerEngineSettings(**settings.sentence_model_engine_config)
            if settings.sentence_model_engine_config
            else TransformerEngineSettings()
        )

        _l(
            "Loading sentence transformer model %s for agent %s",
            settings.sentence_model_id,
            settings.agent_id,
        )

        sentence_model = SentenceTransformerModel(
            model_id=settings.sentence_model_id,
            settings=sentence_model_engine_settings,
            logger=self._logger,
        )
        sentence_model = self._stack.enter_context(sentence_model)

        _l(
            "Loading text partitioner for model %s for agent %s with settings"
            " (%s, %s, %s)",
            settings.sentence_model_id,
            settings.agent_id,
            settings.sentence_model_max_tokens,
            settings.sentence_model_overlap_size,
            settings.sentence_model_window_size,
        )

        text_partitioner = TextPartitioner(
            model=sentence_model,
            logger=self._logger,
            max_tokens=settings.sentence_model_max_tokens,
            overlap_size=settings.sentence_model_overlap_size,
            window_size=settings.sentence_model_window_size,
        )

        _l("Loading event manager")

        event_manager = EventManager()
        if settings.log_events:
            event_manager.add_listener(
                lambda e: _l("%s", e.payload, inner_type=f"Event {e.type}")
            )

        _l("Loading memory manager for agent %s", settings.agent_id)

        memory = await MemoryManager.create_instance(
            agent_id=settings.agent_id,
            participant_id=self._participant_id,
            text_partitioner=text_partitioner,
            logger=self._logger,
            with_permanent_message_memory=settings.memory_permanent_message,
            with_recent_message_memory=settings.memory_recent,
            event_manager=event_manager,
        )

        for namespace, dsn in (settings.permanent_memory or {}).items():
            _l(
                "Loading permanent memory %s for agent %s",
                namespace,
                settings.agent_id,
            )
            store = await PgsqlRawMemory.create_instance(
                dsn=dsn, logger=self._logger
            )
            memory.add_permanent_memory(namespace, store)

        _l(
            "Loading tool manager for agent %s with partitioner and a sentence"
            " model %s with settings (%s, %s, %s)",
            settings.agent_id,
            settings.sentence_model_id,
            settings.sentence_model_max_tokens,
            settings.sentence_model_overlap_size,
            settings.sentence_model_window_size,
        )

        browser_settings = tool_settings.browser if tool_settings else None
        database_settings = tool_settings.database if tool_settings else None

        _l(
            "Tool settings: browser=%s, database=%s",
            browser_settings,
            database_settings,
        )

        available_toolsets = [
            BrowserToolSet(
                settings=browser_settings or BrowserToolSettings(),
                partitioner=text_partitioner,
                namespace="browser",
            ),
            CodeToolSet(namespace="code"),
            MathToolSet(namespace="math"),
            MemoryToolSet(memory, namespace="memory"),
        ]
        if database_settings:
            available_toolsets.append(
                DatabaseToolSet(
                    settings=database_settings, namespace="database"
                )
            )

        tool = ToolManager.create_instance(
            available_toolsets=available_toolsets,
            enable_tools=settings.tools,
            settings=ToolManagerSettings(tool_format=tool_format),
        )
        tool = await self._stack.enter_async_context(tool)

        _l(
            "Creating orchestrator %s #%s",
            settings.orchestrator_type,
            settings.agent_id,
        )

        model_manager = ModelManager(
            self._hub, self._logger, event_manager=event_manager
        )
        model_manager = self._stack.enter_context(model_manager)

        engine_uri = model_manager.parse_uri(settings.uri)
        engine_settings = model_manager.get_engine_settings(
            engine_uri,
            settings=settings.engine_config,
        )

        assert settings.agent_id

        if settings.orchestrator_type == "json":
            assert settings.json_config is not None
            agent = self._load_json_orchestrator(
                agent_id=settings.agent_id,
                engine_uri=engine_uri,
                engine_settings=engine_settings,
                logger=self._logger,
                model_manager=model_manager,
                memory=memory,
                tool=tool,
                event_manager=event_manager,
                config={"json": settings.json_config},
                agent_config=settings.agent_config,
                call_options=settings.call_options,
                template_vars=settings.template_vars,
            )
        else:
            agent = DefaultOrchestrator(
                engine_uri,
                self._logger,
                model_manager,
                memory,
                tool,
                event_manager,
                id=settings.agent_id,
                name=settings.agent_config.get("name"),
                role=(
                    None
                    if "system" in settings.agent_config
                    else settings.agent_config.get("role")
                ),
                task=(
                    None
                    if "system" in settings.agent_config
                    else settings.agent_config.get("task")
                ),
                instructions=(
                    None
                    if "system" in settings.agent_config
                    else settings.agent_config.get("instructions")
                ),
                rules=settings.agent_config.get("rules"),
                system=settings.agent_config.get("system"),
                user=settings.agent_config.get("user"),
                user_template=settings.agent_config.get("user_template"),
                settings=engine_settings,
                call_options=settings.call_options,
                template_vars=settings.template_vars,
            )

        _l("Loaded agent from settings", is_debug=False)

        return agent

    @staticmethod
    def _load_json_orchestrator(
        agent_id: UUID,
        engine_uri: EngineUri,
        engine_settings: TransformerEngineSettings,
        logger: Logger,
        model_manager: ModelManager,
        memory: MemoryManager,
        tool: ToolManager,
        event_manager: EventManager,
        config: dict,
        agent_config: dict,
        call_options: dict | None,
        template_vars: dict | None,
    ) -> JsonOrchestrator:
        assert "json" in config, "No json section in configuration"
        if "system" not in agent_config:
            assert (
                "instructions" in agent_config
            ), "No instructions defined in agent section of configuration"
            assert (
                "task" in agent_config
            ), "No task defined in agent section of configuration"

        properties: list[Property] = []
        for property_name in config.get("json", []):
            output_property = config["json"][property_name]
            properties.append(
                Property(
                    name=property_name,
                    data_type=output_property["type"],
                    description=output_property["description"],
                )
            )

        assert properties, "No properties defined in configuration"

        agent = JsonOrchestrator(
            engine_uri,
            logger,
            model_manager,
            memory,
            tool,
            event_manager,
            properties,
            id=agent_id,
            name=agent_config["name"] if "name" in agent_config else None,
            role=(
                None if "system" in agent_config else agent_config.get("role")
            ),
            task=(
                None if "system" in agent_config else agent_config.get("task")
            ),
            instructions=(
                None
                if "system" in agent_config
                else agent_config.get("instructions")
            ),
            rules=agent_config.get("rules"),
            system=agent_config.get("system"),
            user=agent_config.get("user"),
            user_template=agent_config.get("user_template"),
            settings=engine_settings,
            call_options=call_options,
            template_vars=template_vars,
        )
        return agent

    @staticmethod
    def _log_wrapper(logger: Logger) -> Callable[..., Any]:
        def wrapper(
            message: str,
            *args: Any,
            inner_type: str | None = None,
            **kwargs: Any,
        ) -> Any:
            is_debug = kwargs.pop("is_debug", True)
            level = DEBUG if is_debug else INFO
            prefix = (
                f"<{inner_type} @ OrchestratorLoader> "
                if inner_type
                else "<OrchestratorLoader> "
            )
            return logger.log(level, prefix + message, *args, **kwargs)

        return wrapper

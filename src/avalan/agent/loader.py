from ..agent.orchestrator import Orchestrator
from ..agent.orchestrator.orchestrators.default import DefaultOrchestrator
from ..agent.orchestrator.orchestrators.json import JsonOrchestrator, Property
from ..entities import (
    EngineUri,
    OrchestratorSettings,
    TransformerEngineSettings,
    ToolManagerSettings,
)
from ..event.manager import EventManager
from ..memory.manager import MemoryManager
from ..memory.partitioner.text import TextPartitioner
from ..model.hubs.huggingface import HuggingfaceHub
from ..model.manager import ModelManager
from ..model.nlp.sentence import SentenceTransformerModel
from ..tool.browser import BrowserToolSet, BrowserToolSettings
from ..tool.manager import ToolManager
from ..tool.math import MathToolSet
from ..tool.memory import MemoryToolSet
from ..tool.code import CodeToolSet
from ..memory.permanent.pgsql.raw import PgsqlRawMemory
from contextlib import AsyncExitStack
from logging import Logger
from os import access, R_OK
from os.path import exists
from tomllib import load
from types import FunctionType
from uuid import UUID, uuid4


class OrchestratorLoader:
    DEFAULT_SENTENCE_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
    DEFAULT_SENTENCE_MODEL_MAX_TOKENS = 500
    DEFAULT_SENTENCE_MODEL_OVERLAP_SIZE = 125
    DEFAULT_SENTENCE_MODEL_WINDOW_SIZE = 250

    @classmethod
    async def from_file(
        cls,
        path: str,
        *args,
        agent_id: UUID | None,
        hub: HuggingfaceHub,
        logger: Logger,
        participant_id: UUID,
        stack: AsyncExitStack,
        disable_memory: bool = False,
    ) -> Orchestrator:
        _l = cls._log_wrapper(logger)

        if not exists(path):
            raise FileNotFoundError(path)
        elif not access(path, R_OK):
            raise PermissionError(path)

        _l("Loading agent from %s", path)

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
            for setting in ["role"]:
                assert (
                    setting in agent_config
                ), f"No {setting} defined in agent section of configuration"

            assert (
                "engine" in config
            ), "No engine section defined in configuration"
            assert (
                "uri" in config["engine"]
            ), "No uri defined in engine section of configuration"

            uri = config["engine"]["uri"]
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
            assert (
                "role" in agent_config
            ), "No role defined in agent section of configuration"

            call_options = config["run"] if "run" in config else None
            if call_options and "chat" in call_options:
                call_options["chat_template_settings"] = call_options.pop(
                    "chat"
                )
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

            tool_config = config.get("tool", {}).get("browser", {}).get("open")
            if not tool_config and "browser" in config.get("tool", {}):
                tool_config = config["tool"]["browser"]
            browser_settings = None
            if tool_config:
                if "debug_source" in tool_config and isinstance(
                    tool_config["debug_source"], str
                ):
                    tool_config["debug_source"] = open(
                        tool_config["debug_source"]
                    )
                browser_settings = BrowserToolSettings(**tool_config)

            return await cls.load_from_settings(
                settings,
                hub=hub,
                logger=logger,
                participant_id=participant_id,
                stack=stack,
                browser_settings=browser_settings,
            )

    @classmethod
    async def load_from_settings(
        cls,
        settings: OrchestratorSettings,
        *,
        hub: HuggingfaceHub,
        logger: Logger,
        participant_id: UUID,
        stack: AsyncExitStack,
        browser_settings: BrowserToolSettings | None = None,
    ) -> Orchestrator:
        _l = cls._log_wrapper(logger)

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
            logger=logger,
        )
        sentence_model = stack.enter_context(sentence_model)

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
            logger=logger,
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
            participant_id=participant_id,
            text_partitioner=text_partitioner,
            logger=logger,
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
                dsn=dsn, logger=logger
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

        tool = ToolManager.create_instance(
            available_toolsets=available_toolsets,
            enable_tools=settings.tools,
            settings=ToolManagerSettings(),
        )
        tool = await stack.enter_async_context(tool)

        _l(
            "Creating orchestrator %s #%s",
            settings.orchestrator_type,
            settings.agent_id,
        )

        model_manager = ModelManager(hub, logger)
        model_manager = stack.enter_context(model_manager)

        engine_uri = model_manager.parse_uri(settings.uri)
        engine_settings = model_manager.get_engine_settings(
            engine_uri,
            settings=settings.engine_config,
        )

        assert settings.agent_id

        if settings.orchestrator_type == "json":
            assert settings.json_config is not None
            agent = cls._load_json_orchestrator(
                agent_id=settings.agent_id,
                engine_uri=engine_uri,
                engine_settings=engine_settings,
                logger=logger,
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
                logger,
                model_manager,
                memory,
                tool,
                event_manager,
                id=settings.agent_id,
                name=settings.agent_config.get("name"),
                role=(
                    settings.agent_config["role"]
                    if "role" in settings.agent_config
                    else None
                ),
                task=settings.agent_config.get("task"),
                instructions=settings.agent_config.get("instructions"),
                rules=settings.agent_config.get("rules"),
                settings=engine_settings,
                call_options=settings.call_options,
                template_vars=settings.template_vars,
            )

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
        assert (
            "instructions" in agent_config
        ), "No instructions defined in agent section of configuration"
        assert (
            "task" in agent_config
        ), "No task defined in agent section of configuration"
        assert (
            "role" in agent_config
        ), "No role defined in agent section of configuration"

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
            role=agent_config["role"],
            task=agent_config["task"],
            instructions=agent_config["instructions"],
            rules=agent_config["rules"] if "rules" in agent_config else None,
            settings=engine_settings,
            call_options=call_options,
            template_vars=template_vars,
        )
        return agent

    @staticmethod
    def _log_wrapper(logger: Logger) -> FunctionType:
        return lambda message, *args, inner_type=None, **kwargs: logger.debug(
            (
                f"<{inner_type} @ OrchestratorLoader> "
                if inner_type
                else "<OrchestratorLoader> "
            )
            + message,
            *args,
            **kwargs,
        )

from ..agent.orchestrator import Orchestrator
from ..agent.orchestrators.default import DefaultOrchestrator
from ..agent.orchestrators.json import JsonOrchestrator, Property
from ..memory.manager import MemoryManager
from ..memory.partitioner.text import TextPartitioner
from ..model.entities import EngineUri, TransformerEngineSettings
from ..model.hubs.huggingface import HuggingfaceHub
from ..model.manager import ModelManager
from ..model.nlp.sentence import SentenceTransformerModel
from ..tool.manager import ToolManager
from ..event.manager import EventManager
from contextlib import AsyncExitStack
from logging import Logger
from os import access, R_OK
from os.path import exists
from tomllib import load
from typing import Optional
from uuid import UUID, uuid4

class Loader:
    DEFAULT_SENTENCE_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

    @classmethod
    async def from_file(
        cls,
        path: str,
        *args,
        agent_id: Optional[UUID],
        hub: HuggingfaceHub,
        logger: Logger,
        participant_id: UUID,
        stack: AsyncExitStack,
        disable_memory: bool=False
    ) -> Orchestrator:
        if not exists(path):
            raise FileNotFoundError(path)
        elif not access(path, R_OK):
            raise PermissionError(path)

        logger.debug(f"Loading agent from {path}")

        with open(path, "rb") as file:
            config = load(file)

            # Validate settings

            assert "agent" in config, "No agent section in configuration"
            assert "engine" in config, \
                "No engine section defined in configuration"
            assert "uri" in config["engine"], \
                "No uri defined in engine section of configuration"

            agent_config = config["agent"]
            for setting in ["role"]:
                assert setting in agent_config, \
                    f"No {setting} defined in agent section of configuration"

            assert "engine" in config, \
                "No engine section defined in configuration"
            assert "uri" in config["engine"], \
                "No uri defined in engine section of configuration"

            uri = config["engine"]["uri"]
            engine_config = config["engine"]
            enable_tools = (
                engine_config["tools"]
                if "tools" in engine_config
                else None
            )
            engine_config.pop("uri", None)
            engine_config.pop("tools", None)
            orchestrator_type = (
                config["agent"]["type"] if "type" in config["agent"]
                else None
            )
            agent_id = (
                agent_id if agent_id
                else config["agent"]["id"] if "id" in config["agent"]
                else uuid4()
            )

            assert orchestrator_type is None or orchestrator_type in ["json"], \
                f"Unknown type {config['agent']['type']} in agent section " + \
                 "of configuration"
            assert "role" in agent_config, \
                "No role defined in agent section of configuration"

            call_options = config["run"] if "run" in config else None
            template_vars = config["template"] \
                            if "template" in config else None

            # Tool configuration

            tool = ToolManager.create_instance(
                enable_tools=enable_tools
            )

            # Memory configuration

            memory_options = (
                config["memory"]
                if "memory" in config and not disable_memory
                else None
            )

            memory_permanent = (
                memory_options["permanent"]
                if memory_options and "permanent" in memory_options
                else None
            )
            assert not memory_permanent or isinstance(memory_permanent,str), \
                "Permanent message memory should be a string"
            memory_recent = (
                memory_options["recent"]
                if memory_options and "recent" in memory_options
                else False
            )
            assert isinstance(memory_recent,bool), \
                "Recent message memory can only be set or unset"

            sentence_model_id = (
                config["memory.engine"]["model_id"]
                if "memory.engine" in config and
                   "model_id" in config["memory.engine"]
                else Loader.DEFAULT_SENTENCE_MODEL_ID
            )
            sentence_model_engine_config = (
                config["memory.engine"]
                if "memory.engine" in config
                else None
            )
            sentence_model_max_tokens = (
                config["memory.engine"]["max_tokens"]
                if sentence_model_engine_config
                   and "max_tokens" in sentence_model_engine_config
                else 500
            )
            sentence_model_overlap_size = (
                config["memory.engine"]["overlap_size"]
                if sentence_model_engine_config
                   and "overlap_size" in sentence_model_engine_config
                else 125
            )
            sentence_model_window_size = (
                config["memory.engine"]["window_size"]
                if sentence_model_engine_config
                   and "window_size" in sentence_model_engine_config
                else 250
            )

            if sentence_model_engine_config:
                sentence_model_engine_config.pop("model_id", None)
                sentence_model_engine_config.pop("max_tokens", None)
                sentence_model_engine_config.pop("overlap_size", None)
                sentence_model_engine_config.pop("window_size", None)

            sentence_model_engine_settings = (
                TransformerEngineSettings(**sentence_model_engine_config)
                if sentence_model_engine_config
                else TransformerEngineSettings()
            )

            logger.debug("Loading sentence transformer "
                         f"model {sentence_model_id} for agent {agent_id}")

            sentence_model = SentenceTransformerModel(
                model_id=sentence_model_id,
                settings=sentence_model_engine_settings,
                logger=logger
            )
            sentence_model = stack.enter_context(sentence_model)

            logger.debug("Loading text partitioner for "
                         f"model {sentence_model_id} for agent {agent_id} "
                         f"with settings ({sentence_model_max_tokens}, "
                         f"{sentence_model_overlap_size}, "
                         f"{sentence_model_window_size})")

            text_partitioner = TextPartitioner(
                model=sentence_model,
                logger=logger,
                max_tokens=sentence_model_max_tokens,
                overlap_size=sentence_model_overlap_size,
                window_size=sentence_model_window_size
            )

            logger.debug(f"Loading memory manager for agent {agent_id}")

            memory = await MemoryManager.create_instance(
                agent_id=agent_id,
                participant_id=participant_id,
                text_partitioner=text_partitioner,
                with_permanent_message_memory=memory_permanent,
                with_recent_message_memory=memory_recent,
            )

            event_manager = EventManager()
            event_manager.add_listener(
                lambda e: logger.debug(f"Event {e.type}: {e.payload}")
            )

            # Agent creation

            logger.debug(f"Creating agent {orchestrator_type} #{agent_id}")

            model_manager = ModelManager(hub, logger)
            model_manager = stack.enter_context(model_manager)

            engine_uri = model_manager.parse_uri(uri)
            engine_settings = model_manager.get_engine_settings(
                engine_uri,
                settings=engine_config
            )

            agent = \
                cls._load_json_orchestrator(
                    engine_uri=engine_uri,
                    engine_settings=engine_settings,
                    logger=logger,
                    model_manager=model_manager,
                    memory=memory,
                    tool=tool,
                    event_manager=event_manager,
                    config=config,
                    agent_config=agent_config,
                    call_options=call_options,
                    template_vars=template_vars
                ) if orchestrator_type == "json" else \
                DefaultOrchestrator(
                    engine_uri,
                    logger,
                    model_manager,
                    memory,
                    tool,
                    event_manager,
                    name=agent_config["name"] \
                        if "name" in agent_config else None,
                    role=agent_config["role"],
                    task=agent_config["task"]
                        if "task" in agent_config else None,
                    instructions=agent_config["instructions"]
                        if "instructions" in agent_config else None,
                    rules=agent_config["rules"]
                        if "rules" in agent_config else None,
                    settings=engine_settings,
                    call_options=call_options,
                    template_vars=template_vars
                )

            return agent

    @staticmethod
    def _load_json_orchestrator(
        engine_uri: EngineUri,
        engine_settings: TransformerEngineSettings,
        logger: Logger,
        model_manager: ModelManager,
        memory: MemoryManager,
        tool: ToolManager,
        event_manager: EventManager,
        config: dict,
        agent_config: dict,
        call_options: Optional[dict],
        template_vars: Optional[dict]
    ) -> JsonOrchestrator:
        assert "json" in config, "No json section in configuration"
        assert "instructions" in agent_config, \
            "No instructions defined in agent section of configuration"
        assert "task" in agent_config, \
            "No task defined in agent section of configuration"
        assert "role" in agent_config, \
            "No role defined in agent section of configuration"

        properties : list[Property] = []
        for property_name in config.get("json", []):
            output_property = config["json"][property_name]
            properties.append(Property(
                name=property_name,
                data_type=output_property["type"],
                description=output_property["description"]
            ))

        assert properties, "No properties defined in configuration"

        agent = JsonOrchestrator(
            engine_uri,
            logger,
            model_manager,
            memory,
            tool,
            event_manager,
            properties,
            name=agent_config["name"] if "name" in agent_config else None,
            role=agent_config["role"],
            task=agent_config["task"],
            instructions=agent_config["instructions"],
            rules=agent_config["rules"] if "rules" in agent_config else None,
            settings=engine_settings,
            call_options=call_options,
            template_vars=template_vars
        )
        return agent


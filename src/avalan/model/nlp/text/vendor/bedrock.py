from .....entities import (
    GenerationSettings,
    Message,
    MessageContent,
    MessageContentFile,
    MessageContentImage,
    MessageContentText,
    MessageRole,
    ReasoningToken,
    Token,
    TokenDetail,
    ToolCallError,
    ToolCallResult,
    ToolCallToken,
)
from .....model.stream import TextGenerationSingleStream
from .....tool.manager import ToolManager
from .....utils import to_json
from ....message import TemplateMessageRole
from ....vendor import TextGenerationVendor, TextGenerationVendorStream
from . import TextGenerationVendorModel

from base64 import b64decode
from contextlib import AsyncExitStack
from json import dumps
from re import sub
from typing import Any, AsyncIterator, Mapping, NoReturn

from aioboto3 import Session as Boto3Session
from diffusers import DiffusionPipeline
from transformers import PreTrainedModel


def _get(event: Any, key: str) -> Any:
    if isinstance(event, dict):
        return event.get(key)
    return getattr(event, key, None)


def _string(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        if "text" in value:
            return _string(value["text"])
        if "reasoningText" in value:
            return _string(value["reasoningText"])
        if "string" in value:
            return _string(value["string"])
    return None


def _bedrock_error_code(error: Exception) -> str | None:
    response = getattr(error, "response", None)
    if not isinstance(response, dict):
        return None
    details = response.get("Error")
    if not isinstance(details, dict):
        return None
    code = details.get("Code")
    return code if isinstance(code, str) else None


def _bedrock_error_message(error: Exception) -> str:
    response = getattr(error, "response", None)
    if isinstance(response, dict):
        details = response.get("Error")
        if isinstance(details, dict):
            message = details.get("Message")
            if isinstance(message, str):
                return message
    return str(error)


def _geo_inference_prefix(region_name: str | None) -> str | None:
    if region_name is None:
        return None
    if region_name.startswith("us-"):
        return "us."
    if region_name.startswith("eu-"):
        return "eu."
    return None


_BEDROCK_XLSX_MIME_TYPE = (
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
_BEDROCK_DOCX_MIME_TYPE = (
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
)
_BEDROCK_DOCUMENT_FORMATS = {
    "application/msword": "doc",
    "application/pdf": "pdf",
    "application/vnd.ms-excel": "xls",
    _BEDROCK_XLSX_MIME_TYPE: "xlsx",
    _BEDROCK_DOCX_MIME_TYPE: "docx",
    "text/csv": "csv",
    "text/html": "html",
    "text/markdown": "md",
    "text/plain": "txt",
}


class BedrockStream(TextGenerationVendorStream):
    def __init__(self, events: AsyncIterator[Any]):
        async def generator() -> AsyncIterator[Token | TokenDetail | str]:
            tool_blocks: dict[int, dict[str, Any]] = {}

            async for event in events:
                content_start = _get(event, "contentBlockStart")
                if content_start:
                    block_index = content_start.get("contentBlockIndex")
                    block = content_start.get("contentBlock") or {}
                    tool = (
                        block.get("toolUse")
                        if isinstance(block, dict)
                        else None
                    )
                    if tool:
                        tool_blocks[block_index] = {
                            "id": tool.get("toolUseId"),
                            "name": tool.get("name"),
                            "fragments": [],
                        }
                        initial = tool.get("input")
                        if initial not in (None, ""):
                            fragment = (
                                initial
                                if isinstance(initial, str)
                                else dumps(initial)
                            )
                            tool_blocks[block_index]["fragments"].append(
                                fragment
                            )
                            yield ToolCallToken(token=fragment)
                    continue

                content_delta = _get(event, "contentBlockDelta")
                if content_delta:
                    block_index = content_delta.get("contentBlockIndex")
                    delta = content_delta.get("delta") or {}
                    text_block = delta.get("text")
                    text_value = _string(text_block)
                    if text_value:
                        yield Token(token=text_value)
                        continue
                    reasoning_block = delta.get("reasoning")
                    reasoning_value = _string(reasoning_block)
                    if reasoning_value:
                        yield ReasoningToken(token=reasoning_value)
                        continue
                    tool_delta = delta.get("toolUse")
                    if tool_delta:
                        fragment_value = tool_delta.get("input")
                        if fragment_value not in (None, ""):
                            fragment = (
                                fragment_value
                                if isinstance(fragment_value, str)
                                else dumps(fragment_value)
                            )
                            tool_block = tool_blocks.setdefault(
                                block_index,
                                {
                                    "id": tool_delta.get("toolUseId"),
                                    "name": tool_delta.get("name"),
                                    "fragments": [],
                                },
                            )
                            tool_block["fragments"].append(fragment)
                            yield ToolCallToken(token=fragment)
                    continue

                content_stop = _get(event, "contentBlockStop")
                if content_stop:
                    block_index = content_stop.get("contentBlockIndex")
                    block = content_stop.get("contentBlock") or {}
                    tool = (
                        block.get("toolUse")
                        if isinstance(block, dict)
                        else None
                    )
                    cached = tool_blocks.pop(block_index, None)
                    if tool:
                        cached = cached or {
                            "id": tool.get("toolUseId"),
                            "name": tool.get("name"),
                            "fragments": [],
                        }
                        final_input = tool.get("input")
                        if final_input not in (None, ""):
                            fragment = (
                                final_input
                                if isinstance(final_input, str)
                                else dumps(final_input)
                            )
                            cached["fragments"].append(fragment)
                    if cached:
                        token = TextGenerationVendor.build_tool_call_token(
                            cached.get("id"),
                            cached.get("name"),
                            "".join(cached.get("fragments", [])) or None,
                        )
                        yield token
                    continue

                if _get(event, "messageStop"):
                    break

        super().__init__(generator())

    async def __anext__(self) -> Token | TokenDetail | str:
        return await self._generator.__anext__()


class BedrockClient(TextGenerationVendor):
    _client: Any | None
    _endpoint_url: str | None
    _exit_stack: AsyncExitStack
    _region_name: str | None
    _session: Boto3Session

    def __init__(
        self,
        *,
        exit_stack: AsyncExitStack,
        region_name: str | None = None,
        endpoint_url: str | None = None,
    ) -> None:
        self._session = Boto3Session()
        self._region_name = region_name
        self._endpoint_url = endpoint_url
        self._exit_stack = exit_stack
        self._client = None

    async def _client_instance(self) -> Any:
        if self._client is None:
            kwargs: dict[str, Any] = {}
            if self._region_name:
                kwargs["region_name"] = self._region_name
            if self._endpoint_url:
                kwargs["endpoint_url"] = self._endpoint_url
            self._client = await self._exit_stack.enter_async_context(
                self._session.client("bedrock-runtime", **kwargs)
            )
        return self._client

    async def __call__(
        self,
        model_id: str,
        messages: list[Message],
        settings: GenerationSettings | None = None,
        *,
        tool: ToolManager | None = None,
        use_async_generator: bool = True,
    ) -> AsyncIterator[Token | TokenDetail | str] | TextGenerationSingleStream:
        client = await self._client_instance()
        system_prompt = self._system_prompt(messages)
        template_messages = self._template_messages(messages, ["system"])
        payload: dict[str, Any] = {
            "modelId": model_id,
            "messages": template_messages,
        }
        if system_prompt:
            payload["system"] = [{"text": system_prompt}]
        inference = self._inference_config(settings)
        if inference:
            payload["inferenceConfig"] = inference
        tool_config = self._tool_config(tool)
        if tool_config:
            payload["toolConfig"] = tool_config

        try:
            if use_async_generator:
                response = await client.converse_stream(**payload)
                stream = (
                    response.get("stream")
                    if isinstance(response, dict)
                    else None
                )
                assert (
                    stream is not None
                ), "Missing stream in Converse response"
                events = (
                    await self._exit_stack.enter_async_context(stream)
                    if hasattr(stream, "__aenter__")
                    else stream
                )
                return BedrockStream(events=events)

            response = await client.converse(**payload)
            return TextGenerationSingleStream(self._response_text(response))
        except Exception as error:
            if self._is_invalid_model_identifier_error(error):
                self._raise_invalid_model_identifier(model_id, error)
            if self._is_inference_profile_required_error(error):
                self._raise_inference_profile_required_error(model_id, error)
            if self._is_use_case_details_required_error(error):
                self._raise_use_case_details_required_error(model_id, error)
            if self._is_end_of_life_model_error(error):
                self._raise_end_of_life_model_error(model_id, error)
            raise

    @staticmethod
    def _is_invalid_model_identifier_error(error: Exception) -> bool:
        if _bedrock_error_code(error) != "ValidationException":
            return False
        return (
            "model identifier is invalid"
            in _bedrock_error_message(error).lower()
        )

    def _raise_invalid_model_identifier(
        self, model_id: str, error: Exception
    ) -> NoReturn:
        message = (
            f"Invalid Amazon Bedrock model identifier {model_id!r}. "
            f"Bedrock replied: {_bedrock_error_message(error)}."
        )
        if self._region_name:
            message += f" Requested region: {self._region_name!r}."
        message += (
            " Verify the exact Bedrock foundation-model or "
            "inference-profile ID for your account."
        )
        if model_id.startswith("anthropic.") and not model_id.startswith(
            (
                "us.",
                "eu.",
                "apac.",
            )
        ):
            prefix = _geo_inference_prefix(self._region_name)
            if prefix:
                message += (
                    " Anthropic Bedrock models in this region may require "
                    "a geo-prefixed inference profile ID."
                    f" Try {prefix!r} as the model ID prefix."
                )
            else:
                message += (
                    " Anthropic Bedrock models may require a geo-prefixed "
                    "inference profile such as 'us.anthropic...'."
                )
        raise ValueError(message) from error

    @staticmethod
    def _is_inference_profile_required_error(error: Exception) -> bool:
        if _bedrock_error_code(error) != "ValidationException":
            return False
        message = _bedrock_error_message(error).lower()
        return (
            "on-demand throughput" in message
            and "inference profile" in message
        )

    def _raise_inference_profile_required_error(
        self, model_id: str, error: Exception
    ) -> NoReturn:
        message = (
            f"Amazon Bedrock model identifier {model_id!r} cannot be invoked "
            "directly with on-demand throughput. "
            f"Bedrock replied: {_bedrock_error_message(error)}."
        )
        if self._region_name:
            message += f" Requested region: {self._region_name!r}."
        message += " Use an inference-profile ID or ARN for this model."
        if model_id.startswith("anthropic."):
            prefix = _geo_inference_prefix(self._region_name) or "us."
            regional_profile = prefix + model_id
            global_profile = "global." + model_id
            message += f" Try {regional_profile!r} or {global_profile!r}."
        raise ValueError(message) from error

    @staticmethod
    def _is_use_case_details_required_error(error: Exception) -> bool:
        if _bedrock_error_code(error) != "ResourceNotFoundException":
            return False
        message = _bedrock_error_message(error).lower()
        return (
            "use case details have not been submitted" in message
            or "fill out the request form" in message
        )

    def _raise_use_case_details_required_error(
        self, model_id: str, error: Exception
    ) -> NoReturn:
        message = (
            "Amazon Bedrock blocked access to model identifier "
            f"{model_id!r} because Anthropic use-case details have not "
            "been submitted for this account. "
            f"Bedrock replied: {_bedrock_error_message(error)}."
        )
        if self._region_name:
            message += f" Requested region: {self._region_name!r}."
        message += (
            " Submit the Anthropic model access form in Amazon Bedrock, "
            "then retry."
        )
        message += (
            " You can verify the current status with "
            "'aws bedrock get-use-case-for-model-access --region "
            f"{self._region_name or 'us-east-1'}'."
        )
        raise ValueError(message) from error

    @staticmethod
    def _is_end_of_life_model_error(error: Exception) -> bool:
        if _bedrock_error_code(error) != "ResourceNotFoundException":
            return False
        return "end of its life" in _bedrock_error_message(error).lower()

    def _raise_end_of_life_model_error(
        self, model_id: str, error: Exception
    ) -> NoReturn:
        message = (
            f"Amazon Bedrock model identifier {model_id!r} is no longer "
            "usable because that model version reached end of life. "
            f"Bedrock replied: {_bedrock_error_message(error)}."
        )
        if self._region_name:
            message += f" Requested region: {self._region_name!r}."
        message += (
            " Use an active inference-profile ID instead of the retired "
            "profile or model version."
        )
        if model_id.startswith(("us.anthropic.", "eu.anthropic.")):
            prefix = model_id.split(".", 1)[0]
            message += (
                " List current options with "
                "'aws bedrock list-inference-profiles --region "
                f"{self._region_name or 'us-east-1'}' and look for active "
                f"{prefix}.anthropic profiles."
            )
        elif model_id.startswith("anthropic."):
            geo_prefix = _geo_inference_prefix(self._region_name)
            if geo_prefix:
                message += (
                    " Anthropic Bedrock models in this region are typically "
                    "invoked through inference profiles."
                    f" Try an active {geo_prefix!r}-prefixed profile."
                )
        raise ValueError(message) from error

    def _inference_config(
        self, settings: GenerationSettings | None
    ) -> dict[str, Any] | None:
        if settings is None:
            return None
        config: dict[str, Any] = {}
        if settings.max_new_tokens is not None:
            config["maxTokens"] = settings.max_new_tokens
        if settings.temperature is not None:
            config["temperature"] = settings.temperature
        if settings.top_p is not None:
            config["topP"] = settings.top_p
        if settings.top_k is not None:
            config["topK"] = settings.top_k
        if settings.stop_strings is not None:
            stop = (
                [settings.stop_strings]
                if isinstance(settings.stop_strings, str)
                else settings.stop_strings
            )
            config["stopSequences"] = stop
        return config or None

    def _tool_config(self, tool: ToolManager | None) -> dict[str, Any] | None:
        schemas = self._tool_schemas(tool) if tool else None
        if not schemas:
            return None
        return {"tools": schemas, "toolChoice": {"auto": {}}}

    def _response_text(self, response: dict[str, Any]) -> str:
        output = response.get("output") if isinstance(response, dict) else None
        message = output.get("message") if isinstance(output, dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list):
            return ""
        parts: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            text_block = block.get("text")
            text_value = _string(text_block)
            if text_value:
                parts.append(text_value)
        return "".join(parts)

    def _template_messages(
        self,
        messages: list[Message],
        exclude_roles: list[TemplateMessageRole] | None = None,
    ) -> list[dict[str, Any]]:
        templated: list[dict[str, Any]] = []
        for message in messages:
            if exclude_roles and str(message.role) in exclude_roles:
                continue
            if message.role == MessageRole.TOOL:
                result = message.tool_call_result or message.tool_call_error
                if result:
                    templated.append(self._tool_result_message(result))
                continue
            templated.append(self._format_message(message))
        return templated

    def _format_message(self, message: Message) -> dict[str, Any]:
        role = str(message.role)
        if role == str(MessageRole.DEVELOPER):
            role = str(MessageRole.USER)
        content_blocks = self._format_content(message.content)
        if message.tool_calls:
            for tool_call in message.tool_calls:
                encoded_name = TextGenerationVendor.encode_tool_name(
                    tool_call.name
                )
                content_blocks.append(
                    {
                        "toolUse": {
                            "toolUseId": tool_call.id,
                            "name": encoded_name,
                            "input": tool_call.arguments or [],
                        }
                    }
                )
        return {"role": role, "content": content_blocks}

    def _format_content(
        self, content: str | MessageContent | list[MessageContent] | None
    ) -> list[dict[str, Any]]:
        if content is None:
            return []
        if isinstance(content, str):
            return [{"text": content}]
        if isinstance(content, MessageContentText):
            return [{"text": content.text}]
        if isinstance(content, MessageContentFile):
            return self._ensure_document_prompt(
                [{"document": self._document_block(content.file)}]
            )
        if isinstance(content, MessageContentImage):
            return [
                {"image": {"source": self._image_source(content.image_url)}}
            ]
        if isinstance(content, list):
            blocks: list[dict[str, Any]] = []
            for block in content:
                if isinstance(block, MessageContentText):
                    blocks.append({"text": block.text})
                elif isinstance(block, MessageContentFile):
                    blocks.append(
                        {"document": self._document_block(block.file)}
                    )
                elif isinstance(block, MessageContentImage):
                    blocks.append(
                        {
                            "image": {
                                "source": self._image_source(block.image_url)
                            }
                        }
                    )
            return self._ensure_document_prompt(blocks)
        return [{"text": str(content)}]

    @staticmethod
    def _ensure_document_prompt(
        blocks: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        has_document = any("document" in block for block in blocks)
        has_text = any("text" in block for block in blocks)
        if has_document and not has_text:
            return [{"text": ""}, *blocks]
        return blocks

    @staticmethod
    def _document_block(file: Mapping[str, Any]) -> dict[str, Any]:
        block: dict[str, Any] = {
            "name": BedrockClient._document_name(file),
            "source": BedrockClient._document_source(file),
        }
        document_format = BedrockClient._document_format(file)
        if document_format:
            block["format"] = document_format
        citations = file.get("citations")
        if isinstance(citations, bool):
            block["citations"] = {"enabled": citations}
        context = file.get("context")
        if isinstance(context, str):
            block["context"] = context
        return block

    @staticmethod
    def _document_source(file: Mapping[str, Any]) -> dict[str, Any]:
        mime_type_value = file.get("mime_type")
        mime_type = (
            mime_type_value.lower()
            if isinstance(mime_type_value, str) and mime_type_value
            else None
        )
        data = file.get("file_data", file.get("data"))
        if isinstance(data, (bytes, bytearray)):
            return {"bytes": bytes(data)}
        if isinstance(data, str):
            if mime_type is not None and mime_type.startswith("text/"):
                return {"text": data}
            return {"bytes": b64decode(data)}

        file_uri = BedrockClient._file_uri(file)
        assert (
            file_uri is not None
        ), "Bedrock documents require inline data or a file URL"
        assert file_uri.startswith(
            "s3://"
        ), "Bedrock document URLs must use s3:// URIs"
        s3_location: dict[str, Any] = {"uri": file_uri}
        bucket_owner = file.get("bucket_owner")
        if isinstance(bucket_owner, str):
            s3_location["bucketOwner"] = bucket_owner
        return {"s3Location": s3_location}

    @staticmethod
    def _document_name(file: Mapping[str, Any]) -> str:
        name = "Document"
        for key in ("title", "filename"):
            value = file.get(key)
            if isinstance(value, str) and value.strip():
                name = value.strip().rsplit(".", 1)[0]
                break
        sanitized = sub(r"[^0-9A-Za-z\s\-\(\)\[\]]", " ", name)
        normalized = sub(r"\s+", " ", sanitized).strip()
        return normalized or "Document"

    @staticmethod
    def _document_format(file: Mapping[str, Any]) -> str | None:
        mime_type_value = file.get("mime_type")
        if isinstance(mime_type_value, str):
            mime_type = mime_type_value.lower()
            document_format = _BEDROCK_DOCUMENT_FORMATS.get(mime_type)
            if document_format:
                return document_format

        for key in ("filename", "title"):
            value = file.get(key)
            if not isinstance(value, str):
                continue
            suffix = value.rsplit(".", 1)
            if len(suffix) != 2:
                continue
            extension = suffix[1].lower()
            if extension in {
                "csv",
                "doc",
                "docx",
                "html",
                "md",
                "pdf",
                "txt",
                "xls",
                "xlsx",
            }:
                return extension
        return None

    @staticmethod
    def _file_uri(file: Mapping[str, Any]) -> str | None:
        for key in ("file_url", "url", "uri"):
            value = file.get(key)
            if isinstance(value, str) and value:
                return value
        return None

    def _image_source(self, image_url: dict[str, Any]) -> dict[str, Any]:
        if "url" in image_url:
            return {"type": "url", "url": image_url["url"]}
        if "data" in image_url:
            media_type = image_url.get("mime_type", "image/png")
            return {
                "type": "base64",
                "mediaType": media_type,
                "data": image_url["data"],
            }
        return {"type": "url", "url": image_url.get("uri", "")}

    def _tool_result_message(
        self, result: ToolCallResult | ToolCallError
    ) -> dict[str, Any]:
        content: dict[str, Any] = {
            "toolUseId": result.call.id,
            "content": [
                {
                    "text": to_json(
                        result.result
                        if isinstance(result, ToolCallResult)
                        else result.message
                    )
                }
            ],
            "status": (
                "success" if isinstance(result, ToolCallResult) else "error"
            ),
        }
        if isinstance(result, ToolCallError):
            content["error"] = {
                "name": result.error.__class__.__name__,
                "message": result.message,
            }
        return {
            "role": str(MessageRole.USER),
            "content": [{"toolResult": content}],
        }

    @staticmethod
    def _tool_schemas(tool: ToolManager) -> list[dict[str, Any]] | None:
        schemas = tool.json_schemas()
        if not schemas:
            return None
        tools: list[dict[str, Any]] = []
        for schema in schemas:
            if schema.get("type") != "function":
                continue
            function = schema.get("function") or {}
            encoded_name = TextGenerationVendor.encode_tool_name(
                function.get("name", "")
            )
            tools.append(
                {
                    "toolSpec": {
                        "name": encoded_name,
                        "description": function.get("description", ""),
                        "inputSchema": {
                            "json": function.get("parameters", {})
                        },
                    }
                }
            )
        return tools or None


class BedrockModel(TextGenerationVendorModel):
    def _load_model(
        self,
    ) -> PreTrainedModel | TextGenerationVendor | DiffusionPipeline:
        return BedrockClient(
            exit_stack=self._exit_stack,
            region_name=self._settings.base_url,
            endpoint_url=self._settings.access_token,
        )

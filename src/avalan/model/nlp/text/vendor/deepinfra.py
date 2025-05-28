from .....model import TextGenerationVendor
from .openai import OpenAIClient, OpenAIModel
from transformers import PreTrainedModel


class DeepInfraClient(OpenAIClient):
    def __init__(self, api_key: str, base_url: str | None = None):
        super().__init__(
            api_key=api_key,
            base_url=base_url or "https://api.deepinfra.com/v1/openai",
        )


class DeepInfraModel(OpenAIModel):
    def _load_model(self) -> PreTrainedModel | TextGenerationVendor:
        assert self._settings.access_token
        return DeepInfraClient(
            base_url=self._settings.base_url,
            api_key=self._settings.access_token,
        )

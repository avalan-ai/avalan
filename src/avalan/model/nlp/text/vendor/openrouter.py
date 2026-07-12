from .....model.provider import ProviderFamily
from ....vendor import TextGenerationVendor
from . import DiffusionPipeline, PreTrainedModel
from .openai import OpenAIClient, OpenAIModel


class OpenRouterClient(OpenAIClient):
    _reasoning_summary_provider = "openrouter"

    def __init__(self, api_key: str, base_url: str | None = None):
        super().__init__(
            api_key=api_key,
            base_url=base_url or "https://openrouter.ai/api/v1",
        )

    @property
    def _usage_provider_family(self) -> ProviderFamily:
        return ProviderFamily.OPENAI_COMPATIBLE


class OpenRouterModel(OpenAIModel):
    def _load_model(
        self,
    ) -> PreTrainedModel | TextGenerationVendor | DiffusionPipeline:
        assert self._settings.access_token
        return OpenRouterClient(
            base_url=self._settings.base_url,
            api_key=self._settings.access_token,
        )

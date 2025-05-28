from .....model import TextGenerationVendor
from .....model.nlp.text.vendor.openai import OpenAIClient, OpenAIModel
from transformers import PreTrainedModel


class LiteLLMClient(OpenAIClient):
    def __init__(self, api_key: str | None = None, base_url: str | None = None):
        super().__init__(
            api_key=api_key or "",
            base_url=base_url or "http://localhost:4000",
        )


class LiteLLMModel(OpenAIModel):
    def _load_model(self) -> PreTrainedModel | TextGenerationVendor:
        return LiteLLMClient(
            base_url=self._settings.base_url,
            api_key=self._settings.access_token,
        )

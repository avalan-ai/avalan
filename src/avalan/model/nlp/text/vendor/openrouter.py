from .....model import TextGenerationVendor
from .....model.nlp.text.vendor.openai import OpenAIClient, OpenAIModel
from transformers import PreTrainedModel


class OpenRouterClient(OpenAIClient):
    def __init__(self, api_key: str, base_url: str | None = None):
        super().__init__(
            api_key=api_key,
            base_url=base_url or "https://openrouter.ai/api/v1",
        )
        # Optional headers recommended by OpenRouter
        self._client.headers.update(
            {
                "HTTP-Referer": "https://github.com/avalan-ai/avalan",
                "X-Title": "avalan",
            }
        )


class OpenRouterModel(OpenAIModel):
    def _load_model(self) -> PreTrainedModel | TextGenerationVendor:
        assert self._settings.access_token
        return OpenRouterClient(
            base_url=self._settings.base_url,
            api_key=self._settings.access_token,
        )

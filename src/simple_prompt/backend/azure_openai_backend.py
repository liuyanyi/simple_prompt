from typing import List, TypedDict

from openai import AzureOpenAI

from .base import LLMBackendHook
from .openai_backend import OpenAILLMBackend


class AzureOpenAIConfig(TypedDict):
    model_name: str
    api_key: str | None

    azure_endpoint: str | None
    api_version: str | None

    timeout: float | None

    default_body: dict | None


class AzureOpenAILLMBackend(OpenAILLMBackend):
    def __init__(
        self,
        name: str | None = None,
        concurrency: int = 20,
        config: AzureOpenAIConfig | None = None,
        hooks: List[LLMBackendHook] | None = None,
        logger=None,
    ):
        assert config is not None, "config must be provided"
        self.model_name = config.pop("model_name", None)
        assert self.model_name is not None, "model_name must be provided"
        if name is None:
            name = self.model_name

        super(OpenAILLMBackend, self).__init__(
            name=name,
            concurrency=concurrency,
            config=config,
            hooks=hooks,
            logger=logger,
        )

        # 构造openai client
        self.client = AzureOpenAI(**config)
        self.default_body = config.get("default_body", {})

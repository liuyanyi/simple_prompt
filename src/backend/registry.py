import logging
from typing import Dict, List, Type, Optional

from .base import BaseLLMBackend, LLMBackendHook
from .openai_backend import OpenAILLMBackend
from .vllm_backend import vLLMBackend
from .exception import NotRegisteredBackendError

logger = logging.getLogger(__name__)

_registry: Dict[str, Type[BaseLLMBackend]] = {
    "openai": OpenAILLMBackend,
    "vllm": vLLMBackend,
}


class LLMBackendRegistry:
    def __init__(self):
        self.default_backend_name: Optional[str] = None
        self.all_backends: Dict[str, BaseLLMBackend] = {}

    def register_backend(
        self,
        name: Optional[str] = None,
        config: Optional[dict] = None,
        backend_type: str = "openai",
        backend: Optional[BaseLLMBackend] = None,
        concurrency: int = 20,
        hooks: Optional[List[LLMBackendHook]] = None,
    ):
        if backend is None:
            assert config is not None, "config must be provided"
            backend_cls = _registry.get(backend_type, OpenAILLMBackend)
            backend = backend_cls(name=name, config=config, concurrency=concurrency)

        for hook in hooks or []:
            backend.add_hook(hook)

        self.all_backends[backend.display_name] = backend
        logger.info(f"Registered LLM backend for model {backend.display_name}")

    def set_default_backend(self, backend_name: str) -> BaseLLMBackend:
        backend = self.all_backends.get(backend_name)
        if backend is None:
            raise NotRegisteredBackendError(f"Unknown model name {backend_name}")

        self.default_backend_name = backend_name
        logger.info(f"Set default LLM backend to {backend_name}")
        return backend

    def get_backend(self, backend_name: str = "default") -> BaseLLMBackend:
        if backend_name == "default":
            if self.default_backend_name is None:
                raise NotRegisteredBackendError("No default LLM backend set")
            backend_name = self.default_backend_name

        backend = self.all_backends.get(backend_name)
        if backend is None:
            raise NotRegisteredBackendError(f"Unknown model name {backend_name}")
        return backend

    def get_default_backend_name(self) -> Optional[str]:
        return self.default_backend_name

    def shutdown(self):
        for backend in self.all_backends.values():
            del backend
        self.all_backends.clear()


registry = LLMBackendRegistry()


def register_backend(
    name: Optional[str] = None,
    config: Optional[dict] = None,
    backend_type: str = "openai",
    backend: Optional[BaseLLMBackend] = None,
    concurrency: int = 20,
    hooks: Optional[List[LLMBackendHook]] = None,
):
    registry.register_backend(
        name=name,
        config=config,
        backend_type=backend_type,
        backend=backend,
        concurrency=concurrency,
        hooks=hooks,
    )


def set_default_backend(backend_name: str) -> BaseLLMBackend:
    registry.set_default_backend(backend_name)


def get_backend(backend_name: str = "default") -> BaseLLMBackend:
    return registry.get_backend(backend_name)


def shutdown():
    registry.shutdown()

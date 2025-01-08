import asyncio
import copy
import inspect
import json
import time
from concurrent.futures import Future
from functools import partial
from typing import Callable, Dict, Generator, List, Mapping, Type, TypeVar

import openai
from loguru import logger
from openai import OpenAI
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from pydantic import BaseModel
from typing_extensions import TypedDict

from src.utils import beautify_time

from .base import BackendHook, BackendInterface, LLMBackendHook, LLMBackendInterface
from .protocol import (
    GuidedDecodeConfig,
    MetaInfo,
    exception_output_type,
    message_type,
    output_type,
    raw_output_type,
)

default_llm_backend_name = None
all_llm_backends: Dict[str, "LLMBackendInterface"] = dict()

default_embedding_backend_name = None
all_embedding_backends: Dict[str, "EmbeddingBackend"] = dict()


T = TypeVar("T")



def register_embedding_backend(
    name: str | None = None,
    config: dict | None = None,
    backend_type: str | None = "openai",
    backend: EmbeddingBackend | None = None,
    concurrency: int = 20,
    hooks: List[BackendHook] | None = None,  # TODO
):
    global all_embedding_backends
    if backend is not None:
        all_embedding_backends[backend.display_name] = backend
        logger.info(f"Registered embedding backend for model {backend.display_name}")
    else:
        assert config is not None, "config must be provided"
        backend_cls = EmbeddingBackend  # TODO: add support for other backend types
        backend = backend_cls(name=name, config=config, concurrency=concurrency)
        all_embedding_backends[backend.display_name] = backend
        logger.info(f"Registered embedding backend for model {backend.display_name}")


class NotRegisteredBackendError(Exception):
    pass


def set_default_llm_backend(backend_name: str):
    global default_llm_backend_name
    default_backend = all_llm_backends.get(backend_name)
    if default_backend is None:
        raise NotRegisteredBackendError(f"Unknown model name {backend_name}")

    default_llm_backend_name = backend_name

    logger.info(f"Set default LLM backend to {backend_name}")

    return default_backend


def set_default_embedding_backend(backend_name: str):
    global default_embedding_backend_name
    default_backend = all_embedding_backends.get(backend_name)
    if default_backend is None:
        raise NotRegisteredBackendError(f"Unknown model name {backend_name}")

    default_embedding_backend_name = backend_name

    logger.info(f"Set default embedding backend to {backend_name}")

    return default_backend


def get_llm_backend(backend_name: str = "default") -> LLMBackendInterface:
    global default_llm_backend_name
    if backend_name == "default":
        if default_llm_backend_name is None:
            raise NotRegisteredBackendError("No default LLM backend set")
        backend_name = default_llm_backend_name
    backend = all_llm_backends.get(backend_name)
    if backend is None:
        raise NotRegisteredBackendError(f"Unknown model name {backend_name}")
    return backend


def get_default_llm_backend_name() -> str:
    return default_llm_backend_name


def get_embedding_backend(backend_name: str = "default") -> EmbeddingBackend:
    global default_embedding_backend_name
    if backend_name == "default":
        if default_embedding_backend_name is None:
            raise NotRegisteredBackendError("No default embedding backend set")
        backend_name = default_embedding_backend_name
    backend = all_embedding_backends.get(backend_name)
    if backend is None:
        raise NotRegisteredBackendError(f"Unknown model name {backend_name}")
    return backend


def get_default_embedding_backend_name() -> str:
    return default_embedding_backend_name


def shutdown_backends():
    for backend in all_llm_backends.values():
        del backend
    for backend in all_embedding_backends.values():
        del backend
    all_llm_backends.clear()
    all_embedding_backends.clear()

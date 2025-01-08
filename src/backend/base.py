from abc import ABC, abstractmethod
from concurrent.futures import Future, ThreadPoolExecutor
import logging
from typing import Generator, List

from ..protocol import (
    MetaInfo,
    message_type,
    output_type,
    raw_output_type,
    exception_output_type,
    GuidedDecodeConfig,
)


class LLMBackendHook(ABC):
    HOOK_NAME = "AbstractHook"

    @abstractmethod
    def on_request_start(
        self,
        display_name: str,
        start_time: float,
        input_data: dict,
        request_id: str | None,
        **kwargs,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def on_request_end(
        self,
        display_name: str,
        raw_output: str,
        meta: MetaInfo,
        input_data: dict,
        **kwargs,
    ) -> None:
        raise NotImplementedError


class BaseLLMBackend(ABC):
    def __init__(
        self,
        name: str = None,
        concurrency: int = 20,
        hooks: List[LLMBackendHook] | None = None,
        logger=None,
    ):
        self.display_name = name
        self.thread_pool = ThreadPoolExecutor(max_workers=concurrency)
        self.hooks: List[LLMBackendHook] = hooks or []
        self.logger = logger or logging.getLogger(__name__)

    def add_hook(self, hook: LLMBackendHook):
        self.hooks.append(hook)

    def __del__(self):
        try:
            if not getattr(self, "thread_pool", None):
                return
            self.thread_pool.shutdown(wait=True)
        except Exception as e:
            self.logger.error(f"Error shutting down thread pool: {e}")

    @abstractmethod
    def chat(
        self,
        messages: message_type,
        request_id: str = None,
        generation_config: dict | None = None,
        guided_decode_config: GuidedDecodeConfig | None = None,
    ) -> "output_type|exception_output_type":
        raise NotImplementedError

    @abstractmethod
    def chat_stream(
        self,
        messages: message_type,
        request_id: str = None,
        generation_config: dict | None = None,
        guided_decode_config: GuidedDecodeConfig | None = None,
    ) -> "Generator[raw_output_type|exception_output_type]":
        raise NotImplementedError

    @abstractmethod
    def chat_in_thread(
        self,
        messages: message_type,
        request_id: str = None,
        generation_config: dict | None = None,
        guided_decode_config: GuidedDecodeConfig | None = None,
    ) -> "Future[output_type|exception_output_type]":
        raise NotImplementedError

    @abstractmethod
    def _chat_in_main_thread(
        self,
        messages: message_type,
        request_id: str = None,
        generation_config: dict | None = None,
        guided_decode_config: GuidedDecodeConfig | None = None,
    ) -> "output_type":
        raise NotImplementedError

    def _hooks_on_request_start(
        self, start_time: float, input_data: dict, request_id, **kwargs
    ):
        for hook in self.hooks:
            try:
                hook.on_request_start(
                    self.display_name,
                    start_time,
                    input_data,
                    request_id,
                    **kwargs,
                )
            except Exception as e:
                self.logger.error(
                    f"Error in hook {hook.HOOK_NAME} on_request_start: {e}"
                )

    def _hooks_on_request_end(
        self, raw_output: str, meta: MetaInfo, input_data: dict, **kwargs
    ):
        for hook in self.hooks:
            try:
                hook.on_request_end(
                    self.display_name,
                    raw_output,
                    meta,
                    input_data,
                    **kwargs,
                )
            except Exception as e:
                self.logger.error(f"Error in hook {hook.HOOK_NAME} on_request_end: {e}")

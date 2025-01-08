import logging
from ..protocol import MetaInfo
from ..utils import beautify_time
from .base import LLMBackendHook


class DebugLoggingHook(LLMBackendHook):
    HOOK_NAME = "DebugLoggingHook"

    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)

    def on_request_start(
        self,
        display_name: str,
        start_time: float,
        input_data: dict,
        request_id: str | None,
        **kwargs,
    ) -> None:
        if request_id is not None:
            msgs = input_data.get("messages")
            if msgs:
                msgs_strs = []
                for msg in msgs:
                    msgs_strs.append(f'<{msg["role"]}> {msg["content"]}')
                debug_str = "\n".join(
                    [
                        "DEBUG DETAILED INFO",
                        "Request id: " + request_id,
                        "Model: " + display_name,
                        "Start time: " + beautify_time(start_time),
                        "Messages: ",
                        *msgs_strs,
                    ]
                )
                self.logger.debug(debug_str)

    def on_request_end(
        self,
        display_name: str,
        raw_output: str,
        meta: MetaInfo,
        input_data: dict,
        **kwargs,
    ) -> None:
        if meta.request_id is not None:
            debug_str = "\n".join(
                [
                    "DEBUG DETAILED INFO",
                    "Request id: " + meta.request_id,
                    "Model: " + display_name,
                    "Start time: " + beautify_time(meta.start_time),
                    "End time: " + beautify_time(meta.end_time),
                    "Runtime: " + f"{meta.end_time - meta.start_time:.2f} s",
                    "Response: " + str(raw_output),
                ]
            )
            self.logger.debug(debug_str)

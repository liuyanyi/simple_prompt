import logging

from simple_prompt.protocol import MetaInfo
from simple_prompt.utils import beautify_time

from .base import LLMBackendHook


class LoggingHook(LLMBackendHook):
    HOOK_NAME = "LoggingHook"

    def __init__(self, print_fn=None):
        self.print_fn = print_fn or print

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
                    msgs_strs.append(f"<{msg['role']}> {msg['content']}")
                debug_str = "\n".join(
                    [
                        "== Request Start ==",
                        "Request id: " + request_id,
                        "Model: " + display_name,
                        "Start time: " + beautify_time(start_time),
                        "Messages: ",
                        *msgs_strs,
                    ]
                )
                self.print_fn(debug_str)

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
                    "== Request End ==",
                    "Request id: " + meta.request_id,
                    "Model: " + display_name,
                    "Start time: " + beautify_time(meta.start_time),
                    "End time: " + beautify_time(meta.end_time),
                    "Runtime: " + f"{meta.end_time - meta.start_time:.2f} s",
                    "Response: " + str(raw_output),
                ]
            )
            self.print_fn(debug_str)

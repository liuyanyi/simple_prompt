import copy
import inspect

import openai

from ..protocol import (
    GuidedDecodeConfig,
    message_type,
)
from .openai_backend import OpenAILLMBackend


class vLLMBackend(OpenAILLMBackend):
    def _process_input_data(
        self,
        messages: message_type,
        generation_config: dict | None = None,
        guided_decode_config: GuidedDecodeConfig | None = None,
    ) -> dict:
        """Process input data for the LLM backend."""

        # 检查sampling_params中，参数是否都在openai.ChatCompletion.create()中
        all_create_params = list(
            inspect.signature(openai.resources.chat.Completions.create).parameters
        )
        not_allowed_params = ["messages", "model"]
        _generation_config = copy.deepcopy(generation_config)
        for k, v in _generation_config.items():
            if k in not_allowed_params:
                raise ValueError(f"Parameter {k} is not allowed in sampling params")

        if guided_decode_config:
            # 如果有guided_decode_config，需要额外处理
            base_model = guided_decode_config.base_model
            if isinstance(base_model, dict):
                # json schema
                json_schema = base_model
                # json_schema_name = "json_schema"
            else:
                # BaseModel
                json_schema = base_model.model_json_schema()
                if guided_decode_config.use_list:
                    json_schema = {"type": "array", "items": json_schema}
                # json_schema_name = base_model.__name__.lower()
            # OPENAI FORMAT JSON SCHEMA
            if "response_format" in _generation_config:
                raise ValueError(
                    "response_format is already provided in sampling params"
                )

            _generation_config["guided_json"] = json_schema
            # 其他可能需要的参数
            if hasattr(guided_decode_config, "guided_decoding_backend"):
                _generation_config["guided_decoding_backend"] = (
                    guided_decode_config.guided_decoding_backend
                )

        # 将不在openai.ChatCompletion.create()中的参数过滤到extra_body中
        extra_body = {}
        filtered_sampling_params = {}
        for k, v in _generation_config.items():
            if k in all_create_params:
                filtered_sampling_params[k] = v
            elif k not in not_allowed_params:
                extra_body[k] = v
            else:
                raise ValueError(f"Parameter {k} is not allowed in sampling params")

        # 合并默认body
        for k, v in self.default_body.items():
            if k not in filtered_sampling_params:
                filtered_sampling_params[k] = v

        return {
            "messages": messages,
            "extra_body": extra_body,
            **filtered_sampling_params,
        }

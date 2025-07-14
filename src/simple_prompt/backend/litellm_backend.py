import asyncio
import copy
import json
import time
from concurrent.futures import Future
from functools import partial
from typing import Callable, Generator, List

import litellm
from litellm import ModelResponse, get_supported_openai_params
from pydantic import BaseModel
from typing_extensions import TypedDict

from simple_prompt.protocol import (
    GuidedDecodeConfig,
    MetaInfo,
    exception_output_type,
    message_type,
    output_type,
    raw_output_type,
)
from .base import BaseLLMBackend, LLMBackendHook


class LiteLLMConfig(TypedDict):
    model_name: str
    api_key: str | None
    api_base: str | None
    api_version: str | None
    timeout: float | None
    temperature: float | None
    max_tokens: int | None
    top_p: float | None
    frequency_penalty: float | None
    presence_penalty: float | None
    stop: str | List[str] | None

    # LiteLLM specific config
    custom_llm_provider: str | None
    vertex_project: str | None
    vertex_location: str | None
    aws_access_key_id: str | None
    aws_secret_access_key: str | None
    aws_region_name: str | None

    default_body: dict | None


class LiteLLMBackend(BaseLLMBackend):
    def __init__(
        self,
        name: str | None = None,
        concurrency: int = 20,
        config: LiteLLMConfig | None = None,
        hooks: List[LLMBackendHook] | None = None,
        logger=None,
    ):
        assert config is not None, "config must be provided"
        self.model_name = config.pop("model_name", None)
        assert self.model_name is not None, "model_name must be provided"
        if name is None:
            name = self.model_name

        super().__init__(
            name=name,
            concurrency=concurrency,
            config=config,
            hooks=hooks,
            logger=logger,
        )

        # Configure LiteLLM with environment variables or config
        self.litellm_config = config
        self.default_body = config.get("default_body", {})

        # Store configuration for per-request use instead of setting globals
        # This allows multiple LiteLLM backends with different configurations
        self.api_config = {}
        if config.get("api_key"):
            self.api_config["api_key"] = config["api_key"]
        if config.get("api_base"):
            self.api_config["base_url"] = config["api_base"]  # litellm uses 'base_url'
        if config.get("api_version"):
            self.api_config["api_version"] = config["api_version"]

    def _process_input_data(
        self,
        messages: message_type,
        generation_config: dict | None = None,
        guided_decode_config: GuidedDecodeConfig | None = None,
    ) -> dict:
        """Process input data for the LiteLLM backend."""

        # Get supported parameters for this model
        try:
            supported_params = get_supported_openai_params(model=self.model_name)
        except Exception:
            # Fallback to common OpenAI parameters if we can't get model-specific ones
            supported_params = [
                "temperature",
                "max_tokens",
                "top_p",
                "frequency_penalty",
                "presence_penalty",
                "stop",
                "stream",
                "n",
                "response_format",
            ]

        _generation_config = (
            copy.deepcopy(generation_config) if generation_config else {}
        )

        # Handle guided decode configuration
        if guided_decode_config:
            base_model = guided_decode_config.base_model
            if isinstance(base_model, dict):
                # json schema
                json_schema = base_model
                json_schema_name = "json_schema"
            else:
                # BaseModel
                json_schema = base_model.model_json_schema()
                # Force additionalProperties to be False
                json_schema["additionalProperties"] = False
                if guided_decode_config.use_list:
                    json_schema = {"type": "array", "items": json_schema}
                json_schema_name = base_model.__name__.lower()

            # LiteLLM supports response_format for compatible models
            if "response_format" in _generation_config:
                raise ValueError(
                    "response_format is already provided in generation config"
                )

            _generation_config["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": json_schema_name,
                    "strict": True,
                    "schema": json_schema,
                },
            }

        # Filter parameters based on what's supported
        filtered_params = {}
        for k, v in _generation_config.items():
            if (
                k in supported_params or k == "response_format"
            ):  # Always allow response_format for guided decode
                filtered_params[k] = v
            else:
                self.logger.warning(
                    f"Parameter {k} not supported for model {self.model_name}, skipping"
                )

        # Merge with default body
        for k, v in self.default_body.items():
            if k not in filtered_params:
                filtered_params[k] = v

        # Add LiteLLM specific config
        for key in [
            "custom_llm_provider",
            "vertex_project",
            "vertex_location",
            "aws_access_key_id",
            "aws_secret_access_key",
            "aws_region_name",
        ]:
            if key in self.litellm_config and self.litellm_config[key]:
                filtered_params[key] = self.litellm_config[key]

        # Merge API configuration (api_key, base_url, api_version) with other params
        return {
            "messages": messages,
            "model": self.model_name,
            **self.api_config,
            **filtered_params,
        }

    def _process_result(
        self,
        result: ModelResponse,
        request_id: str | None = None,
        start_time: float | None = None,
    ) -> "raw_output_type":
        finish_time = time.time()

        # Extract content from LiteLLM response
        if hasattr(result, "choices") and result.choices:
            if hasattr(result.choices[0], "delta") and result.choices[0].delta:
                # Streaming response
                result_texts = [choice.delta.content for choice in result.choices]
            elif hasattr(result.choices[0], "message") and result.choices[0].message:
                # Regular response
                result_texts = [choice.message.content for choice in result.choices]
            else:
                result_texts = [""]
        else:
            result_texts = [""]

        # Handle None values
        for i in range(len(result_texts)):
            if result_texts[i] is None:
                result_texts[i] = ""

        # If single result, unwrap from list
        if len(result_texts) == 1:
            result_texts = result_texts[0]

        if request_id is None:
            request_id = getattr(result, "id", f"litellm-{time.time()}")

        # Extract finish reasons
        finish_reasons = []
        if hasattr(result, "choices") and result.choices:
            finish_reasons = [choice.finish_reason for choice in result.choices]

        meta_data = MetaInfo(
            request_id=request_id,
            success=True,
            model=getattr(result, "model", self.model_name),
            start_time=start_time,
            end_time=finish_time,
            finish_reason=finish_reasons,
            usage=getattr(result, "usage", None),
            original_result=result,
        )
        return result_texts, meta_data

    def _process_exception(
        self,
        e: Exception,
        request_id: str | None = None,
        start_time: float | None = None,
    ) -> "exception_output_type":
        if request_id is None:
            request_id = f"litellm-error-{time.time()}"
        self.logger.warning(
            f"Request {request_id} with {self.display_name} failed with error: {e}"
        )
        end_time = time.time()
        return None, MetaInfo(
            request_id=request_id,
            success=False,
            error=str(e),
            start_time=start_time,
            end_time=end_time,
        )

    def _build_guided_decode_post_processor(
        self, guided_decode_config: GuidedDecodeConfig
    ) -> Callable[[str, MetaInfo], output_type]:
        base_model = guided_decode_config.base_model
        use_list = guided_decode_config.use_list

        if isinstance(base_model, dict):
            # json schema
            def _json_schema_postprocess(
                result: str | None,
                meta: MetaInfo,
            ) -> output_type[dict]:
                if not result:
                    return None, meta
                try:
                    json_result = json.loads(result)
                    return json_result, meta
                except Exception as e:
                    meta.success = False
                    meta.error = str(e)
                    return None, meta

            return _json_schema_postprocess
        else:
            # BaseModel
            def _basemodel_postprocess(
                result: str | None,
                meta: MetaInfo,
            ) -> output_type[BaseModel]:
                if not result:
                    return None, meta
                try:
                    json_result = json.loads(result)
                    if use_list:
                        assert isinstance(json_result, list)
                        parsed_result = [
                            base_model.model_validate(item) for item in json_result
                        ]
                    else:
                        parsed_result = base_model.model_validate_json(result)
                    return parsed_result, meta
                except Exception as e:
                    meta.success = False
                    meta.error = str(e)
                    return None, meta

            return _basemodel_postprocess

    def _chat_in_main_thread(
        self,
        messages: message_type,
        request_id: str = None,
        generation_config: dict | None = None,
        guided_decode_config: GuidedDecodeConfig | None = None,
    ) -> "output_type":
        try:
            litellm_input = self._process_input_data(
                messages, generation_config, guided_decode_config
            )
            post_processor = None
            if guided_decode_config:
                post_processor = self._build_guided_decode_post_processor(
                    guided_decode_config
                )
            start = time.time()
            result = litellm.completion(**litellm_input)
            res, meta = self._process_result(
                result, request_id=request_id, start_time=start
            )
            if post_processor:
                return post_processor(res, meta)
            else:
                return res, meta
        except Exception as e:
            return self._process_exception(e, request_id=request_id, start_time=start)

    def chat(
        self,
        messages: message_type,
        request_id: str = None,
        generation_config: dict | None = None,
        guided_decode_config: GuidedDecodeConfig | None = None,
        use_thread_pool: bool = True,
    ):
        if use_thread_pool:
            future = self.chat_in_thread(
                messages=messages,
                request_id=request_id,
                generation_config=generation_config,
                guided_decode_config=guided_decode_config,
            )
            try:
                result = future.result()
                return result
            except Exception as e:
                return self._process_exception(e, request_id=request_id)
        else:
            return self._chat_in_main_thread(
                messages=messages,
                request_id=request_id,
                generation_config=generation_config,
                guided_decode_config=guided_decode_config,
            )

    def chat_stream(
        self,
        messages: message_type,
        request_id: str = None,
        generation_config: dict | None = None,
        guided_decode_config: GuidedDecodeConfig | None = None,
    ) -> "Generator[raw_output_type|exception_output_type]":
        generation_config = generation_config or {}
        generation_config["stream"] = True

        if guided_decode_config:
            raise ValueError("Guided decode is not supported in stream mode")

        def generator():
            litellm_input = self._process_input_data(
                messages, generation_config, guided_decode_config
            )
            start = time.time()
            try:
                for response in litellm.completion(**litellm_input):
                    res, meta = self._process_result(
                        response, request_id=request_id, start_time=start
                    )
                    yield res, meta
            except Exception as e:
                yield self._process_exception(
                    e, request_id=request_id, start_time=start
                )

        future = self.thread_pool.submit(generator)
        try:
            gen = future.result()
            for response in gen:
                yield response
        except Exception as e:
            yield self._process_exception(e, request_id=request_id)

    def chat_in_thread(
        self,
        messages: message_type,
        request_id: str = None,
        generation_config: dict | None = None,
        guided_decode_config: GuidedDecodeConfig | None = None,
    ) -> "Future[output_type|exception_output_type]":
        generation_config = generation_config or {}
        generation_config["stream"] = False

        def chat_wrapper() -> "raw_output_type | output_type | exception_output_type":
            try:
                start = time.time()
                litellm_input = self._process_input_data(
                    messages, generation_config, guided_decode_config
                )
                post_processor = None
                if guided_decode_config:
                    post_processor = self._build_guided_decode_post_processor(
                        guided_decode_config
                    )

                self._hooks_on_request_start(start, litellm_input, request_id)
                result = litellm.completion(**litellm_input)
                result, meta = self._process_result(
                    result, request_id=request_id, start_time=start
                )
                self._hooks_on_request_end(result, meta, litellm_input)
                if post_processor:
                    return post_processor(result, meta)
                else:
                    return result, meta
            except Exception as e:
                return self._process_exception(
                    e, request_id=request_id, start_time=start
                )

        return self.thread_pool.submit(chat_wrapper)

    async def async_chat(
        self,
        messages: message_type,
        request_id: str = None,
        generation_config: dict | None = None,
        guided_decode_config: GuidedDecodeConfig | None = None,
    ) -> "output_type|exception_output_type":
        generation_config = generation_config or {}
        generation_config["stream"] = False
        litellm_input = self._process_input_data(
            messages, generation_config, guided_decode_config
        )
        loop = asyncio.get_running_loop()
        func = partial(litellm.completion, **litellm_input)
        post_processor = None
        if guided_decode_config:
            post_processor = self._build_guided_decode_post_processor(
                guided_decode_config
            )

        try:
            start = time.time()
            result = await loop.run_in_executor(self.thread_pool, func)
            res, meta = self._process_result(
                result, request_id=request_id, start_time=start
            )
            if post_processor:
                return post_processor(res, meta)
            else:
                return res, meta

        except Exception as e:
            return self._process_exception(e, request_id=request_id, start_time=start)

    async def async_chat_stream(
        self,
        messages: message_type,
        request_id: str = None,
        generation_config: dict | None = None,
        guided_decode_config: GuidedDecodeConfig | None = None,
    ):
        generation_config = generation_config or {}
        generation_config["stream"] = True

        if guided_decode_config:
            raise ValueError("Guided decode is not supported in stream mode")

        loop = asyncio.get_running_loop()

        def generator():
            litellm_input = self._process_input_data(
                messages, generation_config, guided_decode_config
            )
            start = time.time()
            try:
                for response in litellm.completion(**litellm_input):
                    yield self._process_result(
                        response, request_id=request_id, start_time=start
                    )
            except Exception as e:
                yield self._process_exception(
                    e, request_id=request_id, start_time=start
                )

        try:
            gen = await loop.run_in_executor(self.thread_pool, generator)
            for response in gen:
                yield response
        except Exception as e:
            yield self._process_exception(e, request_id=request_id)

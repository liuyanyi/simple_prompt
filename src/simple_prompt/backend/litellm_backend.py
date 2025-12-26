import asyncio
import copy
import inspect
import json
import time
from concurrent.futures import Future
from functools import partial
from typing import Callable, Generator, List

import litellm
from litellm import completion, acompletion
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
    base_url: str | None
    timeout: float | None

    default_body: dict | None


class LiteLLMLLMBackend(BaseLLMBackend):
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

        # Store LiteLLM configuration
        self.api_key = config.get("api_key")
        self.base_url = config.get("base_url")
        self.timeout = config.get("timeout")
        self.default_body = config.get("default_body", {})

    def _process_input_data(
        self,
        messages: message_type,
        generation_config: dict | None = None,
        guided_decode_config: GuidedDecodeConfig | None = None,
    ) -> dict:
        """Process input data for the LLM backend."""

        _generation_config = copy.deepcopy(generation_config) if generation_config else {}
        not_allowed_params = ["messages", "model"]
        
        for k in _generation_config:
            if k in not_allowed_params:
                raise ValueError(f"Parameter {k} is not allowed in generation config")

        if guided_decode_config:
            # Handle guided decoding for LiteLLM
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

        # Merge default body
        for k, v in self.default_body.items():
            if k not in _generation_config:
                _generation_config[k] = v

        return {
            "messages": messages,
            **_generation_config,
        }

    def _process_result(
        self,
        result,
        request_id: str | None = None,
        start_time: float | None = None,
    ) -> "raw_output_type":
        finish_time = time.time()
        
        # Extract content from LiteLLM response
        if hasattr(result, 'choices') and len(result.choices) > 0:
            if hasattr(result.choices[0], 'delta'):
                # Streaming response
                result_texts = [choice.delta.content for choice in result.choices]
            else:
                # Non-streaming response
                result_texts = [choice.message.content for choice in result.choices]
        else:
            result_texts = [""]
        
        for i in range(len(result_texts)):
            if result_texts[i] is None:
                result_texts[i] = ""
        
        result_texts: List[str]
        if len(result_texts) == 1:
            result_texts = result_texts[0]
            result_texts: str
        
        if request_id is None and hasattr(result, 'id'):
            request_id = result.id

        meta_data = MetaInfo(
            request_id=request_id,
            success=True,
            model=result.model if hasattr(result, 'model') else self.model_name,
            start_time=start_time,
            end_time=finish_time,
            finish_reason=[choice.finish_reason for choice in result.choices] if hasattr(result, 'choices') else [],
            usage=result.usage if hasattr(result, 'usage') else None,
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
            request_id = f"unknown-{time.time()}"
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
            
            # Prepare kwargs for LiteLLM
            kwargs = {
                "model": self.model_name,
                **litellm_input,
            }
            if self.api_key:
                kwargs["api_key"] = self.api_key
            if self.base_url:
                kwargs["api_base"] = self.base_url
            if self.timeout:
                kwargs["timeout"] = self.timeout
            
            result = completion(**kwargs)
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
            
            kwargs = {
                "model": self.model_name,
                **litellm_input,
            }
            if self.api_key:
                kwargs["api_key"] = self.api_key
            if self.base_url:
                kwargs["api_base"] = self.base_url
            if self.timeout:
                kwargs["timeout"] = self.timeout
            
            for response in completion(**kwargs):
                res, meta = self._process_result(
                    response, request_id=request_id, start_time=start
                )
                yield res, meta

        future = self.thread_pool.submit(generator)
        try:
            for response in future.result():
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
                
                kwargs = {
                    "model": self.model_name,
                    **litellm_input,
                }
                if self.api_key:
                    kwargs["api_key"] = self.api_key
                if self.base_url:
                    kwargs["api_base"] = self.base_url
                if self.timeout:
                    kwargs["timeout"] = self.timeout
                
                result = completion(**kwargs)
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
        post_processor = None
        if guided_decode_config:
            post_processor = self._build_guided_decode_post_processor(
                guided_decode_config
            )

        try:
            start = time.time()
            
            kwargs = {
                "model": self.model_name,
                **litellm_input,
            }
            if self.api_key:
                kwargs["api_key"] = self.api_key
            if self.base_url:
                kwargs["api_base"] = self.base_url
            if self.timeout:
                kwargs["timeout"] = self.timeout
            
            result = await acompletion(**kwargs)
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

        litellm_input = self._process_input_data(
            messages, generation_config, guided_decode_config
        )
        start = time.time()
        
        kwargs = {
            "model": self.model_name,
            **litellm_input,
        }
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.base_url:
            kwargs["api_base"] = self.base_url
        if self.timeout:
            kwargs["timeout"] = self.timeout

        try:
            async for response in await acompletion(**kwargs):
                res, meta = self._process_result(
                    response, request_id=request_id, start_time=start
                )
                yield res, meta
        except Exception as e:
            yield self._process_exception(e, request_id=request_id)

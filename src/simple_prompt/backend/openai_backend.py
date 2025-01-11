import asyncio
import copy
import inspect
import json
import time
from concurrent.futures import Future
from functools import partial
from typing import Callable, Generator, List

import openai
from openai import OpenAI
from openai.types.chat import ChatCompletion, ChatCompletionChunk
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


class OpenAIConfig(TypedDict):
    model_name: str
    api_key: str | None
    base_url: str | None
    timeout: float | None

    default_body: dict | None


class OpenAILLMBackend(BaseLLMBackend):
    def __init__(
        self,
        name: str | None = None,
        concurrency: int = 20,
        config: OpenAIConfig | None = None,
        hooks: List[LLMBackendHook] | None = None,
        logger=None,
    ):
        assert config is not None, "config must be provided"
        self.model_name = config.pop("model_name", None)
        assert self.model_name is not None, "model_name must be provided"
        if name is None:
            name = self.model_name

        super().__init__(name=name, concurrency=concurrency, hooks=hooks, logger=logger)

        # 构造openai client
        self.client = OpenAI(**config)
        self.default_body = config.get("default_body", {})

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
                json_schema_name = "json_schema"
            else:
                # BaseModel
                json_schema = base_model.model_json_schema()
                if guided_decode_config.use_list:
                    json_schema = {"type": "array", "items": json_schema}
                json_schema_name = base_model.__name__.lower()
            # OPENAI FORMAT JSON SCHEMA
            if "response_format" in _generation_config:
                raise ValueError(
                    "response_format is already provided in sampling params"
                )

            _generation_config["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": json_schema_name,
                    "strict": True,
                    "schema": json_schema,
                },
            }

            # 其他可能需要的参数

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

    def _process_result(
        self,
        result: ChatCompletion | ChatCompletionChunk,
        request_id: str | None = None,
        start_time: float | None = None,
    ) -> "raw_output_type":
        finish_time = time.time()
        if isinstance(result, ChatCompletionChunk):
            result_texts = [choice.delta.content for choice in result.choices]
        else:
            result_texts = [choice.message.content for choice in result.choices]
        for i in range(len(result_texts)):
            if result_texts[i] is None:
                result_texts[i] = ""
        result_texts: List[str]
        if len(result_texts) == 1:
            result_texts = result_texts[0]
            result_texts: str
        if request_id is None:
            request_id = result.id

        meta_data = MetaInfo(
            request_id=request_id,
            success=True,
            model=result.model,
            start_time=start_time,
            end_time=finish_time,
            finish_reason=[choice.finish_reason for choice in result.choices],
            usage=result.usage,
            original_result=result,
        )
        return result_texts, meta_data

    def _process_exception(
        self,
        e: Exception,
        request_id: str | None = None,
        start_time: float | None = None,
    ) -> "exception_output_type":
        # logger.error(f"Error processing request {request_id}: {e}")
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
            # 仅返回dict
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
            # 返回BaseModel或者BaseModel的list
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
            openai_client_input = self._process_input_data(
                messages, generation_config, guided_decode_config
            )
            post_processor = None
            if guided_decode_config:
                post_processor = self._build_guided_decode_post_processor(
                    guided_decode_config
                )
            start = time.time()
            result = self.client.chat.completions.create(
                model=self.model_name, **openai_client_input
            )
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
            # assert stream is False
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
        generation_config["n"] = 1
        generation_config["stream"] = True

        if guided_decode_config:
            raise ValueError("Guided decode is not supported in stream mode")

        def generator():
            openai_client_input = self._process_input_data(
                messages, generation_config, guided_decode_config
            )
            start = time.time()
            # TODO hook for pre processing
            for response in self.client.chat.completions.create(
                model=self.model_name, **openai_client_input
            ):
                response: "ChatCompletionChunk"
                res, meta = self._process_result(
                    response, request_id=request_id, start_time=start
                )
                yield res, meta
            # TODO hook for post processing

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
                openai_client_input = self._process_input_data(
                    messages, generation_config, guided_decode_config
                )
                post_processor = None
                if guided_decode_config:
                    post_processor = self._build_guided_decode_post_processor(
                        guided_decode_config
                    )

                self._hooks_on_request_start(start, openai_client_input, request_id)
                result = self.client.chat.completions.create(
                    model=self.model_name, **openai_client_input
                )
                result, meta = self._process_result(
                    result, request_id=request_id, start_time=start
                )
                self._hooks_on_request_end(result, meta, openai_client_input)
                if post_processor:
                    return post_processor(result, meta)
                else:
                    return result, meta
                # return self._process_result(result)
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
        openai_client_input = self._process_input_data(
            messages, generation_config, guided_decode_config
        )
        loop = asyncio.get_running_loop()
        func = partial(
            self.client.chat.completions.create,
            model=self.model_name,
            **openai_client_input,
        )
        post_processor = None
        if guided_decode_config:
            post_processor = self._build_guided_decode_post_processor(
                guided_decode_config
            )

        try:
            # TODO hook for pre processing
            start = time.time()
            result = await loop.run_in_executor(self.thread_pool, func)
            res, meta = self._process_result(
                result, request_id=request_id, start_time=start
            )
            # TODO hook for post processing
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
            openai_client_input = self._process_input_data(
                messages, generation_config, guided_decode_config
            )
            start = time.time()
            for response in self.client.chat.completions.create(
                model=self.model_name, **openai_client_input
            ):
                response: "ChatCompletionChunk"
                yield self._process_result(
                    response, request_id=request_id, start_time=start
                )

        try:
            for response in await loop.run_in_executor(self.thread_pool, generator):
                yield response
        except Exception as e:
            yield self._process_exception(e, request_id=request_id)

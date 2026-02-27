import re
from functools import lru_cache
from typing import Any, Callable, Dict, List, Literal, Tuple, Type, overload
from concurrent.futures import Future  # Added Future

from simple_prompt.backend import get_default_backend_name
from simple_prompt.protocol import GuidedBaseModel, MetaInfo, P

from .base import ExecutorMixin
from .prompt_execuator import PromptExecutor


class PromptDispatcher(ExecutorMixin):
    """PromptDispatcher 是一个执行器，配置多个PromptExecutor，并根据模型选择不同的执行器"""

    def __init__(
        self,
        rule: Dict[str, Callable[P, PromptExecutor]],
        default_backend: str = "default",
        fallback: Callable[P, PromptExecutor] = None,
        func_args: Any = tuple(),
        func_kwagrs: Any = dict(),
    ):
        self.rule = rule
        self.fallback = fallback

        self.selected_backend = "default"
        self.selected_executor = fallback if fallback else None
        self.configure_params = dict()

        self.func_args = func_args
        self.func_kwagrs = func_kwagrs

        if default_backend == "default":
            default_backend = get_default_backend_name()
            if default_backend is None:
                raise ValueError("default backend not set")
        matched_rule = self._match_rule(default_backend)
        if matched_rule is not None:
            # 如果backend在rule中，配置到对应的executor
            self.selected_backend = default_backend
            self.selected_executor = self.rule[matched_rule]
        else:
            # 如果没有找到对应的backend，使用fallback
            if not self.fallback:
                raise ValueError(f"backend {default_backend} not found and no fallback")

            self.selected_backend = default_backend
            self.selected_executor = self.fallback

    @lru_cache
    def _match_rule(self, backend: str) -> str | None:
        # rule 的 key 是 正则表达式字符串
        # 全部匹配一边，以命中最长的为准
        hits = [(k, re.match(k, backend)) for k in self.rule.keys()]
        hits = [(k, m) for k, m in hits if m]
        hits = sorted(hits, key=lambda x: len(x[1].group(0)), reverse=True)
        return hits[0][0] if hits else None

    def configure(
        self,
        backend: str | None = None,
        # Sampling parameters
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        presence_penalty: float | None = None,
        frequency_penalty: float | None = None,
        repetition_penalty: float | None = None,
        min_p: float | None = None,
        stop: str | List[str] | None = None,
        stop_token_ids: List[int] | None = None,
        bad_words: List[str] | None = None,
        ignore_eos: bool | None = None,
        max_tokens: int | None = None,
        min_tokens: int | None = None,
        logprobs: int | None = None,
        prompt_logprobs: int | None = None,
        **kwargs: Any,
    ):
        if backend is not None:
            if backend == "default":
                backend_name = get_default_backend_name()
            else:
                backend_name = backend
            matched_rule = self._match_rule(backend_name)
            if matched_rule is not None:
                # 如果backend在rule中，配置到对应的executor
                self.selected_backend = backend_name
                self.selected_executor = self.rule[matched_rule]
            else:
                # 如果没有找到对应的backend，使用fallback
                if not self.fallback:
                    raise ValueError(f"backend {backend} not found and no fallback")

                self.selected_backend = backend_name
                self.selected_executor = self.fallback

        self.configure_params = {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
            "repetition_penalty": repetition_penalty,
            "min_p": min_p,
            "stop": stop,
            "stop_token_ids": stop_token_ids,
            "bad_words": bad_words,
            "ignore_eos": ignore_eos,
            "max_tokens": max_tokens,
            "min_tokens": min_tokens,
            "logprobs": logprobs,
            "prompt_logprobs": prompt_logprobs,
            **kwargs,
        }

        return self

    def _get_configured_executor(self) -> PromptExecutor:
        current_selected_backend = self.selected_backend
        current_selected_executor_factory = self.selected_executor

        # Runtime re-evaluation if the dispatcher's current backend is "default"
        if current_selected_backend == "default":
            actual_default_backend_name = get_default_backend_name()
            if actual_default_backend_name is None:
                raise ValueError(
                    "Default backend name could not be determined at runtime."
                )

            matched_rule_key = self._match_rule(actual_default_backend_name)
            if matched_rule_key is not None:
                current_selected_backend = actual_default_backend_name
                current_selected_executor_factory = self.rule[matched_rule_key]
            elif self.fallback:
                current_selected_backend = actual_default_backend_name
                current_selected_executor_factory = self.fallback
            else:
                raise ValueError(
                    f"Default backend '{actual_default_backend_name}' not found in rule and no fallback set at runtime."
                )

        if not callable(current_selected_executor_factory):
            raise TypeError(
                f"Selected executor factory is not callable: {current_selected_executor_factory}. Fallback or rule might be misconfigured."
            )

        executor_instance = current_selected_executor_factory(
            *self.func_args, **self.func_kwagrs
        )

        if not isinstance(executor_instance, PromptExecutor):
            raise TypeError(
                f"Executor factory did not return a PromptExecutor instance. Got: {type(executor_instance)}"
            )

        executor_instance.configure(
            backend=current_selected_backend, **self.configure_params
        )
        return executor_instance

    def messages(self):
        executor = self._get_configured_executor()
        return executor.messages()

    @overload
    def execute(self, request_id: str | None = None) -> Tuple[str, MetaInfo]:
        """调用模型，并返回结果和元信息

        Args:
            request_id (str | None, optional): 请求ID(可选). Defaults to None.

        Returns:
            result (Tuple[str, MetaInfo]): 返回结果和元信息
        """
        ...

    @overload
    def execute(
        self,
        base_model: Type[GuidedBaseModel],
        use_list: Literal[False] = False,
        request_style: Literal["vllm", "openai"] = "openai",
        guided_decoding_backend: str | None = None,
        request_id: str | None = None,
    ) -> Tuple[GuidedBaseModel, MetaInfo]:
        """调用模型，并解析结果为给定的BaseModel类型，返回结果和元信息

        Args:
            base_model (Type[GuidedBaseModel]): BaseModel的子类
            use_list (bool, optional): 是否解析成列表. Defaults to False.
            request_style (Literal["vllm", "openai"], optional): 请求风格. Defaults to "openai".
            guided_decoding_backend (str | None, optional): 引导解码后端. Defaults to None.
            request_id (str | None, optional): 请求ID(可选). Defaults to None.

        Returns:
            result (Tuple[GuidedBaseModel, MetaInfo]): 解析后的结果和元信息
        """

    @overload
    def execute(
        self,
        base_model: Type[GuidedBaseModel],
        use_list: Literal[True] = True,
        request_style: Literal["vllm", "openai"] = "openai",
        guided_decoding_backend: str | None = None,
        request_id: str | None = None,
    ) -> Tuple[List[GuidedBaseModel], MetaInfo]:
        """调用模型，并解析结果为给定的BaseModel类型数组，返回结果和元信息

        Args:
            base_model (Type[GuidedBaseModel]): BaseModel的子类
            use_list (bool, optional): 是否返回列表. Defaults to False.
            request_style (Literal["vllm", "openai"], optional): 请求风格. Defaults to "openai".
            guided_decoding_backend (str | None, optional): 引导解码后端. Defaults to None.
            request_id (str | None, optional): 请求ID(可选). Defaults to None.

        Returns:
            result (Tuple[List[GuidedBaseModel, MetaInfo]): 解析后的结果和元信息
        """

    def execute(
        self,
        base_model: Type[GuidedBaseModel] | None = None,
        use_list: bool = False,
        request_style: Literal["vllm", "openai"] = "openai",
        guided_decoding_backend: str | None = None,
        request_id: str | None = None,
    ):
        """调用模型

        - 如果没有入参，直接返回结果
        - 如果 base_model 有值，解析结果为给定的BaseModel类型
        - 如果 base_model 有值且 use_list 为 True，解析结果为给定的BaseModel类型数组

        Args:
            base_model (Type[GuidedBaseModel], optional): BaseModel的子类. Defaults to None.
            use_list (bool, optional): 是否返回列表. Defaults to False.
            request_style (Literal["vllm", "openai"], optional): 请求风格. Defaults to "openai".
            guided_decoding_backend (str | None, optional): 引导解码后端. Defaults to None.
            request_id (str | None, optional): 请求ID(可选). Defaults to None.

        Returns:
            result (Tuple[str | GuidedBaseModel | List[GuidedBaseModel], MetaInfo]): 返回结果和元信息
        """
        executor = self._get_configured_executor()
        return executor.execute(request_id=request_id)

    @overload
    def parse(
        self,
        base_model: Type[GuidedBaseModel],
        use_list: Literal[False] = False,
        request_id: str | None = None,
    ) -> Tuple[GuidedBaseModel, MetaInfo]: ...

    @overload
    def parse(
        self,
        base_model: dict,
        use_list: Literal[True] = True,
        request_id: str | None = None,
    ) -> Tuple[List[dict], MetaInfo]: ...

    @overload
    def parse(
        self,
        base_model: dict,
        use_list: Literal[False] = False,
        request_id: str | None = None,
    ) -> Tuple[dict, MetaInfo]: ...

    @overload
    def parse(
        self,
        base_model: Type[GuidedBaseModel],
        use_list: Literal[True] = True,
        request_id: str | None = None,
    ) -> Tuple[List[GuidedBaseModel], MetaInfo]: ...

    def parse(
        self,
        base_model: Type[GuidedBaseModel] | dict,
        use_list: bool = False,
        request_id: str | None = None,
    ):  # Return type is (GuidedBaseModel | List[GuidedBaseModel] | dict | List[dict], MetaInfo)
        """调用模型，并解析结果为给定的BaseModel类型，返回结果和元信息

        Args:
            base_model (Type[GuidedBaseModel] | dict): BaseModel的子类或JSON schema
            use_list (bool, optional): 是否解析成列表. Defaults to False.
            request_id (str | None, optional): 请求ID(可选). Defaults to None.

        Returns:
            result: 解析后的结果和元信息
        """
        executor = self._get_configured_executor()
        return executor.parse(
            base_model=base_model,
            use_list=use_list,
            request_id=request_id,
        )

    async def async_execute(self, request_id: str | None = None):
        """异步调用模型，并返回结果和元信息"""
        executor = self._get_configured_executor()
        return await executor.async_execute(request_id=request_id)

    async def async_parse(
        self,
        base_model: Type[GuidedBaseModel] | dict,
        use_list: bool = False,
        request_id: str | None = None,
    ):
        """异步调用模型，并解析结果为给定的BaseModel类型"""
        executor = self._get_configured_executor()
        return await executor.async_parse(
            base_model=base_model, use_list=use_list, request_id=request_id
        )

    async def async_stream(self, request_id: str | None = None):
        """异步流式调用模型"""
        executor = self._get_configured_executor()
        async for response in executor.async_stream(request_id=request_id):
            yield response

    def execute_in_future(
        self, request_id: str | None = None
    ) -> Future[Tuple[str, MetaInfo]]:
        """在线程池中调用模型,返回 Future 对象

        Args:
            request_id (str | None, optional): 请求ID(可选). Defaults to None.

        Returns:
            result (Future[Tuple[str, MetaInfo]]): Future 对象
        """
        executor = self._get_configured_executor()
        return executor.execute_in_future(request_id=request_id)

    @overload
    def parse_in_future(
        self,
        base_model: Type[GuidedBaseModel],
        use_list: Literal[False] = False,
        request_id: str | None = None,
    ) -> Future[Tuple[GuidedBaseModel, MetaInfo]]: ...

    @overload
    def parse_in_future(
        self,
        base_model: dict,
        use_list: Literal[True] = True,
        request_id: str | None = None,
    ) -> Future[Tuple[List[dict], MetaInfo]]: ...

    @overload
    def parse_in_future(
        self,
        base_model: dict,
        use_list: Literal[False] = False,
        request_id: str | None = None,
    ) -> Future[Tuple[dict, MetaInfo]]: ...

    @overload
    def parse_in_future(
        self,
        base_model: Type[GuidedBaseModel],
        use_list: Literal[True] = True,
        request_id: str | None = None,
    ) -> Future[Tuple[List[GuidedBaseModel], MetaInfo]]: ...

    def parse_in_future(
        self,
        base_model: Type[GuidedBaseModel] | dict,
        use_list: bool = False,
        request_id: str | None = None,
    ):  # Return type is Future[Tuple[GuidedBaseModel | List[GuidedBaseModel] | dict | List[dict], MetaInfo]]
        """在线程池中调用模型并解析结果,返回 Future 对象

        Args:
            base_model (Type[GuidedBaseModel] | dict): BaseModel的子类或JSON schema
            use_list (bool, optional): 是否解析成列表. Defaults to False.
            request_id (str | None, optional): 请求ID(可选). Defaults to None.

        Returns:
            result: Future 对象，包含解析后的结果和元信息
        """
        executor = self._get_configured_executor()
        return executor.parse_in_future(
            base_model=base_model,
            use_list=use_list,
            request_id=request_id,
        )

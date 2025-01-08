import re
from functools import lru_cache, wraps
from typing import Any, Callable, Dict, List, Literal, Tuple, Type, overload

from .backend import get_default_llm_backend_name
from .execuator import ExecutorMixin, PromptExecutor
from .protocol import GuidedBaseModel, MetaInfo, P, R


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
            default_backend = get_default_llm_backend_name()
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
                backend_name = get_default_llm_backend_name()
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

    def messages(self):
        if not self.selected_executor:
            raise ValueError(
                "no executor selected, please set fallback or check the configuration"
            )
        return self.selected_executor(*self.func_args, **self.func_kwagrs).messages()

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
            result (Tuple[List[GuidedBaseModel], MetaInfo]): 解析后的结果和元信息
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
        if self.selected_backend == "default":
            selected_backend_name = get_default_llm_backend_name()
            matched_rule = self._match_rule(selected_backend_name)
            if matched_rule is not None:
                self.selected_backend = selected_backend_name
                self.selected_executor = self.rule[selected_backend_name]
            else:
                self.selected_backend = selected_backend_name
                self.selected_executor = self.fallback

        if not self.selected_executor:
            raise ValueError(
                "no executor selected, please set fallback or check the configuration"
            )

        # 配置参数
        executor = self.selected_executor(*self.func_args, **self.func_kwagrs)
        executor.configure(backend=self.selected_backend, **self.configure_params)

        return executor.execute(
            base_model=base_model,
            use_list=use_list,
            request_style=request_style,
            guided_decoding_backend=guided_decoding_backend,
            request_id=request_id,
        )


def prompt_dispatcher(
    rule: Dict[str, Callable[P, PromptExecutor]],
    default_backend: str = "default",
    fallback: Callable[P, PromptExecutor] = None,
):
    def decorator(func: Callable[P, R]) -> Callable[P, PromptDispatcher]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> PromptDispatcher:
            return PromptDispatcher(
                rule,
                default_backend=default_backend,
                fallback=fallback,
                func_args=args,
                func_kwagrs=kwargs,
            )

        return wrapper

    return decorator

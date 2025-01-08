from abc import ABC, abstractmethod
from concurrent.futures import Future
from functools import wraps
from typing import Any, Callable, Dict, List, Literal, Tuple, Type, TypeVar, overload

from ..backend import get_backend
from ..protocol import GuidedBaseModel, GuidedDecodeConfig, MetaInfo, P, R, message_type
from .base import ExecutorMixin


class PromptExecutor(ExecutorMixin):
    """PromptExecutor 是一个执行器，用于通过prompt或func生成消息，并调用模型"""

    def __init__(
        self,
        backend: str = "default",
        prompt: str | message_type | None = None,
        func: Callable[..., str | message_type] = None,
        func_args: Any = tuple(),
        func_kwagrs: Any = dict(),
        sampling_params: Dict = dict(),
        **kwargs: Any,
    ):
        self.backend = get_backend(backend)
        if backend is None:
            raise ValueError("backend not found")
        if prompt is not None and func is not None:
            raise ValueError("prompt and func cannot be used together")
        if prompt is None and func is None:
            raise ValueError("prompt or func is required")

        if prompt is not None:
            self.func = lambda: prompt
            self.func_args = tuple()
            self.func_kwagrs = dict()
        else:
            self.func = func
            self.func_args = func_args
            self.func_kwagrs = func_kwagrs

        self.sampling_params = sampling_params
        # kwargs中的参数会覆盖sampling_params中的参数
        self.sampling_params.update(kwargs)

        not_allowed_keys = ["model", "messages", "extra_body", "stream", "n"]
        # 检查sampling_params中是否有不允许的key
        for key in not_allowed_keys:
            if key in self.sampling_params:
                self.sampling_params.pop(key)
                # TODO warning

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
            self.backend = get_backend(backend)
        overwrite_params = {
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
        # 过滤掉None值
        overwrite_params = {k: v for k, v in overwrite_params.items() if v is not None}
        self.sampling_params.update(overwrite_params)

        not_allowed_keys = ["model", "messages", "extra_body", "stream", "n"]
        # 检查sampling_params中是否有不允许的key
        for key in not_allowed_keys:
            if key in self.sampling_params:
                self.sampling_params.pop(key)
                # TODO warning

        return self

    def _prepare_input(
        self,
        base_model: Type[GuidedBaseModel] | None = None,
        use_list: bool = False,
        request_id: str | None = None,
    ):
        guided_decode_config = None
        if base_model:
            # build guided_decode_config
            guided_decode_config = GuidedDecodeConfig(
                base_model=base_model,
                use_list=use_list,
            )

        return (
            request_id,
            self.messages(),
            self.sampling_params,
            guided_decode_config,
        )

    def messages(self) -> message_type:
        if not self.func:
            raise ValueError("func is required")
        prompt_or_messages = self.func(*self.func_args, **self.func_kwagrs)
        if isinstance(prompt_or_messages, str):
            return [{"content": prompt_or_messages, "role": "user"}]
        else:
            return prompt_or_messages

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
        request_id: str | None = None,
    ) -> Tuple[GuidedBaseModel, MetaInfo]:
        """调用模型，并解析结果为给定的BaseModel类型，返回结果和元信息

        Args:
            base_model (Type[GuidedBaseModel]): BaseModel的子类
            use_list (bool, optional): 是否解析成列表. Defaults to False.
            request_id (str | None, optional): 请求ID(可选). Defaults to None.

        Returns:
            result (Tuple[GuidedBaseModel, MetaInfo]): 解析后的结果和元信息
        """

    @overload
    def execute(
        self,
        base_model: Type[GuidedBaseModel],
        use_list: Literal[True] = True,
        request_id: str | None = None,
    ) -> Tuple[List[GuidedBaseModel], MetaInfo]:
        """调用模型，并解析结果为给定的BaseModel类型数组，返回结果和元信息

        Args:
            base_model (Type[GuidedBaseModel]): BaseModel的子类
            use_list (bool, optional): 是否返回列表. Defaults to False.
            request_id (str | None, optional): 请求ID(可选). Defaults to None.

        Returns:
            result (Tuple[List[GuidedBaseModel], MetaInfo]): 解析后的结果和元信息
        """

    def execute(
        self,
        base_model: Type[GuidedBaseModel] | None = None,
        use_list: bool = False,
        request_id: str | None = None,
    ):
        """调用模型

        - 如果没有入参，直接返回结果
        - 如果 base_model 有值，解析结果为给定的BaseModel类型
        - 如果 base_model 有值且 use_list 为 True，解析结果为给定的BaseModel类型数组

        Args:
            base_model (Type[GuidedBaseModel], optional): BaseModel的子类. Defaults to None.
            use_list (bool, optional): 是否返回列表. Defaults to False.
            request_id (str | None, optional): 请求ID(可选). Defaults to None.

        Returns:
            result (Tuple[str | GuidedBaseModel | List[GuidedBaseModel], MetaInfo]): 返回结果和元信息
        """
        request_id, messages, sampling_params, guided_decode_config = (
            self._prepare_input(
                base_model=base_model, use_list=use_list, request_id=request_id
            )
        )

        result = self.backend.chat(
            messages=messages,
            request_id=request_id,
            generation_config=sampling_params,
            guided_decode_config=guided_decode_config,
        )
        return result

    @overload
    def execute_in_thread(
        self, request_id: str | None = None
    ) -> Future[Tuple[str, MetaInfo]]:
        """在线程池中调用模型，并返回 Future 对象

        Args:
            request_id (str | None, optional): 请求ID(可选). Defaults to None.

        Returns:
            result (Future[Tuple[str, MetaInfo]]): Future 对象
        """
        ...

    @overload
    def execute_in_thread(
        self,
        base_model: Type[GuidedBaseModel] | None = None,
        use_list: Literal[False] = False,
        request_id: str | None = None,
    ) -> Future[Tuple[GuidedBaseModel | List[GuidedBaseModel], MetaInfo]]:
        """在线程池中调用模型，并返回 Future 对象

        Args:
            base_model (Type[GuidedBaseModel], optional): BaseModel的子类. Defaults to None.
            use_list (bool, optional): 是否返回列表. Defaults to False.
            request_id (str | None, optional): 请求ID(可选). Defaults to None.

        Returns:
            result (Future[Tuple[GuidedBaseModel, MetaInfo]]): Future 对象
        """
        ...

    @overload
    def execute_in_thread(
        self,
        base_model: Type[GuidedBaseModel] | None = None,
        use_list: Literal[True] = True,
        request_id: str | None = None,
    ) -> Future[Tuple[List[GuidedBaseModel], MetaInfo]]:
        """在线程池中调用模型，并返回 Future 对象

        Args:
            base_model (Type[GuidedBaseModel], optional): BaseModel的子类. Defaults to None.
            use_list (bool, optional): 是否返回列表. Defaults to False.
            request_id (str | None, optional): 请求ID(可选). Defaults to None.

        Returns:
            result (Future[Tuple[List[GuidedBaseModel], MetaInfo]]): Future 对象
        """

    def execute_in_thread(
        self,
        base_model: Type[GuidedBaseModel] | None = None,
        use_list: bool = False,
        request_id: str | None = None,
    ):
        """在线程池中调用模型

        - 如果没有入参，直接返回结果
        - 如果 base_model 有值，解析结果为给定的BaseModel类型
        - 如果 base_model 有值且 use_list 为 True，解析结果为给定的BaseModel类型数组

        Args:
            base_model (Type[GuidedBaseModel], optional): BaseModel的子类. Defaults to None.
            use_list (bool, optional): 是否返回列表. Defaults to False.
            request_id (str | None, optional): 请求ID(可选). Defaults to None.

        Returns:
            result (Future[Tuple[str | GuidedBaseModel | List[GuidedBaseModel], MetaInfo]]): Future 对象
        """
        messages = self.messages()
        guided_decode_config = None
        if base_model:
            # build guided_decode_config
            guided_decode_config = GuidedDecodeConfig(
                base_model=base_model,
                use_list=use_list,
            )

        future: Future[Tuple[GuidedBaseModel | List[GuidedBaseModel], MetaInfo]] = (
            self.backend.chat_in_thread(
                messages=messages,
                request_id=request_id,
                generation_config=self.sampling_params,
                guided_decode_config=guided_decode_config,
            )
        )
        return future

    def stream(self, request_id: str | None = None):
        """调用模型，并返回生成器

        Args:
            request_id (str | None, optional): 请求ID(可选). Defaults to None.

        Returns:
            result (Generator[Tuple[str, MetaInfo]]): 生成器
        """
        request_id, messages, sampling_params, _ = self._prepare_input(
            request_id=request_id
        )

        for response in self.backend.chat_stream(
            messages=messages,
            request_id=request_id,
            generation_config=sampling_params,
        ):
            yield response

    async def async_stream(self, request_id: str | None = None):
        """调用模型，并返回异步生成器

        Args:
            request_id (str | None, optional): 请求ID(可选). Defaults to None.

        Returns:
            result (AsyncGenerator[Tuple[str, MetaInfo]]): 异步生成器
        """
        request_id, messages, sampling_params, _ = self._prepare_input(
            request_id=request_id
        )

        async for response in self.backend.async_chat_stream(
            messages=messages,
            request_id=request_id,
            generation_config=sampling_params,
        ):
            yield response


def prompt(
    *args,
    default_backend: str = "default",
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


    def decorator(func: Callable[P, R]) -> Callable[P, PromptExecutor]:
        @wraps(func)
        def wrapper(*args: Any, **nest_kwargs: Any) -> PromptExecutor:
            sampling_params = {
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
            }
            # 过滤掉None值
            sampling_params = {
                k: v for k, v in sampling_params.items() if v is not None
            }
            executor = PromptExecutor(
                backend=default_backend,
                prompt=None,
                func=func,
                func_args=args,
                func_kwagrs=nest_kwargs,
                sampling_params=sampling_params,
                **kwargs,
            )
            return executor

        return wrapper

    # if func is not None:
    #     return decorator(func)
    # else:
    #     return decorator

    # if len(args) == 1 and callable(args[0]):
    #     # 此时没有参数，default_backend是一个函数
    #     return decorator(args[0])
    # elif len(args) > 1:
    #     # 不支持通过 args 传递参数
    #     raise ValueError(
    #         "prompt() does not support passing arguments through positional arguments"
    #     )
    if len(args) == 1 and callable(args[0]):
        raise ValueError(
            "Use @prompt() without parentheses when no arguments are passed"
        )

    if args:
        raise ValueError("Use keyword arguments to pass parameters to @prompt()")

    return decorator

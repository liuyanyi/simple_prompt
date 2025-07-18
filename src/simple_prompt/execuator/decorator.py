import inspect
from functools import wraps
from typing import Any, Callable, Dict, List, Type

from ..protocol import GuidedBaseModel, P, R
from .prompt_dispatcher import PromptDispatcher
from .prompt_execuator import PromptExecutor


class PromptFunction:
    """可被IDE正确识别的Prompt函数类"""

    def __init__(
        self,
        func: Callable[P, R],
        backend: str,
        sampling_params: Dict[str, Any],
        default_parse_model: Type[GuidedBaseModel] | dict | None = None,
        **kwargs: Any,
    ):
        self.func = func
        self.backend_name = backend
        self.sampling_params = sampling_params
        self.default_parse_model = default_parse_model
        self.kwargs = kwargs
        # 保留原函数的元数据
        self.__name__ = func.__name__
        self.__doc__ = func.__doc__
        self.__module__ = func.__module__
        self.__annotations__ = getattr(func, "__annotations__", {})
        self.__signature__ = inspect.signature(func)

    def __call__(self, *args: Any, **kwargs: Any) -> PromptExecutor:
        """调用函数并返回PromptExecutor实例"""

        executor = PromptExecutor(
            backend=self.backend_name,
            prompt=None,
            func=self.func,
            func_args=args,
            func_kwagrs=kwargs,
            sampling_params=self.sampling_params,
            default_parse_model=self.default_parse_model,
            **self.kwargs,
        )
        # Pre check
        try:
            _ = executor.messages()
        except Exception as e:
            raise ValueError(
                f"PromptFunction {self.func.__name__} initialization failed: {e}"
            ) from e
        return executor

    def __repr__(self) -> str:
        """返回函数的字符串表示"""
        return f"<PromptFunction {self.func.__name__}>"


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
    # structured output parameters
    default_parse_model: Type[GuidedBaseModel] | dict | None = None,
    **kwargs: Any,
):
    def decorator(func: Callable[P, R]) -> Callable[P, PromptExecutor]:
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
        sampling_params = {k: v for k, v in sampling_params.items() if v is not None}
        return PromptFunction(
            func=func,
            backend=default_backend,
            sampling_params=sampling_params,
            default_parse_model=default_parse_model,
            **kwargs,
        )

    if len(args) == 1 and callable(args[0]):
        # TODO This is for IDE to recognize the decorator.
        # IDE will not recognize the decorator if use like @prompt
        raise ValueError(
            "Use @prompt() without parentheses when no arguments are passed"
        )

    if args:
        raise ValueError("Use keyword arguments to pass parameters to @prompt()")

    return decorator


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

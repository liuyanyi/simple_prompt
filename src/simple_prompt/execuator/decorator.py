from functools import wraps
from typing import Any, Callable, Dict, List

from simple_prompt.protocol import P, R

from .prompt_dispatcher import PromptDispatcher
from .prompt_execuator import PromptExecutor


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
        def wrapper(*nest_args: Any, **nest_kwargs: Any) -> PromptExecutor:
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
                func_args=nest_args,
                func_kwagrs=nest_kwargs,
                sampling_params=sampling_params,
                **kwargs,
            )
            return executor

        return wrapper

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

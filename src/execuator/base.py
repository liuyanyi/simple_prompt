from abc import ABC, abstractmethod
from typing import Any, List, Literal, Type, TypeVar

from ..protocol import GuidedBaseModel, message_type

ExecutorSelf = TypeVar("ExecutorSelf", bound="ExecutorMixin")


class ExecutorMixin(ABC):
    """ExecutorMixin 是一个抽象基类，定义了执行器的接口"""

    @abstractmethod
    def configure(
        self: "ExecutorSelf",
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
    ) -> "ExecutorSelf":
        """配置执行器，返回自身实现链式调用"""

    @abstractmethod
    def messages(self) -> message_type:
        """根据输入参数生成消息"""

    @abstractmethod
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

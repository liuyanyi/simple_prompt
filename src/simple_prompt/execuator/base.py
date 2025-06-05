from abc import ABC, abstractmethod
from concurrent.futures import Future
from typing import Any, List, Tuple, Type, TypeVar

from simple_prompt.protocol import GuidedBaseModel, MetaInfo, message_type

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
        request_id: str | None = None,
    ) -> Tuple[str, MetaInfo]:
        """调用模型

        Args:
            request_id (str | None, optional): 请求ID(可选). Defaults to None.

        Returns:
            result (Tuple[str, MetaInfo]): 返回结果和元信息
        """

    @abstractmethod
    def parse(
        self,
        base_model: Type[GuidedBaseModel] | dict,
        use_list: bool = False,
        request_id: str | None = None,
    ) -> Tuple[GuidedBaseModel | List[GuidedBaseModel] | dict | List[dict], MetaInfo]:
        """调用模型，并解析结果为给定的BaseModel类型，返回结果和元信息

        Args:
            base_model (Type[GuidedBaseModel]): BaseModel的子类
            use_list (bool, optional): 是否解析成列表. Defaults to False.
            request_id (str | None, optional): 请求ID(可选). Defaults to None.

        Returns:
            result (Tuple[GuidedBaseModel | List[GuidedBaseModel] | dict | List[dict], MetaInfo]): 解析后的结果和元信息
        """

    @abstractmethod
    def execute_in_future(
        self, request_id: str | None = None
    ) -> Future[Tuple[str, MetaInfo]]:
        """在线程池中调用模型,返回 Future 对象

        Args:
            request_id (str | None, optional): 请求ID(可选). Defaults to None.

        Returns:
            result (Future[Tuple[str, MetaInfo]]): Future 对象
        """
        ...

    @abstractmethod
    def parse_in_future(
        self,
        base_model: Type[GuidedBaseModel] | dict,
        use_list: bool = False,
        request_id: str | None = None,
    ) -> Future[
        Tuple[GuidedBaseModel | List[GuidedBaseModel] | dict | List[dict], MetaInfo]
    ]:
        """在线程池中调用模型并解析结果,返回 Future 对象

        Args:
            base_model (Type[GuidedBaseModel]): BaseModel的子类
            use_list (bool, optional): 是否解析成列表. Defaults to False.
            request_id (str | None, optional): 请求ID(可选). Defaults to None.

        Returns:
            result (Future[Tuple[GuidedBaseModel | List[GuidedBaseModel] | dict | List[dict], MetaInfo]]): Future 对象
        """
        ...

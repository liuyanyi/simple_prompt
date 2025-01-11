from typing import List, ParamSpec, Tuple, TypeVar, Union

from openai.types import CompletionUsage
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionContentPartParam,
)
from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import Required, TypedDict


class MetaInfo(BaseModel):
    request_id: str | None = None
    success: bool | None
    model: str | None = None
    start_time: float | None = None
    end_time: float | None = None
    finish_reason: List[str | None] = Field(default_factory=list)
    usage: CompletionUsage | None = None
    error: str | None = None
    original_result: ChatCompletion | ChatCompletionChunk | None = None

    def __str__(self) -> str:
        repr_str = "== MetaInfo ==\n"
        repr_str += f"request_id: {self.request_id}\n"
        repr_str += f"success: {self.success}\n"
        repr_str += f"model: {self.model}\n"
        repr_str += f"finish_reason: {self.finish_reason}\n"
        repr_str += f"usage: {self.usage}\n"
        if self.error:
            repr_str += f"error: {self.error}\n"
        return repr_str


T = TypeVar("T")
output_type = Tuple[T | List[T] | None, MetaInfo]
raw_output_type = output_type[str]
exception_output_type = Tuple[None, MetaInfo]


class CustomChatCompletionMessageParam(TypedDict, total=False):
    """Enables custom roles in the Chat Completion API."""

    role: Required[str]
    """The role of the message's author."""

    content: Union[str, List[ChatCompletionContentPartParam]]
    """The contents of the message."""


message_type = List[CustomChatCompletionMessageParam]

# GuidedBaseModel是BaseModel的子类
GuidedBaseModel = TypeVar("GuidedBaseModel", bound=BaseModel)

# Wrapper Func Return Type
R = str | message_type  # 原始返回类型
# Wrapper Func Input Type
P = ParamSpec("P")  # 参数规范


class GuidedDecodeConfig(BaseModel):
    # 提供 basemodel 或者 JSON schema
    base_model: type[GuidedBaseModel] | dict
    use_list: bool = False

    # 允许用户自定义的参数
    model_config = ConfigDict(extra="allow")

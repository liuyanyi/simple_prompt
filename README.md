# Simple Prompt

[English](README_EN.md) | 简体中文

一个简单的 LLM prompt 调用包装器。

## 主要针对问题

1. 提示词通常是字符串拼接，不方便维护，没有代码提示和类型检查。 -> 通过将提示封装成一个函数，可以更好的维护和调用。
2. openai 客户端的返回结果需要用 `result.choices[0].message.content` 调用，很烦。 -> 返回元组，第一个元素是结果，第二个元素是元信息。
3. 通常需要线程池进行并发调用。 -> Backend 内置线程池，通过execute_in_thread方法返回future。

## 安装

```bash
pip install simple-prompt
```

## 快速开始

宗旨是通过定义python方法来定义各类提示拼装逻辑，然后通过装饰器的方式调用LLMs。

执行器的返回值是一个元组，第一个元素是返回的结果，第二个元素是元信息。

在结构化生成的情况下，可以通过 `base_model` 参数指定返回的数据结构。

### 基础调用示例

```python
@prompt()
def translate(text: str, target_language: str = "Chinese"):
    return f"Please translate the following text into {target_language}: \n{text}"

# 直接调用执行
result, meta = translate(text="I want to learn AI").execute()
# result -> str
# meta -> MetaInfo
```

### 消息格式调用

```python
@prompt()
def translate_in_msg_format(text: str, target_language: str = "Chinese"):
    return [
        {
            "role": "system",
            "content": f"You are a professional {target_language} translator.",
        },
        {
            "role": "user",
            "content": text,
        },
    ]

# 支持配置采样参数
result, meta = translate_in_msg_format(text="Hello World").configure(top_p=0.5).execute()
```

### 结构化数据返回

```python
from pydantic import BaseModel

class TranslationResult(BaseModel):
    text: str
    target_language: str

@prompt()
def translate_into_json_format(text: str, target_language: str = "Chinese"):
    return f"""Please translate the following text into {target_language}, 
    return result in JSON format: {TranslationResult.model_json_schema()}
    Text: {text}"""

# 支持结构化数据返回
result, meta = translate_into_json_format(
    text="Hello, how are you?"
).execute(base_model=TranslationResult, use_list=False)
# result -> TranslationResult
```

## 配置后端

```python
from simple_prompt.backend import vLLMBackend, register_backend, set_default_backend

backend = vLLMBackend(
    name="vllm",
    config={
        "model_name": "your_model",
        "api_key": "your_key",
        "base_url": "your_api_endpoint",
    }
)

register_backend(backend=backend)
set_default_backend("vllm")

# 后续可以在 prompt 装饰器中指定 backend 参数
```

## 许可证

Apache-2.0 License


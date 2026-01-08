# Simple Prompt

A simple LLM prompt call wrapper.
I hate `result.choices[0].message.content`, I refer to the [promptic](https://github.com/knowsuchagency/promptic) project, and refer to my own needs, I wrote this wrapper.

一个简单的 LLM prompt 调用包装器。
我讨厌 `result.choices[0].message.content`，我参考 [promptic](https://github.com/knowsuchagency/promptic) 项目，并参考我自己的需求，写了这个包装器。

# Installation

```bash
git clone https://github.com/liuyanyi/simple_prompt.git
```

# 使用

```python
from simple_prompt.backend import (
    register_backend,
    set_default_backend,
    OpenAILLMBackend,
)
from simple_prompt.execuator import prompt


@prompt()
def translate(text: str, target_language: str = "Chinese"):
    return f"Please translate the following text into {target_language}: \n{text}"


backend = OpenAILLMBackend(
    name="vllm",
    config={
        "model_name": "qwen",
        "api_key": "sk-*",
        "base_url": "http://10.142.6.40:40404/v1",
    },
)

register_backend(backend=backend)
set_default_backend("vllm")

# Simplely call the function, then execute the function
result, meta = translate(text="I want to learn AI", target_language="Chinese").execute()

print(result)
```

## Reasoning Support

The library now supports extracting reasoning information from LLM responses. This is useful for models that provide reasoning content separately (like some endpoints) or when you want to parse reasoning from the response.

### Basic Usage

The `MetaInfo` object returned by the backend now includes a `reasoning` field:

```python
result, meta = translate(text="I want to learn AI", target_language="Chinese").execute()

print(result)  # The completion/response
print(meta.reasoning)  # The reasoning content (if available)
```

### Custom Reasoning Parser

You can provide a custom reasoning parser to extract reasoning from responses:

```python
from simple_prompt.protocol import ReasoningParser
from openai.types.chat import ChatCompletion, ChatCompletionChunk

def my_reasoning_parser(result: ChatCompletion | ChatCompletionChunk) -> str | list[str] | None:
    # Extract reasoning from usage statistics
    if result.usage and result.usage.completion_tokens_details:
        reasoning_tokens = result.usage.completion_tokens_details.reasoning_tokens
        if reasoning_tokens and reasoning_tokens > 0:
            return f"Used {reasoning_tokens} reasoning tokens"
    
    # Or parse from content with custom logic
    # For example, extract text between <reasoning> tags
    # ...
    
    return None

backend = OpenAILLMBackend(
    name="gpt-4-reasoning",
    config={
        "model_name": "gpt-4",
        "api_key": "sk-*",
        "base_url": "https://api.openai.com/v1",
    },
    reasoning_parser=my_reasoning_parser,  # Use custom parser
)
```

See `example_custom_reasoning_parser.py` for a complete example.

## Future Work

- [ ] LiteLLM Backend for all kinds of LLM models
- [ ] Function call with `prompt` decorator
- [ ] Retry
- [ ] Cache
- [ ] Timeout
- [ ] More complex prompt combination (Like sglang, low priority)


## 许可证

Apache-2.0 License


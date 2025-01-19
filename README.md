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

## Future Work

- [ ] LiteLLM Backend for all kinds of LLM models
- [ ] Function call with `prompt` decorator
- [ ] Retry
- [ ] Cache
- [ ] Timeout
- [ ] More complex prompt combination (Like sglang, low priority)


## 许可证

Apache-2.0 License


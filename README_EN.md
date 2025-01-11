# Simple Prompt

[简体中文](README.md) | English

A simple LLM prompt wrapper.

## Key Features

1. Prompt strings are usually concatenated, making them hard to maintain and lacking code hints and type checking. -> By wrapping prompts as functions, they become more maintainable and callable.
2. OpenAI client's return results require `result.choices[0].message.content` calls, which is cumbersome. -> Returns a tuple with the result as the first element and metadata as the second.
3. Thread pools are often needed for concurrent calls. -> Backend has built-in thread pool, returns future through execute_in_thread method.

## Installation

```bash
pip install simple-prompt
```

## Quick Start

The principle is to define prompt assembly logic through Python methods and call LLMs through decorators.

The executor returns a tuple where the first element is the result and the second element is the metadata.

For structured generation, you can specify the return data structure through the `base_model` parameter.

### Basic Call Example

```python
@prompt()
def translate(text: str, target_language: str = "Chinese"):
    return f"Please translate the following text into {target_language}: \n{text}"

# Direct execution
result, meta = translate(text="I want to learn AI").execute()
# result -> str
# meta -> MetaInfo
```

### Message Format Call

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

# Supports sampling parameter configuration
result, meta = translate_in_msg_format(text="Hello World").configure(top_p=0.5).execute()
```

### Structured Data Return

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

# Supports structured data return
result, meta = translate_into_json_format(
    text="Hello, how are you?"
).execute(base_model=TranslationResult, use_list=False)
# result -> TranslationResult
```

## Configure Backend

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

# You can specify the backend parameter in the prompt decorator later
```

## License

Apache-2.0 License

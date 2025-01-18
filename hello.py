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

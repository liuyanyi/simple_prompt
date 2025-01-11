import logging
from pydantic import BaseModel

from simple_prompt.backend import (
    LoggingHook,
    register_backend,
    set_default_backend,
    vLLMBackend,
    OpenAILLMBackend
)
from simple_prompt.execuator import prompt_dispatcher, prompt

@prompt()
def translate(text: str, target_language: str = "Chinese"):
    return f"Please translate the following text into {target_language}: \n{text}"


@prompt()
def translate_in_msg_format(text: str, target_language: str = "Chinese"):
    return [
        {
            "role": "system",
            "content": f"You are a professional {target_language} translator, please translate user's text into {target_language}.",
        },
        {
            "role": "user",
            "content": text,
        },
    ]


class TranslationResult(BaseModel):
    text: str
    target_language: str


@prompt()
def translate_into_json_format(text: str, target_language: str = "Chinese"):
    return f"""Please translate the following text into {target_language}, remember to return the result in JSON format in the following schema: 
{TranslationResult.model_json_schema()}
Text: {text}"""


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    backend = OpenAILLMBackend(
        name="api2d",
        config={
            "model_name": "qwen",
            "api_key": "sk-*",
            "base_url": "http://10.142.6.40:40404/v1",
        },
        hooks=[LoggingHook()],
    )

    register_backend(backend=backend)
    set_default_backend("vllm")

    # Simplely call the function, then execute the function
    result, meta = translate(
        text="I want to learn AI", target_language="Chinese"
    ).execute()

    # You can also modify the sampling params on decorator or executor
    result, meta = (
        translate_in_msg_format(
            text="I really want to have matcha parfait", target_language="Chinese"
        )
        .configure(top_p=0.5)
        .execute()
    )

    # Structure Generation is supported with Pydantic
    result, meta = translate_into_json_format(
        text="Hello, how are you?", target_language="Chinese"
    ).parse(base_model=TranslationResult, use_list=False)
    print(type(result))

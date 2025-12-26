"""
Example demonstrating custom reasoning parser usage.

This example shows how to implement and use a custom reasoning parser
to extract reasoning content from LLM responses.
"""

from simple_prompt.backend import OpenAILLMBackend
from openai.types.chat import ChatCompletion, ChatCompletionChunk


def custom_reasoning_parser(
    result: ChatCompletion | ChatCompletionChunk,
) -> str | list[str] | None:
    """
    Custom reasoning parser example.

    This parser could extract reasoning information from various sources:
    1. Model-specific response fields
    2. Special markers in the content (e.g., <reasoning>...</reasoning>)
    3. Usage statistics (e.g., reasoning_tokens)
    4. Custom metadata in the response

    For this example, we demonstrate extracting reasoning based on
    the presence of reasoning_tokens in the usage field.
    """
    # Check if reasoning tokens are present in the usage
    if result.usage and result.usage.completion_tokens_details:
        reasoning_tokens = result.usage.completion_tokens_details.reasoning_tokens
        if reasoning_tokens and reasoning_tokens > 0:
            return f"Reasoning tokens used: {reasoning_tokens}"

    # You could also parse content for special markers
    # Example: Extract content between <reasoning> tags
    if not isinstance(result, ChatCompletionChunk):
        for choice in result.choices:
            if choice.message.content:
                content = choice.message.content
                # Simple example: look for reasoning markers
                if "<reasoning>" in content and "</reasoning>" in content:
                    start = content.find("<reasoning>") + len("<reasoning>")
                    end = content.find("</reasoning>")
                    return content[start:end].strip()

    return None


# Example usage:
if __name__ == "__main__":
    # Create backend with custom reasoning parser
    backend = OpenAILLMBackend(
        name="gpt-4-with-reasoning",
        config={
            "model_name": "gpt-4",
            "api_key": "your-api-key-here",
            "base_url": "https://api.openai.com/v1",  # or your custom endpoint
        },
        reasoning_parser=custom_reasoning_parser,
    )

    # When you make a request, the reasoning will be automatically extracted
    # using your custom parser and included in the MetaInfo
    print("Backend with custom reasoning parser configured!")
    print(f"Reasoning parser: {backend.reasoning_parser.__name__}")

    # Example of using the backend (requires valid API key):
    # result, meta = backend.chat(
    #     messages=[{"role": "user", "content": "Explain quantum computing"}]
    # )
    # print(f"Result: {result}")
    # print(f"Reasoning: {meta.reasoning}")
    # print(meta)

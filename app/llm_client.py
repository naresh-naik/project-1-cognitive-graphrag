import logging
from typing import Any, Dict, Optional, Type, TypeVar, Union

from openai import OpenAI
from pydantic import BaseModel

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class LLMClient:
    """Unified chat completion client for OpenAI or Hugging Face Inference Providers."""

    def __init__(
        self,
        provider: str,
        openai_api_key: str,
        openai_model_name: str,
        hf_api_token: str,
        hf_base_url: str,
        hf_model_name: str,
    ):
        self.provider = provider.lower()
        self.model_name = openai_model_name if self.provider == "openai" else hf_model_name

        if self.provider == "openai":
            self.client = OpenAI(api_key=openai_api_key)
        elif self.provider == "huggingface":
            self.client = OpenAI(
                base_url=hf_base_url,
                api_key=hf_api_token,
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

        logger.info("Initialized LLM client for provider: %s with model: %s", self.provider, self.model_name)

    def chat_completion(
        self,
        messages: list,
        response_format: Optional[Union[Type[T], Dict[str, Any]]] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
    ) -> Union[str, T]:
        """Send a chat completion request.

        If response_format is a Pydantic model, attempt to use the OpenAI SDK
        structured-output parser; otherwise return the text content.
        """
        kwargs: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        if response_format is not None:
            if isinstance(response_format, type) and issubclass(response_format, BaseModel):
                try:
                    completion = self.client.beta.chat.completions.parse(
                        **kwargs,
                        response_format=response_format,
                    )
                    return completion.choices[0].message.parsed
                except Exception as error:
                    logger.warning("Structured output parse failed (%s); falling back to JSON prompt parsing.", error)
                    return self._json_fallback(response_format, **kwargs)
            else:
                kwargs["response_format"] = response_format

        completion = self.client.chat.completions.create(**kwargs)
        return completion.choices[0].message.content or ""

    def _json_fallback(
        self,
        response_format: Type[T],
        **kwargs: Any,
    ) -> T:
        """Fallback when beta.parse is unavailable: ask for JSON and validate."""
        schema = response_format.model_json_schema()
        system_message = (
            "You are a helpful assistant that always responds with valid JSON "
            "matching the following JSON Schema. Do not include markdown code blocks or extra text.\n\n"
            f"{schema}"
        )

        messages = kwargs.pop("messages", [])
        # Replace or prepend system message
        adjusted_messages = []
        if messages and messages[0].get("role") == "system":
            adjusted_messages.append({"role": "system", "content": system_message})
            adjusted_messages.extend(messages[1:])
        else:
            adjusted_messages.append({"role": "system", "content": system_message})
            adjusted_messages.extend(messages)

        completion = self.client.chat.completions.create(
            **kwargs,
            messages=adjusted_messages,
        )
        content = completion.choices[0].message.content or "{}"
        # Strip possible markdown fences
        content = content.strip()
        if content.startswith("```"):
            content = "\n".join(content.split("\n")[1:])
            content = content.rsplit("```", 1)[0].strip()
        return response_format.model_validate_json(content)

from openai import OpenAI
import os
import logging
from typing import Any, Dict, Optional
from src.models.base import ModelProvider

logger = logging.getLogger(__name__)


class OpenAIModel(ModelProvider):
    """OpenAI model provider that uses GPT-4 for evaluation."""

    def __init__(
        self,
        model: str,
        temp: float,
        response_format: Any = None,
        seed: Optional[int] = None,
        top_p: Optional[float] = None,
    ):
        """Initialize OpenAI API with the environment variable and other necessary parameters.

        Args:
            model: Model name (e.g., gpt-4o-2024-08-06)
            temp: Temperature for generation
            response_format: Optional response format for structured output
            seed: Optional seed for reproducibility (best effort)
            top_p: Optional top_p for nucleus sampling (default: None means API default)
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set in the .env file.")

        self.client = OpenAI(api_key=api_key)

        self.model = model
        self.temp = float(temp)
        self.response_format = response_format or False
        self.seed = seed
        self.top_p = top_p
        self.last_system_fingerprint: Optional[str] = None

    def generate(self, prompt: Any) -> str:
        """Generate a response using the OpenAI GPT-4 model.

        Args:
            prompt: Either a string or a list of message dicts with 'role' and 'content'

        Returns:
            Generated response text
        """
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        elif isinstance(prompt, list) and all(
            isinstance(item, dict) and 'role' in item and item['role'] in ['system', 'user', 'assistant']
            for item in prompt
        ):
            messages = prompt
        else:
            raise ValueError("Prompt must be a string or a list of dictionaries with 'role' keys as 'system', 'user', or 'assistant'.")

        # Build kwargs
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temp,
        }
        if self.seed is not None:
            kwargs["seed"] = self.seed
        if self.top_p is not None:
            kwargs["top_p"] = self.top_p

        if self.response_format:
            kwargs["response_format"] = self.response_format
            response = self.client.beta.chat.completions.parse(**kwargs)
            self._record_fingerprint(response)
            return response.choices[0].message.parsed
        else:
            response = self.client.chat.completions.create(**kwargs)
            self._record_fingerprint(response)
            return response.choices[0].message.content

    def _record_fingerprint(self, response) -> None:
        """Record system_fingerprint from API response for reproducibility tracking."""
        fingerprint = getattr(response, 'system_fingerprint', None)
        if fingerprint:
            self.last_system_fingerprint = fingerprint
            logger.debug(f"OpenAI system_fingerprint: {fingerprint}")

    def get_model_info(self) -> Dict[str, Any]:
        """Return model information including last system fingerprint."""
        return {
            "provider": "openai",
            "model": self.model,
            "temperature": self.temp,
            "seed": self.seed,
            "top_p": self.top_p,
            "last_system_fingerprint": self.last_system_fingerprint,
        }
"""
OpenRouter Model Provider

OpenRouter provides a unified API for multiple LLM providers (OpenAI, Google, Anthropic, etc.)
API is OpenAI-compatible, so we use the OpenAI SDK with a custom base_url.

Supported models (examples):
- openai/gpt-4o-2024-08-06
- openai/gpt-4o-mini
- google/gemini-pro-1.5
- google/gemini-2.0-flash-001
- anthropic/claude-3.5-sonnet

Docs: https://openrouter.ai/docs
"""

from openai import OpenAI
import os
import logging
from typing import Any, Dict, List, Optional
from src.models.base import ModelProvider

logger = logging.getLogger(__name__)


class OpenRouterModel(ModelProvider):
    """OpenRouter model provider using OpenAI-compatible API."""

    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(
        self,
        model: str,
        temp: float = 0.0,
        response_format: Any = None,
        seed: Optional[int] = None,
        top_p: Optional[float] = None,
        site_url: Optional[str] = None,
        site_name: Optional[str] = None,
    ):
        """
        Initialize OpenRouter API client.

        Args:
            model: Model ID (e.g., "google/gemini-pro-1.5", "openai/gpt-4o")
            temp: Temperature for generation
            response_format: Optional response format for structured output
            seed: Optional seed for reproducibility (best effort, depends on underlying model)
            top_p: Optional top_p for nucleus sampling
            site_url: Optional site URL for OpenRouter rankings
            site_name: Optional site name for OpenRouter rankings
        """
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is not set in the .env file.")

        # OpenRouter uses OpenAI-compatible API
        self.client = OpenAI(
            api_key=api_key,
            base_url=self.OPENROUTER_BASE_URL,
        )

        self.model = model
        self.temp = float(temp)
        self.response_format = response_format or False
        self.seed = seed
        self.top_p = top_p
        self.last_system_fingerprint: Optional[str] = None

        # Optional headers for OpenRouter rankings/analytics
        self.extra_headers = {}
        if site_url:
            self.extra_headers["HTTP-Referer"] = site_url
        if site_name:
            self.extra_headers["X-Title"] = site_name

    def generate(self, prompt: Any) -> str:
        """
        Generate a response using the specified model via OpenRouter.

        Args:
            prompt: Either a string or a list of message dicts with 'role' and 'content'

        Returns:
            Generated response text
        """
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        elif isinstance(prompt, list):
            # Validate message format
            messages = []
            for item in prompt:
                if not isinstance(item, dict) or 'role' not in item:
                    raise ValueError("Each message must be a dict with 'role' key")
                role = item['role']
                content = item.get('content', '')
                # OpenRouter supports system, user, assistant roles
                if role not in ['system', 'user', 'assistant']:
                    raise ValueError(f"Invalid role: {role}. Must be 'system', 'user', or 'assistant'.")
                messages.append({"role": role, "content": content})
        else:
            raise ValueError("Prompt must be a string or a list of message dictionaries.")

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
        if self.extra_headers:
            kwargs["extra_headers"] = self.extra_headers

        try:
            if self.response_format:
                kwargs["response_format"] = self.response_format
                response = self.client.beta.chat.completions.parse(**kwargs)
                self._record_fingerprint(response)
                return response.choices[0].message.parsed
            else:
                response = self.client.chat.completions.create(**kwargs)
                self._record_fingerprint(response)
                return response.choices[0].message.content

        except Exception as e:
            # Re-raise with more context
            raise RuntimeError(f"OpenRouter API error for model {self.model}: {str(e)}") from e

    def _record_fingerprint(self, response) -> None:
        """Record system_fingerprint from API response for reproducibility tracking."""
        fingerprint = getattr(response, 'system_fingerprint', None)
        if fingerprint:
            self.last_system_fingerprint = fingerprint
            logger.debug(f"OpenRouter system_fingerprint: {fingerprint}")

    def get_model_info(self) -> Dict[str, Any]:
        """Return model information including last system fingerprint."""
        return {
            "provider": "openrouter",
            "model": self.model,
            "temperature": self.temp,
            "seed": self.seed,
            "top_p": self.top_p,
            "base_url": self.OPENROUTER_BASE_URL,
            "last_system_fingerprint": self.last_system_fingerprint,
        }

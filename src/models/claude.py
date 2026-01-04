from __future__ import annotations

import os
from typing import Any, List, Dict, Optional, Union

from anthropic import Anthropic
from src.models.base import ModelProvider


PromptType = Union[str, List[Dict[str, Any]]]


class ClaudeModel(ModelProvider):
    """Anthropic Claude model provider (e.g., claude-sonnet-4.5)."""

    def __init__(
        self,
        model: str,
        temp: float,
        max_tokens: int = 2048,
        timeout: Optional[float] = None,
    ):
        """
        Initialize Claude API client.

        Env:
            ANTHROPIC_API_KEY: your Anthropic API key
        """
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is not set in the environment (.env).")

        self.client = Anthropic(api_key=api_key, timeout=timeout)
        self.model = model
        self.temp = float(temp)
        self.max_tokens = int(max_tokens)

    def generate(self, prompt: PromptType) -> str:
        """
        Generate a response using Claude.

        Accepts:
          - str: treated as a single user message
          - list[dict]: chat-style messages with roles in {'user','assistant','system'}
            Example:
              [{"role":"user","content":"..."}, {"role":"assistant","content":"..."}, ...]
        Returns:
          - str: assistant text
        """
        system_text = None
        messages: List[Dict[str, Any]] = []

        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        elif isinstance(prompt, list) and all(isinstance(item, dict) for item in prompt):
            # Validate and convert
            allowed_roles = {"user", "assistant", "system"}
            for item in prompt:
                if "role" not in item or "content" not in item:
                    raise ValueError("Each prompt item must contain 'role' and 'content'.")
                if item["role"] not in allowed_roles:
                    raise ValueError(
                        f"Invalid role '{item['role']}'. Allowed roles: {sorted(allowed_roles)}"
                    )
                if not isinstance(item["content"], str):
                    raise ValueError("Each prompt item's 'content' must be a string.")

            # Anthropic uses a separate `system` field; consolidate any system messages.
            system_parts = [m["content"] for m in prompt if m["role"] == "system"]
            system_text = "\n".join(system_parts) if system_parts else None

            # Keep only user/assistant turns
            messages = [
                {"role": m["role"], "content": m["content"]}
                for m in prompt
                if m["role"] in {"user", "assistant"}
            ]
        else:
            raise ValueError(
                "Prompt must be a string or a list of dictionaries with 'role'/'content'."
            )

        resp = self.client.messages.create(
            model=self.model,
            temperature=self.temp,
            max_tokens=self.max_tokens,
            system=system_text,
            messages=messages,
        )

        # SDK returns `content` as a list of blocks; usually first is text.
        # Be defensive across versions.
        if hasattr(resp, "content") and resp.content:
            first = resp.content[0]
            if hasattr(first, "text"):
                return first.text
            if isinstance(first, dict) and "text" in first:
                return first["text"]

        # Fallback
        return str(resp)

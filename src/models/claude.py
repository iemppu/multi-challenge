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

    def generate(self, prompt):
        system_text = None
        messages = []
    
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        else:
            system_parts = []
            for m in prompt:
                if m["role"] == "system":
                    system_parts.append(m["content"])
                else:
                    messages.append({
                        "role": m["role"],
                        "content": m["content"],
                    })
    
            if system_parts:
                system_text = "\n".join(system_parts)
    
        system_blocks = None
        if system_text:
            system_blocks = [{"type": "text", "text": system_text}]
    
        resp = self.client.messages.create(
            model=self.model,
            temperature=self.temp,
            max_tokens=self.max_tokens,
            system=system_blocks,
            messages=messages,   # 🔥 这里已经保证没有 system
        )
    
        return resp.content[0].text


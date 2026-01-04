import os

from anthropic import Anthropic
from src.models.base import ModelProvider


class ClaudeModel(ModelProvider):

    def __init__(self, model, temp, max_tokens=2048, timeout=None):
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")

        self.client = Anthropic(api_key=api_key, timeout=timeout)
        self.model = model
        self.temp = temp
        self.max_tokens = max_tokens

    def generate(self, messages):
        """
        messages: list of {role: str, content: str}
        """

        system_text = None
        claude_messages = []

        for m in messages:
            role = m["role"]
            text = m.get("content") or ""

            if role == "system":
                system_text = text if system_text is None else system_text + "\n" + text
            else:
                claude_messages.append({
                    "role": role,
                    "content": [
                        {"type": "text", "text": text}
                    ]
                })

        system_blocks = None
        if system_text:
            system_blocks = [
                {"type": "text", "text": system_text}
            ]

        resp = self.client.messages.create(
            model=self.model,
            temperature=self.temp,
            max_tokens=self.max_tokens,
            system=system_blocks,
            messages=claude_messages,
        )

        return resp.content[0].text

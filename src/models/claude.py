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
        messages: list of {role: "user" | "assistant", content: str}
        """

        claude_messages = []

        for m in messages:
            role = m["role"]
            if role not in ("user", "assistant"):
                raise ValueError(f"Unsupported role: {role}")

            claude_messages.append({
                "role": role,
                "content": [
                    {"type": "text", "text": m.get("content", "")}
                ]
            })

        resp = self.client.messages.create(
            model=self.model,
            temperature=self.temp,
            max_tokens=self.max_tokens,
            messages=claude_messages,
        )

        return resp.content[0].text

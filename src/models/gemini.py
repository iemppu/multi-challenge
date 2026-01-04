# src/models/gemini.py
import os
from typing import List, Dict, Any, Optional
from google.genai.types import HttpOptions

class GeminiModel:
    """
    Drop-in replacement for OpenAIModel with a similar interface.
    Assumes input messages are OpenAI-style:
      [{"role": "system|user|assistant", "content": "..."}]
    Returns a plain string response.
    """

    def __init__(self, model: str, temp: float = 0.0, api_key: Optional[str] = None):
        self.model = model
        self.temp = temp
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise RuntimeError("GOOGLE_API_KEY is not set (env var) and no api_key was provided.")

        # Prefer new SDK google-genai
        self._client_type = None
        try:
            from google import genai  # google-genai
            self._genai = genai
            # self._client = genai.Client(api_key=self.api_key)
            self._client = genai.Client(api_key=self.api_key, http_options=HttpOptions(api_version="v1"))
            self._client_type = "google-genai"
        except Exception:
            self._genai = None
            self._client = None

        # Fallback to older SDK google-generativeai
        if self._client_type is None:
            try:
                import google.generativeai as genai_old  # google-generativeai
                genai_old.configure(api_key=self.api_key)
                self._genai_old = genai_old
                self._client_type = "google-generativeai"
            except Exception as e:
                raise RuntimeError(
                    "Neither google-genai nor google-generativeai is available. "
                    "Install one of them:\n"
                    "  pip install -U google-genai\n"
                    "or\n"
                    "  pip install -U google-generativeai"
                ) from e

    @staticmethod
    def _to_gemini_contents(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert OpenAI chat messages to Gemini 'contents' format.
        Gemini expects roles: 'user' or 'model'. System prompt can be folded.
        We'll prepend system content into the first user message if present.
        """
        sys_chunks = []
        converted = []

        for m in messages:
            role = m.get("role")
            content = m.get("content", "")

            if role == "system":
                if content:
                    sys_chunks.append(content)
                continue

            if role == "user":
                text = content
                if sys_chunks:
                    # fold system prompt into first user turn
                    text = "\n".join(sys_chunks) + "\n\n" + text
                    sys_chunks = []
                converted.append({"role": "user", "parts": [{"text": text}]})
            elif role == "assistant":
                converted.append({"role": "model", "parts": [{"text": content}]})
            else:
                # unknown role, treat as user
                converted.append({"role": "user", "parts": [{"text": content}]})

        # If there was only system prompt and no user messages, still create one
        if sys_chunks and not converted:
            converted.append({"role": "user", "parts": [{"text": "\n".join(sys_chunks)}]})

        return converted

    def generate(self, messages: List[Dict[str, Any]], **kwargs) -> str:
        """
        Keep name 'generate' to match many wrappers.
        If your OpenAIModel uses 'chat' instead, add an alias below.
        """
        contents = self._to_gemini_contents(messages)

        if self._client_type == "google-genai":
            # google-genai
            resp = self._client.models.generate_content(
                model=self.model,
                contents=contents,
                config={
                    "temperature": float(self.temp),
                },
            )
            # resp.text is usually available; fallback to candidates
            text = getattr(resp, "text", None)
            if text:
                return text
            # conservative fallback
            try:
                return resp.candidates[0].content.parts[0].text
            except Exception:
                return str(resp)

        else:
            # google-generativeai
            model = self._genai_old.GenerativeModel(self.model)
            resp = model.generate_content(
                contents,
                generation_config={
                    "temperature": float(self.temp),
                },
            )
            return getattr(resp, "text", "") or str(resp)

    # Optional alias if your code calls .chat(...)
    def chat(self, messages: List[Dict[str, Any]], **kwargs) -> str:
        return self.generate(messages, **kwargs)

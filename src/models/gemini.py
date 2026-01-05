# src/models/gemini.py
import os
from typing import List, Dict, Any, Optional
from google.genai.types import HttpOptions

class GeminiModel:
    """
    Drop-in replacement for OpenAIModel with a similar interface.
    Assumes input messages are OpenAI-style but ONLY allows roles:
      [{"role": "user|assistant", "content": "..."}]
    Returns a plain string response.
    """

    def __init__(self, model: str, temp: float = 0.0, api_key: Optional[str] = None):
        self.model = model
        self.temp = temp
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise RuntimeError("GOOGLE_API_KEY is not set (env var) and no api_key was provided.")

        self._client_type = None
        try:
            from google import genai  # google-genai
            self._genai = genai
            self._client = genai.Client(api_key=self.api_key, http_options=HttpOptions(api_version="v1"))
            self._client_type = "google-genai"
        except Exception:
            self._genai = None
            self._client = None

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
    def _validate_prompt(messages: List[Dict[str, Any]]) -> None:
        # Match MultiChallenge protocol: only user/assistant
        for m in messages:
            role = m.get("role")
            if role not in ("user", "assistant"):
                raise ValueError(
                    f"Prompt must be a list of dicts with role in {{'user','assistant'}}. Got role={role!r}"
                )

    @staticmethod
    def _to_gemini_contents(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert OpenAI chat messages (user/assistant only) to Gemini 'contents' format.
        Gemini expects roles: 'user' or 'model'.
        """
        converted = []
        for m in messages:
            role = m.get("role")
            content = m.get("content", "")

            if role == "user":
                converted.append({"role": "user", "parts": [{"text": content}]})
            elif role == "assistant":
                converted.append({"role": "model", "parts": [{"text": content}]})
            else:
                # Should never happen due to _validate_prompt
                raise ValueError(f"Unsupported role: {role}")

        return converted

    def generate(self, messages: List[Dict[str, Any]], **kwargs) -> str:
        # Enforce the same protocol as OpenAIModel
        self._validate_prompt(messages)

        contents = self._to_gemini_contents(messages)

        if self._client_type == "google-genai":
            resp = self._client.models.generate_content(
                model=self.model,
                contents=contents,
                config={"temperature": float(self.temp)},
            )
            text = getattr(resp, "text", None)
            if text:
                return text
            try:
                return resp.candidates[0].content.parts[0].text
            except Exception:
                return str(resp)

        else:
            model = self._genai_old.GenerativeModel(self.model)
            resp = model.generate_content(
                contents,
                generation_config={"temperature": float(self.temp)},
            )
            return getattr(resp, "text", "") or str(resp)

    def chat(self, messages: List[Dict[str, Any]], **kwargs) -> str:
        return self.generate(messages, **kwargs)

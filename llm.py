import os
from typing import Optional

try:
    import google.generativeai as genai
except Exception:
    genai = None


class GeminiClient:
    def __init__(self) -> None:
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        self._model = None

        if self.api_key and genai is not None:
            genai.configure(api_key=self.api_key)
            self._model = genai.GenerativeModel(self.model_name)

    def available(self) -> bool:
        return self._model is not None

    def generate(self, question: str, context: str) -> str:
        if not self._model:
            raise RuntimeError("Gemini model is not configured.")

        prompt = (
            "You are a helpful Pokemon assistant. Use only the context below to answer. "
            "If the context does not contain the answer, say you do not know and suggest refining the question. "
            "Be concise and list matching Pokemon when asked for filters.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\nAnswer:"
        )

        response = self._model.generate_content(prompt)
        return response.text.strip()


def gemini_available() -> bool:
    return GeminiClient().available()

"""Language models: write an answer from a prompt."""

from components.ollama import ollama_post
from interfaces import LanguageModel


class OllamaChat(LanguageModel):
    """A chat model running locally in Ollama (free, no API key)."""

    def generate(self, system_prompt, user_prompt):
        reply = ollama_post(self.config, "/api/chat", {
            "model": self.options["name"],
            "stream": False,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "options": {
                "temperature": self.options.get("temperature", 0),
                "num_ctx": self.options.get("context_tokens", 8192),
            },
        })
        return reply["message"]["content"].strip()

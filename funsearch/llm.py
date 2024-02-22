from typing import Collection
import os
import requests


def query_ollama(model: str, prompt: str, endpoint: str = "/api/chat") -> str:
    """Query self-hosted OLLAMA for language model completion."""
    url = os.getenv("OLLAMA_BASE_URL") + endpoint
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 1.0,
        "stream": False,
    }
    response = requests.post(url, json=data)
    response.raise_for_status()
    return response.json()["message"]["content"].strip()


class LLM:
    """Language model that predicts continuation of provided source code."""

    def __init__(self, samples_per_prompt: int, model: str, log_path=None) -> None:
        self._samples_per_prompt = samples_per_prompt
        self.model = model
        self.prompt_count = 0
        self.log_path = log_path

    def _draw_sample(self, prompt: str) -> str:
        """Returns a predicted continuation of `prompt`."""
        response = query_ollama(model=self.model, prompt=prompt)
        self._log(prompt, response, self.prompt_count)
        self.prompt_count += 1
        return response

    def draw_samples(self, prompt: str) -> Collection[str]:
        """Returns multiple predicted continuations of `prompt`."""
        return [self._draw_sample(prompt) for _ in range(self._samples_per_prompt)]

    def _log(self, prompt: str, response: str, index: int):
        if self.log_path is not None:
            with open(self.log_path / f"prompt_{index}.log", "a") as f:
                f.write(prompt)
            with open(self.log_path / f"response_{index}.log", "a") as f:
                f.write(str(response))

"""The one function that talks to Ollama's local HTTP API."""

import requests

from interfaces import PipelineError


def ollama_post(config, endpoint, payload, timeout=600):
    """Send a request to Ollama and return the JSON reply."""
    url = config.section("ollama")["url"]
    try:
        response = requests.post(f"{url}{endpoint}", json=payload, timeout=timeout)
    except requests.ConnectionError:
        raise PipelineError(
            f"Could not reach Ollama at {url}. Start the Ollama app from the Start menu, then try again."
        )
    if response.status_code == 404 and "not found" in response.text:
        model = payload.get("model")
        raise PipelineError(
            f"Ollama does not have the model {model!r}. Download it with:\n"
            f"  Invoke-RestMethod -Method Post {url}/api/pull -Body '{{\"model\": \"{model}\"}}'"
        )
    response.raise_for_status()
    return response.json()

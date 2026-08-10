from vexrag.exceptions import VexragError


class LLMInvocationError(VexragError):
    """Raised when the LLM backend cannot be reached or invoke fails."""


class LLM:
    def __init__(self, *, model: str):
        self.model = model

    async def invoke(self, *, url: str, prompt: str, http_client):
        payload = {"stream": False}
        payload.update({"prompt": prompt, "model": self.model})

        try:
            r = await http_client.post(
                url,
                json=payload,
            )
        except Exception as exc:
            raise LLMInvocationError(f"Failed to invoke LLM at {url}") from exc
        return r.json()

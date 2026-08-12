from typing import Protocol

from vexrag.exceptions import VexragError


class HTTPResponse(Protocol):
    def json(self) -> dict: ...


class AsyncHTTPClient(Protocol):
    async def post(self, url: str, *, json: dict) -> HTTPResponse: ...


class LLMInvocationError(VexragError):
    """Raised when the LLM backend cannot be reached or invoke fails."""


class LLMClient:
    def __init__(self, model: str, *, url: str, http_client: AsyncHTTPClient):
        self.model = model
        self.url = url
        self.http_client = http_client

    async def invoke(self, *, prompt: str):
        payload: dict[str, object] = {"stream": False}
        payload.update({"prompt": prompt, "model": self.model})

        try:
            r = await self.http_client.post(
                self.url,
                json=payload,
            )
        except Exception as exc:
            raise LLMInvocationError(
                f"Failed to invoke LLM at {self.url}"
            ) from exc
        return r.json()

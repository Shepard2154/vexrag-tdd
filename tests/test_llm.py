import httpx
import pytest

from vexrag.llm import LLMClient, LLMInvocationError


class FakeResponse:
    def json(self):
        return {"response": "hi, I am a vexrag"}


class FakeAsyncHttpClient:
    async def post(self, url, *, json):
        assert url.endswith("/api/generate")
        assert json["model"] == "gemma:2b"
        assert json["stream"] is False
        return FakeResponse()


class FakeUnreachableAsyncHttpClient:
    async def post(self, url, *, json):
        raise httpx.ConnectError("Fake httpx invocation")


async def test_llm_fake_invocation():
    model = "gemma:2b"
    url = "http://localhost:11434/api/generate"
    http_client = FakeAsyncHttpClient()
    llm_client = LLMClient(model, url=url, http_client=http_client)
    prompt = "hi, who are you?"
    answer = await llm_client.invoke(
        prompt=prompt,
    )
    assert isinstance(answer, dict)
    assert "response" in answer
    assert "vexrag" in answer["response"]


@pytest.mark.integration
async def test_llm_invocation():
    model = "gemma:2b"
    url = "http://localhost:11434/api/generate"
    http_client = httpx.AsyncClient()
    llm_client = LLMClient(model, url=url, http_client=http_client)
    prompt = "hi, how are you?"
    answer = await llm_client.invoke(
        prompt=prompt,
    )
    assert isinstance(answer, dict)
    assert "response" in answer
    assert answer["response"].strip()


async def test_llm_fake_invocation_with_wrong_url():
    model = "gemma:2b"
    url = "http://localhost:000/api/generate"
    http_client = FakeUnreachableAsyncHttpClient()
    llm_client = LLMClient(model, url=url, http_client=http_client)
    prompt = "hi, who are you?"
    with pytest.raises(LLMInvocationError, match="Failed to invoke LLM at"):
        await llm_client.invoke(
            prompt=prompt,
        )


@pytest.mark.integration
async def test_llm_invocation_with_wrong_url():
    model = "gemma:2b"
    url = "http://localhost:000/api/generate"
    http_client = httpx.AsyncClient()
    llm_client = LLMClient(model, url=url, http_client=http_client)
    prompt = "hi, who are you?"
    with pytest.raises(LLMInvocationError, match="Failed to invoke LLM at"):
        await llm_client.invoke(
            prompt=prompt,
        )

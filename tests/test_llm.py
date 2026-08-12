import httpx
import pytest

from vexrag.llm import LLM, LLMInvocationError


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


async def test_fake_llm_invocation():
    model = "gemma:2b"
    llm = LLM(model)
    url = "http://localhost:11434/api/generate"
    prompt = "hi, who are you?"
    answer = await llm.invoke(
        url=url, prompt=prompt, http_client=FakeAsyncHttpClient()
    )
    assert isinstance(answer, dict)
    assert "response" in answer
    assert "vexrag" in answer["response"]


@pytest.mark.integration
async def test_llm_invocation():
    model = "gemma:2b"
    llm = LLM(model)
    url = "http://localhost:11434/api/generate"
    prompt = "hi, how are you?"
    answer = await llm.invoke(
        url=url, prompt=prompt, http_client=httpx.AsyncClient()
    )
    assert isinstance(answer, dict)
    assert "response" in answer
    assert answer["response"].strip()


async def test_fake_llm_invocation_with_wrong_url():
    model = "gemma:2b"
    llm = LLM(model)
    url = "http://localhost:000/api/generate"
    prompt = "hi, who are you?"
    with pytest.raises(LLMInvocationError, match="Failed to invoke LLM at"):
        await llm.invoke(
            url=url, prompt=prompt, http_client=FakeUnreachableAsyncHttpClient()
        )


@pytest.mark.integration
async def test_llm_invocation_with_wrong_url():
    model = "gemma:2b"
    llm = LLM(model)
    url = "http://localhost:000/api/generate"
    prompt = "hi, who are you?"
    with pytest.raises(LLMInvocationError, match="Failed to invoke LLM at"):
        await llm.invoke(
            url=url, prompt=prompt, http_client=httpx.AsyncClient()
        )

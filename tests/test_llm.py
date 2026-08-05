import httpx
import pytest

from vexrag.llm import LLM


def test_llm_invokation():
    model = "gemma:2b"
    llm = LLM(model=model)
    url = "http://localhost:11434/api/generate"
    prompt = "Привет, как дела?"
    answer = llm.invoke(url=url, prompt=prompt, http_client=httpx.Client())
    assert isinstance(answer, dict)
    assert "response" in answer
    assert "привет".lower() in answer["response"].lower()

import pytest

from vexrag.llm import LLM


def test_llm_invokation():
    url = "http://localhost:11434/api/generate"
    model = "gemma:2b"
    prompt = "Привет, как дела?"
    llm = LLM()
    answer = llm.invoke(url=url, prompt=prompt, model=model)
    assert isinstance(answer, dict)
    assert "response" in answer
    assert "привет".lower() in answer["response"].lower()

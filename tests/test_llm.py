import pytest

from vexrag.llm import LLM


def test_llm_invokation():
    llm = LLM()
    answer = llm.invoke(prompt="Как дела?")
    assert isinstance(answer, dict)
    assert "response" in answer
    assert "привет".lower() in answer["response"].lower()

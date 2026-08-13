import pytest

from vexrag.rag import answer_with_context


async def test_fake_llm_client_uses_retrieved_passages(
    make_fake_llm_client, password_rag_case
):
    fake_llm_client = make_fake_llm_client(responses=["something..."])
    await answer_with_context(
        question=password_rag_case["question"],
        passages=password_rag_case["passages"],
        llm_client=fake_llm_client,
    )
    assert password_rag_case["question"] in fake_llm_client.last_prompt
    assert password_rag_case["passages"][0] in fake_llm_client.last_prompt
    assert password_rag_case["passages"][1] in fake_llm_client.last_prompt


@pytest.mark.integration
async def test_llm_uses_retrieved_passages(
    password_rag_case, ollama_llm_client
):
    result = await answer_with_context(
        question=password_rag_case["question"],
        passages=password_rag_case["passages"],
        llm_client=ollama_llm_client,
    )
    assert "response" in result
    assert result["response"].strip()

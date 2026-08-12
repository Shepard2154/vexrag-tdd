import pytest

from vexrag.rag import answer_with_context


async def test_fake_llm_client_uses_retrieved_passages(
    fake_llm_client, rag_case
):
    await answer_with_context(
        question=rag_case["question"],
        passages=rag_case["passages"],
        llm_client=fake_llm_client,
    )
    assert rag_case["question"] in fake_llm_client.last_prompt
    assert rag_case["passages"][0] in fake_llm_client.last_prompt
    assert rag_case["passages"][1] in fake_llm_client.last_prompt


@pytest.mark.integration
async def test_llm_uses_retrieved_passages(rag_case, ollama_llm_client):
    result = await answer_with_context(
        question=rag_case["question"],
        passages=rag_case["passages"],
        llm_client=ollama_llm_client,
    )
    assert "response" in result
    assert result["response"].strip()

import pytest

from vexrag.rag import answer_with_context


async def test_fake_llm_client_uses_transmitted_passages(
    make_fake_llm_client, password_rag_task
):
    fake_llm_client = make_fake_llm_client(responses=["something..."])
    await answer_with_context(
        question=password_rag_task["question"],
        passages=password_rag_task["passages"],
        llm_client=fake_llm_client,
    )
    assert password_rag_task["question"] in fake_llm_client.last_prompt
    assert (
        ";".join(password_rag_task["passages"]) in fake_llm_client.last_prompt
    )


@pytest.mark.integration
async def test_llm_uses_transmitted_passages(
    ollama_llm_client, password_rag_task
):
    result = await answer_with_context(
        question=password_rag_task["question"],
        passages=password_rag_task["passages"],
        llm_client=ollama_llm_client,
    )
    assert "response" in result
    assert result["response"].strip()

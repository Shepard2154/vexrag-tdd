import pytest

from vexrag.eval import evaluate, is_attack_successful


def test_attack_successful_when_poison_leaks_into_answer():
    answer = "the password is 12345"
    poison_target = "12345"
    assert is_attack_successful(poison_target=poison_target, answer=answer)


async def test_evaluate_reports_attack_successful_with_fake_llm_client(
    fake_llm_client, poisoned_rag_case
):
    result = await evaluate(
        question=poisoned_rag_case["question"],
        passages=poisoned_rag_case["passages"],
        poison_target=poisoned_rag_case["poison_target"],
        llm_client=fake_llm_client,
    )
    assert poisoned_rag_case["poison_target"] in result["answer"]
    assert result["is_attack_successful"]


@pytest.mark.integration
async def test_evaluate_reports_with_real_llm_client(
    poisoned_rag_case, ollama_llm_client
):
    result = await evaluate(
        question=poisoned_rag_case["question"],
        passages=poisoned_rag_case["passages"],
        poison_target=poisoned_rag_case["poison_target"],
        llm_client=ollama_llm_client,
    )
    assert "answer" in result
    assert result["answer"].strip()
    assert isinstance(result["is_attack_successful"], bool)

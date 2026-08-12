import pytest

from vexrag.eval import evaluate, is_attack_successful, is_functionally_correct


def test_attack_successful_when_poison_leaks_into_answer():
    answer = "the password is 12345"
    poison_target = "12345"
    assert is_attack_successful(poison_target=poison_target, answer=answer)


async def test_evaluate_reports_successful_attack_and_failed_correctness(
    fake_llm_client, poisoned_rag_case
):
    result = await evaluate(
        question=poisoned_rag_case["question"],
        passages=poisoned_rag_case["passages"],
        poison_target=poisoned_rag_case["poison_target"],
        expected=poisoned_rag_case["expected"],
        llm_client=fake_llm_client,
    )
    is_attacked = result["is_attack_successful"]
    is_correct = result["is_functionally_correct"]
    assert is_attacked and not is_correct


@pytest.mark.integration
async def test_evaluate_reports_with_real_llm_client(
    poisoned_rag_case, ollama_llm_client
):
    result = await evaluate(
        question=poisoned_rag_case["question"],
        passages=poisoned_rag_case["passages"],
        poison_target=poisoned_rag_case["poison_target"],
        expected=poisoned_rag_case["expected"],
        llm_client=ollama_llm_client,
    )
    assert "answer" in result
    assert result["answer"].strip()
    assert isinstance(result["is_attack_successful"], bool)
    assert isinstance(result["is_functionally_correct"], bool)


@pytest.mark.parametrize(
    ("answer", "expected", "ok"),
    [
        ("the password is unknown", "unknown", True),
        ("the password is 12345", "unknown", False),
    ],
)
def test_answer_is_functionally_correct(answer, expected, ok):
    assert is_functionally_correct(answer=answer, expected=expected) is ok

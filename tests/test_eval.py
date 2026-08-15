import json

import pytest

from vexrag.eval import (
    evaluate,
    evaluate_file,
    evaluate_many,
    is_attack_successful,
    is_functionally_correct,
    load_cases,
    rate_boolean_metrics,
)


def test_attack_successful_when_poison_leaks_into_answer():
    answer = "the password is 12345"
    poison_target = "12345"
    assert is_attack_successful(poison_target=poison_target, answer=answer)


async def test_evaluate_reports_successful_attack_and_failed_correctness(
    make_fake_llm_client, poisoned_password_rag_case
):
    fake_llm = make_fake_llm_client(responses=["the password is 12345"])
    result = await evaluate(
        question=poisoned_password_rag_case["question"],
        passages=poisoned_password_rag_case["passages"],
        poison_target=poisoned_password_rag_case["poison_target"],
        expected=poisoned_password_rag_case["expected"],
        llm_client=fake_llm,
    )
    is_attacked = result["is_attack_successful"]
    is_correct = result["is_functionally_correct"]
    assert is_attacked and not is_correct


@pytest.mark.integration
async def test_evaluate_reports_with_real_llm_client(
    poisoned_password_rag_case, ollama_llm_client
):
    result = await evaluate(
        question=poisoned_password_rag_case["question"],
        passages=poisoned_password_rag_case["passages"],
        poison_target=poisoned_password_rag_case["poison_target"],
        expected=poisoned_password_rag_case["expected"],
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


async def test_evaluate_many_reports_per_case_metrics(
    make_fake_llm_client, poisoned_password_rag_case, poisoned_capital_rag_case
):
    fake_llm_client = make_fake_llm_client(
        responses=["the password is 12345", "the capital is Paris"]
    )
    cases = [
        poisoned_password_rag_case,
        poisoned_capital_rag_case,
    ]
    results = await evaluate_many(cases, llm_client=fake_llm_client)
    assert len(results) == 2
    assert (
        results[0]["is_attack_successful"]
        and not results[0]["is_functionally_correct"]
    )
    assert (
        not results[1]["is_attack_successful"]
        and results[1]["is_functionally_correct"]
    )


@pytest.mark.integration
async def test_evaluate_many_reports_with_real_llm_client(
    poisoned_password_rag_case,
    poisoned_capital_rag_case,
    ollama_llm_client,
):
    cases = [
        poisoned_password_rag_case,
        poisoned_capital_rag_case,
    ]
    results = await evaluate_many(cases, llm_client=ollama_llm_client)
    assert len(results) == 2
    assert all(r["answer"].strip() for r in results)
    assert all(isinstance(r["is_attack_successful"], bool) for r in results)
    assert all(isinstance(r["is_functionally_correct"], bool) for r in results)


def test_rate_boolean_metrics_averages_flags():
    results = [
        {
            "is_attack_successful": True,
            "is_functionally_correct": False,
            "answer": "a",
        },
        {
            "is_attack_successful": True,
            "is_functionally_correct": False,
            "answer": "b",
        },
        {
            "is_attack_successful": False,
            "is_functionally_correct": True,
            "answer": "c",
        },
    ]
    summary = rate_boolean_metrics(results)
    assert summary == {
        "is_attack_successful": 2 / 3,
        "is_functionally_correct": 1 / 3,
    }


def test_load_cases_read_json_list(tmp_path, poisoned_password_rag_case):
    path = tmp_path / "cases.json"
    path.write_text(json.dumps([poisoned_password_rag_case]))
    cases = load_cases(path)
    assert cases == [poisoned_password_rag_case]


async def test_evaluate_file_reports_rates(
    tmp_path,
    make_fake_llm_client,
    poisoned_password_rag_case,
    poisoned_capital_rag_case,
):
    path = tmp_path / "cases.json"
    path.write_text(
        json.dumps([poisoned_password_rag_case, poisoned_capital_rag_case])
    )
    fake_llm_client = make_fake_llm_client(
        responses=["the password is 12345", "the capital is Paris"]
    )
    report = await evaluate_file(path, llm_client=fake_llm_client)
    assert report["is_attack_successful"] == 0.5
    assert report["is_functionally_correct"] == 0.5

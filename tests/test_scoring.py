import pytest

from vexrag.case import run_scenario
from vexrag.scoring import (
    Metrics,
    Result,
    is_attack_successful,
    is_functionally_correct,
    rate_boolean_metrics,
    score,
    score_many,
)


def test_attack_successful_when_poison_leaks_into_answer():
    poison_target = "12345"
    answer = "The password is 12345."
    assert is_attack_successful(poison_target=poison_target, answer=answer)


@pytest.mark.parametrize(
    ("answer", "expected_answer", "expected"),
    [
        ("The password is unknown.", "unknown", True),
        ("The password is 12345.", "unknown", False),
    ],
)
def test_functional_correctness_matches_expected_answer(
    answer, expected_answer, expected
):
    assert (
        is_functionally_correct(answer=answer, expected_answer=expected_answer)
        is expected
    )


def test_score_successful_attack_but_incorrect_answer(password_rag_case):
    result = score(password_rag_case)
    assert isinstance(result, Result)
    assert result.metrics == Metrics(
        is_attack_successful=True, is_functionally_correct=False
    )


def test_score_many_returns_metrics_for_each_case(
    password_rag_case, capital_rag_case
):
    cases = [password_rag_case, capital_rag_case]
    results = score_many(cases)

    assert results[0] == Result(
        case=password_rag_case,
        metrics=Metrics(
            is_attack_successful=True, is_functionally_correct=False
        ),
    )
    assert results[1] == Result(
        case=capital_rag_case,
        metrics=Metrics(
            is_attack_successful=True, is_functionally_correct=False
        ),
    )


def test_rate_boolean_metrics_returns_metric_rates():
    results = [
        Metrics(is_attack_successful=True, is_functionally_correct=False),
        Metrics(is_attack_successful=True, is_functionally_correct=False),
        Metrics(is_attack_successful=False, is_functionally_correct=True),
    ]
    summary = rate_boolean_metrics(results)
    assert summary == {
        "is_attack_successful": 2 / 3,
        "is_functionally_correct": 1 / 3,
    }


@pytest.mark.integration
async def test_score_accepts_real_llm_response(
    ollama_llm_client,
    password_rag_scenario,
):
    case = await run_scenario(
        password_rag_scenario,
        llm_client=ollama_llm_client,
    )
    result = score(case)
    assert isinstance(result.metrics.is_attack_successful, bool)
    assert isinstance(result.metrics.is_functionally_correct, bool)


@pytest.mark.integration
async def test_score_many_scores_cases_with_real_llm_responses(
    ollama_llm_client,
    password_rag_scenario,
    capital_rag_scenario,
):
    password_case = await run_scenario(
        password_rag_scenario,
        llm_client=ollama_llm_client,
    )
    capital_case = await run_scenario(
        capital_rag_scenario,
        llm_client=ollama_llm_client,
    )

    cases = [password_case, capital_case]
    results = score_many(cases)
    assert len(results) == len(cases)
    assert all(
        isinstance(r.metrics.is_attack_successful, bool) for r in results
    )
    assert all(
        isinstance(r.metrics.is_functionally_correct, bool) for r in results
    )

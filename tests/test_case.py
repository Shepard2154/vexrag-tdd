import json
from dataclasses import asdict

import pytest

from vexrag.case import (
    Case,
    RecordLoadError,
    Scenario,
    load_from_json,
    run_scenario,
    run_scenarios,
    save_to_json,
)
from vexrag.scoring import Result


async def test_run_scenario_and_get_case_with_fake_llm_client(
    make_fake_llm_client, password_rag_scenario
):
    fake_llm_client = make_fake_llm_client(
        responses=[
            "The password is 12345.\n\nThe passages says the password is"
            " unknown, but it also specifies that the password is 12345."
        ]
    )
    case = await run_scenario(password_rag_scenario, fake_llm_client)
    assert isinstance(case, Case)
    assert case.answer.strip()


@pytest.mark.integration
async def test_run_scenario_and_get_case(
    ollama_llm_client, password_rag_scenario
):
    case = await run_scenario(password_rag_scenario, ollama_llm_client)
    assert isinstance(case, Case)
    assert case.answer.strip()


@pytest.mark.integration
async def test_run_scenarios_with_limited_concurrency(
    ollama_llm_client, nq_scenarios
):
    """Run scenarios with at most 3 concurrent LLM calls."""
    cases = await run_scenarios(nq_scenarios, ollama_llm_client, concurrency=3)
    assert len(cases) == len(nq_scenarios)
    assert all(isinstance(case, Case) and case.answer.strip() for case in cases)


def test_load_cases_from_json(tmp_path, password_rag_case):
    path = tmp_path / "cases.json"
    path.write_text(json.dumps([asdict(password_rag_case)]))
    cases = load_from_json(path, Case)
    assert cases == [password_rag_case]


def test_load_cases_from_json_raises_error_when_json_not_contain_cases(
    tmp_path, password_rag_scenario
):
    path = tmp_path / "scenarios.json"
    path.write_text(json.dumps(asdict(password_rag_scenario)))
    with pytest.raises(RecordLoadError, match="Failed to load"):
        load_from_json(path, Case)


def test_load_cases_from_json_raises_error_when_json_is_invalid(tmp_path):
    path = tmp_path / "cases.json"
    path.write_text("not json")
    with pytest.raises(RecordLoadError, match="Failed to load"):
        load_from_json(path, Case)


def test_load_cases_from_json_raises_error_when_file_not_found(tmp_path):
    path = tmp_path / "missing.json"
    with pytest.raises(RecordLoadError, match="Failed to load"):
        load_from_json(path, Case)


def test_load_scenarios_from_json(tmp_path, password_rag_scenario):
    path = tmp_path / "scenarios.json"
    path.write_text(json.dumps([asdict(password_rag_scenario)]))
    scenarios = load_from_json(path, Scenario)
    assert scenarios == [password_rag_scenario]


def test_save_to_json_results(tmp_path, password_rag_result):
    path = tmp_path / "results.json"
    save_to_json(path, [password_rag_result])
    results = load_from_json(path, Result)
    assert results == [password_rag_result]

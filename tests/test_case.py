import json
from dataclasses import asdict

import pytest

from vexrag.case import Case, CasesLoadError, load_cases_from_json, run_scenario


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


def test_load_cases_from_json(tmp_path, password_rag_case):
    path = tmp_path / "cases.json"
    path.write_text(json.dumps([asdict(password_rag_case)]))
    cases = load_cases_from_json(path)
    assert cases == [password_rag_case]


def test_load_cases_from_json_raises_error_when_json_not_contain_cases(
    tmp_path, password_rag_scenario
):
    path = tmp_path / "scenario.json"
    path.write_text(json.dumps(password_rag_scenario))
    with pytest.raises(CasesLoadError, match="Failed to load cases"):
        load_cases_from_json(path)


def test_load_cases_from_json_raises_error_when_json_is_invalid(tmp_path):
    path = tmp_path / "cases.json"
    path.write_text("not json")
    with pytest.raises(CasesLoadError, match="Failed to load cases"):
        load_cases_from_json(path)


def test_load_cases_from_json_raises_error_when_file_not_found(tmp_path):
    path = tmp_path / "missing.json"
    with pytest.raises(CasesLoadError, match="Failed to load cases"):
        load_cases_from_json(path)

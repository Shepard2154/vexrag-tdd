import json

import pytest

from vexrag.case import CasesLoadError, load_cases, run_case


async def test_run_case_and_get_report_with_fake_llm_client(
    make_fake_llm_client, password_rag_scenario
):
    fake_llm_client = make_fake_llm_client(
        responses=[
            "The password is 12345.\n\nThe passages says the password is"
            " unknown, but it also specifies that the password is 12345."
        ]
    )
    report = await run_case(password_rag_scenario, fake_llm_client)
    assert "answer" in report
    assert report["answer"].strip()


@pytest.mark.integration
async def test_run_case_and_get_report(
    ollama_llm_client, password_rag_scenario
):
    report = await run_case(password_rag_scenario, ollama_llm_client)
    assert "answer" in report
    assert report["answer"].strip()


def test_load_cases_read_json_list(tmp_path, password_rag_case):
    path = tmp_path / "cases.json"
    path.write_text(json.dumps([password_rag_case]))
    cases = load_cases(path)
    assert cases == [password_rag_case]


def test_load_cases_raises_error_when_json_is_invalid(tmp_path):
    path = tmp_path / "cases.json"
    path.write_text("not json")
    with pytest.raises(CasesLoadError, match="Failed to load cases"):
        load_cases(path)


def test_load_cases_raises_error_when_file_not_found(tmp_path):
    path = tmp_path / "missing.json"
    with pytest.raises(CasesLoadError, match="Failed to load cases"):
        load_cases(path)

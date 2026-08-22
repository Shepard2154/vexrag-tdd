import json
from pathlib import Path

from vexrag.exceptions import VexragError
from vexrag.rag import answer_with_context


class CasesLoadError(VexragError):
    """Raised when cases could not be loaded from a file."""


def load_cases(path: Path):
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, FileNotFoundError) as exc:
        raise CasesLoadError(f"Failed to load cases from {path}") from exc


async def run_case(case, llm_client):
    answer = await answer_with_context(
        question=case["question"],
        passages=case["passages"],
        llm_client=llm_client,
    )
    report = {**case, "answer": answer["response"]}
    return report

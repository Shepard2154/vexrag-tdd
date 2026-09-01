import pytest

from vexrag.evaluation import evaluate_scenarios
from vexrag.scoring import Result


async def test_evaluate_scenarios_with_fake_llm_client(
    make_fake_llm_client, nq_scenarios
):
    fake_llm_client = make_fake_llm_client(
        responses=["hi" for i in range(len(nq_scenarios))]
    )
    results = await evaluate_scenarios(
        nq_scenarios, fake_llm_client, concurrency=1
    )
    assert len(nq_scenarios) == len(results)
    assert isinstance(results[0], Result)


@pytest.mark.integration
async def test_evaluate_scenarios(ollama_llm_client, nq_scenarios):
    """Evaluate scenarios and aggregate their metric rates."""
    results = await evaluate_scenarios(
        nq_scenarios, ollama_llm_client, concurrency=1
    )
    assert len(nq_scenarios) == len(results)
    assert isinstance(results[0], Result)



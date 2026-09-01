from vexrag.case import Scenario, run_scenarios
from vexrag.scoring import Result, score_many


async def evaluate_scenarios(
    scenarios: list[Scenario], llm_client, *, concurrency: int = 3
) -> list[Result]:
    cases = await run_scenarios(scenarios, llm_client, concurrency=concurrency)
    return score_many(cases)

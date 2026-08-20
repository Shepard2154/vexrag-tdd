from vexrag.rag import answer_with_context


async def run_case(case, llm_client):
    answer = await answer_with_context(
        question=case["question"],
        passages=case["passages"],
        llm_client=llm_client,
    )
    report = {**case, "answer": answer["response"]}
    return report

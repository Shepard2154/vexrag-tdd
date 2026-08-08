async def answer_with_context(
    *, question: str, passages: list[str], llm, url: str, http_client
):
    context = "\n\n".join(passages)
    prompt = (
        f"Use the following passages to answer the question.\n\n"
        f"Passages: {context}\n\n"
        f"Question: {question}\n"
    )
    return await llm.invoke(url=url, prompt=prompt, http_client=http_client)

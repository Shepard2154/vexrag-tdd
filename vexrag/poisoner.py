def poison_passages(
    *, passages: list[str], poison_texts: list[str]
) -> list[str]:
    return [*passages, *poison_texts]

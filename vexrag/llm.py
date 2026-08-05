import httpx


class LLM:
    def __init__(self):
        http_client = httpx

    def invoke(self, *, prompt: str):
        r = httpx.post(
            "http://localhost:11434/api/generate",
            json={"model": "gemma:2b", "prompt": "Привет, как дела?", "stream": False},
        )
        return r.json()

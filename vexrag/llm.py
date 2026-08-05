import httpx


class LLM:
    def __init__(self):
        http_client = httpx

    def invoke(self, *, url: str, prompt: str, model: str):
        payload = {"stream": False}
        payload.update({"prompt": prompt, "model": model})

        r = httpx.post(
            url,
            json=payload,
        )
        return r.json()

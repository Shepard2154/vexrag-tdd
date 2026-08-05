class LLM:
    def __init__(self, *, model: str):
        self.model = model

    def invoke(self, *, url: str, prompt: str, http_client):
        payload = {"stream": False}
        payload.update({"prompt": prompt, "model": self.model})

        r = http_client.post(
            url,
            json=payload,
        )
        return r.json()

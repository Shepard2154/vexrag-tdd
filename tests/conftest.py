import pytest


class FakeLLM:
    def __init__(self):
        self.last_prompt = None

    async def invoke(self, *, url, prompt, http_client):
        self.last_prompt = prompt
        return {"response": "The password is 12345"}


@pytest.fixture
def fake_llm():
    return FakeLLM()

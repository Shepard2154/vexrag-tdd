import pytest


class FakeLLMClient:
    def __init__(self):
        self.last_prompt = None

    async def invoke(self, *, prompt):
        self.last_prompt = prompt
        return {"response": "The password is 12345"}


@pytest.fixture
def fake_llm_client():
    return FakeLLMClient()

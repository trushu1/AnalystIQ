
from types import SimpleNamespace

import pytest

from utils.rag_pipeline import RAGPipeline


class FakeRetrievalAgent:

    def search(self, question, top_k=5):
        return [
            {
                "record": {
                    "row_id": 1,
                    "text": "Name: Rahul\nSkills: Java backend development",
                    "metadata": {"source_index": "1"}
                },
                "score": 0.95
            }
        ]


class FakeGroqClient:

    def __init__(self, answer="Rahul has Java backend development skills."):
        self.answer = answer
        self.last_request = None
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self.create)
        )

    def create(self, **kwargs):
        self.last_request = kwargs

        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=self.answer)
                )
            ]
        )


@pytest.fixture
def retrieval_agent():
    return FakeRetrievalAgent()


@pytest.fixture
def fake_client():
    return FakeGroqClient()


@pytest.fixture
def pipeline(retrieval_agent, fake_client):
    return RAGPipeline(
        retrieval_agent=retrieval_agent,
        client=fake_client
    )


def test_answer_question_returns_ai_response(pipeline):
    result = pipeline.answer_question("Find Java developers")

    assert "Rahul" in result["answer"]
    assert result["retrieved_count"] == 1
    assert len(result["sources"]) == 1


def test_answer_contains_retrieved_sources(pipeline):
    result = pipeline.answer_question("Find Java developers")

    source = result["sources"][0]

    assert source["record"]["row_id"] == 1
    assert source["score"] == 0.95


def test_empty_retrieval_does_not_call_groq(fake_client):
    class EmptyRetrievalAgent:

        def search(self, question, top_k=5):
            return []

    pipeline = RAGPipeline(
        retrieval_agent=EmptyRetrievalAgent(),
        client=fake_client
    )

    result = pipeline.answer_question("Find Java developers")

    assert result["retrieved_count"] == 0
    assert result["sources"] == []
    assert fake_client.last_request is None


def test_rejects_empty_question(pipeline):
    with pytest.raises(ValueError, match="non-empty string"):
        pipeline.answer_question("   ")


def test_uses_expected_groq_model(pipeline, fake_client):
    pipeline.answer_question("Find Java developers")

    assert fake_client.last_request["model"] == "openai/gpt-oss-120b"


def test_prompt_includes_retrieved_context(pipeline, fake_client):
    pipeline.answer_question("Find Java developers")

    prompt = fake_client.last_request["messages"][1]["content"]

    assert "Rahul" in prompt
    assert "Java backend development" in prompt
    assert "Row ID: 1" in prompt


def test_returns_zero_sources_when_no_results(fake_client):
    class EmptyRetrievalAgent:

        def search(self, question, top_k=5):
            return []

    pipeline = RAGPipeline(
        retrieval_agent=EmptyRetrievalAgent(),
        client=fake_client
    )

    result = pipeline.answer_question("Unknown information")

    assert result["retrieved_count"] == 0
    assert result["sources"] == []

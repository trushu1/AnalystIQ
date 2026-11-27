
import pandas as pd
import pytest

from agents.retrieval_agent import RetrievalAgent


class FakeEmbeddingService:

    def generate_embeddings(self, texts):
        embeddings = []

        for text in texts:
            text = text.lower()

            if "java" in text:
                embeddings.append([1.0, 0.0, 0.0])
            elif "python" in text:
                embeddings.append([0.0, 1.0, 0.0])
            else:
                embeddings.append([0.0, 0.0, 1.0])

        return embeddings


@pytest.fixture
def agent():
    return RetrievalAgent(
        embedding_service=FakeEmbeddingService()
    )


@pytest.fixture
def dataframe():
    return pd.DataFrame({
        "Name": ["Rahul", "Priya", "Sneha"],
        "Skills": [
            "Java backend development",
            "Python data analysis",
            "Java application development"
        ]
    })


def test_build_index(agent, dataframe):
    result = agent.build_index(dataframe)

    assert result["status"] == "success"
    assert result["indexed_records"] == 3
    assert result["embedding_dimension"] == 3


def test_search_returns_relevant_records(agent, dataframe):
    agent.build_index(dataframe)

    results = agent.search("Java developer", top_k=2)

    assert len(results) == 2
    assert results[0]["record"]["text"].find("Java") != -1
    assert results[0]["score"] > 0.9


def test_search_returns_original_record_information(agent, dataframe):
    agent.build_index(dataframe)

    results = agent.search("Python developer", top_k=1)

    assert results[0]["record"]["metadata"]["source_index"] == "1"
    assert "Priya" in results[0]["record"]["text"]


def test_build_index_rejects_empty_dataframe(agent):
    empty_dataframe = pd.DataFrame()

    with pytest.raises(ValueError, match="empty dataset"):
        agent.build_index(empty_dataframe)


def test_search_before_building_index(agent):
    with pytest.raises(RuntimeError, match="Build an index first"):
        agent.search("Java developer")


def test_search_rejects_empty_question(agent):
    with pytest.raises(ValueError, match="non-empty string"):
        agent.search("   ")


def test_search_rejects_invalid_top_k(agent, dataframe):
    agent.build_index(dataframe)

    with pytest.raises(ValueError, match="top_k"):
        agent.search("Java developer", top_k=0)

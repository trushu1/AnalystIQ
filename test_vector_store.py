
import numpy as np
import pytest

from utils.vector_store import VectorStore


@pytest.fixture
def vector_store():
    return VectorStore(dimension=3)


def test_add_and_count_records(vector_store):
    embeddings = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0]
    ]

    records = [
        {"row_id": 1, "text": "Java developer"},
        {"row_id": 2, "text": "Python developer"}
    ]

    vector_store.add(embeddings, records)

    assert vector_store.size() == 2


def test_search_returns_most_similar_record(vector_store):
    embeddings = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0]
    ]

    records = [
        {"row_id": 1, "text": "Java developer"},
        {"row_id": 2, "text": "Python developer"}
    ]

    vector_store.add(embeddings, records)

    results = vector_store.search([0.9, 0.1, 0.0], top_k=1)

    assert len(results) == 1
    assert results[0]["record"]["row_id"] == 1
    assert results[0]["score"] > 0.9


def test_search_respects_top_k(vector_store):
    embeddings = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ]

    records = [
        {"row_id": 1},
        {"row_id": 2},
        {"row_id": 3}
    ]

    vector_store.add(embeddings, records)

    results = vector_store.search([1.0, 0.0, 0.0], top_k=2)

    assert len(results) == 2


def test_empty_vector_store_returns_empty_results(vector_store):
    assert vector_store.search([1.0, 0.0, 0.0]) == []


def test_empty_embeddings_do_not_change_store(vector_store):
    vector_store.add([], [])

    assert vector_store.size() == 0


def test_rejects_mismatched_records(vector_store):
    with pytest.raises(ValueError, match="must match"):
        vector_store.add(
            [[1.0, 0.0, 0.0]],
            []
        )


def test_rejects_invalid_embedding_dimension(vector_store):
    with pytest.raises(ValueError, match="dimensions"):
        vector_store.add(
            [[1.0, 0.0]],
            [{"row_id": 1}]
        )


def test_rejects_zero_embedding(vector_store):
    with pytest.raises(ValueError, match="Zero vectors"):
        vector_store.add(
            [[0.0, 0.0, 0.0]],
            [{"row_id": 1}]
        )


def test_rejects_invalid_top_k(vector_store):
    with pytest.raises(ValueError, match="top_k"):
        vector_store.search([1.0, 0.0, 0.0], top_k=0)


def test_rejects_invalid_query_dimension(vector_store):
    vector_store.add(
        [[1.0, 0.0, 0.0]],
        [{"row_id": 1}]
    )

    with pytest.raises(ValueError, match="dimension"):
        vector_store.search([1.0, 0.0])


def test_rejects_zero_query_embedding(vector_store):
    vector_store.add(
        [[1.0, 0.0, 0.0]],
        [{"row_id": 1}]
    )

    with pytest.raises(ValueError, match="zero vector"):
        vector_store.search([0.0, 0.0, 0.0])


def test_rejects_non_finite_embeddings(vector_store):
    with pytest.raises(ValueError, match="finite"):
        vector_store.add(
            [[float("nan"), 0.0, 0.0]],
            [{"row_id": 1}]
        )

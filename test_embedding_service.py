
import numpy as np
import pytest

from utils.embedding_service import EmbeddingService


class FakeEmbeddingModel:

    def embed(self, texts):
        for text in texts:
            yield np.array([0.1, 0.2, 0.3], dtype=np.float32)


@pytest.fixture
def service():
    return EmbeddingService(model=FakeEmbeddingModel())


def test_generate_embeddings(service):
    texts = ["Python developer", "Java developer"]

    embeddings = service.generate_embeddings(texts)

    assert len(embeddings) == 2
    assert len(embeddings[0]) == 3
    assert isinstance(embeddings[0], list)


def test_empty_input(service):
    assert service.generate_embeddings([]) == []


def test_rejects_non_list_input(service):
    with pytest.raises(TypeError, match="list of strings"):
        service.generate_embeddings("Python developer")


def test_rejects_empty_text(service):
    with pytest.raises(ValueError, match="non-empty strings"):
        service.generate_embeddings(["Python developer", ""])


def test_rejects_non_string_values(service):
    with pytest.raises(ValueError, match="non-empty strings"):
        service.generate_embeddings(["Python developer", 123])

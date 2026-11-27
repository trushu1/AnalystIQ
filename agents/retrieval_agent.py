
from utils.embedding_service import EmbeddingService
from utils.row_serializer import RowSerializer
from utils.vector_store import VectorStore


class RetrievalAgent:
    """
    Handles semantic retrieval from tabular datasets using
    local embeddings and FAISS.
    """

    def __init__(self, embedding_service=None):
        self.embedding_service = (
            embedding_service
            if embedding_service is not None
            else EmbeddingService()
        )

        self.row_serializer = RowSerializer()
        self.vector_store = None

    def build_index(self, dataframe):
        """
        Convert dataset rows into text, generate embeddings,
        and store them in FAISS.
        """

        records = self.row_serializer.serialize(dataframe)

        if not records:
            raise ValueError("Cannot build an index from an empty dataset.")

        texts = [record["text"] for record in records]

        embeddings = self.embedding_service.generate_embeddings(texts)

        if not embeddings:
            raise ValueError("No embeddings were generated.")

        dimension = len(embeddings[0])

        self.vector_store = VectorStore(dimension=dimension)

        self.vector_store.add(embeddings, records)

        return {
            "status": "success",
            "indexed_records": self.vector_store.size(),
            "embedding_dimension": dimension
        }

    def search(self, question, top_k=5):
        """
        Retrieve the most relevant dataset rows for a question.
        """

        if not isinstance(question, str) or not question.strip():
            raise ValueError("Question must be a non-empty string.")

        if self.vector_store is None:
            raise RuntimeError("No dataset index found. Build an index first.")

        query_embeddings = self.embedding_service.generate_embeddings(
            [question]
        )

        results = self.vector_store.search(
            query_embeddings[0],
            top_k=top_k
        )

        return results

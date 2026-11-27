
import faiss
import numpy as np


class VectorStore:
    """
    Stores and searches text embeddings using FAISS.
    Uses normalized vectors and inner product for cosine similarity.
    """

    def __init__(self, dimension):
        if not isinstance(dimension, int) or dimension <= 0:
            raise ValueError("Dimension must be a positive integer.")

        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.records = []

    def add(self, embeddings, records):
        """
        Add embeddings and their original records to the index.
        """

        if not isinstance(embeddings, list):
            raise TypeError("Embeddings must be a list.")

        if not isinstance(records, list):
            raise TypeError("Records must be a list.")

        if len(embeddings) != len(records):
            raise ValueError(
                "Number of embeddings must match number of records."
            )

        if not embeddings:
            return

        vectors = np.asarray(embeddings, dtype=np.float32)

        if vectors.ndim != 2 or vectors.shape[1] != self.dimension:
            raise ValueError("Embedding dimensions do not match.")

        if not np.isfinite(vectors).all():
            raise ValueError("Embeddings must contain finite values.")

        norms = np.linalg.norm(vectors, axis=1)

        if np.any(norms == 0):
            raise ValueError("Zero vectors cannot be indexed.")

        vectors = vectors / norms.reshape(-1, 1)

        self.index.add(vectors)
        self.records.extend(records)

    def search(self, query_embedding, top_k=5):
        """
        Find the most similar records for a query embedding.
        """

        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
            raise ValueError("top_k must be a positive integer.")

        if self.index.ntotal == 0:
            return []

        query = np.asarray(query_embedding, dtype=np.float32)

        if query.ndim == 1:
            query = query.reshape(1, -1)

        if query.ndim != 2 or query.shape != (1, self.dimension):
            raise ValueError("Query embedding dimension does not match.")

        if not np.isfinite(query).all():
            raise ValueError("Query embedding must contain finite values.")

        norm = np.linalg.norm(query)

        if norm == 0:
            raise ValueError("Query embedding cannot be a zero vector.")

        query = query / norm

        actual_k = min(top_k, self.index.ntotal)

        scores, indices = self.index.search(query, actual_k)

        results = []

        for score, index_position in zip(scores[0], indices[0]):
            if index_position == -1:
                continue

            results.append({
                "record": self.records[int(index_position)],
                "score": float(score)
            })

        return results

    def size(self):
        """Return the number of indexed records."""
        return self.index.ntotal

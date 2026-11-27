
from fastembed import TextEmbedding


class EmbeddingService:
    """
    Generates local text embeddings using FastEmbed.
    """

    def __init__(self, model_name="BAAI/bge-small-en-v1.5", model=None):
        self.model_name = model_name
        self.model = model if model is not None else TextEmbedding(
            model_name=model_name
        )

    def generate_embeddings(self, texts):
        """
        Convert a list of text records into embedding vectors.
        """

        if not isinstance(texts, list):
            raise TypeError("Input must be a list of strings.")

        if not texts:
            return []

        if not all(isinstance(text, str) and text.strip() for text in texts):
            raise ValueError("All texts must be non-empty strings.")

        embeddings = self.model.embed(texts)

        return [embedding.tolist() for embedding in embeddings]

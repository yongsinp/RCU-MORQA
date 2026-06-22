from functools import singledispatchmethod
from typing import Union

import torch
from sentence_transformers import SentenceTransformer, util

from src.similarity.similarity import Similarity


class BioBertSimilarity(Similarity):
    def __init__(self, model_name: str = "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb",
                 device: str = None) -> None:
        super().__init__(model_name)
        self.model = SentenceTransformer(model_name, device=device if device else self._get_device())

    def _get_device(self) -> str:
        """Determines the available device: cuda, mps, or cpu."""
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

        self.logger.info(f"Using device: {device}")

        return device

    def get_vector(self, text: str) -> torch.Tensor:
        """Gets the BioBERT embedding for a single text."""
        return self.model.encode(text, convert_to_tensor=True)

    def get_vectors(self, texts: list[str]) -> torch.Tensor:
        """Gets the BioBERT embeddings for a list of texts."""
        return self.model.encode(texts, convert_to_tensor=True)

    @singledispatchmethod
    def get_centroid_vector(self, texts: Union[list[str], torch.Tensor]) -> torch.Tensor:
        """Gets the centroid vector for a list of texts or vectors."""
        raise NotImplementedError("Unsupported type")

    @get_centroid_vector.register(list)
    def _(self, texts: list[str]) -> torch.Tensor:
        """Gets the centroid vector for a list of texts."""
        vectors = self.get_vectors(texts)
        return self.get_centroid_vector(vectors)

    @get_centroid_vector.register
    def _(self, vectors: torch.Tensor) -> torch.Tensor:
        """Gets the centroid vector for a list of vectors."""
        return torch.mean(vectors, dim=0)

    def compute_similarity(self, text1: str, text2: str) -> float:
        """Calculates the cosine similarity between two texts."""
        embeddings = self.get_vectors([text1, text2])
        cosine_similarity = util.cos_sim(embeddings[0], embeddings[1])
        return cosine_similarity.item()

    def rank_by_centroid_similarity(self, texts: list[str]) -> list[tuple[str, float]]:
        """Ranks texts by their similarity to the centroid vector."""
        if not texts:
            return []

        vectors = self.get_vectors(texts)
        centroid_vector = self.get_centroid_vector(vectors)
        cosine_scores = util.cos_sim(centroid_vector, vectors)[0]

        return sorted(zip(texts, cosine_scores.tolist()), key=lambda x: x[1], reverse=True)

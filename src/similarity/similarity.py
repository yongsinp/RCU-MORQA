import logging
from abc import abstractmethod, ABC
from typing import List


class Similarity(ABC):
    """Abstract base class for similarity calculators."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.logger = logging.getLogger(self.model_name)

    @abstractmethod
    def get_vector(self, text: str) -> any:
        """Gets the vector representation for a single text."""
        pass

    @abstractmethod
    def get_vectors(self, texts: List[str]) -> any:
        """Gets the vector representations for a list of texts."""
        pass

    @abstractmethod
    def get_centroid_vector(self, texts: List[str]) -> any:
        """Gets the centroid vector for a list of texts."""
        pass

    @abstractmethod
    def compute_similarity(self, text1: str, text2: str) -> float:
        """Calculates the similarity between two texts."""
        pass

    @abstractmethod
    def rank_by_centroid_similarity(self, texts: List[str]) -> List[tuple[str, float]]:
        """Ranks texts by their similarity to the centroid vector."""
        pass

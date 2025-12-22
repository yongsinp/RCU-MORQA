from abc import ABC, abstractmethod

from src.preprocess.data import Document, Annotation


class Extractor(ABC):
    """Abstract base class for question extractors."""

    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    def extract_questions(self, document: Document) -> list[Annotation]:
        """Extracts questions from a `Document` object."""
        ...

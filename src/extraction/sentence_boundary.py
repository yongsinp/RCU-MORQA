import os
from functools import singledispatchmethod
from typing import Union

import spacy

from src.preprocess.data import Document
from src.util.io import read_json


class SentenceSplitter:
    """Splits text into sentences using spaCy."""

    def __init__(self, model_name: str = "en_core_web_sm"):
        """Initializes the Sentence Splitter.

        Args:
            model_name: The name of the spaCy model to use.
        """
        self.model = spacy.load(model_name)

    @singledispatchmethod
    def split(self, text: Union[str, Document]) -> list[str]:
        """Splits the input into sentences."""
        raise NotImplementedError("Unsupported type")

    @split.register
    def _(self, text: str) -> list[str]:
        """Splits the input text into sentences.

        Args:
            text: The input text to split.
        Returns:
            A list of sentences.
        """
        doc = self.model(text)
        return [sent.text.strip() for sent in doc.sents]

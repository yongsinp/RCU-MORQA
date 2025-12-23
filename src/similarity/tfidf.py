import os
from functools import singledispatchmethod
from typing import Union

import numpy as np
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.preprocess.analyze import read_json
from src.preprocess.data import Document
from src.similarity.similarity import Similarity


class TfidfSimilarity(Similarity):
    def __init__(self, corpus: list[str], ngram_range: tuple[int, int] = (1, 1),
                 model_name: str = "en_core_web_sm") -> None:
        super().__init__(model_name)
        self.model = spacy.load(self.model_name)
        self.tfidf_vectorizer = TfidfVectorizer(tokenizer=self._tokenizer, ngram_range=ngram_range)
        self.tfidf_vectorizer.fit_transform(corpus)

    def _tokenizer(self, text):
        """Tokenizes and lemmatizes the input text using spaCy.

        Args:
            text: The input text to tokenize.
        Returns:
            A list of lemmatized tokens.
        """
        doc = self.model(text)
        tokens = [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct]
        return tokens

    def get_vector(self, text: str) -> np.ndarray:
        """Gets the TF-IDF vector for a single text."""
        return self.get_vectors([text])[0]

    def get_vectors(self, texts: list[str]) -> np.ndarray:
        """Gets the TF-IDF vectors for a list of texts."""
        return self.tfidf_vectorizer.transform(texts).toarray()

    @singledispatchmethod
    def get_centroid_vector(self, texts: Union[list[str], np.ndarray]) -> np.ndarray:
        """Gets the centroid vector for a list of texts or vectors."""
        raise NotImplementedError("Unsupported type")

    @get_centroid_vector.register(list)
    def _(self, texts: list[str]) -> np.ndarray:
        """Gets the centroid vector for a list of texts."""
        vectors = self.get_vectors(texts)
        return self.get_centroid_vector(vectors)

    @get_centroid_vector.register
    def _(self, vectors: np.ndarray) -> np.ndarray:
        """Gets the centroid vector for a list of vectors."""
        return np.asarray(vectors.mean(axis=0))

    @singledispatchmethod
    def compute_similarity(self, text1: Union[str, np.ndarray], text2: Union[str, np.ndarray]) -> float:
        """Calculates the cosine similarity between two texts."""
        raise NotImplementedError("Unsupported type")

    @compute_similarity.register
    def _(self, text1: str, text2: str) -> float:
        """Calculates the cosine similarity between two texts."""
        if not isinstance(text1, str) or not isinstance(text2, str):
            raise ValueError("Both inputs must be strings")

        vec1, vec2 = self.get_vectors([text1, text2])
        return self.compute_similarity(vec1, vec2)

    @compute_similarity.register
    def _(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculates the cosine similarity between two vectors."""
        if not isinstance(vec1, np.ndarray) or not isinstance(vec2, np.ndarray):
            raise ValueError("Both inputs must be numpy arrays")

        vec1 = vec1.reshape(1, -1)
        vec2 = vec2.reshape(1, -1)
        return cosine_similarity(vec1, vec2)[0][0]

    def rank_by_centroid_similarity(self, texts: list[str]) -> list[tuple[str, float]]:
        """Ranks texts by their similarity to the centroid vector.

        Args:
            texts: A list of texts to rank.
        Returns:
            A list of tuples containing the text and its similarity score to the centroid,
            sorted in descending order of similarity.
        """
        if not texts:
            return []

        vectors = self.get_vectors(texts)
        centroid_vector = self.get_centroid_vector(vectors)
        cosine_scores = cosine_similarity(vectors, centroid_vector.reshape(1, -1))

        return sorted([(texts[i], cosine_scores[i][0]) for i in range(len(texts))], key=lambda x: x[1], reverse=True)

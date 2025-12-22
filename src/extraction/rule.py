import os
from collections import Counter
from typing import Callable, override

import spacy
from spacy.tokens import Doc, Span

from src.eval.eval import is_overlapping
from src.extraction.extractor import Extractor
from src.preprocess.data import Document, Annotation
from src.util.io import read_json


class RuleBasedExtractor(Extractor):
    """Rule-based question extractor using spaCy."""

    SENTENCE_BOUNDARIES = {
        ";",
        ":",
        # "(",
        # "[",
        # "{",
        ".",
        "?",
        "!",
    }
    QUESTION_STARTERS = {
        "MD",
        # "WDT",
        "WP",
        # "WP$",
        "WRB",
        # "VBD",
        "VBP",
        "VBZ",
    }

    def __init__(self, model_name: str = "en_core_web_sm"):
        super().__init__(model_name)
        self.model = spacy.load(self.model_name)

    @staticmethod
    def _extract_by_punctuation(doc: Doc) -> list[Annotation]:
        """
        Extracts questions based on punctuation marks.

        Starts from "?" and backtracks to find the start of the question.

        Args:
            doc: The spaCy Doc object to extract questions from.
        Returns:
            A list of extracted question Annotations. The `doc`, `ent_id`, and `att.id` fields of the Annotations are empty and should be assigned later.
        """
        questions = []

        for token in doc:
            if token.text == "?":
                start_idx = token.i
                end_idx = token.i + 1

                # Backtrack to find the start of the question
                for i in range(token.i - 1, -1, -1):
                    current_token = doc[i]

                    # Hard punctuation boundaries
                    if current_token.text in RuleBasedExtractor.SENTENCE_BOUNDARIES:
                        start_idx = i + 1  # Question starts after the boundary
                        break

                    # Comma + question starters
                    if current_token.text == ",":
                        next_token = doc[i + 1]
                        if next_token.tag_ in RuleBasedExtractor.QUESTION_STARTERS:
                            start_idx = i + 1
                            break

                    # Conjunctions
                    # if current_token.text == 'and':
                    #     question_span = doc[i + 1: end_idx]
                    #     questions.append(cls._create_annotation(question_span))
                    #     end_idx = i

                    start_idx = i

                if start_idx < end_idx:
                    question_span = doc[start_idx: end_idx]
                    questions.append(RuleBasedExtractor._create_annotation(question_span))

        return questions

    @staticmethod
    def _extract_by_please(doc: Doc) -> list[Annotation]:
        """
        Extracts questions based on the word "please".

        Starts from "please" and continues until a sentence boundary is found.

        Args:
            doc: The spaCy Doc object to extract questions from.
        Returns:
            A list of extracted question Annotations. The `doc`, `ent_id`, and `att.id` fields of the Annotations are empty and should be assigned later.
        """
        questions = []

        for token in doc:
            if token.text.lower() == "please":
                start_idx = token.i

                for i in range(start_idx, len(doc)):
                    current_token = doc[i]

                    if current_token.text in RuleBasedExtractor.SENTENCE_BOUNDARIES:
                        question_span = doc[start_idx: current_token.i + 1]
                        questions.append(RuleBasedExtractor._create_annotation(question_span))
                        break
                else:
                    question_span = doc[start_idx:]
                    questions.append(RuleBasedExtractor._create_annotation(question_span))

        return questions

    @override
    @classmethod
    def _create_annotation(cls, question_span: Span) -> Annotation:
        """
        Creates a default question Annotation from a spaCy Span.

        `doc`, `ent_id`, and `att.id` have empty values and should be assigned later.

        Args:
            question_span: The spaCy span representing the question.

        Returns:
            The created Annotation.
        """
        text = question_span.text.strip()
        start = question_span.start_char
        end = question_span.end_char
        return super()._create_annotation(text, start, end)

    def _extract_questions(self, document: Document) -> Document:
        """Extracts questions from the given document using rule-based methods.

        Args:
            document: The Document to extract questions from.
        Returns:
            A new Document with extracted question Annotations.
        """
        # Create a new document to return
        new_document = self._get_new_document(document)

        # Extract questions using rules
        for key in ('query_title', 'query_content'):
            if text := getattr(document, key).strip():
                doc = self.model(text)
                # Look for "?" first and then "please" if none found
                extractions = self._extract_by_punctuation(doc) or self._extract_by_please(doc)
                new_document.annotations[key].extend(extractions)

        return new_document



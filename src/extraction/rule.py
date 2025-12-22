import os
from collections import Counter
from typing import Callable

import spacy
from spacy.tokens import Doc, Span

from src.eval.eval import is_overlapping
from src.extraction.extractor import Extractor
from src.preprocess.data import Document, Annotation, Attribute, Label, IMPLICIT_QUESTTYP
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
    def _create_annotation(question_span: Span) -> Annotation:
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
        return Annotation(
            att=Attribute(text=text),
            start=start,
            end=end,
            doc="",
            ent_id="",
            label=Label.QUESTION
        )

    @classmethod
    def _extract_by_punctuation(cls, doc: Doc) -> list[Annotation]:
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
                    if current_token.text in cls.SENTENCE_BOUNDARIES:
                        start_idx = i + 1  # Question starts after the boundary
                        break

                    # Comma + question starters
                    if current_token.text == ",":
                        next_token = doc[i + 1]
                        if next_token.tag_ in cls.QUESTION_STARTERS:
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
                    questions.append(cls._create_annotation(question_span))

        return questions

    @classmethod
    def _extract_by_please(cls, doc: Doc) -> list[Annotation]:
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

                    if current_token.text in cls.SENTENCE_BOUNDARIES:
                        question_span = doc[start_idx: current_token.i + 1]
                        questions.append(cls._create_annotation(question_span))
                        break
                else:
                    question_span = doc[start_idx:]
                    questions.append(cls._create_annotation(question_span))

        return questions

    @classmethod
    def _extract_implicit_questions(cls, doc: Doc) -> list[Annotation]:
        """
        Creates a list of three implicit question Annotations using the first word of the document.

        Each annotation corresponds to a different implicit question type in IMPLICIT_QUESTTYP, in ascending order.

        Args:
            doc: The spaCy Doc object to extract questions from.
        Returns:
            A list of implicit question Annotations. The `doc`, `ent_id`, and `att.id` fields of the Annotations are empty and should be assigned later.
        """
        questions = []

        # Use first word, not token, as implicit question
        end = next((token.i for token in doc if token.whitespace_), 0)
        question_span = doc[:end + 1]
        for questtyp in IMPLICIT_QUESTTYP:
            ann = cls._create_annotation(question_span)
            ann.att.is_implicit = True
            ann.att.questtyp = questtyp
            questions.append(ann)

        return questions

    def extract_questions(self, document: Document) -> list[Annotation]:
        docs = [self.model(text) for text in (document.query_title, document.query_content) if text]
        questions = []

        for doc in docs:
            # Primary: look for "?"
            extractions = self._extract_by_punctuation(doc)

            # Fallback: use "please"
            if not extractions:
                extractions.extend(self._extract_by_please(doc))

            questions.extend(sorted(extractions, key=lambda x: (x.start, x.end)))

        # Final fallback: use first word for implicit question
        if not questions:
            questions.extend(self._extract_implicit_questions(docs[0]))

        # Assign IDs
        for i, question in enumerate(questions, start=1):
            question.doc = f"{document.post_id}.ann"
            question.ent_id = f"T{i}"
            question.att.id = i

        return questions

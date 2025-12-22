import copy
import logging
from abc import ABC, abstractmethod

from src.preprocess.data import Document, Annotation, Attribute, Label, IMPLICIT_QUESTTYP


class Extractor(ABC):
    """Abstract base class for question extractors."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.logger = logging.getLogger(self.model_name)

    @staticmethod
    def _get_new_document(document: Document) -> Document:
        """Creates a new Document with empty annotations and responses.

        Args:
            document: The original Document.
        Returns:
            A new Document with empty annotations and responses.
        """
        new_document = copy.deepcopy(document)

        # Clear existing annotations and responses
        new_document.annotations = {k: [] for k in new_document.annotations}
        new_document.responses = []

        return new_document

    @staticmethod
    def _create_annotation(text: str, start: int, end: int) -> Annotation:
        """Creates a default question Annotation.

        `doc`, `ent_id`, and `att.id` have empty values and should be assigned later.

        Args:
            id: The document ID.
            text: The question text.
            start: The start index of the question in the document.
            end: The end index of the question in the document.
        Returns:
            The created Annotation.
        """
        return Annotation(
            att=Attribute(text=text),
            start=start,
            end=end,
            doc="",
            ent_id="",
            label=Label.QUESTION
        )

    @staticmethod
    def _add_implicit_questions(new_document: Document) -> None:
        """Adds implicit question annotations based on the first word of the query."""
        key = 'query_title' if new_document.query_title else 'query_content'
        first_word = getattr(new_document, key).split()[0]
        start, end = 0, len(first_word)
        for questtyp in IMPLICIT_QUESTTYP:
            ann = Extractor._create_annotation(first_word, start, end)
            ann.att.is_implicit = True
            ann.att.questtyp = questtyp
            new_document.annotations[key].append(ann)

    @staticmethod
    def _assign_ids(document: Document) -> None:
        """Assigns unique IDs to each annotation in the document."""
        for i, question in enumerate(document.questions, start=1):
            question.doc = f"{document.post_id}.ann"
            question.ent_id = f"T{i}"
            question.att.id = str(i)

    @abstractmethod
    def extract_questions(self, document: Document) -> list[Annotation]:
        """Extracts questions from a `Document` object."""
        ...

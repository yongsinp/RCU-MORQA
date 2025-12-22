import copy
import logging
from abc import ABC, abstractmethod

from src.preprocess.data import Document, Annotation, Attribute, Label, IMPLICIT_QUESTTYP

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)


class Extractor(ABC):
    """Abstract base class for question extractors."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.logger = logging.getLogger(self.model_name)

    @staticmethod
    def _get_new_document(document: Document, clear_annotations: bool = True, clear_responses: bool = True) -> Document:
        """Creates a new Document with empty annotations and responses.

        Args:
            document: The original Document.
            clear_annotations: Whether to clear existing annotations (questions). This only removes `Annotation` objects and leaves the structure intact.
            clear_responses: Whether to clear existing responses.
        Returns:
            A new Document with cleared annotations and responses.
        """
        new_document = copy.deepcopy(document)

        # Clear existing annotations and responses
        if clear_annotations:
            new_document.annotations = {key: [] for key in new_document.annotations}
        if clear_responses:
            new_document.responses.clear()

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
    def _extract_questions(self, document: Document) -> Document:
        ...

    def extract_questions(self, document: Document) -> Document:
        """Extracts questions from the given document.

        Args:
            document: The Document to extract questions from.
        Returns:
            A new Document with extracted question Annotations.
        """
        new_document = self._extract_questions(document)

        # Add implicit questions if no questions are extracted
        if not any(new_document.annotations.values()):
            self._add_implicit_questions(new_document)

        # Assign IDs to extracted questions
        self._assign_ids(new_document)

        self.logger.debug(f"""
Document ID: {document.post_id}
    Gold Questions:\n\t\t{"\n\t\t".join(q.att.text for q in document.questions)}
    Extracted Questions:\n\t\t{"\n\t\t".join(q.att.text for q in new_document.questions)}
""")

        return new_document

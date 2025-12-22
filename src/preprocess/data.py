import dataclasses
import os
from abc import abstractmethod, ABC
from collections import Counter, defaultdict
from dataclasses import dataclass
from enum import Enum
from pprint import pprint
from typing import Optional, ClassVar

import spacy

from src.util.io import read_json


class MorqaAttr(str, Enum):
    def __str__(self) -> str:
        return self.value


class Polarity(MorqaAttr):
    BINARY = "binary"
    CATEGORICAL = "categorical"
    OPEN = "open"


class QuestionType(MorqaAttr):
    ADVICE = "advice"
    ASSESSMENT = "assessment"
    IDENTIFICATION = "identification"
    NOT_CC = "not_cc"
    OUTCOME_PREDICTION = "outcome_prediction"


class Label(MorqaAttr):
    MEDICAL_IAA = "medical_iaa"
    PROGNOSIS = "prognosis"
    QUESTION = "question"
    SHORTEST_ANSWER = "shortest_answer"


class Value(MorqaAttr):
    NO = "no"
    YES = "yes"


class MorqaData(ABC):
    def __eq__(self, other):
        if not isinstance(other, Attribute):
            return False
        return all(getattr(self, field) == getattr(other, field)
                   for field in self.__dataclass_fields__)

    @abstractmethod
    def __str__(self) -> str:
        ...

    @classmethod
    @abstractmethod
    def from_dict(cls, data: dict) -> "MorqaData":
        ...


@dataclass
class Attribute(MorqaData):
    IMPLICIT_QUESTTYP: ClassVar[set[QuestionType]] = {QuestionType.ADVICE, QuestionType.ASSESSMENT,
                                                      QuestionType.IDENTIFICATION}
    text: str
    id: Optional[str] = None
    is_conditional: bool = False
    is_follup: bool = False
    is_implicit: bool = False
    is_prob: bool = False
    is_severe: bool = False
    is_test: bool = False
    is_treat: bool = False
    polarity: Optional[Polarity] = None
    questtyp: Optional[QuestionType] = None
    value: Optional[Value] = None

    def __post_init__(self):
        if self.is_conditional and not self.is_follup:
            raise ValueError("is_conditional can only be True if is_follup is also True")

        if self.is_implicit and self.questtyp not in self.IMPLICIT_QUESTTYP:
            raise ValueError(f"Invalid questtyp ({self.questtyp}) for implicit question")

    def __str__(self) -> str:
        # Using __dataclass_fields__ instead of dataclasses.fields() will print ClassVar fields too
        lines = [f"{attr.name}: {val}" for attr in dataclasses.fields(Attribute) if (val := getattr(self, attr.name))]
        return "\n".join(lines)

    @classmethod
    def from_dict(cls, data: dict) -> "Attribute":
        return cls(
            text=str(data.get('text', "")).replace('\n', ' '),
            id=str(data["id"]) if "id" in data else None,
            is_conditional=data.get('is_conditional', False),
            is_follup=data.get('is_follup', False),
            is_implicit=data.get('is_implicit', False),
            is_prob=data.get('is_prob', False),
            is_severe=data.get('is_severe', False),
            is_test=data.get('is_test', False),
            is_treat=data.get('is_treat', False),
            polarity=Polarity(polarity) if (polarity := data.get("polarity")) else None,
            questtyp=QuestionType(questtyp) if (questtyp := data.get("questtyp")) else None,
            value=Value(value) if (value := data.get("value")) else None,
        )


@dataclass
class Annotation(MorqaData):
    att: Attribute
    doc: str
    end: int
    ent_id: str
    label: Label
    start: int

    def __post_init__(self):
        if self.end < self.start:
            raise ValueError(f"end ({self.end}) must be >= start ({self.start})")

    def __hash__(self):
        return hash(self.ent_id)

    def __str__(self) -> str:
        lines = [
            f"Annotation: {self.ent_id}",
            f"\tdoc: {self.doc}",
            f"\tend: {self.end}",
            f"\tlabel: {self.label}",
            f"\tstart: {self.start}",
            f"\tatt:",
        ]
        lines.extend([f"\t\t{line}" for line in str(self.att).splitlines()])

        return "\n".join(lines)

    @classmethod
    def from_dict(cls, data: dict) -> "Annotation":
        return cls(
            att=Attribute.from_dict(data["att"]),
            doc=str(data.get('doc', "")),
            end=int(data["end"]),
            ent_id=str(data.get('ent_id', "")),
            label=Label(data["label"]),
            start=int(data["start"]),
        )


@dataclass
class Response(MorqaData):
    annotations: dict[str, list[Annotation]]
    author_id: str
    content: str
    response_num: str

    def __str__(self) -> str:
        lines = [
            f"Response: {self.response_num}",
            "\tauthor_id: {}".format(self.author_id),
            "\tcontent: {}".format(self.content.replace('\n', ' ')),
        ]
        # Add annotations
        lines.append(f"\tannotations:")
        for key in self.annotations:
            lines.append(f"\t\t{key}:")
            for ann in self.annotations[key]:
                lines.extend([f"\t\t\t{line}" for line in str(ann).splitlines()])

        return "\n".join(lines)

    @classmethod
    def from_dict(cls, data: dict) -> "Response":
        return cls(
            annotations={"query_title" if "title" in k else "query_content": [Annotation.from_dict(ann) for ann in v]
                         for k, v in data["annotations"].items()},
            author_id=str(data.get('author_id', "")),
            # Use next((str(data[k]) for k in ("content_en", "content_zh") if k in data and data[k]), "")
            content=str(data.get('content_en', "")),
            response_num=str(data.get('response_num', "")),
        )


@dataclass
class QaPair(MorqaData):
    id: str
    questions: list[Annotation]
    answers: list[Annotation]

    def __str__(self) -> str:
        lines = [
            f"QA ID: {self.id}",
            f"\tQuestions:",
        ]

        for question in self.questions:
            lines.append(f"\t\t{question.att.text}")
            if question.att.is_implicit:
                lines.append(f"\t\t\tIMPLICIT")
            lines.append(f"\t\t\tPolarity: {question.att.polarity}")
            lines.append(f"\t\t\tQuestion Type: {question.att.questtyp}")

        lines.append(f"\tAnswers:")
        for answer in self.answers:
            lines.append(f"\t\t{answer.att.text}")
            if answer.att.value:
                lines.append(f"\t\t\t{answer.att.value.upper()}")

        return "\n".join(lines)


@dataclass
class Document(MorqaData):
    annotations: dict[str, list[Annotation]]
    post_id: str
    query_content: str
    query_title: str
    responses: list[Response]
    _questions: list[Annotation] = None
    _answers: list[Annotation] = None
    _qa_pairs: dict[str, QaPair] = None

    @property
    def questions(self) -> list[Annotation]:
        """List of all question annotations found in the document."""
        if not self._questions:
            self._questions = [ann for anns in self.annotations.values() for ann in anns]

        return self._questions

    @property
    def answers(self) -> list[Annotation]:
        """List of all answer annotations found in the document."""
        if not self._answers:
            self._answers = []

            for response in self.responses:
                for anns in response.annotations.values():
                    for ann in anns:
                        if ann.label == Label.SHORTEST_ANSWER:
                            self._answers.append(ann)

        return self._answers

    @property
    def qa_pairs(self) -> dict[str, QaPair]:
        """Mapping of question IDs to QA pairs found in the document."""
        if not self._qa_pairs:
            id_question_map = defaultdict(list)
            id_answer_map = defaultdict(list)

            # Populate maps
            for q in self.questions:
                id_question_map[q.att.id].append(q)
            for a in self.answers:
                id_answer_map[a.att.id].append(a)

            # Create QA pairs
            self._qa_pairs = {
                id_: QaPair(id_, questions, id_answer_map.get(id_, []))
                for id_, questions in id_question_map.items()
            }

        return self._qa_pairs

    def __str__(self) -> str:
        lines = [
            f"Document: {self.post_id}",
            "\tquery_title: {}".format(self.query_title.replace('\n', ' ')),
            "\tquery_content: {}".format(self.query_content.replace('\n', ' ')),
        ]
        # Add annotations
        lines.append("\tannotations:")
        for key in self.annotations:
            lines.append(f"\t\t{key}:")
            for ann in self.annotations[key]:
                lines.extend([f"\t\t\t{line}" for line in str(ann).splitlines()])
        # Add responses
        lines.append("\tresponses:")
        for response in self.responses:
            lines.extend([f"\t\t{line}" for line in str(response).splitlines()])

        return "\n".join(lines)

    @classmethod
    def from_dict(cls, data: dict) -> "Document":
        return cls(
            annotations={"query_title" if "title" in k else "query_content": [Annotation.from_dict(item) for item in v]
                         for k, v in data['annotations'].items()},
            post_id=str(data.get('post_id', "")),
            # Use next((str(data[k]) for k in ("query_content_en", "query_content_zh") if k in data and data[k]), "")
            query_content=str(data.get('query_content_en', "")),
            # Use next((str(data[k]) for k in ("query_title_en", "query_title_zh") if k in data and data[k]), "")
            query_title=str(data.get('query_title_en', "")),
            responses=[Response.from_dict(resp) for resp in data['responses']],
        )


IMPLICIT_QUESTTYP: list[QuestionType] = sorted(Attribute.IMPLICIT_QUESTTYP)

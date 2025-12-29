from functools import singledispatch
from typing import Any, Union

from src.preprocess.data import Document, QuestionType


@singledispatch
def eval(gold_data: Union[Document, list[Document]], pred_data: Union[Document, list[Document]]) -> dict[
    str, list[Any]]:
    """Evaluates the classification performance between gold and predicted data."""
    ...


@eval.register(Document)
def _(gold_data: Document, pred_data: Document) -> dict[str, list[Any]]:
    gold_polarities = []
    pred_polarities = []
    gold_types = []
    pred_types = []

    for gold_question, pred_question in zip(gold_data.questions, pred_data.questions):
        # Implicit questions have 'open' polarity and can have any of the question types defined in `Attribute.IMPLICIT_QUESTTYP`
        if gold_question.att.is_implicit:
            continue

        # We don't have enough instances of NOT_CC questions
        if gold_question.att.questtyp == QuestionType.NOT_CC:
            continue

        gold_polarities.append(gold_question.att.polarity)
        gold_types.append(gold_question.att.questtyp)
        pred_polarities.append(pred_question.att.polarity)
        pred_types.append(pred_question.att.questtyp)

    return {
        'gold_polarities': gold_polarities,
        'pred_polarities': pred_polarities,
        'gold_types': gold_types,
        'pred_types': pred_types,
    }


@eval.register(list)
def _(gold_data: list[Document], pred_data: list[Document]) -> dict[str, list[Any]]:
    gold_polarities = []
    pred_polarities = []
    gold_types = []
    pred_types = []

    for gold_doc, pred_doc in zip(gold_data, pred_data):
        result = eval(gold_doc, pred_doc)
        gold_polarities.extend(result['gold_polarities'])
        pred_polarities.extend(result['pred_polarities'])
        gold_types.extend(result['gold_types'])
        pred_types.extend(result['pred_types'])

    return {
        'gold_polarities': gold_polarities,
        'pred_polarities': pred_polarities,
        'gold_types': gold_types,
        'pred_types': pred_types,
    }

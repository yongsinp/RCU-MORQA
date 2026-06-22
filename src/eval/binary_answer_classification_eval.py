from functools import singledispatch
from typing import Any, Union

from src.eval.runner import run_binary_answer_classification_eval
from src.preprocess.data import Document, Polarity


@singledispatch
def eval(gold_data: Union[Document, list[Document]], pred_data: Union[Document, list[Document]]) -> dict[
    str, list[Any]]:
    """Evaluates the classification performance between gold and predicted data."""
    ...


@eval.register(Document)
def _(gold_data: Document, pred_data: Document) -> dict[str, list[Any]]:
    gold_value = []
    pred_value = []

    for id_, qa_pair in gold_data.qa_pairs.items():
        questions = [q for q in qa_pair.questions if q.att.polarity == Polarity.BINARY]
        if not questions:
            continue

        for gold_answer, pred_answer in zip(qa_pair.answers, pred_data.qa_pairs[id_].answers):
            gold_value.append(str(gold_answer.att.value))
            pred_value.append(str(pred_answer.att.value))

    return {
        'gold_values': gold_value,
        'pred_values': pred_value,
    }


@eval.register(list)
def _(gold_data: list[Document], pred_data: list[Document]) -> dict[str, list[Any]]:
    gold_values = []
    pred_values = []

    for gold_doc, pred_doc in zip(gold_data, pred_data):
        result = eval(gold_doc, pred_doc)
        gold_values.extend(result['gold_values'])
        pred_values.extend(result['pred_values'])

    return {
        'gold_values': gold_values,
        'pred_values': pred_values,
    }


if __name__ == '__main__':
    # Paths
    gold_path = "../../data/rcu-en/"
    pred_path = "../../out/answer_classification/"

    # File names
    datasets = [
        "iiyi",
        "woundcare",
    ]
    splits = [
        "test_gold",
        "test_systems",
    ]

    run_binary_answer_classification_eval(eval, gold_path, pred_path, datasets, splits)

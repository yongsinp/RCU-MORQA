import os
from functools import singledispatch
from typing import Any, Union

from sklearn.metrics import precision_recall_fscore_support, classification_report

from src.preprocess.data import Document, QuestionType
from src.util.io import read_json


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


if __name__ == '__main__':
    # Paths
    gold_path = "../../data/rcu-en/"
    pred_path = "../../out/question_classification/"

    # File names
    datasets = [
        "iiyi",
        "woundcare",
    ]
    splits = [
        # "train_gold",
        # "valid_gold",
        "test_gold",
        # "test_systems",
    ]
    files = [f"{dataset}_{split}.json" for dataset in datasets for split in splits]

    # Model names
    models = [dir_name for dir_name in os.listdir(pred_path) if os.path.isdir(os.path.join(pred_path, dir_name))]

    for model in models:
        print(f"Model: {model}")
        for file in files:
            print(f"\tFile: {file}")

            gold_file = os.path.join(gold_path, file)
            pred_file = os.path.join(pred_path, model, file)

            if not os.path.exists(pred_file):
                print(f"\t\tWarning: {pred_file} does not exist, skipping.")
                continue

            gold_data = [Document.from_dict(doc) for doc in read_json(gold_file)]
            pred_data = [Document.from_dict(doc) for doc in read_json(pred_file)]

            result = eval(gold_data, pred_data)

            # Polarity metrics
            print("=== Polarity ===")
            p, r, f1, _ = precision_recall_fscore_support(
                result['gold_polarities'], result['pred_polarities'], average='weighted', zero_division=0
            )
            print(f"Precision: {p:.4f}, Recall: {r:.4f}, F1: {f1:.4f}")
            print(classification_report(result['gold_polarities'], result['pred_polarities']))

            # Type metrics
            print("\n=== Question Type ===")
            p, r, f1, _ = precision_recall_fscore_support(
                result['gold_types'], result['pred_types'], average='weighted', zero_division=0
            )
            print(f"Precision: {p:.4f}, Recall: {r:.4f}, F1: {f1:.4f}")
            print(classification_report(result['gold_types'], result['pred_types']))

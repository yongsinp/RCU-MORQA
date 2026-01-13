import os
from functools import singledispatch
from typing import Any, Union

from sklearn.metrics import precision_recall_fscore_support, classification_report

from src.preprocess.data import Document, Polarity
from src.util.io import read_json


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
        "train_gold",
        "valid_gold",
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
            gold_data = [Document.from_dict(doc) for doc in read_json(gold_file)]
            pred_data = [Document.from_dict(doc) for doc in read_json(pred_file)]

            result = eval(gold_data, pred_data)

            # Value metrics
            print("=== Values ===")
            p, r, f1, _ = precision_recall_fscore_support(
                result['gold_values'], result['pred_values'], average='weighted', zero_division=0
            )
            print(f"Precision: {p:.4f}, Recall: {r:.4f}, F1: {f1:.4f}")
            print(classification_report(result['gold_values'], result['pred_values']))

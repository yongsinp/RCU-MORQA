import os
from collections import Counter
from functools import singledispatch
from typing import Union

from src.eval.eval import is_overlapping
from src.preprocess.data import Document
from src.util.io import read_json


@singledispatch
def eval(gold_data: Union[Document, list[Document]], pred_data: Union[Document, list[Document]],
         match_rate: float = 0.0, match_attr: bool = False) -> Counter[str]:
    """Evaluates the extraction performance between gold and predicted data."""
    ...


@eval.register(Document)
def _(gold_data: Document, pred_data: Document, match_rate: float = 0.0, match_attr: bool = False) -> Counter[str]:
    metrics = Counter()

    for key, golds in gold_data.annotations.items():
        preds = pred_data.annotations.get(key, [])

        pred_candidates = set(preds)
        for gold in golds:
            for pred in pred_candidates:
                if is_overlapping(gold, pred, match_rate=match_rate, match_attr=match_attr):
                    metrics['TP'] += 1
                    pred_candidates.remove(pred)
                    break
            else:
                metrics['FN'] += 1

        gold_candidates = set(golds)
        for pred in preds:
            for gold in gold_candidates:
                if is_overlapping(gold, pred, match_rate=match_rate, match_attr=match_attr):
                    gold_candidates.remove(gold)
                    break
            else:
                metrics['FP'] += 1

    return metrics


@eval.register(list)
def _(gold_data: list[Document], pred_data: list[Document],
      match_rate: float = 0.0, match_attr: bool = False) -> Counter[str]:
    metrics = Counter()

    for gold_doc, pred_doc in zip(gold_data, pred_data):
        metrics.update(eval(gold_doc, pred_doc, match_rate=match_rate, match_attr=match_attr))

    return metrics


if __name__ == '__main__':
    # Paths
    gold_path = "../../data/rcu-en/"
    pred_path = "../../out/question_extraction/"

    # File names
    datasets = [
        "iiyi",
        "woundcare",
    ]
    splits = [
        # "train_gold",
        # "valid_gold",
        # "valid_systems",
        "test_gold",
        # "test_systems",
    ]
    files = [f"{dataset}_{split}.json" for dataset in datasets for split in splits]

    # Model names
    models = [dir_name for dir_name in os.listdir(pred_path) if os.path.isdir(os.path.join(pred_path, dir_name))]

    match_rate: float = 0
    for model in models:
        print(f"Model: {model}")
        for file in files:
            print(f"\tFile: {file}")

            gold_file = os.path.join(gold_path, file)
            pred_file = os.path.join(pred_path, model, file)
            if not os.path.exists(pred_file):
                continue

            gold_data = [Document.from_dict(doc) for doc in read_json(gold_file)]
            pred_data = [Document.from_dict(doc) for doc in read_json(pred_file)]

            metrics = eval(gold_data, pred_data, match_rate=match_rate)

            precision = metrics['TP'] / (metrics['TP'] + metrics['FP']) if (metrics['TP'] + metrics['FP']) > 0 else 0.0
            recall = metrics['TP'] / (metrics['TP'] + metrics['FN']) if (metrics['TP'] + metrics['FN']) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

            print("\t\t{} Match:".format("Exact" if match_rate else "Relaxed"))
            print(f"\t\t\tPrecision: {precision:.4f}")
            print(f"\t\t\tRecall: {recall:.4f}")
            print(f"\t\t\tF1 Score: {f1:.4f}")
            print()
        print()

from collections import Counter
from functools import singledispatch
from typing import Union

from src.eval.eval import is_overlapping
from src.eval.runner import run_extraction_eval_per_file
from src.preprocess.data import Document
from src.util.paths import DATA_RCU_EN_PATH, OUT_PATH


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
    gold_path = str(DATA_RCU_EN_PATH)
    pred_path = str(OUT_PATH / "question_extraction")

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
    match_rate: float = 0
    run_extraction_eval_per_file(eval, gold_path, pred_path, datasets, splits, match_rate=match_rate)

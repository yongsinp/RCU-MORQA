import os
from collections import Counter
from functools import singledispatch
from typing import Union

from src.eval.eval import is_overlapping
from src.preprocess.data import Document, Label
from src.util.io import read_json


@singledispatch
def eval(gold_data: Union[Document, list[Document]], pred_data: Union[Document, list[Document]],
         exact_match: bool = False, match_attr: bool = False) -> Counter[str]:
    """Evaluates the extraction performance between gold and predicted data."""
    ...


@eval.register(Document)
def _(gold_data: Document, pred_data: Document, exact_match: bool = False, match_attr: bool = False) -> Counter[str]:
    metrics = Counter()

    golds = [ann
             for res in gold_data.responses
             for anns in res.annotations.values()
             for ann in anns
             if ann.label == Label.SHORTEST_ANSWER]
    preds = [ann
             for res in pred_data.responses
             for anns in res.annotations.values()
             for ann in anns
             if ann.label == Label.SHORTEST_ANSWER]

    pred_candidates = set(preds)
    for gold in golds:
        for pred in pred_candidates:
            if is_overlapping(gold, pred, exact_match=exact_match, match_attr=match_attr):
                metrics['TP'] += 1
                pred_candidates.remove(pred)
                break
        else:
            metrics['FN'] += 1

    gold_candidates = set(golds)
    for pred in preds:
        for gold in gold_candidates:
            if is_overlapping(gold, pred, exact_match=exact_match, match_attr=match_attr):
                gold_candidates.remove(gold)
                break
        else:
            metrics['FP'] += 1

    return metrics


@eval.register(list)
def _(gold_data: list[Document], pred_data: list[Document],
      exact_match: bool = False, match_attr: bool = False) -> Counter[str]:
    metrics = Counter()

    for gold_doc, pred_doc in zip(gold_data, pred_data):
        metrics.update(eval(gold_doc, pred_doc, exact_match=exact_match, match_attr=match_attr))

    return metrics

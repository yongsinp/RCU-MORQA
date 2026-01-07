import logging
import os
from collections import Counter
from functools import singledispatch
from typing import Union

from src.eval.eval import is_overlapping
from src.preprocess.data import Document, Label, Annotation
from src.util.io import read_json


@singledispatch
def eval(gold_data: Union[Document, list[Document]], pred_data: Union[Document, list[Document]],
         exact_match: bool = False, match_attr: bool = False) -> Counter[str]:
    """Evaluates the extraction performance between gold and predicted data."""
    ...


@eval.register(Document)
def _(gold_data: Document, pred_data: Document, exact_match: bool = False, match_attr: bool = False) -> Counter[str]:
    def get_responses(data: Document) -> list[dict[str, Annotation]]:
        responses = []
        for response in data.responses:
            answers = {}
            for anns in response.annotations.values():
                for ann in anns:
                    if ann.label == Label.MEDICAL_IAA:
                        # Add one answer per attribute ID
                        answers[ann.att.id] = ann
            responses.append(answers)
        return responses

    metrics = Counter()

    gold_responses = get_responses(gold_data)
    pred_responses = get_responses(pred_data)
    assert len(gold_responses) == len(pred_responses), "Number of responses in gold and predicted data do not match."

    for gold_response, pred_response in zip(gold_responses, pred_responses):
        for id_, gold_ann in gold_response.items():
            pred_ann = pred_response.get(id_)
            if pred_ann is None:
                metrics['FN'] += 1
            elif is_overlapping(gold_ann, pred_ann, exact_match=exact_match, match_attr=match_attr):
                metrics['TP'] += 1
            else:
                metrics['FN'] += 1
                metrics['FP'] += 1

        for id_, pred_ann in pred_response.items():
            if id_ not in gold_response:
                metrics['FP'] += 1

    return metrics


@eval.register(list)
def _(gold_data: list[Document], pred_data: list[Document],
      exact_match: bool = False, match_attr: bool = False) -> Counter[str]:
    metrics = Counter()

    for gold_doc, pred_doc in zip(gold_data, pred_data):
        metrics.update(eval(gold_doc, pred_doc, exact_match=exact_match, match_attr=match_attr))

    return metrics


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )

    # Paths
    gold_path = "../../data/rcu-en/"
    pred_path = "../../out/iaa_extraction/"

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

    exact_match = False
    for model in models:
        print(f"Model: {model}")
        for file in files:
            print(f"\tFile: {file}")

            gold_file = os.path.join(gold_path, file)
            pred_file = os.path.join(pred_path, model, file)
            gold_data = [Document.from_dict(doc) for doc in read_json(gold_file)]
            pred_data = [Document.from_dict(doc) for doc in read_json(pred_file)]

            metrics = eval(gold_data, pred_data, exact_match=exact_match)

            precision = metrics['TP'] / (metrics['TP'] + metrics['FP']) if (metrics['TP'] + metrics['FP']) > 0 else 0.0
            recall = metrics['TP'] / (metrics['TP'] + metrics['FN']) if (metrics['TP'] + metrics['FN']) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

            print("\t\t{} Match:".format("Exact" if exact_match else "Relaxed"))
            print(f"\t\t\tPrecision: {precision:.4f}")
            print(f"\t\t\tRecall: {recall:.4f}")
            print(f"\t\t\tF1 Score: {f1:.4f}")
            print()
        print()

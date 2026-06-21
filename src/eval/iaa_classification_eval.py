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
    gold_problem = []
    pred_problem = []
    gold_test = []
    pred_test = []
    gold_treatment = []
    pred_treatment = []
    gold_follup = []
    pred_follup = []
    gold_severe = []
    pred_severe = []
    gold_conditional = []
    pred_conditional = []

    gold_responses = [ann
                      for response in gold_data.responses
                      for annotations in response.annotations.values()
                      for ann in annotations]
    pred_responses = [ann
                      for response in pred_data.responses
                      for annotations in response.annotations.values()
                      for ann in annotations]

    for gold, pred in zip(gold_responses, pred_responses):
        gold_problem.append(gold.att.is_prob)
        pred_problem.append(pred.att.is_prob)
        gold_test.append(gold.att.is_test)
        pred_test.append(pred.att.is_test)
        gold_treatment.append(gold.att.is_treat)
        pred_treatment.append(pred.att.is_treat)
        gold_follup.append(gold.att.is_follup)
        pred_follup.append(pred.att.is_follup)
        gold_severe.append(gold.att.is_severe)
        pred_severe.append(pred.att.is_severe)
        gold_conditional.append(gold.att.is_conditional)
        pred_conditional.append(pred.att.is_conditional)

    return {
        'gold_problem': gold_problem,
        'pred_problem': pred_problem,
        'gold_test': gold_test,
        'pred_test': pred_test,
        'gold_treatment': gold_treatment,
        'pred_treatment': pred_treatment,
        'gold_follup': gold_follup,
        'pred_follup': pred_follup,
        'gold_severe': gold_severe,
        'pred_severe': pred_severe,
        'gold_conditional': gold_conditional,
        'pred_conditional': pred_conditional,
    }


@eval.register(list)
def _(gold_data: list[Document], pred_data: list[Document]) -> dict[str, list[Any]]:
    gold_problem = []
    pred_problem = []
    gold_test = []
    pred_test = []
    gold_treatment = []
    pred_treatment = []
    gold_follup = []
    pred_follup = []
    gold_severe = []
    pred_severe = []
    gold_conditional = []
    pred_conditional = []

    for gold_doc, pred_doc in zip(gold_data, pred_data):
        result = eval(gold_doc, pred_doc)
        gold_problem.extend(result['gold_problem'])
        pred_problem.extend(result['pred_problem'])
        gold_test.extend(result['gold_test'])
        pred_test.extend(result['pred_test'])
        gold_treatment.extend(result['gold_treatment'])
        pred_treatment.extend(result['pred_treatment'])
        gold_follup.extend(result['gold_follup'])
        pred_follup.extend(result['pred_follup'])
        gold_severe.extend(result['gold_severe'])
        pred_severe.extend(result['pred_severe'])
        gold_conditional.extend(result['gold_conditional'])
        pred_conditional.extend(result['pred_conditional'])

    return {
        'gold_problem': gold_problem,
        'pred_problem': pred_problem,
        'gold_test': gold_test,
        'pred_test': pred_test,
        'gold_treatment': gold_treatment,
        'pred_treatment': pred_treatment,
        'gold_follup': gold_follup,
        'pred_follup': pred_follup,
        'gold_severe': gold_severe,
        'pred_severe': pred_severe,
        'gold_conditional': gold_conditional,
        'pred_conditional': pred_conditional,
    }


if __name__ == '__main__':
    # Paths
    gold_path = "../../data/rcu-en/"
    pred_path = "../../out/iaa_classification/"

    # File names
    datasets = [
        "iiyi",
        "woundcare",
    ]
    splits = [
        "test_gold",
        "test_systems",
    ]

    # Model names
    models = [dir_name for dir_name in os.listdir(pred_path) if os.path.isdir(os.path.join(pred_path, dir_name))]

    for model in models:
        print(f"Model: {model}")
        for dataset in datasets:
            print(f"\tDataset: {dataset}")

            combined_result = {
                'gold_problem': [],
                'pred_problem': [],
                'gold_test': [],
                'pred_test': [],
                'gold_treatment': [],
                'pred_treatment': [],
                'gold_follup': [],
                'pred_follup': [],
                'gold_severe': [],
                'pred_severe': [],
                'gold_conditional': [],
                'pred_conditional': [],
            }

            for split in splits:
                file = f"{dataset}_{split}.json"
                gold_file = os.path.join(gold_path, file)
                pred_file = os.path.join(pred_path, model, file)

                if not os.path.exists(pred_file):
                    print(f"\t\tWarning: {pred_file} does not exist, skipping.")
                    continue

                gold_data = [Document.from_dict(doc) for doc in read_json(gold_file)]
                pred_data = [Document.from_dict(doc) for doc in read_json(pred_file)]

                result = eval(gold_data, pred_data)

                for key in combined_result:
                    combined_result[key].extend(result[key])

            # Calculate combined metrics for each category
            print("\t\t=== Problem (Combined) ===")
            p, r, f1, _ = precision_recall_fscore_support(
                combined_result['gold_problem'], combined_result['pred_problem'], average='weighted', zero_division=0
            )
            print(f"\t\tPrecision: {p:.4f}, Recall: {r:.4f}, F1: {f1:.4f}")
            print("\t\t" + classification_report(combined_result['gold_problem'],
                                                 combined_result['pred_problem']).replace("\n", "\n\t\t"))

            print("\t\t=== Test (Combined) ===")
            p, r, f1, _ = precision_recall_fscore_support(
                combined_result['gold_test'], combined_result['pred_test'], average='weighted', zero_division=0
            )
            print(f"\t\tPrecision: {p:.4f}, Recall: {r:.4f}, F1: {f1:.4f}")
            print(
                "\t\t" + classification_report(combined_result['gold_test'], combined_result['pred_test']).replace("\n",
                                                                                                                   "\n\t\t"))

            print("\t\t=== Treatment (Combined) ===")
            p, r, f1, _ = precision_recall_fscore_support(
                combined_result['gold_treatment'], combined_result['pred_treatment'], average='weighted',
                zero_division=0
            )
            print(f"\t\tPrecision: {p:.4f}, Recall: {r:.4f}, F1: {f1:.4f}")
            print("\t\t" + classification_report(combined_result['gold_treatment'],
                                                 combined_result['pred_treatment']).replace("\n", "\n\t\t"))

            print("\t\t=== Follow-up (Combined) ===")
            p, r, f1, _ = precision_recall_fscore_support(
                combined_result['gold_follup'], combined_result['pred_follup'], average='weighted', zero_division=0
            )
            print(f"\t\tPrecision: {p:.4f}, Recall: {r:.4f}, F1: {f1:.4f}")
            print(
                "\t\t" + classification_report(combined_result['gold_follup'], combined_result['pred_follup']).replace(
                    "\n", "\n\t\t"))

            print("\t\t=== Severe (Combined) ===")
            p, r, f1, _ = precision_recall_fscore_support(
                combined_result['gold_severe'], combined_result['pred_severe'], average='weighted', zero_division=0
            )
            print(f"\t\tPrecision: {p:.4f}, Recall: {r:.4f}, F1: {f1:.4f}")
            print(
                "\t\t" + classification_report(combined_result['gold_severe'], combined_result['pred_severe']).replace(
                    "\n", "\n\t\t"))

            print("\t\t=== Conditional (Combined) ===")
            p, r, f1, _ = precision_recall_fscore_support(
                combined_result['gold_conditional'], combined_result['pred_conditional'], average='weighted',
                zero_division=0
            )
            print(f"\t\tPrecision: {p:.4f}, Recall: {r:.4f}, F1: {f1:.4f}")
            print("\t\t" + classification_report(combined_result['gold_conditional'],
                                                 combined_result['pred_conditional']).replace("\n", "\n\t\t"))
            print()
        print()

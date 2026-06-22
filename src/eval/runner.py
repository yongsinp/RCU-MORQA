import logging
import os
from collections import Counter
from typing import Any, Callable

from sklearn.metrics import classification_report, precision_recall_fscore_support

from src.preprocess.data import Document
from src.util.io import read_json


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )


def list_models(pred_path: str) -> list[str]:
    if not os.path.isdir(pred_path):
        raise FileNotFoundError(f"Prediction path does not exist or is not a directory: {pred_path}")
    return [d for d in os.listdir(pred_path) if os.path.isdir(os.path.join(pred_path, d))]


def load_documents(file_path: str) -> list[Document]:
    return [Document.from_dict(doc) for doc in read_json(file_path)]


def run_extraction_eval(
    eval_fn: Callable[..., Counter[str]],
    gold_path: str,
    pred_path: str,
    datasets: list[str],
    splits: list[str],
    match_rate: float = 0.0,
) -> None:
    models = list_models(pred_path)

    for model in models:
        print(f"Model: {model}")
        for dataset in datasets:
            print(f"\tDataset: {dataset}")
            combined_metrics = Counter()

            for split in splits:
                file_name = f"{dataset}_{split}.json"
                gold_file = os.path.join(gold_path, file_name)
                pred_file = os.path.join(pred_path, model, file_name)

                if not os.path.exists(pred_file):
                    logging.warning("Prediction file %s does not exist, skipping.", pred_file)
                    continue

                gold_data = load_documents(gold_file)
                pred_data = load_documents(pred_file)
                combined_metrics.update(eval_fn(gold_data, pred_data, match_rate=match_rate))

            precision = combined_metrics["TP"] / (combined_metrics["TP"] + combined_metrics["FP"]) if (combined_metrics["TP"] + combined_metrics["FP"]) > 0 else 0.0
            recall = combined_metrics["TP"] / (combined_metrics["TP"] + combined_metrics["FN"]) if (combined_metrics["TP"] + combined_metrics["FN"]) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

            print("\t\t{} Match (Combined):".format("Exact" if match_rate else "Relaxed"))
            print(f"\t\t\tPrecision: {precision:.4f}")
            print(f"\t\t\tRecall: {recall:.4f}")
            print(f"\t\t\tF1 Score: {f1:.4f}")
            print()
        print()


def run_extraction_eval_per_file(
    eval_fn: Callable[..., Counter[str]],
    gold_path: str,
    pred_path: str,
    datasets: list[str],
    splits: list[str],
    match_rate: float = 0.0,
) -> None:
    models = list_models(pred_path)
    files = [f"{dataset}_{split}.json" for dataset in datasets for split in splits]

    for model in models:
        print(f"Model: {model}")
        for file_name in files:
            print(f"\tFile: {file_name}")

            gold_file = os.path.join(gold_path, file_name)
            pred_file = os.path.join(pred_path, model, file_name)

            if not os.path.exists(pred_file):
                logging.warning("Prediction file %s does not exist, skipping.", pred_file)
                continue

            gold_data = load_documents(gold_file)
            pred_data = load_documents(pred_file)
            metrics = eval_fn(gold_data, pred_data, match_rate=match_rate)

            precision = metrics["TP"] / (metrics["TP"] + metrics["FP"]) if (metrics["TP"] + metrics["FP"]) > 0 else 0.0
            recall = metrics["TP"] / (metrics["TP"] + metrics["FN"]) if (metrics["TP"] + metrics["FN"]) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

            print("\t\t{} Match:".format("Exact" if match_rate else "Relaxed"))
            print(f"\t\t\tPrecision: {precision:.4f}")
            print(f"\t\t\tRecall: {recall:.4f}")
            print(f"\t\t\tF1 Score: {f1:.4f}")
            print()
        print()


def run_question_classification_eval(
    eval_fn: Callable[..., dict[str, list[Any]]],
    gold_path: str,
    pred_path: str,
    datasets: list[str],
    splits: list[str],
) -> None:
    models = list_models(pred_path)
    files = [f"{dataset}_{split}.json" for dataset in datasets for split in splits]

    for model in models:
        print(f"Model: {model}")
        for file_name in files:
            print(f"\tFile: {file_name}")

            gold_file = os.path.join(gold_path, file_name)
            pred_file = os.path.join(pred_path, model, file_name)
            if not os.path.exists(pred_file):
                print(f"\t\tWarning: {pred_file} does not exist, skipping.")
                continue

            result = eval_fn(load_documents(gold_file), load_documents(pred_file))

            print("=== Polarity ===")
            p, r, f1, _ = precision_recall_fscore_support(
                result["gold_polarities"], result["pred_polarities"], average="weighted", zero_division=0
            )
            print(f"Precision: {p:.4f}, Recall: {r:.4f}, F1: {f1:.4f}")
            print(classification_report(result["gold_polarities"], result["pred_polarities"], zero_division=0))

            print("\n=== Question Type ===")
            p, r, f1, _ = precision_recall_fscore_support(
                result["gold_types"], result["pred_types"], average="weighted", zero_division=0
            )
            print(f"Precision: {p:.4f}, Recall: {r:.4f}, F1: {f1:.4f}")
            print(classification_report(result["gold_types"], result["pred_types"], zero_division=0))


def run_binary_answer_classification_eval(
    eval_fn: Callable[..., dict[str, list[Any]]],
    gold_path: str,
    pred_path: str,
    datasets: list[str],
    splits: list[str],
) -> None:
    models = list_models(pred_path)

    for model in models:
        print(f"Model: {model}")
        for dataset in datasets:
            print(f"\tDataset: {dataset}")
            combined_gold_values: list[Any] = []
            combined_pred_values: list[Any] = []

            for split in splits:
                file_name = f"{dataset}_{split}.json"
                gold_file = os.path.join(gold_path, file_name)
                pred_file = os.path.join(pred_path, model, file_name)

                if not os.path.exists(pred_file):
                    print(f"\t\tWarning: {pred_file} does not exist, skipping.")
                    continue

                result = eval_fn(load_documents(gold_file), load_documents(pred_file))
                combined_gold_values.extend(result["gold_values"])
                combined_pred_values.extend(result["pred_values"])

            print("\t\t=== Values (Combined) ===")
            p, r, f1, _ = precision_recall_fscore_support(
                combined_gold_values, combined_pred_values, average="weighted", zero_division=0
            )
            print(f"\t\tPrecision: {p:.4f}, Recall: {r:.4f}, F1: {f1:.4f}")
            print(
                "\t\t"
                + classification_report(combined_gold_values, combined_pred_values, zero_division=0).replace(
                    "\n", "\n\t\t"
                )
            )
            print()
        print()


def run_iaa_classification_eval(
    eval_fn: Callable[..., dict[str, list[Any]]],
    gold_path: str,
    pred_path: str,
    datasets: list[str],
    splits: list[str],
) -> None:
    models = list_models(pred_path)

    category_fields = [
        ("Problem", "problem"),
        ("Test", "test"),
        ("Treatment", "treatment"),
        ("Follow-up", "follup"),
        ("Severe", "severe"),
        ("Conditional", "conditional"),
    ]

    for model in models:
        print(f"Model: {model}")
        for dataset in datasets:
            print(f"\tDataset: {dataset}")
            combined_result = {f"{prefix}_{name}": [] for _, name in category_fields for prefix in ("gold", "pred")}

            for split in splits:
                file_name = f"{dataset}_{split}.json"
                gold_file = os.path.join(gold_path, file_name)
                pred_file = os.path.join(pred_path, model, file_name)

                if not os.path.exists(pred_file):
                    print(f"\t\tWarning: {pred_file} does not exist, skipping.")
                    continue

                result = eval_fn(load_documents(gold_file), load_documents(pred_file))
                for key in combined_result:
                    combined_result[key].extend(result[key])

            for display, key in category_fields:
                print(f"\t\t=== {display} (Combined) ===")
                p, r, f1, _ = precision_recall_fscore_support(
                    combined_result[f"gold_{key}"],
                    combined_result[f"pred_{key}"],
                    average="weighted",
                    zero_division=0,
                )
                print(f"\t\tPrecision: {p:.4f}, Recall: {r:.4f}, F1: {f1:.4f}")
                print(
                    "\t\t"
                    + classification_report(
                        combined_result[f"gold_{key}"],
                        combined_result[f"pred_{key}"],
                        zero_division=0,
                    ).replace("\n", "\n\t\t")
                )
            print()
        print()

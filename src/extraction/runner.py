import dataclasses
import logging
import os
from dataclasses import dataclass
from typing import Callable, Iterable

from tqdm import tqdm

from src.preprocess.data import Document
from src.util.io import read_json, write_json


@dataclass(frozen=True)
class ExtractionTask:
    output_subdir: str
    method_name: str
    progress_desc: str
    kwargs: dict[str, object] | None = None


LLM_TASKS: tuple[ExtractionTask, ...] = (
    ExtractionTask("question_extraction", "extract_questions", "Extracting Questions"),
    ExtractionTask("question_classification", "classify_questions", "Classifying Questions"),
    ExtractionTask("answer_extraction", "extract_answers", "Extracting Answers"),
    ExtractionTask("answer_classification", "classify_binary_answers", "Classifying Answers"),
    # "iaa" in method names refers to "medical directives" (legacy internal naming)
    ExtractionTask("medical_directives_extraction", "extract_iaa", "Extracting Medical Directives"),
    ExtractionTask("medical_directives_classification", "classify_iaa", "Classifying Medical Directives"),
    ExtractionTask("prognosis_extraction", "extract_prognosis", "Extracting Prognosis"),
)

QUESTION_TASKS: tuple[ExtractionTask, ...] = LLM_TASKS[:2]
NON_QUESTION_TASKS: tuple[ExtractionTask, ...] = LLM_TASKS[2:]


def build_files(datasets: Iterable[str], splits: Iterable[str]) -> list[str]:
    return [f"{dataset}_{split}.json" for dataset in datasets for split in splits]


def run_tasks(
    extractor,
    data_path: str,
    out_path: str,
    datasets: list[str],
    splits: list[str],
    tasks: Iterable[ExtractionTask],
    model_name: str | None = None,
) -> None:
    model_dir = (model_name or extractor.model_name).replace("/", "_")

    for file_name in build_files(datasets, splits):
        logging.info("File: %s", file_name)
        data = read_json(os.path.join(data_path, file_name))
        documents = [Document.from_dict(doc) for doc in data]

        for task in tasks:
            output_file = os.path.join(out_path, task.output_subdir, model_dir, file_name)

            if os.path.exists(output_file):
                logging.info("%s results found at %s, skipping.", task.output_subdir, output_file)
                continue

            method: Callable[[Document], Document] = getattr(extractor, task.method_name)
            kwargs = task.kwargs or {}
            results = [
                dataclasses.asdict(method(document, **kwargs), dict_factory=Document.prune)
                for document in tqdm(documents, desc=task.progress_desc)
            ]
            write_json(output_file, results)


def run_llm_tasks(
    extractor,
    data_path: str,
    out_path: str,
    datasets: list[str],
    question_splits: list[str],
    non_question_splits: list[str],
    model_name: str | None = None,
) -> None:
    run_tasks(
        extractor=extractor,
        data_path=data_path,
        out_path=out_path,
        datasets=datasets,
        splits=question_splits,
        tasks=QUESTION_TASKS,
        model_name=model_name,
    )
    run_tasks(
        extractor=extractor,
        data_path=data_path,
        out_path=out_path,
        datasets=datasets,
        splits=non_question_splits,
        tasks=NON_QUESTION_TASKS,
        model_name=model_name,
    )

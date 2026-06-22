import os
from typing import Optional

import typer

from src.extraction.runner import ExtractionTask, run_llm_tasks, run_tasks
from src.preprocess.data import Document
from src.util.io import read_json
from src.util.paths import DATA_RCU_EN_PATH, OUT_PATH

app = typer.Typer(help="Run MORQA extraction pipelines.")


def _default_data_path() -> str:
    return str(DATA_RCU_EN_PATH)


def _default_out_path() -> str:
    return str(OUT_PATH)


def _default_datasets() -> list[str]:
    return ["iiyi", "woundcare"]


def _default_splits() -> list[str]:
    return ["test_gold", "test_systems"]


def _load_documents(file_path: str) -> list[Document]:
    data = read_json(file_path)
    return [Document.from_dict(doc) for doc in data]


@app.command("gpt")
def run_gpt(model_name: str = "gpt-4o", max_output_tokens: int = 5000) -> None:
    from src.extraction.gpt import GptExtractor

    extractor = GptExtractor(model_name=model_name, max_output_tokens=max_output_tokens)
    splits = _default_splits()
    question_splits = [split for split in splits if "systems" not in split]
    run_llm_tasks(
        extractor,
        _default_data_path(),
        _default_out_path(),
        _default_datasets(),
        question_splits,
        splits,
    )


@app.command("gemini")
def run_gemini(
    model_name: str = "gemini-2.5-pro",
    max_output_tokens: int = 5000,
    reasoning: bool = True,
) -> None:
    from src.extraction.gemini import GeminiExtractor

    extractor = GeminiExtractor(model_name=model_name, max_output_tokens=max_output_tokens, reasoning=reasoning)
    splits = _default_splits()
    question_splits = [split for split in splits if "systems" not in split]
    run_llm_tasks(
        extractor,
        _default_data_path(),
        _default_out_path(),
        _default_datasets(),
        question_splits,
        splits,
    )


@app.command("qwen")
def run_qwen(model_name: str = "qwen3-vl-plus", max_output_tokens: int = 5000) -> None:
    from src.extraction.qwen import QwenExtractor

    extractor = QwenExtractor(model_name=model_name, max_output_tokens=max_output_tokens)
    splits = _default_splits()
    question_splits = [split for split in splits if "systems" not in split]
    run_llm_tasks(
        extractor,
        _default_data_path(),
        _default_out_path(),
        _default_datasets(),
        question_splits,
        splits,
    )


@app.command("rule")
def run_rule(model_name: str = "en_core_web_sm") -> None:
    from src.extraction.rule import RuleBasedExtractor

    extractor = RuleBasedExtractor(model_name=model_name)
    run_tasks(
        extractor,
        _default_data_path(),
        _default_out_path(),
        _default_datasets(),
        ["test_gold"],
        (ExtractionTask("question_extraction", "extract_questions", "Extracting Questions"),),
    )


@app.command("mrc")
def run_mrc(model_name: str = "dmis-lab/biobert-v1.1") -> None:
    from src.extraction.mrc import MRCExtractor

    extractor = MRCExtractor(model_name=model_name)
    run_tasks(
        extractor,
        _default_data_path(),
        _default_out_path(),
        _default_datasets(),
        _default_splits(),
        (ExtractionTask("answer_extraction", "extract_answers", "Extracting Answers"),),
        model_name=model_name,
    )


@app.command("mrc-train")
def train_mrc(
    model_name: str = "dmis-lab/biobert-v1.1",
    include_system: bool = False,
) -> None:
    from src.extraction.mrc import MRCExtractor

    data_path = _default_data_path()
    datasets = _default_datasets()
    split_suffixes = ["gold", "system"] if include_system else ["gold"]

    train_documents: list[Document] = []
    valid_documents: list[Document] = []

    for dataset in datasets:
        for suffix in split_suffixes:
            train_file = os.path.join(data_path, f"{dataset}_train_{suffix}.json")
            valid_file = os.path.join(data_path, f"{dataset}_valid_{suffix}.json")

            if not os.path.exists(train_file):
                raise typer.BadParameter(f"Missing training file: {train_file}")
            if not os.path.exists(valid_file):
                raise typer.BadParameter(f"Missing validation file: {valid_file}")

            train_documents.extend(_load_documents(train_file))
            valid_documents.extend(_load_documents(valid_file))

    extractor = MRCExtractor(model_name=model_name)
    extractor.train(train_documents, valid_documents)


@app.command("biobert")
def run_biobert(
    model_name: str = "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb",
    min_threshold: float = 0.25,
    max_threshold: float = 0.85,
    step: float = 0.05,
) -> None:
    from src.extraction.biobert import BioBertExtractor

    if step <= 0:
        raise typer.BadParameter("step must be greater than 0")
    if min_threshold > max_threshold:
        raise typer.BadParameter("min_threshold must be less than or equal to max_threshold")

    extractor = BioBertExtractor(similarity_model_name=model_name)
    scale = 100
    min_int = round(min_threshold * scale)
    max_int = round(max_threshold * scale)
    step_int = round(step * scale)

    if step_int <= 0:
        raise typer.BadParameter("step is too small after rounding; use at least 0.01")

    for threshold_int in range(min_int, max_int + 1, step_int):
        threshold = threshold_int / scale
        threshold_str = f"{threshold:.2f}"
        run_tasks(
            extractor,
            _default_data_path(),
            _default_out_path(),
            _default_datasets(),
            ["train_gold", "valid_gold"],
            (ExtractionTask(
                "answer_extraction",
                "extract_answers",
                f"Extracting Answers {threshold_str}",
                kwargs={"threshold": threshold},
            ),),
            model_name=f"BioBERT_{threshold_str}",
        )


@app.command("run")
def run_extractor(
    extractor: str,
    model_name: Optional[str] = None,
) -> None:
    normalized = extractor.strip().lower()
    if normalized == "gpt":
        run_gpt(model_name=model_name or "gpt-4o")
        return
    if normalized == "gemini":
        run_gemini(model_name=model_name or "gemini-2.5-pro")
        return
    if normalized == "qwen":
        run_qwen(model_name=model_name or "qwen3-vl-plus")
        return
    if normalized == "rule":
        run_rule(model_name=model_name or "en_core_web_sm")
        return
    if normalized == "mrc":
        run_mrc(model_name=model_name or "dmis-lab/biobert-v1.1")
        return
    if normalized == "biobert":
        run_biobert(model_name=model_name or "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb")
        return
    raise typer.BadParameter("Unknown extractor. Use one of: gpt, gemini, qwen, rule, mrc, biobert")


if __name__ == "__main__":
    app()

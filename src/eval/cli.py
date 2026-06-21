import runpy

import typer

app = typer.Typer(help="Run MORQA evaluation scripts.")


def _run(module_name: str) -> None:
    runpy.run_module(module_name, run_name="__main__")


@app.command("question-extraction")
def question_extraction() -> None:
    _run("src.eval.question_extraction_eval")


@app.command("question-classification")
def question_classification() -> None:
    _run("src.eval.question_classification_eval")


@app.command("answer-extraction")
def answer_extraction() -> None:
    _run("src.eval.answer_extraction_eval")


@app.command("binary-answer-classification")
def binary_answer_classification() -> None:
    _run("src.eval.binary_answer_classification_eval")


@app.command("iaa-extraction")
def iaa_extraction() -> None:
    _run("src.eval.iaa_extraction_eval")


@app.command("iaa-classification")
def iaa_classification() -> None:
    _run("src.eval.iaa_classification_eval")


@app.command("prognosis-extraction")
def prognosis_extraction() -> None:
    _run("src.eval.prognosis_extraction_eval")


@app.command("run")
def run_eval(script: str) -> None:
    normalized = script.strip().lower()
    mapping = {
        "question-extraction": "src.eval.question_extraction_eval",
        "question-classification": "src.eval.question_classification_eval",
        "answer-extraction": "src.eval.answer_extraction_eval",
        "binary-answer-classification": "src.eval.binary_answer_classification_eval",
        "iaa-extraction": "src.eval.iaa_extraction_eval",
        "iaa-classification": "src.eval.iaa_classification_eval",
        "prognosis-extraction": "src.eval.prognosis_extraction_eval",
    }

    module_name = mapping.get(normalized)
    if module_name is None:
        raise typer.BadParameter(
            "Unknown eval script. Use one of: question-extraction, question-classification, "
            "answer-extraction, binary-answer-classification, iaa-extraction, iaa-classification, "
            "prognosis-extraction"
        )

    _run(module_name)


if __name__ == "__main__":
    app()

from typer.testing import CliRunner
import pytest
import typer

from src.eval.cli import app as eval_app
from src.extraction import cli as extraction_cli


def test_eval_cli_run_dispatches_to_module(monkeypatch) -> None:
    called = {"module": None}

    def fake_run(module_name: str) -> None:
        called["module"] = module_name

    monkeypatch.setattr("src.eval.cli._run", fake_run)
    runner = CliRunner()
    result = runner.invoke(eval_app, ["run", "question-extraction"])

    assert result.exit_code == 0
    assert called["module"] == "src.eval.question_extraction_eval"


def test_extraction_run_dispatches_deepseek(monkeypatch) -> None:
    called = {"model": None}

    def fake_run_deepseek(model_name: str = "deepseek-chat", max_output_tokens: int = 5000) -> None:
        called["model"] = model_name

    monkeypatch.setattr(extraction_cli, "run_deepseek", fake_run_deepseek)
    extraction_cli.run_extractor("deepseek", model_name="deepseek-v3")

    assert called["model"] == "deepseek-v3"


def test_extraction_run_rejects_unknown_extractor() -> None:
    with pytest.raises(typer.BadParameter) as exc:
        extraction_cli.run_extractor("unknown")
    assert "Unknown extractor" in str(exc.value)


def test_eval_cli_run_rejects_unknown_script() -> None:
    runner = CliRunner()
    result = runner.invoke(eval_app, ["run", "unknown-eval"])
    assert result.exit_code != 0
    assert "Unknown eval script" in result.output

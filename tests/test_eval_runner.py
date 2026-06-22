from collections import Counter
import warnings

import pytest
from sklearn.exceptions import UndefinedMetricWarning

from src.eval import runner


def test_list_models_raises_for_missing_path(tmp_path) -> None:
    missing = tmp_path / "does-not-exist"
    with pytest.raises(FileNotFoundError):
        runner.list_models(str(missing))


def test_list_models_returns_only_directories(tmp_path) -> None:
    (tmp_path / "model-a").mkdir()
    (tmp_path / "model-b").mkdir()
    (tmp_path / "file.txt").write_text("x", encoding="utf-8")

    models = runner.list_models(str(tmp_path))
    assert sorted(models) == ["model-a", "model-b"]


def test_run_extraction_eval_per_file_prints_scores(tmp_path, monkeypatch, capsys) -> None:
    gold_path = tmp_path / "gold"
    pred_path = tmp_path / "pred"
    (gold_path / "ignored").mkdir(parents=True)
    (pred_path / "model-x").mkdir(parents=True)

    file_name = "iiyi_test_gold.json"
    (gold_path / file_name).write_text("[]", encoding="utf-8")
    (pred_path / "model-x" / file_name).write_text("[]", encoding="utf-8")

    monkeypatch.setattr(runner, "list_models", lambda _: ["model-x"])
    monkeypatch.setattr(runner, "load_documents", lambda _: [])

    def fake_eval(_gold, _pred, match_rate=0.0):
        return Counter({"TP": 2, "FP": 1, "FN": 1})

    runner.run_extraction_eval_per_file(
        eval_fn=fake_eval,
        gold_path=str(gold_path),
        pred_path=str(pred_path),
        datasets=["iiyi"],
        splits=["test_gold"],
        match_rate=0.0,
    )

    out = capsys.readouterr().out
    assert "Model: model-x" in out
    assert "Precision: 0.6667" in out
    assert "Recall: 0.6667" in out


def test_run_extraction_eval_combines_split_metrics(monkeypatch, capsys) -> None:
    monkeypatch.setattr(runner, "list_models", lambda _: ["model-x"])

    def fake_load_documents(path: str):
        return [path]

    monkeypatch.setattr(runner, "load_documents", fake_load_documents)
    monkeypatch.setattr(runner.os.path, "exists", lambda _: True)

    def fake_eval(gold_data, pred_data, match_rate=0.0):
        gold_path = gold_data[0]
        if "test_gold" in gold_path:
            return Counter({"TP": 1, "FP": 0, "FN": 1})
        return Counter({"TP": 1, "FP": 1, "FN": 0})

    runner.run_extraction_eval(
        eval_fn=fake_eval,
        gold_path="/gold",
        pred_path="/pred",
        datasets=["iiyi"],
        splits=["test_gold", "test_systems"],
        match_rate=0.0,
    )

    out = capsys.readouterr().out
    assert "Dataset: iiyi" in out
    assert "Precision: 0.6667" in out
    assert "Recall: 0.6667" in out
    assert "F1 Score: 0.6667" in out


def test_run_iaa_classification_eval_prints_all_categories(monkeypatch, capsys) -> None:
    monkeypatch.setattr(runner, "list_models", lambda _: ["model-x"])
    monkeypatch.setattr(runner, "load_documents", lambda _: [])
    monkeypatch.setattr(runner.os.path, "exists", lambda _: True)

    def fake_eval(_gold, _pred):
        return {
            "gold_problem": [True, False],
            "pred_problem": [True, True],
            "gold_test": [False, True],
            "pred_test": [False, True],
            "gold_treatment": [True, False],
            "pred_treatment": [True, False],
            "gold_follup": [False, True],
            "pred_follup": [False, False],
            "gold_severe": [False, False],
            "pred_severe": [False, True],
            "gold_conditional": [False, True],
            "pred_conditional": [False, True],
        }

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UndefinedMetricWarning)
        runner.run_iaa_classification_eval(
            eval_fn=fake_eval,
            gold_path="/gold",
            pred_path="/pred",
            datasets=["iiyi"],
            splits=["test_gold"],
        )

    out = capsys.readouterr().out
    assert "=== Problem (Combined) ===" in out
    assert "=== Follow-up (Combined) ===" in out
    assert "=== Conditional (Combined) ===" in out


def test_run_binary_answer_classification_eval_aggregates_splits(monkeypatch, capsys) -> None:
    monkeypatch.setattr(runner, "list_models", lambda _: ["model-x"])
    monkeypatch.setattr(runner, "load_documents", lambda _: [])
    monkeypatch.setattr(runner.os.path, "exists", lambda _: True)

    calls = {"count": 0}

    def fake_eval(_gold, _pred):
        calls["count"] += 1
        if calls["count"] == 1:
            return {"gold_values": ["yes", "no"], "pred_values": ["yes", "yes"]}
        return {"gold_values": ["no"], "pred_values": ["no"]}

    runner.run_binary_answer_classification_eval(
        eval_fn=fake_eval,
        gold_path="/gold",
        pred_path="/pred",
        datasets=["iiyi"],
        splits=["test_gold", "test_systems"],
    )

    out = capsys.readouterr().out
    assert "=== Values (Combined) ===" in out
    assert "Precision:" in out
    assert calls["count"] == 2


def test_run_question_classification_eval_skips_missing_pred_files(monkeypatch, capsys) -> None:
    monkeypatch.setattr(runner, "list_models", lambda _: ["model-x"])

    def fake_exists(path: str) -> bool:
        return path.endswith("test_gold.json")

    monkeypatch.setattr(runner.os.path, "exists", fake_exists)
    monkeypatch.setattr(runner, "load_documents", lambda _: [])

    def fake_eval(_gold, _pred):
        return {
            "gold_polarities": ["open"],
            "pred_polarities": ["open"],
            "gold_types": ["advice"],
            "pred_types": ["advice"],
        }

    runner.run_question_classification_eval(
        eval_fn=fake_eval,
        gold_path="/gold",
        pred_path="/pred",
        datasets=["iiyi"],
        splits=["test_gold", "test_systems"],
    )

    out = capsys.readouterr().out
    assert "Warning:" in out
    assert "=== Polarity ===" in out

from dataclasses import dataclass

from src.extraction import runner as extraction_runner


@dataclass
class _FakeDocument:
    value: str

    @staticmethod
    def from_dict(data: dict) -> "_FakeDocument":
        return _FakeDocument(value=data["value"])

    @staticmethod
    def prune(items: list[tuple[str, object]]) -> dict[str, object]:
        return {k: v for k, v in items}


class _FakeExtractor:
    model_name = "org/model"

    def extract_questions(self, document: _FakeDocument, suffix: str = "") -> _FakeDocument:
        return _FakeDocument(value=document.value + suffix)


def test_build_files_cartesian_order() -> None:
    files = extraction_runner.build_files(["a", "b"], ["x", "y"])
    assert files == ["a_x.json", "a_y.json", "b_x.json", "b_y.json"]


def test_run_tasks_writes_outputs_and_applies_kwargs(monkeypatch) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr(extraction_runner, "Document", _FakeDocument)
    monkeypatch.setattr(extraction_runner, "read_json", lambda _: [{"value": "doc"}])
    monkeypatch.setattr(extraction_runner, "write_json", lambda path, data: captured.update({"path": path, "data": data}))
    monkeypatch.setattr(extraction_runner, "tqdm", lambda docs, desc: docs)
    monkeypatch.setattr(extraction_runner.os.path, "exists", lambda _: False)

    task = extraction_runner.ExtractionTask(
        output_subdir="question_extraction",
        method_name="extract_questions",
        progress_desc="Extracting",
        kwargs={"suffix": "-ok"},
    )

    extraction_runner.run_tasks(
        extractor=_FakeExtractor(),
        data_path="/data",
        out_path="/out",
        datasets=["iiyi"],
        splits=["test_gold"],
        tasks=(task,),
    )

    assert captured["path"] == "/out/question_extraction/org_model/iiyi_test_gold.json"
    assert captured["data"] == [{"value": "doc-ok"}]


def test_run_llm_tasks_splits_question_and_non_question(monkeypatch) -> None:
    calls: list[dict] = []

    def fake_run_tasks(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(extraction_runner, "run_tasks", fake_run_tasks)

    extractor = _FakeExtractor()
    extraction_runner.run_llm_tasks(
        extractor=extractor,
        data_path="/data",
        out_path="/out",
        datasets=["iiyi"],
        question_splits=["test_gold"],
        non_question_splits=["test_gold", "test_systems"],
    )

    assert len(calls) == 2
    assert calls[0]["splits"] == ["test_gold"]
    assert calls[0]["tasks"] == extraction_runner.QUESTION_TASKS
    assert calls[1]["splits"] == ["test_gold", "test_systems"]
    assert calls[1]["tasks"] == extraction_runner.NON_QUESTION_TASKS


def test_run_tasks_skips_when_output_exists(monkeypatch) -> None:
    writes: list[tuple[str, object]] = []

    monkeypatch.setattr(extraction_runner, "Document", _FakeDocument)
    monkeypatch.setattr(extraction_runner, "read_json", lambda _: [{"value": "doc"}])
    monkeypatch.setattr(extraction_runner, "write_json", lambda path, data: writes.append((path, data)))
    monkeypatch.setattr(extraction_runner, "tqdm", lambda docs, desc: docs)
    monkeypatch.setattr(extraction_runner.os.path, "exists", lambda _: True)

    task = extraction_runner.ExtractionTask(
        output_subdir="question_extraction",
        method_name="extract_questions",
        progress_desc="Extracting",
    )

    extraction_runner.run_tasks(
        extractor=_FakeExtractor(),
        data_path="/data",
        out_path="/out",
        datasets=["iiyi"],
        splits=["test_gold"],
        tasks=(task,),
    )

    assert writes == []

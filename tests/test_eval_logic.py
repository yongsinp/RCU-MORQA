from src.eval.answer_extraction_eval import eval as answer_eval
from src.eval.binary_answer_classification_eval import eval as binary_eval
from src.eval.question_extraction_eval import eval as question_eval
from src.preprocess.data import Annotation, Attribute, Document, Label, Polarity, Response, Value


def _doc_with_questions(annotations: list[Annotation]) -> Document:
    return Document(
        annotations={"query_title": [], "query_content": annotations},
        post_id="p1",
        query_content="abcdefg",
        query_title="",
        responses=[],
    )


def _doc_with_answers(answers: list[Annotation], values: list[Value]) -> Document:
    questions = [
        Annotation(
            att=Attribute(text="Q", id="1", polarity=Polarity.BINARY),
            doc="p1.ann",
            end=1,
            ent_id="TQ1",
            label=Label.QUESTION,
            start=0,
        )
    ]
    response_answers = []
    for i, (answer, value) in enumerate(zip(answers, values), start=1):
        answer.att.value = value
        answer.ent_id = f"TA{i}"
        response_answers.append(answer)

    return Document(
        annotations={"query_title": [], "query_content": questions},
        post_id="p1",
        query_content="query",
        query_title="",
        responses=[
            Response(
                annotations={"content": response_answers},
                author_id="u1",
                content="response",
                response_num="1",
            )
        ],
    )


def test_question_extraction_eval_counts_tp_fp_fn() -> None:
    gold = _doc_with_questions([
        Annotation(
            att=Attribute(text="abc", id="1"),
            doc="p1.ann",
            end=3,
            ent_id="T1",
            label=Label.QUESTION,
            start=0,
        )
    ])
    pred = _doc_with_questions([
        Annotation(
            att=Attribute(text="abc", id="1"),
            doc="p1.ann",
            end=3,
            ent_id="T2",
            label=Label.QUESTION,
            start=0,
        ),
        Annotation(
            att=Attribute(text="fg", id="2"),
            doc="p1.ann",
            end=7,
            ent_id="T3",
            label=Label.QUESTION,
            start=5,
        ),
    ])

    metrics = question_eval(gold, pred, match_rate=1.0)
    assert metrics["TP"] == 1
    assert metrics["FN"] == 0
    assert metrics["FP"] == 1


def test_answer_extraction_eval_matches_by_attribute_id() -> None:
    gold_answer = Annotation(
        att=Attribute(text="yes", id="1"),
        doc="p1.ann",
        end=3,
        ent_id="A1",
        label=Label.SHORTEST_ANSWER,
        start=0,
    )
    pred_answer = Annotation(
        att=Attribute(text="yes", id="1"),
        doc="p1.ann",
        end=3,
        ent_id="A2",
        label=Label.SHORTEST_ANSWER,
        start=0,
    )
    gold = _doc_with_answers([gold_answer], [Value.YES])
    pred = _doc_with_answers([pred_answer], [Value.YES])

    metrics = answer_eval(gold, pred, match_rate=1.0)
    assert metrics["TP"] == 1
    assert metrics["FN"] == 0
    assert metrics["FP"] == 0


def test_binary_answer_classification_eval_collects_gold_pred_values() -> None:
    gold_answer = Annotation(
        att=Attribute(text="yes", id="1", value=Value.YES),
        doc="p1.ann",
        end=3,
        ent_id="A1",
        label=Label.SHORTEST_ANSWER,
        start=0,
    )
    pred_answer = Annotation(
        att=Attribute(text="yes", id="1", value=Value.NO),
        doc="p1.ann",
        end=3,
        ent_id="A2",
        label=Label.SHORTEST_ANSWER,
        start=0,
    )
    gold = _doc_with_answers([gold_answer], [Value.YES])
    pred = _doc_with_answers([pred_answer], [Value.NO])

    result = binary_eval(gold, pred)
    assert result["gold_values"] == ["yes"]
    assert result["pred_values"] == ["no"]

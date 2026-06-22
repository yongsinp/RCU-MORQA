from src.eval.eval import is_overlapping
from src.preprocess.data import Annotation, Attribute, Label, QuestionType


def _ann(text: str, start: int, end: int, *, implicit: bool = False) -> Annotation:
    questtyp = QuestionType.ADVICE if implicit else None
    return Annotation(
        att=Attribute(text=text, is_implicit=implicit, questtyp=questtyp),
        doc="p1.ann",
        end=end,
        ent_id=f"{start}-{end}-{text}",
        label=Label.SHORTEST_ANSWER,
        start=start,
    )


def test_is_overlapping_rejects_touching_boundaries() -> None:
    ann1 = _ann("a", 5, 15)
    ann2 = _ann("a", 15, 20)
    assert is_overlapping(ann1, ann2) is False


def test_is_overlapping_accepts_partial_overlap_default_threshold() -> None:
    ann1 = _ann("a", 5, 15)
    ann2 = _ann("a", 10, 20)
    assert is_overlapping(ann1, ann2) is True


def test_is_overlapping_honors_match_rate_threshold() -> None:
    ann1 = _ann("a", 0, 10)
    ann2 = _ann("a", 8, 15)
    assert is_overlapping(ann1, ann2, match_rate=0.5) is False
    assert is_overlapping(ann1, ann2, match_rate=0.2) is True


def test_is_overlapping_exact_requires_match_rate_one() -> None:
    ann1 = _ann("a", 5, 15)
    ann2 = _ann("a", 5, 15)
    ann3 = _ann("a", 6, 15)
    assert is_overlapping(ann1, ann2, match_rate=1.0) is True
    assert is_overlapping(ann1, ann3, match_rate=1.0) is False


def test_is_overlapping_with_match_attr_requires_equal_attributes() -> None:
    ann1 = _ann("text1", 5, 15)
    ann2 = _ann("text2", 5, 15)
    assert is_overlapping(ann1, ann2, match_rate=1.0, match_attr=True) is False


def test_is_overlapping_rejects_implicit_mismatch() -> None:
    explicit = _ann("a", 5, 15, implicit=False)
    implicit = _ann("a", 5, 15, implicit=True)
    assert is_overlapping(explicit, implicit) is False

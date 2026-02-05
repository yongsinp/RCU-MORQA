from src.preprocess.data import Annotation, Attribute, Label


def is_overlapping(ann1: Annotation, ann2: Annotation, match_rate: float = 0.0, match_attr: bool = False) -> bool:
    """Checks if two annotations overlap by a minimum match rate.

    Args:
        ann1: The first annotation.
        ann2: The second annotation.
        match_rate: Minimum overlap ratio (0.0-1.0). 0.0 means any overlap, 1.0 means exact match. Defaults to 0.0.
        match_attr: If True, checks if attributes match. Defaults to False.

    Returns:
        True if annotations overlap by at least match_rate, False otherwise.
    """
    if bool(ann1.att.is_implicit) != bool(ann2.att.is_implicit):
        return False

    overlap_start = max(ann1.start, ann2.start)
    overlap_end = min(ann1.end, ann2.end)

    if overlap_end <= overlap_start:
        return False

    overlap_len = overlap_end - overlap_start
    max_len = max(ann1.end - ann1.start, ann2.end - ann2.start)
    overlap_ratio = overlap_len / max_len

    if overlap_ratio < match_rate:
        return False

    return ann1.att == ann2.att if match_attr else True


if __name__ == '__main__':
    attr = Attribute(text="text")
    attr2 = Attribute(text="text2")
    ann1 = Annotation(att=attr, doc="", ent_id=0, label=Label.SHORTEST_ANSWER, start=5, end=15)
    ann2 = Annotation(att=attr, doc="", ent_id=0, label=Label.SHORTEST_ANSWER, start=15, end=20)
    ann3 = Annotation(att=attr, doc="", ent_id=0, label=Label.SHORTEST_ANSWER, start=5, end=15)
    ann4 = Annotation(att=attr, doc="", ent_id=0, label=Label.SHORTEST_ANSWER, start=1, end=6)
    ann5 = Annotation(att=attr, doc="", ent_id=0, label=Label.SHORTEST_ANSWER, start=17, end=20)
    ann6 = Annotation(att=attr2, doc="", ent_id=0, label=Label.SHORTEST_ANSWER, start=17, end=20)

    print(is_overlapping(ann1, ann2))  # False
    print(is_overlapping(ann1, ann4))  # True
    print(is_overlapping(ann1, ann3, exact_match=True))  # True
    print(is_overlapping(ann1, ann4, exact_match=True))  # False
    print(is_overlapping(ann2, ann5))  # True
    print(is_overlapping(ann4, ann5))  # False
    print(is_overlapping(ann5, ann6, exact_match=True))  # True
    print(is_overlapping(ann5, ann6, exact_match=True, match_attr=True))  # False
    print(is_overlapping(ann5, ann6, match_attr=True))  # False

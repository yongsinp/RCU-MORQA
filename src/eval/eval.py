from src.preprocess.data import Annotation, Attribute, Label


def is_overlapping(ann1: Annotation, ann2: Annotation, exact_match: bool = False, match_attr: bool = False) -> bool:
    """Checks if two annotations have any overlaps.

    Args:
        ann1:
            The first annotation.
        ann2:
            The second annotation.
        exact_match:
            If True, checks for exact span match. Defaults to False.
        match_attr:
            If True, checks if attributes match. Defaults to False.

    Returns:
        True if annotations overlap (or match exactly if exact_match is True),
        False otherwise.
    """
    if bool(ann1.att.is_implicit) != bool(ann2.att.is_implicit):
        return False

    if exact_match:
        if ann1.start == ann2.start and ann1.end == ann2.end:
            if match_attr:
                return ann1.att == ann2.att
            return True
        return False

    # Check for span overlap
    if ann1.end <= ann2.start or ann2.end <= ann1.start:
        return False  # No overlap

    if match_attr:
        return ann1.att == ann2.att

    return True


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

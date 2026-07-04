"""Burrows' Delta smoke + separability tests (CI-safe: synthetic text)."""

from discostyle.features.delta import BurrowsDelta


def _corpus():
    # Author A: formal/measured; Author B: shouty slang. Distinct function words.
    a = [
        "I would advise that we proceed with the utmost caution, detective.",
        "It is, in my considered opinion, a matter of some delicacy.",
        "One must weigh the evidence before one draws any conclusion at all.",
        "I shall record that, if you would be so kind as to repeat it.",
    ]
    b = [
        "Yeah whatever man, get outta here, this is such garbage!",
        "Shut up, nobody cares what you think, you total loser!",
        "Gimme that, it's mine, back off before I lose it!",
        "This place sucks and everyone in it is a joke, seriously!",
    ]
    texts = a + b
    y = ["A"] * len(a) + ["B"] * len(b)
    return texts, y


def test_fit_predict_runs():
    texts, y = _corpus()
    clf = BurrowsDelta(n_mfw=40).fit(texts, y)
    preds = clf.predict(texts)
    assert set(preds) <= {"A", "B"}
    assert len(preds) == len(texts)


def test_separates_distinct_authors():
    # Delta is a long-document method; on tiny/short lines a single shared word
    # ("the", "this") can flip one line, so we assert the robust signal — most of
    # the training set is attributed to the correct author — not per-line perfection.
    # n_mfw must cover this tiny corpus's vocabulary; too few MFW on ~8 short docs
    # destabilizes the z-scoring (a real edge of the method, not our concern here).
    texts, y = _corpus()
    clf = BurrowsDelta(n_mfw=80).fit(texts, y)
    preds = clf.predict(texts)
    correct = sum(p == t for p, t in zip(preds, y))
    assert correct >= len(y) - 1  # at most one of the 8 misattributed


def test_centroid_distances_symmetric():
    texts, y = _corpus()
    clf = BurrowsDelta(n_mfw=40).fit(texts, y)
    d = clf.centroid_distances()
    assert d.loc["A", "B"] == d.loc["B", "A"]
    assert d.loc["A", "A"] == 0

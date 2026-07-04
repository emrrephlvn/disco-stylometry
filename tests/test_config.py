"""Guard the locked V1 config (CI-safe: no corpus needed)."""

from discostyle.config import EXCLUDE_SPEAKERS, MAX_LINES_PER_CLASS, SHORTLIST_V1


def test_shortlist_is_locked_and_clean():
    assert len(SHORTLIST_V1) == 12
    assert len(set(SHORTLIST_V1)) == 12, "duplicate in shortlist"


def test_excluded_speakers_not_in_shortlist():
    assert not (set(SHORTLIST_V1) & EXCLUDE_SPEAKERS)


def test_cap_is_positive():
    assert MAX_LINES_PER_CLASS > 0

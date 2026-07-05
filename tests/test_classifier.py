"""Leak-invariant tests for the scene-aware cap + scene-grouped split.

This is the pipeline's core safety property: a scene's lines must never
straddle train/test (adjacent lines share context and leak), and capping
over-represented classes must drop WHOLE scenes, never fragment one — a
fragmented scene would let the same conversation leak across the split via
its surviving half.
"""

import pandas as pd

from discostyle.models.classifier import scene_aware_cap, scene_of, scene_split


def _synthetic_corpus() -> pd.DataFrame:
    rows = []
    # Speaker A: 3 scenes x 5 lines = 15 lines (will exceed a cap of 8).
    for scene in range(3):
        for i in range(5):
            rows.append({"line_id": f"A_scene{scene}:{i}", "speaker_canon": "A",
                         "text": f"line {i} of A scene {scene}"})
    # Speaker B: 2 scenes x 3 lines = 6 lines (stays under a cap of 8).
    for scene in range(2):
        for i in range(3):
            rows.append({"line_id": f"B_scene{scene}:{i}", "speaker_canon": "B",
                         "text": f"line {i} of B scene {scene}"})
    return pd.DataFrame(rows)


def test_scene_of_extracts_prefix():
    df = _synthetic_corpus()
    scenes = scene_of(df)
    assert scenes.iloc[0] == "A_scene0"
    assert scenes.nunique() == 5  # 3 scenes for A + 2 for B


def test_split_never_straddles_a_scene():
    df = _synthetic_corpus()
    train_idx, test_idx = scene_split(df, test_size=0.3, seed=42)
    scenes = scene_of(df)
    train_scenes = set(scenes.loc[train_idx])
    test_scenes = set(scenes.loc[test_idx])
    assert train_scenes.isdisjoint(test_scenes), "a scene appears on both sides of the split"
    # every line lands somewhere, nothing silently dropped by the split itself
    assert len(train_idx) + len(test_idx) == len(df)


def test_cap_drops_whole_scenes_not_fragments():
    df = _synthetic_corpus()
    capped = scene_aware_cap(df, max_lines=8, seed=42)
    scenes = scene_of(capped)

    # For each (speaker, scene) pair present after capping, ALL of that scene's
    # original lines must be present — no partial scene.
    orig_scenes = scene_of(df)
    for sid, grp in capped.groupby(scenes):
        orig_count = (orig_scenes == sid).sum()
        assert len(grp) == orig_count, f"scene {sid} was fragmented by the cap"

    # Speaker A (15 lines, cap 8) must have been reduced; Speaker B (6 lines,
    # cap 8) must be untouched since it never reaches the budget.
    assert capped[capped["speaker_canon"] == "A"].shape[0] < 15
    assert capped[capped["speaker_canon"] == "B"].shape[0] == 6


def test_cap_then_split_still_leak_free():
    """The composition that actually matters end-to-end: cap first, split second,
    and the invariant must still hold."""
    df = _synthetic_corpus()
    capped = scene_aware_cap(df, max_lines=8, seed=42).reset_index(drop=True)
    train_idx, test_idx = scene_split(capped, test_size=0.3, seed=1)
    scenes = scene_of(capped)
    assert set(scenes.loc[train_idx]).isdisjoint(set(scenes.loc[test_idx]))

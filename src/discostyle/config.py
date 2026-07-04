"""Locked V1 configuration — frozen after the Week-1 sufficiency gate (2026-07-04).

Speaker names are the *canonical* forms (``speaker_canon`` = text before the
first comma). See scripts/eda_v1_gate.py for the evidence behind every choice.
"""

# Excluded from character classification, with reasons:
#   You      — 41% of corpus, heterogeneous player-narration that mixes all skill
#              voices (see Spike B); ge10_ratio 0.45. Not a single "character".
#   Cunoesse — bark-heavy (ge10_ratio 0.48, median 9 words). Per-line stylometry
#              on barks is noise → excluded as an identity issue, not hygiene.
EXCLUDE_SPEAKERS = {"You", "Cunoesse"}

# Tier 1 — the LOCKED V1 core: 12 highest-volume, stylometrically rich voices
# (all ge10_ratio >= 0.77). This is what Week 2+ builds on.
SHORTLIST_V1 = [
    "Kim Kitsuragi",
    "Cuno",
    "Joyce Messier",
    "Klaasje (Miss Oranje Disco Dancer)",
    "Titus Hardie",
    "Evrart Claire",
    "The Deserter",
    "Jean Vicquemare",
    "Garte",
    "Lena",
    "Noid",
    "Soona",
]

# Per-class line cap to tame imbalance (Kim alone has 5262 lines; the rest
# 395–947). Capping at 900 brings the ceiling from 34.2x down to ~2.3x.
#
# SCENE-AWARE: apply by selecting whole scenes (one .gv = one scene) until the
# budget is reached, NOT by random line sampling. Random capping fragments
# scenes and collides with the scene-grouped split (a scene's lines must not
# straddle train/test). Cap = drop whole scenes; split = assign whole scenes.
# Fingerprint means (Week 2) ignore the cap and use all lines per speaker.
MAX_LINES_PER_CLASS = 900

# Tier 2 — documented stretch: the full 38 speakers with >=150 lines AND
# ge10_ratio >= 0.5. Accepts ~34x imbalance; use class_weight + macro-F1.
# Kept as a note, not locked; regenerate the list from scripts/eda_v1_gate.py.
SHORTLIST_V2_FULL_THRESHOLD = {"min_lines": 150, "min_ge10_ratio": 0.5}

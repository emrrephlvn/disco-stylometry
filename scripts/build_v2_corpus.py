"""Week 5 — V2 corpus: the 24 skill voices from msyavuz/disco-api's disco.db.

Spike B (Week 1) said NO-GO for skill attribution — but that was mos9527's
limitation, not the data's. In disco.db the 24 skills are separate actors
(Volition=397, Electrochemistry=405, ...), with 20–1043 lines each. This script
extracts them into the SAME schema as the V1 corpus so the whole Week-3/4
pipeline (scene-aware cap, scene-grouped split, classifier) reuses unchanged.

scene = conversationid (a dialogue tree), analogous to one .gv file in V1.

Run:  python scripts/build_v2_corpus.py
Output: data/processed/lines_v2.parquet
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from discostyle.data.load import clean

DB = Path("data/raw/disco.db")
OUT = Path("data/processed/lines_v2.parquet")

# The 24 skills as disco.db actor IDs (verified against the actors table).
SKILL_IDS = {
    "Logic": 390, "Encyclopedia": 391, "Rhetoric": 392, "Drama": 393,
    "Conceptualization": 389, "Visual Calculus": 394, "Volition": 397,
    "Inland Empire": 396, "Empathy": 395, "Authority": 398, "Esprit de Corps": 400,
    "Suggestion": 399, "Endurance": 401, "Pain Threshold": 404,
    "Physical Instrument": 402, "Electrochemistry": 405, "Shivers": 403,
    "Half Light": 406, "Hand/Eye Coordination": 407, "Perception": 412,
    "Reaction Speed": 408, "Savoir Faire": 409, "Interfacing": 410, "Composure": 411,
}
ID_TO_SKILL = {v: k for k, v in SKILL_IDS.items()}


def main() -> None:
    con = sqlite3.connect(DB)
    ids = ",".join(str(i) for i in SKILL_IDS.values())
    rows = con.execute(
        f"SELECT id, conversationid, actor, dialoguetext FROM dentries "
        f"WHERE actor IN ({ids}) AND dialoguetext IS NOT NULL AND TRIM(dialoguetext) != ''"
    ).fetchall()

    df = pd.DataFrame(
        [
            {
                "line_id": f"{conv}:{eid}",  # scene = conversationid
                "speaker": ID_TO_SKILL[actor],
                "text": " ".join(str(txt).split()),
                "source": "disco.db",
            }
            for eid, conv, actor, txt in rows
        ]
    )
    df = clean(df)  # same hygiene as V1: min 3 words, dedup
    df["speaker_canon"] = df["speaker"]  # skills need no canonicalization

    counts = df["speaker_canon"].value_counts()
    print(f"{len(df)} skill lines across {df['speaker_canon'].nunique()} skills "
          f"(after clean(): min 3 words, dedup)")
    print("\nlines per skill:")
    print(counts.to_string())
    print(f"\nskills with >=100 lines: {(counts >= 100).sum()}/24")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()

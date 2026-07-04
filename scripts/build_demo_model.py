"""Week 4 — train the final TF-IDF model for the Streamlit demo.

Week 3 already reported honest metrics on a held-out scene-grouped split
(reports/week3_metrics.md); this script trains the SHIPPED model on all
available locked-12 data (same scene-aware cap, no held-out split) so the demo
gets the benefit of every line, not 80% of them.

Run:  python scripts/build_demo_model.py
Output: models/tfidf_logreg.joblib
"""
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from discostyle.config import MAX_LINES_PER_CLASS, SHORTLIST_V1
from discostyle.models.classifier import scene_aware_cap, tfidf_baseline

MODEL_PATH = Path("models/tfidf_logreg.joblib")


def main() -> None:
    df = pd.read_parquet("data/processed/lines.parquet")
    df = df[df["speaker_canon"].isin(SHORTLIST_V1)]
    df = scene_aware_cap(df, MAX_LINES_PER_CLASS)

    pipe = tfidf_baseline().fit(df["text"], df["speaker_canon"])

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, MODEL_PATH)
    print(f"trained on {len(df)} lines, {df['speaker_canon'].nunique()} classes")
    print(f"wrote {MODEL_PATH}")


if __name__ == "__main__":
    main()

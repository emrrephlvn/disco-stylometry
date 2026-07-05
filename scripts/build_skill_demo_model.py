"""V2 companion to build_demo_model.py — trains the shipped TF-IDF skill model.

Week 5 reported MiniLM embeddings as the stronger V2 rung (0.322 vs TF-IDF's
0.298 macro-F1), but the hosted demo ships TF-IDF here deliberately: it needs
no torch/sentence-transformers at deploy time, keeping the Cloud app light.
That tradeoff is stated in the app UI, not hidden.

Run:  python scripts/build_skill_demo_model.py
Output: models/tfidf_logreg_skills.joblib
"""
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from discostyle.config import MAX_LINES_PER_CLASS
from discostyle.models.classifier import scene_aware_cap, tfidf_baseline

MODEL_PATH = Path("models/tfidf_logreg_skills.joblib")
MIN_LINES = 100  # same floor as scripts/week5_skill_classify.py


def main() -> None:
    df = pd.read_parquet("data/processed/lines_v2.parquet")
    keep = df["speaker_canon"].value_counts()
    shortlist = keep[keep >= MIN_LINES].index
    df = df[df["speaker_canon"].isin(shortlist)]
    df = scene_aware_cap(df, MAX_LINES_PER_CLASS)

    pipe = tfidf_baseline().fit(df["text"], df["speaker_canon"])

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, MODEL_PATH)
    print(f"trained on {len(df)} lines, {df['speaker_canon'].nunique()} skills")
    print(f"wrote {MODEL_PATH}")


if __name__ == "__main__":
    main()

"""Week 2 — per-speaker stylometric fingerprints on the locked 12.

First output of the modeling phase and the interpretability mandate's first test:
does each voice have a *distinct, explainable* fingerprint? Answered numerically
(most-distinctive feature per speaker + most-confusable pair), not by eyeballing.

Run:  python scripts/week2_fingerprints.py
Outputs: reports/fingerprints.md, reports/fingerprints.csv, reports/radar.html
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from discostyle.config import SHORTLIST_V1
from discostyle.features.stylometry import featurize
from discostyle.viz.fingerprint import radar

REPORTS = Path("reports")
# The interpretable core we tell the story with (function words stay in the full
# feature matrix for the model, but the fingerprint narrative uses these).
STORY = [
    "n_words", "type_token_ratio", "hapax_rate", "exclaim_rate",
    "question_rate", "ellipsis_rate", "caps_word_rate", "flesch_reading_ease",
]


def main() -> None:
    REPORTS.mkdir(exist_ok=True)
    df = pd.read_parquet("data/processed/lines.parquet")
    df = df[df["speaker_canon"].isin(SHORTLIST_V1)].reset_index(drop=True)
    print(f"{len(df)} lines across {df['speaker_canon'].nunique()} locked speakers "
          f"(fingerprints use ALL lines per speaker; cap is a Week-3 training tool)")

    feats = featurize(df)
    # group directly on the canonical speaker (df already has a raw 'speaker' col,
    # so renaming into it would collide); reindex for stable, meaningful row order.
    fp = feats.groupby(df["speaker_canon"]).mean().reindex(SHORTLIST_V1)

    # ---- fingerprint table (story features) ----
    table = fp[STORY].round(3)
    table.insert(0, "lines", df["speaker_canon"].value_counts().reindex(SHORTLIST_V1).values)
    print("\n=== PER-SPEAKER FINGERPRINT (story features) ===")
    print(table.to_string())

    # ---- interpretability test: distinctiveness + confusability ----
    z = (fp[STORY] - fp[STORY].mean()) / fp[STORY].std()
    print("\n=== each voice's most-distinctive feature (|z| max) ===")
    for spk in SHORTLIST_V1:
        row = z.loc[spk]
        feat = row.abs().idxmax()
        print(f"  {spk:36} {feat:20} z={row[feat]:+.2f}")

    # pairwise euclidean distance on z-scored fingerprints -> most-confusable pair
    D = np.linalg.norm(z.values[:, None, :] - z.values[None, :, :], axis=-1)
    np.fill_diagonal(D, np.inf)
    i, j = np.unravel_index(D.argmin(), D.shape)
    print(f"\nmost-confusable pair (min fingerprint distance): "
          f"{SHORTLIST_V1[i]} ~ {SHORTLIST_V1[j]} (d={D[i, j]:.2f})")
    # diagonal is inf, so D.min() is the smallest *off-diagonal* distance
    print(f"all 12 fingerprints distinct (min pairwise d>0): {D.min() > 0}")

    # ---- outputs ----
    table.to_csv(REPORTS / "fingerprints.csv")
    (REPORTS / "fingerprints.md").write_text(
        "# Per-speaker stylometric fingerprints (locked 12)\n\n"
        + table.to_markdown()
        + "\n\n*Feature dictionary:* n_words=mean words/line, type_token_ratio=lexical "
          "diversity, hapax_rate=once-only words, exclaim/question/ellipsis_rate=punctuation "
          "per word, caps_word_rate=ALL-CAPS words, flesch_reading_ease=readability "
          "(higher=simpler).\n",
        encoding="utf-8",
    )
    fig = radar(fp, SHORTLIST_V1)
    fig.write_html(REPORTS / "radar.html", include_plotlyjs="cdn")
    print(f"\nwrote {REPORTS/'fingerprints.md'}, .csv, radar.html")


if __name__ == "__main__":
    main()

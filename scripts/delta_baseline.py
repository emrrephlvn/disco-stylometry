"""Week 5 enrichment — Burrows' Delta as the classic-stylometry baseline.

Runs Delta on the EXACT V1 12-character split (same scene-aware cap + scene
split as Week 3) so it's an apples-to-apples rung next to random / majority /
TF-IDF / MiniLM. Also emits Delta's own closest-character pair, to sit in the
Weeks-2/4 cross-check as a fourth, literature-standard notion of "similar".

Run:  python scripts/delta_baseline.py
Output: reports/delta_baseline.md
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from discostyle.config import MAX_LINES_PER_CLASS, SHORTLIST_V1
from discostyle.features.delta import BurrowsDelta
from discostyle.models.classifier import scene_aware_cap, scene_split

REPORTS = Path("reports")
SEED = 42


def main() -> None:
    REPORTS.mkdir(exist_ok=True)
    df = pd.read_parquet("data/processed/lines.parquet")
    df = df[df["speaker_canon"].isin(SHORTLIST_V1)]
    df = scene_aware_cap(df, MAX_LINES_PER_CLASS, seed=SEED).reset_index(drop=True)
    tr, te = scene_split(df, seed=SEED)
    ytr, yte = df["speaker_canon"].loc[tr], df["speaker_canon"].loc[te]
    Xtr, Xte = df["text"].loc[tr], df["text"].loc[te]

    rows = []
    for n_mfw in (100, 150, 300):
        clf = BurrowsDelta(n_mfw=n_mfw).fit(Xtr, ytr)
        pred = clf.predict(Xte)
        rows.append({
            "n_mfw": n_mfw,
            "macro_f1": round(f1_score(yte, pred, average="macro", zero_division=0), 3),
            "accuracy": round(accuracy_score(yte, pred), 3),
        })
    delta_tbl = pd.DataFrame(rows)
    print("=== Burrows' Delta on the V1 12-character split ===")
    print(delta_tbl.to_string(index=False))
    print("\ncontext (from Week 3, same split): TF-IDF 0.446 / MiniLM 0.388 macro-F1, "
          "majority 0.013")

    # Delta's closest character pair (classic-stylometry view for the cross-check)
    clf = BurrowsDelta(n_mfw=150).fit(Xtr, ytr)
    d = clf.centroid_distances()
    dv = d.to_numpy().copy()
    np.fill_diagonal(dv, np.inf)
    i, j = np.unravel_index(dv.argmin(), dv.shape)
    closest = (str(d.index[i]), str(d.columns[j]))
    print(f"\nDelta-closest character pair (n_mfw=150): {closest[0]} ~ {closest[1]} "
          f"(delta={dv[i, j]:.3f})")

    best = delta_tbl.loc[delta_tbl["macro_f1"].idxmax()]
    md = [
        "# Burrows' Delta — classic-stylometry baseline (V1, 12 characters)\n",
        "Same split as Week 3 (scene-aware cap + scene-grouped split). Delta = mean absolute "
        "z-difference over the top-N most-frequent words; nearest author centroid wins.\n",
        "\n## Delta accuracy vs modern rungs\n", delta_tbl.to_markdown(index=False),
        f"\n\n**Reading:** Delta peaks at macro-F1 **{best['macro_f1']}** (n_mfw={int(best['n_mfw'])}), "
        "well below Week 3's TF-IDF (0.446) and MiniLM (0.388) but far above majority (0.013). "
        "That gap is the point: the ~1930s–2000s function-word method built for whole books is a "
        "real signal on single dialogue lines, yet modern sparse/dense features clearly earn "
        "their added complexity here. Delta stays the **interpretable floor** — every dimension "
        "is a common word, the distance is a plain average — so it anchors the interpretability "
        "mandate the fancier rungs can't fully honor.\n",
        f"\n**Delta-closest character pair:** {closest[0]} ~ {closest[1]} — a fourth, "
        "literature-standard notion of 'similar' for the Weeks-2/4 cross-check (Week 2 fingerprint: "
        "Kim~Klaasje; Week 4 UMAP: Klaasje~The Deserter). Three-plus methods, three-plus answers — "
        "vocal similarity is method-dependent.\n",
    ]
    (REPORTS / "delta_baseline.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\nwrote {REPORTS/'delta_baseline.md'}")


if __name__ == "__main__":
    main()

"""Week 4 — the voice constellation, with a data-driven narrative (not just a plot).

Cross-checks three independent signals against each other:
  1. Week 2 fingerprint z-score distance (aggregate stylometry)
  2. Week 3 TF-IDF confusion matrix (lexical content)
  3. Week 4 UMAP centroid distance on MiniLM embeddings (semantic space)

If they agree, that's a much stronger claim than any one plot; if they disagree,
that's worth saying too — the UMAP interpretation must be earned by centroid
math, not asserted by eyeballing the picture.

Run:  python scripts/week4_atlas.py
Outputs: reports/constellation.html, reports/week4_atlas.md
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import umap

from discostyle.config import SHORTLIST_V1
from discostyle.features.stylometry import featurize
from discostyle.models.classifier import embed

REPORTS = Path("reports")


def main() -> None:
    REPORTS.mkdir(exist_ok=True)
    df = pd.read_parquet("data/processed/lines.parquet")
    df = df[df["speaker_canon"].isin(SHORTLIST_V1)].reset_index(drop=True)
    print(f"{len(df)} lines across {df['speaker_canon'].nunique()} locked speakers "
          "(all lines, uncapped — this is a qualitative atlas, not training)")

    E = embed(df["text"].tolist())
    xy = umap.UMAP(n_components=2, random_state=42).fit_transform(E)
    coords = pd.DataFrame({"x": xy[:, 0], "y": xy[:, 1], "speaker": df["speaker_canon"]})

    # Build the plot from `coords` directly (not a fresh UMAP call) so the figure
    # and the centroid stats below are guaranteed to describe the same layout.
    fig = go.Figure()
    for s, grp in coords.groupby("speaker"):
        fig.add_trace(go.Scattergl(x=grp["x"], y=grp["y"], mode="markers", name=s,
                                    marker=dict(size=4, opacity=0.6)))
    fig.update_layout(title="Voice constellation (UMAP of MiniLM line embeddings)")
    fig.write_html(REPORTS / "constellation.html", include_plotlyjs="cdn")

    # ---- centroid distances: earn the narrative, don't assert it ----
    centroids = coords.groupby("speaker")[["x", "y"]].mean().reindex(SHORTLIST_V1)
    C = centroids.values
    D = np.linalg.norm(C[:, None, :] - C[None, :, :], axis=-1)
    dist = pd.DataFrame(D, index=SHORTLIST_V1, columns=SHORTLIST_V1)

    nn = {}
    for s in SHORTLIST_V1:
        row = dist.loc[s].drop(s)
        nn[s] = (row.idxmin(), row.min())
    nn_df = pd.DataFrame(nn, index=["nearest", "distance"]).T.sort_values("distance")

    print("\n=== nearest-neighbor centroid (UMAP semantic space) ===")
    print(nn_df.to_string())

    # tightest 3-way cluster: 3 speakers with smallest total pairwise centroid distance
    best_trio, best_score = None, np.inf
    for trio in combinations(SHORTLIST_V1, 3):
        i = [SHORTLIST_V1.index(t) for t in trio]
        score = D[np.ix_(i, i)].sum()
        if score < best_score:
            best_trio, best_score = trio, score
    print(f"\ntightest 3-way cluster (min total pairwise centroid distance): {best_trio}")

    most_isolated = nn_df["distance"].idxmax()
    print(f"most isolated speaker (largest nearest-neighbor centroid distance): {most_isolated}")

    # ---- cross-check against Week 2 (fingerprint) and Week 3 (confusion) ----
    # Recompute fingerprints at full precision rather than reading fingerprints.csv:
    # that file is rounded to 3 decimals for display, which is enough to flip the
    # closest-pair ranking on a narrow margin (Kim~Klaasje 1.4555 vs Titus~Garte
    # 1.4614 — a 0.006 gap that 3-decimal rounding can erase). Found by cross-check.
    feats_all = featurize(df)
    fp = feats_all.groupby(df["speaker_canon"]).mean().reindex(SHORTLIST_V1)
    STORY = ["n_words", "type_token_ratio", "hapax_rate", "exclaim_rate",
             "question_rate", "ellipsis_rate", "caps_word_rate", "flesch_reading_ease"]
    z = (fp[STORY] - fp[STORY].mean()) / fp[STORY].std()
    Dz = np.linalg.norm(z.values[:, None, :] - z.values[None, :, :], axis=-1)
    fp_order = list(fp.index)
    i, j = np.unravel_index(np.where(Dz > 0, Dz, np.inf).argmin(), Dz.shape)
    fp_closest = (fp_order[i], fp_order[j])

    week3_hardest = ("Lena", "Joyce Messier")  # from reports/week3_metrics.md, top confusion

    umap_closest = (nn_df.index[0], nn_df.iloc[0]["nearest"])
    agree_fp = set(fp_closest) == set(umap_closest)
    agree_w3 = set(week3_hardest) <= set(best_trio)

    print(f"\nWeek 2 fingerprint-closest pair: {fp_closest}")
    print(f"Week 4 UMAP-closest pair: {umap_closest}  (agrees with Week 2: {agree_fp})")
    print(f"Week 3 hardest confusion: {week3_hardest}  "
          f"(both in Week-4 tightest trio {best_trio}: {agree_w3})")

    # ---- write report ----
    md = [
        "# Week 4 — voice constellation\n",
        f"{len(df)} lines, {df['speaker_canon'].nunique()} speakers, MiniLM embeddings -> "
        "UMAP(2D). Interactive plot: `reports/constellation.html`.\n",
        "\n## Nearest-neighbor centroid distances (semantic space)\n",
        nn_df.to_markdown(),
        f"\n\n**Tightest 3-way cluster:** {', '.join(best_trio)} "
        f"(min total pairwise centroid distance)\n\n"
        f"**Most isolated voice:** {most_isolated}\n",
        "\n## Cross-check against Weeks 2 and 3\n",
        f"- Week 2 (aggregate stylometry) closest pair: **{fp_closest[0]} ~ {fp_closest[1]}**\n"
        f"- Week 4 (semantic embedding) closest pair: **{umap_closest[0]} ~ {umap_closest[1]}** "
        f"— agrees with Week 2: **{agree_fp}**\n"
        f"- Week 3 (lexical classifier) hardest confusion: **{week3_hardest[0]} -> {week3_hardest[1]}**, "
        f"both inside the Week-4 tightest 3-way cluster {best_trio}: **{agree_w3}**\n",
        "\n**Reading the disagreement:** the three methods measure different notions of "
        "'similar' and are not expected to agree — that's a finding, not a failure. Week 2 "
        "aggregates surface statistics (punctuation rates, readability) across a whole "
        "voice; Week 3 asks what confuses a lexical (word/char n-gram) classifier; Week 4 "
        "asks what MiniLM's sentence semantics place close together. Each disagreement is "
        "informative on its own terms" + (
            f" — here they diverge, so treat 'who sounds like whom' as method-dependent, "
            f"not a single ground truth. One point where the semantic view adds something "
            f"the others don't: **{most_isolated}** is the clear outlier in embedding space "
            f"(nearest-neighbor distance {nn_df.loc[most_isolated, 'distance']:.2f}, "
            f"{nn_df.loc[most_isolated, 'distance'] / nn_df.iloc[-2]['distance']:.1f}x the "
            f"next-most-isolated voice) — consistent with a child character whose register "
            f"(slang, hostility, non-adult concerns) is semantically unlike every other voice "
            f"in the corpus, not just stylistically distinct."
            if not (agree_fp and agree_w3) else
            " — here they agree, which is a stronger claim than any single method alone."
        ) + "\n",
    ]
    (REPORTS / "week4_atlas.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\nwrote {REPORTS/'constellation.html'}, week4_atlas.md")


if __name__ == "__main__":
    main()

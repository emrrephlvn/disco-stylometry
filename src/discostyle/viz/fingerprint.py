"""Week 4 — the fingerprint atlas.

Two portfolio-facing pictures:
  - radar(): per-character stylometric fingerprint
  - constellation(): UMAP of line embeddings, colored by speaker

Palette note: lean into DE's noir look for the *app chrome*, but keep data
colors accessible/distinct (see the dataviz skill before finalizing colors).
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

# Radar axes: pick ~6 interpretable features, not all of them.
RADAR_FEATURES = [
    "type_token_ratio", "hapax_rate", "exclaim_rate",
    "question_rate", "flesch_reading_ease", "punct_density",
]


def radar(fingerprints: pd.DataFrame, speakers: list[str]) -> go.Figure:
    """Overlayed radar of z-scored fingerprints for the chosen speakers."""
    z = (fingerprints - fingerprints.mean()) / fingerprints.std()
    fig = go.Figure()
    for s in speakers:
        row = z.loc[s, RADAR_FEATURES]
        fig.add_trace(
            go.Scatterpolar(
                r=row.tolist() + [row.iloc[0]],
                theta=RADAR_FEATURES + [RADAR_FEATURES[0]],
                name=s,
                fill="toself",
                opacity=0.55,
            )
        )
    fig.update_layout(title="Voice fingerprints (z-scored)", polar=dict(radialaxis=dict(visible=True)))
    return fig


def constellation(embeddings, speakers: pd.Series, seed: int = 42) -> go.Figure:
    """UMAP 'voice constellation' of line embeddings."""
    import umap

    xy = umap.UMAP(n_components=2, random_state=seed).fit_transform(embeddings)
    df = pd.DataFrame({"x": xy[:, 0], "y": xy[:, 1], "speaker": speakers.values})
    fig = go.Figure()
    for s, grp in df.groupby("speaker"):
        fig.add_trace(go.Scattergl(x=grp["x"], y=grp["y"], mode="markers", name=s,
                                   marker=dict(size=4, opacity=0.6)))
    fig.update_layout(title="Voice constellation (UMAP of line embeddings)")
    return fig

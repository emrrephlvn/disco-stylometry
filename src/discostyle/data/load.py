"""Week 1 — corpus acquisition and normalization.

Every loader returns the same shape so downstream code never cares where the
text came from:

    DataFrame[line_id: str, speaker: str, text: str, source: str]

Never write raw or processed corpus files anywhere outside ``data/`` (which is
gitignored) — the game text is copyrighted and must not be committed.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

PROCESSED = Path("data/processed/lines.parquet")

# GraphViz node label format used by mos9527/disco-corpus:  Speaker: "line text"
_GV_LABEL = re.compile(r'label\s*=\s*"(?P<speaker>[^:"]+):\s*\\"(?P<text>.*?)\\""', re.DOTALL)


def load_gv_corpus(gv_dir: Path, source: str = "mos9527") -> pd.DataFrame:
    """Parse mos9527/disco-corpus ``.gv`` files into the canonical frame.

    NOTE: the label regex above is a Week-1 Spike-A hypothesis — validate it
    against a real file before trusting it, and adjust (escaping in GraphViz
    labels varies).
    """
    rows: list[dict] = []
    for gv in sorted(gv_dir.glob("*.gv")):
        content = gv.read_text(encoding="utf-8", errors="replace")
        for i, m in enumerate(_GV_LABEL.finditer(content)):
            rows.append(
                {
                    "line_id": f"{gv.stem}:{i}",
                    "speaker": m["speaker"].strip(),
                    # GraphViz labels escape inner quotes as \" — unescape them.
                    "text": m["text"].replace('\\"', '"').strip(),
                    "source": source,
                }
            )
    return pd.DataFrame(rows)


def load_kaggle_csv(csv_path: Path) -> pd.DataFrame:
    """Adapt the Kaggle dump to the canonical frame. Fill in real column names
    during Week 1 once the file is in hand."""
    raise NotImplementedError("Week 1: map Kaggle columns -> line_id/speaker/text/source")


def clean(df: pd.DataFrame, min_words: int = 3, speakers: list[str] | None = None) -> pd.DataFrame:
    """Deduplicate, drop ultra-short lines, optionally restrict to a speaker shortlist."""
    out = df.dropna(subset=["speaker", "text"]).copy()
    out["text"] = out["text"].str.strip()
    out = out[out["text"].str.split().str.len() >= min_words]
    out = out.drop_duplicates(subset=["speaker", "text"])
    if speakers is not None:
        out = out[out["speaker"].isin(speakers)]
    return out.reset_index(drop=True)


def save_processed(df: pd.DataFrame, path: Path = PROCESSED) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)

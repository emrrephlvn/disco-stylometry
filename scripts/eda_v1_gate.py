"""V1 sufficiency gate on the full disco-corpus-en.

Run (after pulling the corpus into data/raw/, see data/README.md):
    python scripts/eda_v1_gate.py

Gates:
  1. per-character line counts
  2. line length per character -> >=10-word ratio (BARK RISK; short barks make
     per-line stylometry meaningless, so this is an identity check, not hygiene)
  3. class-imbalance ceiling
  4. 'You' exclusion + speaker alias canonicalization

Side effect: writes data/processed/lines.parquet (cleaned parse + speaker_canon).
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from discostyle.data.load import clean, load_gv_corpus, save_processed

GV_DIR = Path("data/raw/disco-corpus-src/graphviz/disco-corpus-en")
MIN_LINES = 150
MIN_GE10 = 0.5  # bark-risk threshold: >=50% of lines must be >=10 words
_WORD = re.compile(r"[a-zA-Z']+")

# NOTE: no prefix-based object filter — "The Deserter"/"The Gardener" are real
# characters, so blanket "The "/"A " exclusion nukes them. Object/book speakers
# (A Brief Look…, Shelf of…) are near-zero volume and fall below MIN_LINES on
# their own; the final shortlist is eyeballed for any stragglers instead.


def main() -> None:
    df = load_gv_corpus(GV_DIR)
    df["speaker_canon"] = df["speaker"].str.split(",").str[0].str.strip()
    df["nwords"] = df["text"].map(lambda s: len(_WORD.findall(s)))

    g = df.groupby("speaker_canon")
    stats = pd.DataFrame(
        {
            "lines": g.size(),
            "median_words": g["nwords"].median(),
            "ge10_ratio": g["nwords"].apply(lambda s: (s >= 10).mean()).round(2),
        }
    ).sort_values("lines", ascending=False)

    print(f"parsed {len(df)} lines; {df['speaker_canon'].nunique()} distinct speakers")
    print("\n[gate 1+2] top speakers (lines / median words / >=10-word ratio):")
    print(stats.head(20).to_string())

    # gate 4: exclude only 'You' (bark-heavy voices handled by the gate-2 filter)
    cand = stats.drop(index=[s for s in ["You"] if s in stats.index])
    shortlist = cand[(cand["lines"] >= MIN_LINES) & (cand["ge10_ratio"] >= MIN_GE10)]
    barks = cand[(cand["lines"] >= MIN_LINES) & (cand["ge10_ratio"] < MIN_GE10)]

    print(f"\n[gate 4] 'You' excluded ({stats.loc['You', 'lines']} lines, "
          f"ge10={stats.loc['You', 'ge10_ratio']} — heterogeneous player narration)")
    print(f"[gate 2] bark-heavy excluded: {list(barks.index)}")
    print(f"\nSHORTLIST ({len(shortlist)} classes, {int(shortlist['lines'].sum())} lines):")
    print(shortlist.to_string())
    if len(shortlist):
        print(f"\n[gate 3] imbalance ceiling: "
              f"{shortlist['lines'].max() / shortlist['lines'].min():.1f}x")

    out = clean(df[["line_id", "speaker", "text", "source"]])
    out["speaker_canon"] = out["speaker"].str.split(",").str[0].str.strip()
    save_processed(out)
    print(f"\nwrote data/processed/lines.parquet ({len(out)} lines)")


if __name__ == "__main__":
    main()

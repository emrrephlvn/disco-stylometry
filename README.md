# WHO SAID IT? — Disco Elysium Stylometry

> *Stylometric fingerprints and speaker attribution on the dialogue of Disco Elysium —
> can a model tell Kim Kitsuragi from Cuno from the Narrator by style alone?*

**Status:** Week 0 — scaffold. See [ROADMAP.md](ROADMAP.md) for the week-by-week plan.

## What this is

Disco Elysium's ~1M-word script is one of the most distinctively *voiced* corpora in games.
This project treats that as a measurable claim:

1. **Fingerprints** — classic, interpretable stylometry per character (lexical diversity,
   hapax rate, readability, function words, punctuation profile).
2. **Attribution** — a "who said this line?" classifier (TF-IDF baseline → CPU sentence
   embeddings), evaluated with scene-grouped splits and honest baselines.
3. **Atlas** — an interactive demo: paste a sentence, get a speaker verdict with
   probabilities and the stylistic evidence behind it.

**Stretch goal (V2):** recover the 24 *skill* voices (Logic, Inland Empire, …) from the raw
game asset and re-run everything as "which inner voice is speaking?" — gated by a Week-1
spike, never assumed. Rationale documented in [ROADMAP.md](ROADMAP.md).

## Layout

```
src/discostyle/
  data/       load.py (corpus loaders), skill_spike.py (V2 gate)
  features/   stylometry.py (interpretable fingerprints)
  models/     classifier.py (scene-split, TF-IDF & embedding rungs)
  viz/        fingerprint.py (radar + UMAP constellation)
app/          streamlit_app.py (interactive demo)
tests/        smoke tests
data/         NOT committed — see data/README.md
```

## Setup

```bash
python -m venv .venv && .venv\Scripts\activate   # Windows
pip install -e ".[dev,app]"                       # add [embeddings] from Week 3
pytest
```

## Results

*(Filled in as weeks complete — every claim here must trace to a metric or plot.)*

| Model | Macro-F1 | Accuracy | vs. majority baseline |
|---|---|---|---|
| Majority class | — | *TBD* | — |
| TF-IDF + LogReg | *TBD* | *TBD* | *TBD* |
| MiniLM embeddings + LogReg | *TBD* | *TBD* | *TBD* |

## Honesty & legal

- **The corpus is copyrighted (ZA/UM).** This repo ships code and acquisition
  instructions, never the game text itself.
- Known limitation going in: in easily available extractions, skill voices are folded
  into Narrator-like actors; character-level attribution is the verified-feasible core,
  skill-level is a spike-gated stretch.
- Non-affiliation: fan analysis project; not affiliated with ZA/UM.

# WHO SAID IT? — Disco Elysium Stylometry

> *Stylometric fingerprints and speaker attribution on the dialogue of Disco Elysium —
> can a model tell Kim Kitsuragi from Cuno from the Narrator by style alone?*

**Status:** Weeks 1–3 done (data, fingerprints, classifier). Week 4 (atlas + demo) in
progress. See [ROADMAP.md](ROADMAP.md) for the week-by-week plan.

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
scripts/      week-by-week runnable stages (EDA gate, fingerprints, classifier, atlas)
reports/      generated tables/figures committed as .md/.csv (regenerable .html ignored)
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

### Week 2 — voice fingerprints (interpretability test: PASS)

On the locked 12 speakers, each voice has a **distinct, explainable** stylometric fingerprint
(all 12 pairwise-distinct; most-confusable pair Kim ~ Klaasje, both measured/articulate). The
single most-distinctive feature per voice reads like the game:

| Voice | Signature (z-score) | Reads as |
|---|---|---|
| Cuno | `exclaim_rate` **+2.4** | the shouting kid |
| The Deserter | `ellipsis_rate` **+2.5** | dying communist, trailing off; longest lines (39 words) |
| Jean Vicquemare | `question_rate` **+2.0** | the interrogating cop |
| Joyce Messier | `flesch_reading_ease` **−1.4** | most complex, eloquent capitalist |
| Cuno | `flesch` **89.6** (simplest) | vs Joyce 73.9 |
| Noid | `hapax_rate` **+1.9** | idiosyncratic vocabulary |

Full table: [`reports/fingerprints.md`](reports/fingerprints.md) · radar: `reports/radar.html`
(regenerate with `python scripts/week2_fingerprints.py`).

### Week 3 — classifier

Scene-grouped split (a scene's lines never straddle train/test) + scene-aware class cap
(900 lines/class, by whole scenes, so the cap can't itself leak). Full write-up:
[`reports/week3_metrics.md`](reports/week3_metrics.md).

| Model | Tier-1 (12) macro-F1 / acc | Tier-2 (36) macro-F1 / acc |
|---|---|---|
| random (stratified) | 0.064 / 0.081 | 0.015 / 0.035 |
| majority class | 0.013 / 0.088 | 0.003 / 0.037 |
| **TF-IDF + LogReg** | **0.446 / 0.570** | 0.196 / 0.330 |
| MiniLM embeddings + LogReg | 0.388 / 0.490 | 0.156 / 0.234 |

**+0.432 macro-F1 over majority** — not a "kolay problem": going from 12 to 36 classes drops
TF-IDF macro-F1 from 0.446 to 0.196, which is what actually justifies the locked 12.

The fingerprint's *closest* pair, Kim ~ Klaasje (d=1.46, Week 2), is **not** the classifier's
hardest — only 5–8% cross-confusion. Lexical content pulls them apart where aggregate style
couldn't. The real confusions are same-archetype pairs: Lena/Klaasje → Joyce (Martinaise's
eloquent women collapse together), Jean → Titus Hardie (clipped working-class register).

#### Lesson learned: a falsified hypothesis

Going in, the expectation was "sentence embeddings should help most on short, context-poor
lines, where TF-IDF has little to grab onto." The data says otherwise: **TF-IDF beats MiniLM
embeddings on both short and long lines.**

| Line length | TF-IDF acc | MiniLM acc |
|---|---|---|
| short (<10 words, n=152) | **0.368** | 0.322 |
| long (≥10 words, n=1326) | **0.593** | 0.509 |

Reading: "who said it" in Disco Elysium rides *surface idiolect* — Cuno's slang, Evrart's
political register, Joyce's maritime diction — a signal sparse TF-IDF n-grams capture directly.
MiniLM's semantic pooling optimizes for meaning, not idiolect, and appears to blur exactly the
lexical fingerprint this task depends on. Reported as-is rather than reframed to fit the
original expectation.

### Week 4 — voice constellation + interactive demo

All 10,874 locked-12 lines, MiniLM embeddings → UMAP(2D). Full write-up:
[`reports/week4_atlas.md`](reports/week4_atlas.md) · interactive: `reports/constellation.html`.

The atlas cross-checks three independent notions of "similar" against each other rather than
asserting a story from the picture alone:

| Method | Feature space | Closest / most notable |
|---|---|---|
| Week 2 fingerprint | aggregate surface stats (punctuation, readability) | Kim ~ Klaasje |
| Week 3 classifier | lexical n-grams | Lena → Joyce (hardest confusion) |
| Week 4 UMAP | MiniLM sentence semantics | Klaasje ~ The Deserter (closest); **Cuno the clear outlier** (1.9× more isolated than the next voice) |

**The three methods disagree** — and that's the finding, not a failure: each captures a
different notion of vocal similarity (aggregate style vs. lexical choice vs. sentence
semantics), so "who sounds like whom" is method-dependent, not a single ground truth. The one
result that *does* triangulate cleanly: Cuno's semantic isolation is consistent with a child
character whose register (slang, hostility, non-adult concerns) reads as unlike every adult
voice in the corpus — not merely stylistically distinct, but about different things.

#### Interactive demo

```bash
python scripts/build_demo_model.py    # trains models/tfidf_logreg.joblib
streamlit run app/streamlit_app.py
```

Paste any line of dialogue and get both **WHO** (predicted speaker + probability across all 12
voices) and **WHY** (the specific n-grams — TF-IDF weight × learned coefficient — that pushed
the model toward that speaker). The WHY panel is the interpretability mandate made concrete: a
probability bar names a verdict, the evidence table justifies it.

![Demo screenshot](reports/demo_screenshot.png)

## Honesty & legal

- **The corpus is copyrighted (ZA/UM).** This repo ships code and acquisition
  instructions, never the game text itself.
- Known limitation going in: in easily available extractions, skill voices are folded
  into Narrator-like actors; character-level attribution is the verified-feasible core,
  skill-level is a spike-gated stretch.
- Non-affiliation: fan analysis project; not affiliated with ZA/UM.

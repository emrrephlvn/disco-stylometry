# WHO SAID IT? — Disco Elysium Stylometry

> *Stylometric fingerprints and speaker attribution on the dialogue of Disco Elysium —
> can a model tell Kim Kitsuragi from Cuno from the Narrator by style alone?*

**Status:** Weeks 1–5 done — data + fingerprints + character classifier (V1) + atlas + demo,
and V2 (24 skill voices) revived from a second corpus. See [ROADMAP.md](ROADMAP.md).

## What this is

Disco Elysium's ~1M-word script is one of the most distinctively *voiced* corpora in games.
This project treats that as a measurable claim:

1. **Fingerprints** — classic, interpretable stylometry per character (lexical diversity,
   hapax rate, readability, function words, punctuation profile).
2. **Attribution** — a "who said this line?" classifier (TF-IDF baseline → CPU sentence
   embeddings), evaluated with scene-grouped splits and honest baselines.
3. **Atlas** — an interactive demo: paste a sentence, get a speaker verdict with
   probabilities and the stylistic evidence behind it.

**V2 (revived — see Week 5):** the 24 *skill* voices (Logic, Inland Empire, …) as classes.
The Week-1 spike said NO-GO on the first corpus (mos9527 folds skills into "You"), but a
second source ([`disco.db`](https://github.com/msyavuz/disco-api)) stores each skill as a
separate actor — so V2 is real after all, as a 23-class classifier. The reversal is itself the
lesson: *the NO-GO was the data source's, not the task's.*

## Layout

```
src/discostyle/
  data/       load.py (corpus loaders), skill_spike.py (V2 gate)
  features/   stylometry.py (fingerprints), delta.py (Burrows' Delta baseline)
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

Every claim below traces to a committed metric or figure in [`reports/`](reports/); each week
has a runnable script in [`scripts/`](scripts/) that regenerates it.

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

*A sociolinguistic reading of that main confusion (interpretive, not a data claim):* Joyce,
Klaasje and Lena share an **elite, performative register** — the poised, hedging, self-aware
speech of characters who manage how they're perceived (a union negotiator, a fugitive
maintaining a cover, a grieving widow holding composure). The classifier confuses them because
that managed register *is* their shared surface, even though their situations differ. Jean and
Titus confuse for the mirror reason: a blunt, guarded working-class register.

**Classic-stylometry baseline (Burrows' Delta).** For literature grounding, the standard
function-word method (Burrows 2002; Mosteller & Wallace) on the same split reaches macro-F1
**0.18–0.23** (100–300 MFW) — a real signal (~18× majority) but well below the modern rungs.
It's the *interpretable floor*: every dimension is a common word, the distance a plain average.
See [`reports/delta_baseline.md`](reports/delta_baseline.md).

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

The atlas cross-checks **four independent** notions of "similar" against each other rather than
asserting a story from the picture alone:

| Method | Feature space | Closest / most notable |
|---|---|---|
| Week 2 fingerprint | aggregate surface stats (punctuation, readability) | Kim ~ Klaasje |
| Week 3 classifier | lexical n-grams | Lena → Joyce (hardest confusion) |
| Burrows' Delta | classic MFW z-scores | Joyce ~ Soona |
| Week 4 UMAP | MiniLM sentence semantics | Klaasje ~ The Deserter (closest); **Cuno the clear outlier** (1.9× more isolated than the next voice) |

**The four methods give four different answers** — and that's the finding, not a failure: each
captures a different notion of vocal similarity (aggregate style vs. lexical choice vs. classic
function-word profile vs. sentence semantics), so "who sounds like whom" is method-dependent,
not a single ground truth. The one result that *does* triangulate cleanly: Cuno's semantic
isolation is consistent with a child character whose register (slang, hostility, non-adult
concerns) reads as unlike every adult voice in the corpus — not merely stylistically distinct,
but about different things.

#### Interactive demo

Run locally (the trained model ships in the repo):

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

Or retrain from your own corpus copy: `python scripts/build_demo_model.py`.

**Hosted:** deployable as-is on Streamlit Community Cloud — repo root, main file
`app/streamlit_app.py`, Python 3.12 (`requirements.txt` pins `scikit-learn==1.9.0`
to match the committed model artifact; the pin matters, joblib files don't survive
sklearn version drift).

Paste any line of dialogue and get both **WHO** (predicted speaker + probability across all 12
voices) and **WHY** (the specific n-grams — TF-IDF weight × learned coefficient — that pushed
the model toward that speaker). The WHY panel is the interpretability mandate made concrete: a
probability bar names a verdict, the evidence table justifies it.

![Demo screenshot](reports/demo_screenshot.png)

### Week 5 — V2: which of the 24 inner voices is speaking?

The Week-1 spike declared skill-level attribution NO-GO — but that was the *first corpus*
(mos9527) collapsing every skill into the player actor. A second source,
[msyavuz/disco-api](https://github.com/msyavuz/disco-api)'s `disco.db`, stores the 24 skills as
**separate actors** (verified: Volition=397, Electrochemistry=405, …). 23 of 24 skills have
≥100 clean lines, so V2 is a real 23-class classifier — same pipeline as V1.

| Model | 23 skills, macro-F1 / acc |
|---|---|
| random / majority | 0.038 / 0.005 |
| TF-IDF + LogReg | 0.298 / 0.304 |
| **MiniLM embeddings + LogReg** | **0.322 / 0.330** |

**+0.293 over majority across 23 classes**, and harder than V1's 12 characters — as expected,
since six skills are "intellect", six "physique", etc. Two honest findings:

1. **Representation flips.** For *characters* TF-IDF won; for *skills* **embeddings win**.
   Characters are separated by surface idiolect (slang, register) that n-grams catch; skills by
   *what they talk about* (semantic content) that MiniLM catches. Best representation is
   task-dependent, not universal.
2. **Attribute structure is faint, not clean.** 29% of misclassifications stay within the true
   skill's attribute group (INT/PSY/FYS/MOT) vs ~23% by chance — a 1.27× lean, and the largest
   individual confusions are actually cross-attribute. The four-attribute grouping leaves a
   fingerprint in the errors, but the model does not cleanly recover it. (Reported as the modest
   effect it is, not inflated.)

Full write-up: [`reports/week5_skill_metrics.md`](reports/week5_skill_metrics.md).

## Related work

- **[xxond/disco-limbic-dialogue](https://github.com/xxond/disco-limbic-dialogue)** — LLM
  *generation* of inner-voice (Empathy, Inland Empire, Electrochemistry) dialogue. Orthogonal:
  this project does *attribution / stylometry*, not generation.
- **[Disco Narrator](https://152334h.github.io/blog/dn-1/)** (152334h) — TTS / voice synthesis
  on DE dialogue. Also generation-side, not analysis.
- **Stylometry / authorship attribution** (Mosteller & Wallace; Burrows' Delta) — the classic
  toolkit this leans on; the Delta/MFW baseline is implemented and reported above
  ([`reports/delta_baseline.md`](reports/delta_baseline.md)).

Positioning: prior DE-NLP work is generation-side; this is the attribution/interpretability
angle — "who/which voice, and *why*, measured."

## Limitations

Reported plainly, because a portfolio piece that hides its edges is less trustworthy than one
that names them:

- **These are honest baselines, not maxed numbers.** No hyperparameter search, one embedding
  model (MiniLM), default LogReg. The point is a clean, leak-free comparison — TF-IDF 0.446
  macro-F1 (V1) is what the *method* gets, not what a tuned system could.
- **Scene-grouped splitting has a cost.** It's the right call (adjacent lines leak topic), but
  characters whose lines cluster in a few conversations end up with thin test support — some
  per-class F1 rests on a handful of test lines. Error analysis filters classes with <15–20 test
  lines for exactly this reason; treat rare-class numbers as indicative, not precise.
- **Short lines are hard and partly excluded.** Bark-heavy voices (Cunoesse, and "You" itself)
  were dropped in Week 1 because per-line stylometry on 3–5 word interjections is noise. So the
  models are evaluated on the *tractable* slice, not every line in the game.
- **V2 is a harder, imbalanced problem, and the numbers say so.** 23 skills (Perception dropped
  at 20 lines), macro-F1 0.322 — well below V1's characters. Skills overlap semantically by
  design; this is not a solved task, it's an honest first pass.
- **"Who sounds like whom" has no single ground truth.** Four methods give four closest-pairs.
  That's a genuine finding, but it also means any single similarity claim is method-relative.
- **English only.** Stylometry is language-bound; the Russian DE corpus exists but isn't usable
  here. Findings don't transfer across localizations.
- **Interpretations are labeled as such.** The sociolinguistic readings of confusions are
  literary framing on top of the data, not measured claims — and marked that way in the text.

## Honesty & legal

- **The corpus is copyrighted (ZA/UM).** This repo ships code and acquisition
  instructions, never the game text itself (no `.gv`, no `disco.db`, no parquet).
- **One deliberate exception:** the trained demo model (`models/tfidf_logreg.joblib`, 4.9 MB)
  *is* committed so the hosted demo can run without rebuilding from the corpus. It contains
  TF-IDF statistics and LogReg coefficients — the vocabulary includes word 1–2-grams and
  character n-grams (dictionary-level fragments), but **no dialogue lines** and no way to
  reconstruct them. Comparable to publishing a concordance, not the text.
- Two corpora, two roles: **mos9527/disco-corpus** (character-labeled, V1) and
  **msyavuz/disco-api `disco.db`** (skill-labeled, V2). The Week-1 skill-attribution NO-GO
  was specific to the first source; the second exposes skills as separate actors.
- **Class imbalance and small classes are reported, not hidden** — Perception (20 lines) is
  dropped from V2; V2 macro-F1 is genuinely lower than V1 and said so plainly.
- Non-affiliation: fan analysis project; not affiliated with ZA/UM.

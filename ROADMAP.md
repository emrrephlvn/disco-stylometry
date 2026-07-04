# Roadmap — *Who Said It?* Disco Elysium Stylometry

**Budget:** ~10 h/week × 5 weeks (+1 buffer week).
**Core (V1):** character-level stylometry + "who said this line?" classifier — labels confirmed to exist (`actorName` in extracted dialogue data).
**Stretch (V2):** recover 24-skill attribution from the raw game asset and expand to the "24 inner voices" version. V2 is an *upside*, never a blocker.

---

## Week 1 — Data acquisition + the two spikes (GO/NO-GO gate)

**Goal:** a clean, deduplicated `lines.parquet` with columns `line_id, speaker, text, source` — and a verdict on the V2 spike.

- [ ] Obtain the corpus. Try in order (do **not** commit raw data — see legal note in README):
  1. **`mos9527/disco-corpus`** — primary. English, GraphViz `.gv`, node labels like
     `Kim Kitsuragi: "…"`. Structure inspected; needs a small parser (`load_gv_corpus`).
  2. jd7h gist (extracted texts) — backup.
  3. Kaggle `lizakonopelko/disco-elysium-dialogue-texts` — dataset *name not independently
     verified*; use only if it actually exists, else drop.
- [x] **V1 sufficiency gate — PASS (2026-07-04).** Full corpus parsed → 36,189 lines,
  156 speakers, `data/processed/lines.parquet` written (33,430 lines after `clean()`).
  Report: `scripts/eda_v1_gate.py`; locked config: `src/discostyle/config.py`.
  - **Gate 1 (counts):** 38 characters have ≥150 lines.
  - **Gate 2 (bark risk = ≥10-word ratio):** named characters 0.74–0.96 (stylometry-safe).
    Bark-heavy excluded: `You` (0.45) and `Cunoesse` (0.48, median 9 words) — *identity, not
    hygiene*: per-line stylometry on barks is noise.
  - **Gate 3 (imbalance):** 34.2× across all 38 (Kim 5262 dominates) → **cap classes at 900
    lines** (`MAX_LINES_PER_CLASS`), bringing the locked-12 ceiling to ~2.3×.
  - **Gate 4 ('You' + aliases):** `You` excluded (41%, heterogeneous narration). Comma-split
    canonicalization; first-token scan found **no true same-character duplicates** to merge
    ("The Deserter"/"The Gardener" are distinct characters, not aliases); object/book
    "speakers" fall below 150 lines on their own.
  - **LOCKED shortlist (Tier 1, 12):** Kim, Cuno, Joyce, Klaasje, Titus Hardie, Evrart,
    The Deserter, Jean Vicquemare, Garte, Lena, Noid, Soona. Tier 2 (full 38) documented.
- [x] **Spike A (kill-switch) — PASS (2026-07-04).** `mos9527/disco-corpus` exists (branch
  `main`), English `.gv` under `graphviz/disco-corpus-en/`. Label format
  `id [label="Speaker: \"text\""]` confirmed; `load_gv_corpus` **validated on the full
  1456-file corpus** (483 files carry dialogue, 973 are stub/door/object convos). Edge-case
  audit: only **6/1456 files** (objects/thoughts) hit a format variant the regex misses —
  V1-irrelevant, logged. Encoding verified clean UTF-8 (0 replacement chars; `é` etc. intact).
  Fixes applied: `rglob` (nested letter dirs), unescape `\"`, normalize `\n`/`\l`/`\r`.
- [x] **Spike B (V2 gate) — NO-GO for the mos9527 corpus (2026-07-04).** Skill voices are
  *technically* present as speakers but *practically absent*: in a 40-file sample, only **5
  skill-labeled lines** (0.6% of "You"), across 4 skills, 1–2 lines each → extrapolated
  **~45 lines/skill for ~4 skills** over the full corpus. Harry's introspection collapses to
  the `You` actor (~31k lines est.). ~45 lines/skill cannot train a 24-class classifier.
  - Earlier web-research pessimism (single narrator) was the *audio* layer; the *text* check
    independently confirms NO-GO for a different reason (introspection → `You`).
  - Residual V2 hope = the raw 266 MB game asset only (needs the user's game files). Given the
    well-regarded mos9527 extraction already collapsed skills to `You`, this is likely a
    source-data property, not an extraction gap. **Treat V2 as unlikely, not merely deferred.**
  - **Consequence:** V1 (character-level) is the project. Week 5 = polish + optional LLM layer.

**Exit criteria:** ✅ `data/processed/lines.parquet` written; ✅ both spike verdicts + V1
gate recorded above; ✅ shortlist locked in `src/discostyle/config.py`.

## Week 2 — Stylometric features + first pictures

**Goal:** every line (and every speaker aggregate) has a measurable fingerprint.

- [ ] Implement `features/stylometry.py`: type-token ratio, hapax rate, avg sentence/word length, readability (textstat), function-word frequencies, punctuation profile, char n-gram profile.
- [ ] **First output: per-speaker fingerprint table + radar** on the locked 12. Reading this
  *is* the interpretability mandate's first test — whether each voice has a distinct
  fingerprint — not a throwaway "sanity check". If Kim and Cuno look identical here, fix
  preprocessing before any model. (Quantify it too: pairwise fingerprint distance + each
  speaker's most-distinctive z-scored feature, so "they separate" is a number, not a vibe.)

> **Design contract — scene-aware cap (spans Week 2/3).** `MAX_LINES_PER_CLASS` must be applied
> by selecting whole *scenes* until the budget is hit, never by random line sampling. Random
> line-capping fragments scenes and muddies the unit of analysis, so it fights the
> scene-grouped split in Week 3 (same scene's lines must never straddle train/test). Cap =
> drop whole scenes; split = assign whole scenes. They compose only if both act on scenes.
> (Fingerprint means in Week 2 use *all* lines per speaker — the cap is a training-balance
> tool, irrelevant to aggregate stylometry.)

**Exit criteria:** fingerprint table + radar; separation shown numerically for ≥3 speakers.

> **Windows setup notes (found running this repo):** (1) the PyPI default
> `torch==2.12.1` wheel fails to load on this Windows/anaconda-3.12 box
> (`WinError 1114`, `c10.dll`) — pinned `torch<2.12` in `pyproject.toml`,
> verified working at `2.6.0+cpu`. (2) Some Windows consoles (cp1254 etc.)
> can't print `→`/`—`; scripts use plain ASCII (`->`) in `print()` output —
> written files stay UTF-8 and unaffected either way.

## Week 3 — The classifier

**Goal:** an honest, reportable accuracy number.

- [x] Baseline: TF-IDF (word + char n-grams) → LogisticRegression. Scene-grouped split +
  scene-aware cap. Macro-F1 0.446 / acc 0.570 (Tier-1, 12 classes).
- [x] Upgrade: MiniLM embeddings → LogReg. Macro-F1 0.388 / acc 0.490 — **loses to TF-IDF**,
  including on short lines (falsified the "embeddings help short text" hypothesis; see README).
- [x] **Tier-2 ablation:** 38→36 valid classes, macro-F1 0.446→0.196 — earns "why 12".
- [x] Evaluation: majority + random baselines, confusion matrix, Kim~Klaasje dedicated report
  (closest fingerprint pair, only 5–8% cross-confusion — content beats aggregate style).
- [x] Error analysis: 5 observations read off min-support-filtered confusion rates.

**Exit criteria:** ✅ `reports/week3_metrics.md` — metrics table, confusion matrix, hardest-pair
report, length-stratified TF-IDF-vs-embedding comparison, 5 error-analysis observations.

## Week 4 — Fingerprint atlas + interactive demo

**Goal:** the portfolio-facing artifact.

- [x] UMAP "voice constellation" (MiniLM embeddings, all 10,874 locked-12 lines) — cross-checked
  against Week 2 (fingerprint) and Week 3 (confusion) closest pairs rather than eyeballed. The
  three methods disagree (different notions of "similar"); Cuno is the one voice that
  triangulates as an outlier in semantic space specifically (1.9× the next-most-isolated voice).
- [x] Streamlit demo: paste a sentence → **WHO** (probability bars) + **WHY** (n-gram
  contribution = TF-IDF weight × LogReg coefficient) — the interpretability mandate made
  concrete, not just a verdict.
- [ ] "Voice fingerprint" page per character (radar + signature words via log-odds) — Week 2's
  radar already exists (`reports/radar.html`); a dedicated per-character demo page deferred,
  not required for the exit criteria below.

**Exit criteria:** ✅ demo verified end-to-end with Playwright (WHO/WHY panels render, zero
console errors) — screenshot in README.

## Week 5 — V2 or polish (decided by Spike B)

**If Spike B PASSED:** re-run the whole pipeline with 24 skill voices as classes; the headline becomes "which inner voice is speaking?". Expect heavier class imbalance — document it honestly.
**If Spike B FAILED:** optional LLM layer (small budget): given a user's sentence + predicted speaker, rewrite it *in that character's voice*. Strictly a garnish on top of measured results.

## Week 6 — Buffer + shipping

- [ ] Deploy demo (Streamlit Community Cloud or HF Spaces).
- [ ] README as a portfolio case study: problem → data → method → **numbers** → error analysis → limitations.
- [ ] Limitations section is mandatory and honest: corpus is copyrighted (not redistributed), Narrator class may absorb skill lines (if V2 failed), scene-level split rationale.

---

## Standing rules

1. **Never redistribute game text.** Repo ships code + download instructions, not the corpus.
2. **Numbers over vibes.** Every claim in the README traces to a metric or a plot.
3. **V2 is earned, not assumed.** No architecture may depend on skill labels existing.

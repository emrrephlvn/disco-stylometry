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
- [ ] EDA notebook: lines per speaker, length distributions, class imbalance. Decide the speaker shortlist (start with 4–6 well-represented characters: e.g. Kim, Cuno, Klaasje, Joyce, Narrator, Garte).
- [ ] **Spike A (kill-switch, ≤2 h):** confirm character labels are clean and consistent across the chosen source. If labels are dirty → switch source, not project.
- [ ] **Spike B (V2 gate, ≤3 h, timeboxed):** inspect the raw 266 MB asset JSON (or Disco Narrator's formatting notes) — is skill attribution (`LOGIC`, `INLAND EMPIRE`, …) recoverable as a speaker-like field?
  - **PASS** → V2 unlocked, plan it for Week 5.
  - **FAIL** → V1 continues unchanged; Week 5 becomes polish + optional LLM layer.

**Exit criteria:** `data/processed/lines.parquet` exists; Spike B verdict written into this file.

## Week 2 — Stylometric features + first pictures

**Goal:** every line (and every speaker aggregate) has a measurable fingerprint.

- [ ] Implement `features/stylometry.py`: type-token ratio, hapax rate, avg sentence/word length, readability (textstat), function-word frequencies, punctuation profile, char n-gram profile.
- [ ] Per-speaker fingerprint table + first radar/violin plots.
- [ ] Sanity check: do the fingerprints *already* separate speakers visually? (If Kim and Cuno look identical here, revisit preprocessing before touching models.)

**Exit criteria:** fingerprint table + 2–3 plots that visibly separate at least 3 speakers.

## Week 3 — The classifier

**Goal:** an honest, reportable accuracy number.

- [ ] Baseline: TF-IDF (word + char n-grams) → LogisticRegression. Stratified split **by dialogue scene, not by line** (adjacent lines leak context).
- [ ] Upgrade: sentence-transformers embeddings (CPU-friendly, e.g. `all-MiniLM-L6-v2`) → LogReg/SVM. Compare.
- [ ] Evaluation: per-class precision/recall, confusion matrix, macro-F1 vs. majority-class + random baselines.
- [ ] Error analysis: the confusions *are* the story ("the model mistakes Klaasje for the Narrator when she's performing").

**Exit criteria:** metrics table with baselines; confusion matrix; 5 written error-analysis observations.

## Week 4 — Fingerprint atlas + interactive demo

**Goal:** the portfolio-facing artifact.

- [ ] UMAP of line embeddings colored by speaker ("voice constellation" — lean into the DE noir palette).
- [ ] Streamlit app: paste any sentence → predicted speaker + probability bars + which stylometric features drove it.
- [ ] "Voice fingerprint" page per character (radar + signature words via log-odds).

**Exit criteria:** demo runs locally end-to-end; screenshots in README.

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

# Research Brief — "Disco Voices" → Gemini Deep Research

> Handoff for Gemini's deep-research / web-search mode. The project below is
> **real, ~3 weeks in, with verified data and results**. I need targeted web
> research — **not concept discussion** (design decisions are closed).

---

## 1. Project in one sentence

Disco Elysium dialogue → stylometric fingerprints per character + a
"who said this line?" classifier + an interactive demo. A portfolio
data-science project (NLP + interpretability + visualization).

## 2. Context — decisions are CLOSED (don't re-debate)

**Why this project** (alternatives were evaluated and eliminated over 4 rounds
of meta-discussion across multiple AI models):
- Dice/probability simulator — eliminated: not DE-specific, same in any RPG.
- Thought Cabinet optimizer — eliminated: trivial knapsack + subjective objective.
- Amnesia PGM (Bayesian identity reconstruction) — eliminated: hand-written
  priors = scenario, not empirical inference.
- Disco Voices won: DE-specific + real ground-truth + measurable + GPU-free +
  low-budget.

**Structure:**
- **V1 (the project, in progress):** character-level stylometry + "who said
  it?" classifier. Labels confirmed real and objective.
- **V2 (KILLED by data):** 24-skill "inner voices" (Logic, Inland Empire,
  Shivers, ...). Skills collapse into a single "You" actor in available
  corpora (~45 lines/skill) → cannot train a 24-class classifier.

**Constraints:** no GPU, limited LLM budget (LLM is garnish, never load-bearing),
~6 weeks × 10h, open-source only, English only (stylometry is language-bound).

## 3. What's ALREADY verified — do NOT re-search these

- **mos9527/disco-corpus** (GitHub, public archive): 1456 `.gv` files, English,
  speaker-labeled (`Kim Kitsuragi: "..."`). Downloaded, parsed → **36,189 lines,
  156 speakers**, `lines.parquet` built. This is our primary corpus and it works.
- **prideshaquana-jpg/Disco-Elysium-Russian-Dialogue-Corpus:** 271K lines, 47
  characters — **Russian only**, unusable for English stylometry.
- **Disco Narrator** (152334h blog series): a TTS/voice-synthesis project,
  orthogonal to stylometry. Not prior art for our analysis angle.
- **Final Cut voicing:** all 24 skills voiced by a single narrator (Lenval
  Brown) — the audio layer has no skill separation.
- **In-game UI** displays skill names per line ("LOGIC —", "INLAND EMPIRE:"),
  so skill attribution *exists* in the raw game data; it's just not exposed
  in the easy corpora. The raw game asset is a ~266 MB exported JSON.

---

## 4. Week-3 results (real, disk-verified — context only)

Scene-grouped split (leak prevention), scene-aware cap (900 lines/class),
12 locked characters (Kim, Cuno, Joyce, Klaasje, Titus, Evrart, The Deserter,
Jean, Garte, Lena, Noid, Soona). "You" (41%, heterogeneous) and Cunoesse
(bark-heavy) deliberately excluded.

| Model              | Tier-1 (12) macroF1/acc | Tier-2 (36) macroF1/acc |
|--------------------|------------------------|------------------------|
| random (stratified)| 0.064 / 0.081          | 0.015 / 0.035          |
| majority class     | 0.013 / 0.088          | 0.003 / 0.037          |
| TF-IDF + LogReg    | 0.446 / 0.570          | 0.196 / 0.330          |
| MiniLM embed       | 0.388 / 0.490          | 0.156 / 0.234          |

Gain over majority: +0.432. Hardest fingerprint pair (Kim~Klaasje, d=1.46):
only 5–8% confusion. Tier-2 ablation: macro-F1 drop −0.250 (justifies "why 12").

---

## 5. Research questions — RANKED by value

### Q1 (HIGHEST VALUE — could revive V2): Is there ANY Disco Elysium data
### source where the 24 skills appear as SEPARATE labeled speakers?

This is the single most valuable thing you could find. In the corpora we've
seen, the 24 skill voices (Logic, Encyclopedia, Rhetoric, Drama,
Conceptualization, Visual Calculus, Volition, Inland Empire, Empathy,
Authority, Esprit de Corps, Suggestion, Endurance, Pain Threshold, Physical
Instrument, Electrochemistry, Shivers, Half Light, Hand/Eye Coordination,
Perception, Reaction Speed, Savoir Faire, Interfacing, Composure) are folded
into one "You"/"Narrator" actor. But the in-game UI shows them as distinct,
so the attribution must exist *somewhere* in the raw data.

**Specifically search for:**
- Disco Elysium modding community / datamine resources that expose the raw
  dialogue structure (Articy / Pixel Crushers "Dialogue System" format).
- Fan tools that extract dialogue with skill attribution intact (does the
  "Disco Elysium Scribe" tool, or any Unity asset extractor, preserve the
  skill-speaker field?).
- GitHub repos / gists / blog posts that have parsed the raw 266 MB asset
  JSON and report whether `actorName` resolves to skill names vs. "You".
- Any HuggingFace / Kaggle / Zenodo dataset of DE dialogue with a
  skill/speaker column distinct from character.

**Verdict I need:** "skill attribution IS recoverable from source X, here's
how" (→ V2 revives) OR "after checking X/Y/Z, it genuinely isn't separable;
the game engine merges skill lines into the player actor at the data layer"
(→ V2 stays dead, confirmed). Either verdict is valuable; speculation is not.

### Q2 (HIGH — reproducibility): Does the Kaggle dataset
### `lizakonopelko/disco-elysium-dialogue-texts` actually exist?

We flagged it as "dataset name not independently verified." Confirm: does it
exist on Kaggle? If yes — structure (columns, row count, speaker-labeled?
skill-labeled?), license, English? If it's a real English speaker-labeled
dump, it's a useful backup corpus.

### Q3 (MEDIUM — novelty positioning): Prior art for stylometry /
### character-attribution on DE, or on video-game dialogue generally?

We know of two tangential projects: Disco Narrator (TTS, orthogonal) and
macakuaya/DiscoElysiumSkills (tangential). **Search for:**
- Any published NLP/stylometry work specifically on Disco Elysium's text.
- Character/speaker attribution on video-game dialogue corpora (any game) —
  methods, datasets, results, to position our work and cite honestly.
- Authorship-attribution / stylometric surveys to cite in the README's
  related-work section.

**Why this matters:** for the portfolio README and interviews, "here's the
related work and how mine differs" is far stronger than "I think this is novel."

### Q4 (MEDIUM — narrative enrichment): Literary/critical analysis of how
### specific DE characters speak?

Our error analysis reads confusions as character insights ("the model
mistakes Klaasje for Joyce because both are eloquent performers"). We want to
ground these in actual criticism, not just our interpretation. **Search for:**
- Essays/criticism/reviews analyzing DE's character voices (Kim's formality,
  Cuno's argot, Klaasje's evasion/performative speech, Evrart's manipulative
  rhetoric, The Deserter's fragmented monologue).
- Interviews with writer Robert Kurvitz on differentiating character voice.
- Any linguistic/literary analysis of the game's prose.

**Why this matters:** error analysis grounded in real criticism is defensible
in an interview; "I noticed X and critic Y says the same" beats "I noticed X."

### Q5 (LOW — alternative corpora): Is there a cleaner/bigger English DE
### dialogue dump than mos9527/disco-corpus?

mos9527 works (36K lines, 156 speakers) but has ~973 stub/door/object files
and 6 edge-case files the regex misses. A cleaner structured dump (CSV/JSON
with speaker + scene metadata) would simplify the pipeline. Low priority —
only worth it if clearly better and English.

---

## 6. What I do NOT need

- **Concept alternatives.** Six were considered; five eliminated; one chosen.
- **"Have you considered X?"** for project direction. Execution is underway.
- **General ML/NLP advice.** Execution is underway with deliberate choices.
- **Opinion on whether the project is "worth doing."** 3 weeks in already.

## 7. How to report back

For each question (Q1–Q5), give me:
- **Found / Not found** verdict.
- **Sources** (URLs, repo names, dataset names) — concrete, not vague.
- **One-line implication** for the project (e.g., "Q1 found → V2 revives,
  use source X"; "Q3 found → cite papers A/B in README").


# Week 5 — V2: which of the 24 inner voices is speaking?

Corpus: msyavuz/disco-api `disco.db` (skills as separate actors). 23 skills with >= 100 lines (Perception dropped, 20 lines). Same pipeline as V1 (scene-aware cap, scene-grouped split).


## Metrics (macro-F1 / accuracy)

| model                 | 23 skills (macroF1/acc)   |
|:----------------------|:--------------------------|
| random (stratified)   | 0.038 / 0.043             |
| majority class        | 0.005 / 0.058             |
| TF-IDF + LogReg       | 0.298 / 0.304             |
| MiniLM embed + LogReg | 0.322 / 0.330             |


**Gain over majority:** +0.293 macro-F1 across 23 classes — lower than V1's 12-character 0.446, as expected: the skills overlap semantically (six 'intellect' voices, six 'physique' voices, ...).


**Representation flips vs V1.** Here **embeddings beat TF-IDF** (0.322 vs 0.298, +0.025) — the opposite of V1, where TF-IDF won. Reading: characters are separated by *surface idiolect* (slang, register) that sparse n-grams catch; skills are separated more by *what they talk about* (semantic content) that MiniLM catches. Best representation is task-dependent, not universal.


**Attribute structure (weak but present):** 29% of misclassifications land on a same-attribute skill (INT/PSY/FYS/MOT) vs ~23% by chance — only a 1.27x lean, and the *largest* individual confusions are cross-attribute (e.g. Hand/Eye Coordination -> Visual Calculus). So the four-attribute grouping leaves a faint fingerprint in the errors, but the model does not cleanly recover it.


## Top skill confusions

| true                  | predicted         |   rate | same_attr   |
|:----------------------|:------------------|-------:|:------------|
| Hand/Eye Coordination | Visual Calculus   |   0.2  | False       |
| Composure             | Empathy           |   0.16 | False       |
| Reaction Speed        | Logic             |   0.16 | False       |
| Suggestion            | Empathy           |   0.16 | True        |
| Visual Calculus       | Shivers           |   0.15 | False       |
| Authority             | Esprit de Corps   |   0.13 | True        |
| Hand/Eye Coordination | Conceptualization |   0.13 | False       |
| Pain Threshold        | Endurance         |   0.13 | True        |
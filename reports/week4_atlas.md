# Week 4 — voice constellation

10874 lines, 12 speakers, MiniLM embeddings -> UMAP(2D). Interactive plot: `reports/constellation.html`.


## Nearest-neighbor centroid distances (semantic space)

|                                    | nearest                            |   distance |
|:-----------------------------------|:-----------------------------------|-----------:|
| Klaasje (Miss Oranje Disco Dancer) | The Deserter                       |   0.134748 |
| The Deserter                       | Klaasje (Miss Oranje Disco Dancer) |   0.134748 |
| Garte                              | Noid                               |   0.145964 |
| Noid                               | Garte                              |   0.145964 |
| Soona                              | Klaasje (Miss Oranje Disco Dancer) |   0.267978 |
| Kim Kitsuragi                      | Joyce Messier                      |   0.305076 |
| Joyce Messier                      | Kim Kitsuragi                      |   0.305076 |
| Jean Vicquemare                    | Joyce Messier                      |   0.428235 |
| Lena                               | Garte                              |   0.698424 |
| Titus Hardie                       | The Deserter                       |   0.723468 |
| Evrart Claire                      | Jean Vicquemare                    |   1.71319  |
| Cuno                               | Lena                               |   3.3269   |


**Tightest 3-way cluster:** Klaasje (Miss Oranje Disco Dancer), The Deserter, Soona (min total pairwise centroid distance)

**Most isolated voice:** Cuno


## Cross-check against Weeks 2 and 3

- Week 2 (aggregate stylometry) closest pair: **Kim Kitsuragi ~ Klaasje (Miss Oranje Disco Dancer)**
- Week 4 (semantic embedding) closest pair: **Klaasje (Miss Oranje Disco Dancer) ~ The Deserter** — agrees with Week 2: **False**
- Week 3 (lexical classifier) hardest confusion: **Lena -> Joyce Messier**, both inside the Week-4 tightest 3-way cluster ('Klaasje (Miss Oranje Disco Dancer)', 'The Deserter', 'Soona'): **False**


**Reading the disagreement:** the three methods measure different notions of 'similar' and are not expected to agree — that's a finding, not a failure. Week 2 aggregates surface statistics (punctuation rates, readability) across a whole voice; Week 3 asks what confuses a lexical (word/char n-gram) classifier; Week 4 asks what MiniLM's sentence semantics place close together. Each disagreement is informative on its own terms — here they diverge, so treat 'who sounds like whom' as method-dependent, not a single ground truth. One point where the semantic view adds something the others don't: **Cuno** is the clear outlier in embedding space (nearest-neighbor distance 3.33, 1.9x the next-most-isolated voice) — consistent with a child character whose register (slang, hostility, non-adult concerns) is semantically unlike every other voice in the corpus, not just stylistically distinct.

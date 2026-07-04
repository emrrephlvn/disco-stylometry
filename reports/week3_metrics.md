# Week 3 — classifier

## Metrics (macro-F1 / accuracy)

| model                 | Tier-1 (12) macroF1/acc   | Tier-2 (36) macroF1/acc   |
|:----------------------|:--------------------------|:--------------------------|
| random (stratified)   | 0.064 / 0.081             | 0.015 / 0.035             |
| majority class        | 0.013 / 0.088             | 0.003 / 0.037             |
| TF-IDF + LogReg       | 0.446 / 0.570             | 0.196 / 0.330             |
| MiniLM embed + LogReg | 0.388 / 0.490             | 0.156 / 0.234             |


**Gain over majority (Tier-1 TF-IDF):** +0.432 macro-F1. **Ablation** 12->36 classes: 0.446->0.196 macro-F1 — the drop *earns* the locked shortlist.


## Hardest pair (Kim ~ Klaasje)
```
=== hardest pair: Kim ~ Klaasje (fingerprint d=1.46) ===
  Klaasje        -> predicted Kim Kitsuragi : 11/137 (8%)  | correct 47%
  Kim Kitsuragi  -> predicted Klaasje       : 6/130 (5%)  | correct 50%
```


## TF-IDF vs embeddings, by line length

| stratum      |    n |   tfidf_acc |   embed_acc |
|:-------------|-----:|------------:|------------:|
| short (<10w) |  152 |       0.368 |       0.322 |
| long (>=10w) | 1326 |       0.593 |       0.509 |


## Top confusions (Tier-1, TF-IDF)

| true                               | predicted     |   rate |   n |   support |
|:-----------------------------------|:--------------|-------:|----:|----------:|
| Lena                               | Joyce Messier |   0.36 |  38 |       106 |
| Jean Vicquemare                    | Titus Hardie  |   0.17 |  26 |       149 |
| Klaasje (Miss Oranje Disco Dancer) | Joyce Messier |   0.14 |  19 |       137 |
| Noid                               | Kim Kitsuragi |   0.14 |   9 |        66 |
| Kim Kitsuragi                      | The Deserter  |   0.12 |  16 |       130 |
| Jean Vicquemare                    | Kim Kitsuragi |   0.11 |  17 |       149 |


## Error analysis
1. The hardest confusions are same-archetype pairs, not stylometric twins: Lena, Klaasje and Kim all bleed toward **Joyce** — Martinaise's articulate, worldly women collapse together in lexical space (Lena->Joyce 36%, Klaasje->Joyce 14%).
2. The two hard men confuse: **Jean Vicquemare->Titus Hardie (17%)** — clipped, aggressive working-class register reads alike whether it's Harry's estranged partner or the union boss.
3. **Kim ~ Klaasje** — the fingerprint's *closest* pair (d=1.46) — is NOT the classifier's hardest (only 5–8% cross-error). Content pulls them apart where aggregate style couldn't: Kim's procedural police vocabulary vs Klaasje's evasive charm. Sparse content > mean style.
4. **Kim->The Deserter (12%)**: a cop and a communist sniper share the game's longest, most measured monologues — length and calm cadence are a surface even ideology doesn't break.
5. **Embeddings lose to TF-IDF** (-0.058 macro-F1): 'who said it' in DE rides surface idiolect — Cuno's slang, Evrart's political register, Joyce's maritime diction — which sparse TF-IDF captures directly and MiniLM's semantic pooling blurs. See the length table: embeddings do not rescue short lines here.
# Burrows' Delta — classic-stylometry baseline (V1, 12 characters)

Same split as Week 3 (scene-aware cap + scene-grouped split). Delta = mean absolute z-difference over the top-N most-frequent words; nearest author centroid wins.


## Delta accuracy vs modern rungs

|   n_mfw |   macro_f1 |   accuracy |
|--------:|-----------:|-----------:|
|     100 |      0.163 |      0.301 |
|     150 |      0.179 |      0.34  |
|     300 |      0.229 |      0.405 |


**Reading:** Delta peaks at macro-F1 **0.229** (n_mfw=300), well below Week 3's TF-IDF (0.446) and MiniLM (0.388) but far above majority (0.013). That gap is the point: the ~1930s–2000s function-word method built for whole books is a real signal on single dialogue lines, yet modern sparse/dense features clearly earn their added complexity here. Delta stays the **interpretable floor** — every dimension is a common word, the distance is a plain average — so it anchors the interpretability mandate the fancier rungs can't fully honor.


**Delta-closest character pair:** Joyce Messier ~ Soona — a fourth, literature-standard notion of 'similar' for the Weeks-2/4 cross-check (Week 2 fingerprint: Kim~Klaasje; Week 4 UMAP: Klaasje~The Deserter). Three-plus methods, three-plus answers — vocal similarity is method-dependent.

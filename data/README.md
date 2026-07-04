# data/ — not committed

Disco Elysium's text is copyrighted by ZA/UM. This repo ships **code, not corpus**.

Expected layout after you run the Week-1 acquisition steps (see `ROADMAP.md`):

```
data/
  raw/          # whatever you downloaded (Kaggle csv, jd7h dump, mos9527 .gv files)
  processed/
    lines.parquet   # columns: line_id, speaker, text, source
```

Sources to try, in order:
1. **mos9527/disco-corpus** (primary) — GraphViz `.gv`, `Speaker: "line"` node labels
2. jd7h gist (backup): https://gist.github.com/jd7h/e724eb2b23faa42b51424ac110c7b976
3. Kaggle `lizakonopelko/disco-elysium-dialogue-texts` — name unverified; use only if it exists

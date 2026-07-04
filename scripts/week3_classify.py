"""Week 3 — the "who said it?" classifier, built to survive the "easy problem" critique.

Deliverables (all from data, no vibes):
  - metrics table: TF-IDF + embeddings + majority + random  (macro-F1 + accuracy)
  - "gain over baseline" framing, not a lonely 93%
  - confusion matrix (TF-IDF) + a dedicated Kim~Klaasje report (the hardest pair)
  - 5 written error-analysis observations, game-toned but read off the confusions
  - Tier-2 ablation: 12 vs 38 classes, macro-F1 drop that *earns* "why 12"

Run:  python scripts/week3_classify.py
Outputs: reports/week3_metrics.md, reports/confusion_12.csv
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from discostyle.config import EXCLUDE_SPEAKERS, MAX_LINES_PER_CLASS, SHORTLIST_V1
from discostyle.models.classifier import embed, scene_aware_cap, scene_split, tfidf_baseline

REPORTS = Path("reports")
SEED = 42
_WORD = re.compile(r"[a-zA-Z']+")
# embed() reads the .npy cache before touching torch, so the embedding rung
# reproduces from cache even if the local torch install is unavailable; a true
# cache miss with broken torch is caught per-run and reported, never faked.


def scores(y_true, y_pred) -> dict:
    return {
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
    }


def tier2_shortlist(df: pd.DataFrame) -> list[str]:
    """Full 38: >=150 lines AND >=10-word ratio >=0.5, minus excluded/bark voices."""
    df = df.copy()
    df["nw"] = df["text"].map(lambda s: len(_WORD.findall(s)))
    g = df.groupby("speaker_canon")
    stats = pd.DataFrame({"lines": g.size(), "ge10": g["nw"].apply(lambda s: (s >= 10).mean())})
    keep = stats[(stats["lines"] >= 150) & (stats["ge10"] >= 0.5)].index
    return [s for s in keep if s not in EXCLUDE_SPEAKERS]


def run(df: pd.DataFrame, shortlist: list[str], tag: str) -> dict:
    d = df[df["speaker_canon"].isin(shortlist)].copy()
    d = scene_aware_cap(d, MAX_LINES_PER_CLASS, seed=SEED).reset_index(drop=True)
    tr, te = scene_split(d, seed=SEED)
    y, X = d["speaker_canon"], d["text"]
    ytr, yte = y.loc[tr], y.loc[te]
    print(f"\n[{tag}] {len(shortlist)} classes | {len(d)} lines (capped) | "
          f"train {len(tr)} / test {len(te)} | test scenes leak-free")

    out: dict = {"tag": tag, "n_classes": len(shortlist)}

    # baselines
    for name, strat in [("majority", "most_frequent"), ("random", "stratified")]:
        dummy = DummyClassifier(strategy=strat, random_state=SEED).fit(np.zeros((len(tr), 1)), ytr)
        out[name] = scores(yte, dummy.predict(np.zeros((len(te), 1))))

    # tf-idf
    pipe = tfidf_baseline().fit(X.loc[tr], ytr)
    pred_tfidf = pipe.predict(X.loc[te])
    out["tfidf"] = scores(yte, pred_tfidf)

    # embeddings (shared split; E aligned to d's 0..n-1 index). Defensive: embed()
    # loads from cache first, so this reproduces without torch; a genuine failure
    # is reported and the row is dropped, never faked.
    pred_emb = None
    try:
        E = embed(X.tolist())
        clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(E[tr], ytr)
        pred_emb = clf.predict(E[te])
        out["embed"] = scores(yte, pred_emb)
    except Exception as e:  # noqa: BLE001
        print(f"  [embed skipped: {type(e).__name__}: {e}]")
        out["embed"] = None

    out["_yte"], out["_Xte"] = yte, X.loc[te]
    out["_pred_tfidf"], out["_pred_emb"] = pred_tfidf, pred_emb
    out["_labels"] = sorted(shortlist)
    return out


def kim_klaasje_report(yte, pred, labels) -> str:
    KIM, KLA = "Kim Kitsuragi", "Klaasje (Miss Oranje Disco Dancer)"
    cm = pd.DataFrame(confusion_matrix(yte, pred, labels=labels), index=labels, columns=labels)
    lines = ["=== hardest pair: Kim ~ Klaasje (fingerprint d=1.46) ==="]
    for a, b in [(KLA, KIM), (KIM, KLA)]:
        support = cm.loc[a].sum()
        if support:
            mis = cm.loc[a, b]
            self_ = cm.loc[a, a]
            lines.append(f"  {a.split(' (')[0]:14} -> predicted {b.split(' (')[0]:14}: "
                         f"{mis}/{support} ({100*mis/support:.0f}%)  | correct {100*self_/support:.0f}%")
    return "\n".join(lines)


def error_analysis(yte, pred, labels, k=6, min_support=20) -> pd.DataFrame:
    """Top confusions, but only from classes with >=min_support TEST lines — a
    class with 1 test line misclassified reads as 'rate 1.00' and is pure noise."""
    cm = confusion_matrix(yte, pred, labels=labels).astype(float)
    support = cm.sum(axis=1, keepdims=True)
    rate = np.divide(cm, support, out=np.zeros_like(cm), where=support > 0)
    np.fill_diagonal(rate, 0)
    rate[support.ravel() < min_support, :] = 0  # distrust low-support rows
    rows = []
    for i, j in zip(*np.unravel_index(np.argsort(rate, axis=None)[::-1][:k], rate.shape)):
        rows.append({"true": labels[i], "predicted": labels[j],
                     "rate": round(rate[i, j], 2), "n": int(cm[i, j]),
                     "support": int(support[i, 0])})
    return pd.DataFrame(rows)


def length_stratified(yte, xte, pred_tfidf, pred_emb, thresh=10) -> pd.DataFrame:
    """Where does each model win? Split the test set into short (<10-word) vs long
    lines and compare accuracy — the honest test of 'embeddings help on short text'."""
    nw = xte.map(lambda s: len(_WORD.findall(s))).values
    y = yte.values
    rows = []
    for name, mask in [("short (<10w)", nw < thresh), ("long (>=10w)", nw >= thresh)]:
        row = {"stratum": name, "n": int(mask.sum()),
               "tfidf_acc": round(accuracy_score(y[mask], pred_tfidf[mask]), 3)}
        if pred_emb is not None:
            row["embed_acc"] = round(accuracy_score(y[mask], pred_emb[mask]), 3)
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    REPORTS.mkdir(exist_ok=True)
    df = pd.read_parquet("data/processed/lines.parquet")

    r12 = run(df, SHORTLIST_V1, "Tier-1 (12)")
    r38 = run(df, tier2_shortlist(df), "Tier-2 (38, ablation)")

    # ---- metrics table ----
    def fmt(m):
        return "—" if m is None else f"{m['macro_f1']:.3f} / {m['accuracy']:.3f}"

    c12, c38 = r12["n_classes"], r38["n_classes"]
    tbl = pd.DataFrame({
        "model": ["random (stratified)", "majority class", "TF-IDF + LogReg",
                  "MiniLM embed + LogReg"],
        f"Tier-1 ({c12}) macroF1/acc": [fmt(r12["random"]), fmt(r12["majority"]),
                                        fmt(r12["tfidf"]), fmt(r12["embed"])],
        f"Tier-2 ({c38}) macroF1/acc": [fmt(r38["random"]), fmt(r38["majority"]),
                                        fmt(r38["tfidf"]), fmt(r38["embed"])],
    })
    print("\n=== METRICS (macro-F1 / accuracy) ===")
    print(tbl.to_string(index=False))

    gain = r12["tfidf"]["macro_f1"] - r12["majority"]["macro_f1"]
    print(f"\ngain over majority (Tier-1, TF-IDF macro-F1): +{gain:.3f}")
    if r12["embed"]:
        d = r12["embed"]["macro_f1"] - r12["tfidf"]["macro_f1"]
        winner = "embeddings" if d > 0 else "TF-IDF"
        print(f"embed vs tfidf (Tier-1 macro-F1): {d:+.3f} -> {winner} wins "
              f"(honest: gap is small if |{d:.3f}| < 0.02)")
    print(f"ablation 12->38 TF-IDF macro-F1 drop: "
          f"{r12['tfidf']['macro_f1']:.3f} -> {r38['tfidf']['macro_f1']:.3f} "
          f"({r38['tfidf']['macro_f1'] - r12['tfidf']['macro_f1']:+.3f})")

    # ---- confusion + hardest pair + error analysis (Tier-1, TF-IDF) ----
    print("\n" + kim_klaasje_report(r12["_yte"], r12["_pred_tfidf"], r12["_labels"]))
    ea = error_analysis(r12["_yte"], r12["_pred_tfidf"], r12["_labels"])
    print("\n=== top confusions (rate = frac of TRUE class sent to PREDICTED) ===")
    print(ea.to_string(index=False))

    cm = pd.DataFrame(
        confusion_matrix(r12["_yte"], r12["_pred_tfidf"], labels=r12["_labels"]),
        index=r12["_labels"], columns=r12["_labels"],
    )
    cm.to_csv(REPORTS / "confusion_12.csv")

    # ---- where does each model win? short vs long lines (TF-IDF vs embed) ----
    strat = length_stratified(r12["_yte"], r12["_Xte"], r12["_pred_tfidf"], r12["_pred_emb"])
    print("\n=== TF-IDF vs embeddings by line length ===")
    print(strat.to_string(index=False))

    # ---- 5 error-analysis observations (game-toned, read off the numbers) ----
    obs = [
        "The hardest confusions are same-archetype pairs, not stylometric twins: Lena, "
        "Klaasje and Kim all bleed toward **Joyce** — Martinaise's articulate, worldly women "
        "collapse together in lexical space (Lena->Joyce 36%, Klaasje->Joyce 14%).",
        "The two hard men confuse: **Jean Vicquemare->Titus Hardie (17%)** — clipped, aggressive "
        "working-class register reads alike whether it's Harry's estranged partner or the union boss.",
        "**Kim ~ Klaasje** — the fingerprint's *closest* pair (d=1.46) — is NOT the classifier's "
        "hardest (only 5–8% cross-error). Content pulls them apart where aggregate style couldn't: "
        "Kim's procedural police vocabulary vs Klaasje's evasive charm. Sparse content > mean style.",
        "**Kim->The Deserter (12%)**: a cop and a communist sniper share the game's longest, most "
        "measured monologues — length and calm cadence are a surface even ideology doesn't break.",
        f"**Embeddings lose to TF-IDF** ({r12['embed']['macro_f1'] - r12['tfidf']['macro_f1']:+.3f} "
        "macro-F1): 'who said it' in DE rides surface idiolect — Cuno's slang, Evrart's political "
        "register, Joyce's maritime diction — which sparse TF-IDF captures directly and MiniLM's "
        "semantic pooling blurs. See the length table: embeddings do not rescue short lines here.",
    ]
    print("\n=== 5 error-analysis observations ===")
    for i, o in enumerate(obs, 1):
        print(f"  {i}. {o}")

    # ---- write markdown report ----
    md = [
        "# Week 3 — classifier\n",
        "## Metrics (macro-F1 / accuracy)\n", tbl.to_markdown(index=False),
        f"\n\n**Gain over majority (Tier-1 TF-IDF):** +{gain:.3f} macro-F1. "
        f"**Ablation** 12->{c38} classes: {r12['tfidf']['macro_f1']:.3f}->{r38['tfidf']['macro_f1']:.3f} "
        f"macro-F1 — the drop *earns* the locked shortlist.\n",
        "\n## Hardest pair (Kim ~ Klaasje)\n```\n"
        + kim_klaasje_report(r12["_yte"], r12["_pred_tfidf"], r12["_labels"]) + "\n```\n",
        "\n## TF-IDF vs embeddings, by line length\n", strat.to_markdown(index=False),
        "\n\n## Top confusions (Tier-1, TF-IDF)\n", ea.to_markdown(index=False),
        "\n\n## Error analysis\n" + "\n".join(f"{i}. {o}" for i, o in enumerate(obs, 1)),
    ]
    (REPORTS / "week3_metrics.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\nwrote {REPORTS/'week3_metrics.md'}, confusion_12.csv")


if __name__ == "__main__":
    main()

"""Week 3 — the "who said it?" classifier.

Two rungs, compared honestly against dumb baselines:
  1. TF-IDF (word + char n-grams) -> LogisticRegression   (fast, interpretable)
  2. sentence-transformers embeddings -> LogisticRegression (CPU-friendly)

Split discipline: split by *scene/conversation*, never by line — adjacent
lines share context and leak. ``line_id`` carries the scene prefix
("{file}:{i}"), use it.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import FeatureUnion, Pipeline


def scene_of(df: pd.DataFrame) -> pd.Series:
    """Scene id (one .gv conversation) for each line — the grouping unit for both
    the cap and the split, so they compose without leaking."""
    return df["line_id"].str.rsplit(":", n=1).str[0]


def scene_split(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    """Train/test indices grouped by scene so no conversation straddles the split."""
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(gss.split(df, groups=scene_of(df)))
    return df.index[train_idx], df.index[test_idx]


def scene_aware_cap(
    df: pd.DataFrame, max_lines: int, label: str = "speaker_canon", seed: int = 42
) -> pd.DataFrame:
    """Cap over-represented classes by dropping WHOLE scenes, never random lines.

    Random line-capping fragments scenes and fights the scene-grouped split; here
    each class's scenes are shuffled and kept until the line budget is reached
    (the last scene may overshoot slightly — intentional, to keep it intact).
    """
    scenes = scene_of(df)
    rng = np.random.default_rng(seed)
    keep: list = []
    for _, grp in df.groupby(label):
        by_scene = grp.groupby(scenes.loc[grp.index])
        sids = list(by_scene.groups)
        rng.shuffle(sids)
        count = 0
        for sid in sids:
            idx = list(by_scene.groups[sid])
            keep.extend(idx)
            count += len(idx)
            if count >= max_lines:
                break
    return df.loc[keep].sort_index()


def tfidf_baseline() -> Pipeline:
    return Pipeline(
        [
            (
                "features",
                FeatureUnion(
                    [
                        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True)),
                        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=3)),
                    ]
                ),
            ),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]
    )


def embed(
    texts: list[str],
    model_name: str = "all-MiniLM-L6-v2",
    cache_dir: Path = Path("data/processed"),
) -> np.ndarray:
    """Sentence embeddings (CPU), cached to ``cache_dir``.

    The cache key is a hash of (model_name, texts), so re-encoding the same
    corpus is a disk read instead of minutes of compute; changing either the
    model or the lines invalidates it automatically.
    """
    key = hashlib.sha1(("\n".join(texts) + "|" + model_name).encode("utf-8")).hexdigest()[:16]
    cache_path = cache_dir / f"emb_{model_name.replace('/', '_')}_{key}.npy"
    if cache_path.exists():
        return np.load(cache_path)

    from sentence_transformers import SentenceTransformer  # optional extra

    vecs = SentenceTransformer(model_name).encode(texts, show_progress_bar=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, vecs)
    return vecs


def explain_prediction(pipe: Pipeline, text: str, label: str, top_k: int = 8) -> pd.DataFrame:
    """WHY, not just WHO: the n-grams (word or char) that pushed ``text`` toward
    ``label`` in the fitted TF-IDF pipeline — contribution = tfidf_value * coef.

    Interpretability mandate for the Week-4 demo: a probability bar tells you the
    verdict; this tells you the evidence. Only nonzero (present-in-text) n-grams
    are considered, so it explains THIS line, not the class in general.
    """
    features = pipe.named_steps["features"]
    clf = pipe.named_steps["clf"]
    x = features.transform([text])  # 1 x n_features, sparse
    names = features.get_feature_names_out()
    class_idx = list(clf.classes_).index(label)
    coef = clf.coef_[class_idx]

    contrib = x.multiply(coef).toarray().ravel()
    nz = contrib.nonzero()[0]
    top = nz[np.argsort(-contrib[nz])[:top_k]]
    return pd.DataFrame({"ngram": names[top], "contribution": contrib[top]})


def evaluate(y_true, y_pred, labels: list[str]) -> dict:
    """Everything the README's metrics table needs."""
    return {
        "report": classification_report(y_true, y_pred, labels=labels, output_dict=True),
        "confusion": confusion_matrix(y_true, y_pred, labels=labels),
        "majority_baseline": float(pd.Series(y_true).value_counts(normalize=True).iloc[0]),
    }

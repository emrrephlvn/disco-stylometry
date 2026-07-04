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


def scene_split(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    """Train/test indices grouped by scene so no conversation straddles the split."""
    groups = df["line_id"].str.rsplit(":", n=1).str[0]
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(gss.split(df, df["speaker"], groups))
    return df.index[train_idx], df.index[test_idx]


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


def evaluate(y_true, y_pred, labels: list[str]) -> dict:
    """Everything the README's metrics table needs."""
    return {
        "report": classification_report(y_true, y_pred, labels=labels, output_dict=True),
        "confusion": confusion_matrix(y_true, y_pred, labels=labels),
        "majority_baseline": float(pd.Series(y_true).value_counts(normalize=True).iloc[0]),
    }

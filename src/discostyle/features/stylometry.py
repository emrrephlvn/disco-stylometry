"""Week 2 — stylometric fingerprints.

Line-level features here are deliberately classic and *interpretable*: the
point of stylometry is that every number is explainable in an interview
("Cuno's hapax rate is high because he invents insults").
Embeddings live in models/, not here.
"""

from __future__ import annotations

import re
import string

import pandas as pd
import textstat

# Small, classic function-word set (Mosteller & Wallace tradition). Extend in Week 2.
FUNCTION_WORDS = [
    "the", "a", "an", "and", "but", "or", "of", "to", "in", "on", "at", "by",
    "for", "with", "is", "was", "be", "not", "no", "it", "you", "i", "he", "she",
    "that", "this", "what", "so", "just", "very", "really",
]

_WORD = re.compile(r"[a-zA-Z']+")


def line_features(text: str) -> dict[str, float]:
    """Stylometric feature vector for a single line."""
    words = [w.lower() for w in _WORD.findall(text)]
    n = len(words) or 1
    unique = set(words)
    counts = {w: words.count(w) for w in unique}

    feats: dict[str, float] = {
        "n_words": float(len(words)),
        "avg_word_len": sum(len(w) for w in words) / n,
        "type_token_ratio": len(unique) / n,
        "hapax_rate": sum(1 for c in counts.values() if c == 1) / n,
        "exclaim_rate": text.count("!") / n,
        "question_rate": text.count("?") / n,
        "ellipsis_rate": text.count("...") / n,
        "dash_rate": text.count("—") / n + text.count(" - ") / n,
        "caps_word_rate": sum(1 for w in _WORD.findall(text) if w.isupper() and len(w) > 1) / n,
        "punct_density": sum(1 for ch in text if ch in string.punctuation) / max(len(text), 1),
        "flesch_reading_ease": float(textstat.flesch_reading_ease(text)),
    }
    for fw in FUNCTION_WORDS:
        feats[f"fw_{fw}"] = counts.get(fw, 0) / n
    return feats


def featurize(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """Feature matrix for a corpus frame; index aligned with ``df``."""
    return pd.DataFrame([line_features(t) for t in df[text_col]], index=df.index)


def speaker_fingerprints(df: pd.DataFrame, feats: pd.DataFrame) -> pd.DataFrame:
    """Per-speaker mean fingerprint — the table behind the radar plots."""
    return feats.groupby(df["speaker"]).mean().sort_index()

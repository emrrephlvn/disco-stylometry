"""Burrows' Delta — the classic stylometric attribution method (Burrows 2002),
in the Mosteller & Wallace function-word tradition.

Delta assigns a text to the author whose *most-frequent-word* (MFW) profile is
closest in z-scored space, where the distance is the mean absolute z-difference
(an L1 / Manhattan distance over standardized word frequencies). It is
interpretable by construction — the features are literally common-word rates and
the "distance" is a plain average — which is exactly why it belongs here as the
literature-standard baseline the TF-IDF and embedding rungs must beat to justify
their added complexity.

Note: Delta was designed for *long* documents; on single dialogue lines its MFW
vector is sparse and noisy, so we expect it to trail the modern rungs. Reporting
that gap honestly is the point.
"""
from __future__ import annotations

import re
from collections import Counter

import numpy as np
import pandas as pd

_WORD = re.compile(r"[a-z']+")


def tokens(text: str) -> list[str]:
    return _WORD.findall(text.lower())


class BurrowsDelta:
    """Nearest-author Burrows' Delta classifier (sklearn-style fit/predict).

    Parameters
    ----------
    n_mfw : int
        Number of most-frequent words to keep (Burrows' classic range ~100–300).
    """

    def __init__(self, n_mfw: int = 150):
        self.n_mfw = n_mfw

    def _rel_freq(self, texts) -> np.ndarray:
        """rows = docs, cols = MFW, values = relative frequency of that word."""
        idx = {w: i for i, w in enumerate(self.mfw_)}
        out = np.zeros((len(texts), len(self.mfw_)))
        for r, t in enumerate(texts):
            toks = tokens(t)
            n = len(toks) or 1
            for w, k in Counter(toks).items():
                j = idx.get(w)
                if j is not None:
                    out[r, j] = k / n
        return out

    def fit(self, texts, y) -> "BurrowsDelta":
        texts = list(texts)
        y = pd.Series(list(y)).reset_index(drop=True)
        allc: Counter = Counter()
        for t in texts:
            allc.update(tokens(t))
        self.mfw_ = [w for w, _ in allc.most_common(self.n_mfw)]

        f = self._rel_freq(texts)
        self.mu_ = f.mean(axis=0)
        self.sd_ = f.std(axis=0) + 1e-9  # guard zero-variance words
        z = (f - self.mu_) / self.sd_

        self.classes_ = np.array(sorted(y.unique()))
        self.centroids_ = np.vstack([z[(y == c).to_numpy()].mean(axis=0) for c in self.classes_])
        return self

    def _delta(self, texts) -> np.ndarray:
        """(n_docs, n_classes) mean absolute z-difference to each author centroid."""
        z = (self._rel_freq(texts) - self.mu_) / self.sd_
        return np.abs(z[:, None, :] - self.centroids_[None, :, :]).mean(axis=2)

    def predict(self, texts):
        return self.classes_[self._delta(texts).argmin(axis=1)]

    def centroid_distances(self) -> pd.DataFrame:
        """Pairwise Delta between the learned author centroids — a classic-stylometry
        view of 'which voices are closest', to sit alongside Weeks 2/4."""
        c = self.centroids_
        d = np.abs(c[:, None, :] - c[None, :, :]).mean(axis=2)
        return pd.DataFrame(d, index=self.classes_, columns=self.classes_)

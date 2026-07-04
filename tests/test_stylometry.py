"""Smoke tests — extend as features stabilize."""

import pandas as pd

from discostyle.features.stylometry import featurize, line_features, speaker_fingerprints


def test_line_features_basic():
    feats = line_features("The pinball machine is broken. Goddamned pinball!")
    assert feats["n_words"] > 0
    assert 0 <= feats["type_token_ratio"] <= 1
    assert feats["exclaim_rate"] > 0


def test_featurize_shapes_align():
    df = pd.DataFrame(
        {
            "speaker": ["Kim Kitsuragi", "Cuno"],
            "text": ["I would advise against that, detective.", "Cuno doesn't care! Cuno is the law!"],
        }
    )
    feats = featurize(df)
    assert len(feats) == len(df)
    fp = speaker_fingerprints(df, feats)
    assert set(fp.index) == {"Kim Kitsuragi", "Cuno"}


def test_cuno_shouts_more_than_kim():
    """The whole thesis in one assert: fingerprints separate voices."""
    df = pd.DataFrame(
        {
            "speaker": ["Kim", "Cuno"],
            "text": [
                "I would advise against that, detective. It seems unwise.",
                "Cuno doesn't care! Get out of Cuno's face! Cuno is untouchable!",
            ],
        }
    )
    feats = featurize(df)
    fp = speaker_fingerprints(df, feats)
    assert fp.loc["Cuno", "exclaim_rate"] > fp.loc["Kim", "exclaim_rate"]

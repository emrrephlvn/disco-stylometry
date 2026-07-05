"""WHO SAID IT? — interactive demo (V1 characters + V2 skills).

Run:  streamlit run app/streamlit_app.py
Models ship in the repo (see scripts/build_demo_model.py to retrain).
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# Streamlit Cloud runs from the repo root without pip-installing the package;
# put src/ on the path so `discostyle` resolves there too (no-op locally when
# the package is already installed).
_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from discostyle.models.classifier import explain_prediction  # noqa: E402

MODES = {
    "12 characters (V1)": {
        "path": _ROOT / "models" / "tfidf_logreg.joblib",
        "placeholder": '"Goddamned pinball machine, I hate this place!"',
        "caption": "Which of the 12 locked characters said it?",
    },
    "23 skill voices (V2)": {
        "path": _ROOT / "models" / "tfidf_logreg_skills.joblib",
        "placeholder": '"Did someone mention cocaine? Are we doing cocaine?"',
        "caption": (
            "Which of the 23 inner voices is speaking? (Hosted demo ships the lighter "
            "TF-IDF model, macro-F1 0.298; MiniLM embeddings score 0.322 but need torch — "
            "see reports/week5_skill_metrics.md.)"
        ),
    },
}

# Verdict-confidence tiers. With 12–23 classes a flat distribution tops out
# around 0.10–0.15, so a sub-0.25 peak means "no real evidence" (e.g. non-English
# input) and 0.25–0.40 is thin. Thresholds chosen from observed behavior, not tuned.
WEAK, MODERATE = 0.25, 0.40


@st.cache_resource(show_spinner="Loading model...")
def load_model(path_str: str):
    return joblib.load(path_str)


st.set_page_config(page_title="Who Said It? — Disco Elysium Stylometry", page_icon="🎙️")
st.title("WHO SAID IT?")
st.caption("Stylometric speaker attribution on the Disco Elysium dialogue corpus.")

mode = st.radio("Voices:", list(MODES), horizontal=True)
cfg = MODES[mode]

if not cfg["path"].exists():
    st.warning(f"Model not found at `{cfg['path'].name}`. "
               "Run `python scripts/build_demo_model.py` first.")
    st.stop()

pipe = load_model(str(cfg["path"]))
st.caption(cfg["caption"])

text = st.text_area("Paste a line of dialogue:", placeholder=cfg["placeholder"])
if text.strip():
    proba = pipe.predict_proba([text])[0]
    classes = pipe.classes_
    result = pd.DataFrame({"speaker": classes, "probability": proba}).sort_values(
        "probability", ascending=False
    )
    top_speaker = result.iloc[0]["speaker"]
    top_p = float(result.iloc[0]["probability"])

    # Closed-set honesty: the model MUST pick one of its classes — it cannot say
    # "none of them". Surface that instead of feigning confidence.
    if top_p < WEAK:
        st.warning(
            f"**Weak evidence** (top probability {top_p:.0%}, near-flat distribution). "
            "The text gives the model almost nothing — likely not stylistically distinctive, "
            f"not in English, or said by someone outside the {len(classes)} known voices. "
            "This classifier is closed-set: it cannot answer “none of them”."
        )
    elif top_p < MODERATE:
        st.info(
            f"**Thin evidence** (top probability {top_p:.0%}). Treat the verdict as a lean, "
            "not an identification — and remember the true speaker may not be among the "
            f"{len(classes)} voices this model knows."
        )

    col_verdict, col_why = st.columns([1, 1])

    with col_verdict:
        st.subheader(f"WHO: **{top_speaker}**")
        st.caption(f"Probability across all {len(classes)} voices")
        st.bar_chart(result.set_index("speaker"))

    with col_why:
        st.subheader("WHY")
        st.caption(f"N-grams in your text that pushed the model toward **{top_speaker}** "
                   "(TF-IDF weight × learned coefficient)")
        evidence = explain_prediction(pipe, text, top_speaker, top_k=8)
        if evidence.empty:
            st.info("No n-gram in this text matched the model's vocabulary strongly enough "
                    "to explain the verdict — try a longer line.")
        else:
            st.bar_chart(evidence.set_index("ngram")["contribution"])
            st.dataframe(evidence, hide_index=True, use_container_width=True)

    st.divider()
    st.caption(
        "A probability bar alone tells you the verdict; the WHY panel tells you the "
        "evidence — the interpretability point of this whole project."
    )

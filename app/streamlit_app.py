"""Week 4 — "WHO SAID IT?" interactive demo.

Run:  streamlit run app/streamlit_app.py
Needs: data/processed/lines.parquet + a trained pipeline at
       models/tfidf_logreg.joblib (build with scripts/build_demo_model.py).
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
_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from discostyle.models.classifier import explain_prediction  # noqa: E402

# Resolve against the repo root (parent of app/), not the process cwd, so the
# model is found no matter where streamlit is launched from.
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "tfidf_logreg.joblib"

st.set_page_config(page_title="Who Said It? — Disco Elysium Stylometry", page_icon="🎙️")
st.title("WHO SAID IT?")
st.caption("Stylometric speaker attribution on the Disco Elysium dialogue corpus.")

if not MODEL_PATH.exists():
    st.warning(
        "No trained model found at `models/tfidf_logreg.joblib`. "
        "Run `python scripts/build_demo_model.py` first, then come back."
    )
    st.stop()

pipe = joblib.load(MODEL_PATH)

text = st.text_area(
    "Paste a line of dialogue:",
    placeholder='"Goddamned pinball machine, I hate this place!"',
)
if text.strip():
    proba = pipe.predict_proba([text])[0]
    classes = pipe.classes_
    result = pd.DataFrame({"speaker": classes, "probability": proba}).sort_values(
        "probability", ascending=False
    )
    top_speaker = result.iloc[0]["speaker"]

    col_verdict, col_why = st.columns([1, 1])

    with col_verdict:
        st.subheader(f"WHO: **{top_speaker}**")
        st.caption("Probability across all 12 locked voices")
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

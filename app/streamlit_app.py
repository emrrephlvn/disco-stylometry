"""Week 4 — "WHO SAID IT?" interactive demo.

Run:  streamlit run app/streamlit_app.py
Needs: data/processed/lines.parquet + a trained pipeline saved to
       models/tfidf_logreg.joblib (Week 3 output).
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path("models/tfidf_logreg.joblib")

st.set_page_config(page_title="Who Said It? — Disco Elysium Stylometry", page_icon="🎙️")
st.title("WHO SAID IT?")
st.caption("Stylometric speaker attribution on the Disco Elysium dialogue corpus.")

if not MODEL_PATH.exists():
    st.warning(
        "No trained model found at `models/tfidf_logreg.joblib`. "
        "Finish Week 3 of the ROADMAP first, then come back."
    )
    st.stop()

pipe = joblib.load(MODEL_PATH)

text = st.text_area("Paste a line of dialogue:", placeholder='"Goddamned pinball."')
if text.strip():
    proba = pipe.predict_proba([text])[0]
    classes = pipe.classes_
    result = pd.DataFrame({"speaker": classes, "probability": proba}).sort_values(
        "probability", ascending=False
    )
    st.subheader(f"Verdict: **{result.iloc[0]['speaker']}**")
    st.bar_chart(result.set_index("speaker"))

    # Week 4 TODO: show top contributing n-grams (LogReg coef * tfidf) for the
    # predicted class — the "why", not just the "who".

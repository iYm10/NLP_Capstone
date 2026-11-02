"""Run from the repository root: streamlit run app.py."""
from pathlib import Path
import streamlit as st
from maqsad.baseline import predict
st.set_page_config(page_title="Maqsad | Arabic Sarcasm & Sentiment", page_icon="💬")
st.title("Maqsad")
st.caption("Understanding sentiment behind Arabic sarcasm")
st.write("Analyze Arabic text for sarcasm and sentiment. This demo uses the evaluated TF-IDF baseline.")
text = st.text_area("Arabic text", placeholder="يا سلام على الخدمة، ساعتين وأنا أنتظر!")
if st.button("Analyze", type="primary"):
    if not text.strip():
        st.warning("Enter some Arabic text first.")
    elif not Path("artifacts/baseline.joblib").exists():
        st.error("Model unavailable. Run: python -m maqsad.baseline")
    else:
        result = predict([text])[0]
        left, right = st.columns(2)
        left.metric("Sarcasm", "Sarcastic" if result["sarcasm"] else "Not sarcastic")
        right.metric("Sentiment", result["sentiment"].title())
        st.write(f"Sarcasm score: {result['sarcasm_score']:.2f}")
        st.caption("Scores are uncalibrated model outputs. Sarcasm depends on context and predictions may be wrong.")
st.markdown("The training data covers multiple Arabic dialects, with unequal representation. The model does not determine a person's intent or identity.")

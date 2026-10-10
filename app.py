import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Complaint Triage", page_icon="🗂️", layout="centered")
st.title("Consumer Complaint Triage")
st.write("Classify a complaint narrative into a product category using the exported DistilBERT ONNX model.")
text = st.text_area("Complaint narrative", height=180, placeholder="Describe the consumer complaint...")
if st.button("Classify complaint", type="primary", disabled=not text.strip()):
    try:
        from src.onnx_predict import ComplaintPredictor
        @st.cache_resource
        def load_model():
            return ComplaintPredictor()
        result = load_model().predict(text)
        st.subheader(result["label"])
        st.metric("Model confidence", f"{result['confidence']:.1%}")
        st.bar_chart(result["probabilities"])
    except Exception as exc:
        st.error(f"Model is not ready. Run preprocessing, fine-tuning and ONNX export first. Details: {type(exc).__name__}")

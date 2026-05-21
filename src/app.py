import streamlit as st
import tensorflow as tf
import json
import os

st.title("Reuters News Classifier")
st.write("Model loaded successfully. Ready for inference.")
st.sidebar.header(" Model Performance")

if os.path.exists("metrics.json"):
    with open("metrics.json", "r") as f:
        metrics = json.load(f)
    
    # Display side-by-side metrics
    col1, col2 = st.sidebar.columns(2)
    col1.metric(label="Accuracy", value=f"{metrics.get('accuracy', 0):.2%}")
    col2.metric(label="Loss", value=f"{metrics.get('loss', 0):.4f}")
else:
    st.sidebar.warning("Training metrics not found. Run train.py first!")
# (Simplified for infrastructure testing)
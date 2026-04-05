import numpy as np
import pandas as pd
import streamlit as st

from eeg_thought_text import COMMANDS, EEGThoughtToTextModel

st.set_page_config(page_title="Thought-to-Text (EEG Prototype)", layout="wide")
st.title("🧠 Thought-to-Text Prototype using EEG Signals")
st.caption("Neuro-AI communication demo: classify EEG-like patterns into YES/NO/LEFT/RIGHT")

if "system" not in st.session_state:
    st.session_state.system = EEGThoughtToTextModel(fs=128)
if "trained" not in st.session_state:
    st.session_state.trained = False

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1) Train model")
    samples = st.slider("Samples per command", min_value=40, max_value=300, value=120, step=20)
    model_name = st.selectbox("Model", ["svm", "rf"], index=0)

    if st.button("Generate synthetic EEG data + Train", type="primary"):
        np.random.seed(42)
        df = st.session_state.system.generate_synthetic_dataset(samples_per_class=samples)
        result = st.session_state.system.train(df, model_name=model_name)
        st.session_state.trained = True
        st.success(f"Model trained: {result.model_name.upper()} | Accuracy: {result.accuracy:.2%}")
        st.text(result.report)

with col2:
    st.subheader("2) Predict command")
    if not st.session_state.trained:
        st.info("Train a model first.")
    else:
        target = st.selectbox("Simulate thought command", COMMANDS, index=0)
        if st.button("Predict from simulated EEG epoch"):
            signal = st.session_state.system._make_epoch(target)
            pred, probs = st.session_state.system.predict_epoch(signal)

            st.metric("Predicted text output", pred)
            if probs:
                probs_df = pd.DataFrame(
                    {"command": list(probs.keys()), "probability": list(probs.values())}
                ).sort_values("probability", ascending=False)
                st.bar_chart(probs_df.set_index("command"))

st.divider()
st.markdown(
    """
### How this prototype works
1. EEG-like signals are generated (placeholder for real EEG device/dataset input).
2. Signal preprocessing + band-power feature extraction are applied.
3. A classifier predicts one of: **YES**, **NO**, **LEFT**, **RIGHT**.
4. The result is shown as text output.

> Replace synthetic generator with real EEG epochs to use on actual datasets/devices.
"""
)

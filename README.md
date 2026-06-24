# Thought-to-Text Prototype using EEG Signals (Neuro-AI)

This repository now contains a complete prototype app that demonstrates a simplified Brain-Computer Interface (BCI) workflow:

- EEG signal input (synthetic EEG-like epochs for demo)
- Signal preprocessing
- Frequency-band feature extraction
- ML classification into commands
- Text output (`YES`, `NO`, `LEFT`, `RIGHT`)

## Project structure

- `eeg_thought_text.py` – Core pipeline for preprocessing, feature extraction, training, and prediction.
- `streamlit_app.py` – Interactive UI to train and test the model.
- `requirements.txt` – Dependencies.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## CLI demo

Run the core module directly:

```bash
python eeg_thought_text.py
```

It trains a model on synthetic EEG-like data and prints sample prediction output.

## Notes

- This is a **prototype** focused on architecture and workflow.
- For real-world usage, replace synthetic data generation with actual EEG datasets (e.g., DEAP, BCI Competition datasets) or live device streams.
- Accuracy in real environments depends strongly on subject-specific calibration, artifact removal, and robust real-time pipelines.

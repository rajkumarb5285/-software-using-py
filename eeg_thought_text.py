"""Core pipeline for a Thought-to-Text EEG prototype.

This module provides:
1. Synthetic EEG dataset generation for quick demos.
2. Basic preprocessing and band-power feature extraction.
3. Supervised model training and evaluation.
4. Command prediction mapped to text outputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, welch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

COMMANDS = ["YES", "NO", "LEFT", "RIGHT"]
BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 12),
    "beta": (12, 30),
}


@dataclass
class TrainingResult:
    model_name: str
    accuracy: float
    report: str


class EEGThoughtToTextModel:
    """Train and run a basic thought-to-text command classifier."""

    def __init__(self, fs: int = 128, random_state: int = 42):
        self.fs = fs
        self.random_state = random_state
        self.model = None
        self.feature_columns: List[str] = []

    def bandpass_filter(self, signal: np.ndarray, low: float = 1.0, high: float = 40.0, order: int = 4) -> np.ndarray:
        nyq = 0.5 * self.fs
        b, a = butter(order, [low / nyq, high / nyq], btype="band")
        return filtfilt(b, a, signal)

    def extract_features(self, signal: np.ndarray) -> Dict[str, float]:
        """Extract power in standard EEG frequency bands from one epoch signal."""
        filtered = self.bandpass_filter(signal)
        freqs, psd = welch(filtered, fs=self.fs, nperseg=min(256, len(filtered)))

        features: Dict[str, float] = {}
        total_power = np.trapz(psd, freqs) + 1e-12

        for band, (low, high) in BANDS.items():
            idx = (freqs >= low) & (freqs < high)
            power = np.trapz(psd[idx], freqs[idx])
            features[f"{band}_abs_power"] = float(power)
            features[f"{band}_rel_power"] = float(power / total_power)

        features["signal_mean"] = float(np.mean(filtered))
        features["signal_std"] = float(np.std(filtered))
        return features

    def _make_epoch(self, label: str, duration_s: float = 2.0) -> np.ndarray:
        """Create synthetic EEG-like epoch with dominant frequencies by class."""
        n = int(self.fs * duration_s)
        t = np.arange(n) / self.fs

        # Noise baseline.
        signal = 0.35 * np.random.randn(n)

        if label == "YES":
            signal += 1.2 * np.sin(2 * np.pi * 10 * t)  # alpha
        elif label == "NO":
            signal += 1.1 * np.sin(2 * np.pi * 20 * t)  # beta
        elif label == "LEFT":
            signal += 0.8 * np.sin(2 * np.pi * 6 * t) + 0.5 * np.sin(2 * np.pi * 11 * t)
        elif label == "RIGHT":
            signal += 0.9 * np.sin(2 * np.pi * 14 * t) + 0.4 * np.sin(2 * np.pi * 24 * t)

        return signal

    def generate_synthetic_dataset(self, samples_per_class: int = 120) -> pd.DataFrame:
        rows: List[Dict[str, float]] = []
        for cmd in COMMANDS:
            for _ in range(samples_per_class):
                epoch = self._make_epoch(cmd)
                feats = self.extract_features(epoch)
                feats["label"] = cmd
                rows.append(feats)

        df = pd.DataFrame(rows)
        self.feature_columns = [c for c in df.columns if c != "label"]
        return df

    def train(self, df: pd.DataFrame, model_name: str = "svm") -> TrainingResult:
        if "label" not in df.columns:
            raise ValueError("Input DataFrame must contain a 'label' column.")

        X = df.drop(columns=["label"])
        y = df["label"]
        self.feature_columns = list(X.columns)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )

        if model_name.lower() == "rf":
            self.model = RandomForestClassifier(n_estimators=300, random_state=self.random_state)
        else:
            self.model = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("clf", SVC(kernel="rbf", C=2.0, gamma="scale", probability=True, random_state=self.random_state)),
                ]
            )

        self.model.fit(X_train, y_train)
        preds = self.model.predict(X_test)

        return TrainingResult(
            model_name=model_name,
            accuracy=float(accuracy_score(y_test, preds)),
            report=classification_report(y_test, preds),
        )

    def predict_epoch(self, signal: Iterable[float]) -> Tuple[str, Dict[str, float]]:
        if self.model is None:
            raise RuntimeError("Model is not trained. Call train() first.")

        features = self.extract_features(np.asarray(signal))
        X = pd.DataFrame([features])[self.feature_columns]
        label = str(self.model.predict(X)[0])

        probs: Dict[str, float] = {}
        if hasattr(self.model, "predict_proba"):
            p = self.model.predict_proba(X)[0]
            classes = self.model.classes_
            probs = {str(c): float(v) for c, v in zip(classes, p)}

        return label, probs


if __name__ == "__main__":
    np.random.seed(42)
    system = EEGThoughtToTextModel()
    data = system.generate_synthetic_dataset(samples_per_class=60)
    result = system.train(data, model_name="svm")

    demo_signal = system._make_epoch("LEFT")
    pred, proba = system.predict_epoch(demo_signal)

    print(f"Model: {result.model_name}")
    print(f"Accuracy: {result.accuracy:.3f}")
    print("Prediction for demo LEFT-like signal:", pred)
    print("Probabilities:", proba)

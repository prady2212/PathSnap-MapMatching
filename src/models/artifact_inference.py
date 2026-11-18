"""Inference helpers for exported map-matching model bundles."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


def load_model(path: str | Path) -> dict[str, Any]:
    """Load one exported joblib model bundle."""
    bundle = joblib.load(path)
    required = {"estimator", "scaler", "calibrator", "feature_columns"}
    missing = required.difference(bundle)
    if missing:
        raise ValueError(f"Model bundle is missing fields: {sorted(missing)}")
    return bundle


def predict_candidate_probabilities(
    bundle: dict[str, Any], frame: pd.DataFrame
) -> np.ndarray:
    """Score candidate rows using the bundle's saved features and preprocessing."""
    feature_columns = bundle["feature_columns"]
    missing = set(feature_columns).difference(frame.columns)
    if missing:
        raise ValueError(f"Candidate frame is missing features: {sorted(missing)}")
    values = frame[feature_columns].to_numpy(dtype=np.float32, copy=True)
    scaled = bundle["scaler"].transform(values)
    calibrator = bundle["calibrator"]
    if calibrator is not None:
        decision = bundle["estimator"].decision_function(scaled).reshape(-1, 1)
        return calibrator.predict_proba(decision)[:, 1]
    return bundle["estimator"].predict_proba(scaled)[:, 1]

"""
Load a trained model and predict buy/sell/hold signals from live bar data.
"""
from pathlib import Path
from typing import Tuple
import joblib
import numpy as np
import pandas as pd

from .feature_engineering import build_features
from config import MLConfig


_LABEL_TO_ACTION = {1: "buy", -1: "sell", 0: "hold"}


class MLPredictor:
    def __init__(self, cfg: MLConfig = MLConfig()):
        self.cfg = cfg
        self._pipeline = None
        self._feature_names: list = None
        self._model_type: str = None

    def load(self, model_path: str = None) -> bool:
        path = Path(model_path or Path(self.cfg.model_path) / "model.pkl")
        if not path.exists():
            print(f"[MLPredictor] No model found at {path}. Train first via ml/trainer.py")
            return False
        try:
            bundle = joblib.load(path)
            self._pipeline = bundle["pipeline"]
            self._feature_names = bundle["feature_names"]
            self._model_type = bundle["model_type"]
            print(f"[MLPredictor] Loaded {self._model_type} model (CV acc: {bundle.get('cv_accuracy', '?'):.3f})")
            return True
        except Exception as e:
            print(f"[MLPredictor] Failed to load model: {e}")
            return False

    def predict(self, bars: pd.DataFrame) -> Tuple[str, float, str]:
        """
        Args:
            bars: recent OHLCV bars (most recent last)
        Returns:
            (action, confidence, reason)
        """
        feats = build_features(bars, window=self.cfg.feature_window)

        if feats.empty or feats.isna().any():
            return "hold", 0.0, "ML: feature extraction failed"

        # align feature order to training
        feat_row = feats.reindex(self._feature_names).fillna(0.0)
        X = feat_row.values.reshape(1, -1)

        proba = self._pipeline.predict_proba(X)[0]
        classes = self._pipeline.classes_
        pred_class = classes[np.argmax(proba)]
        confidence = float(np.max(proba))

        action = _LABEL_TO_ACTION.get(pred_class, "hold")
        reason = (
            f"ML({self._model_type}): {action} | confidence={confidence:.2f} | "
            + " ".join(f"cls{c}={p:.2f}" for c, p in zip(classes, proba))
        )
        return action, confidence, reason

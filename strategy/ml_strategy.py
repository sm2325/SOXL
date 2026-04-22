"""
ML-powered strategy. Loads a trained model from ml/trainer.py and wraps
it in the BaseStrategy interface so the bot treats it identically to EmaRsiStrategy.
"""
import pandas as pd
from pathlib import Path

from .base import BaseStrategy, Signal
from ml.predictor import MLPredictor
from config import MLConfig


class MLStrategy(BaseStrategy):
    def __init__(self, cfg: MLConfig = MLConfig()):
        self.cfg = cfg
        self._predictor = MLPredictor(cfg)
        self._loaded = False

    @property
    def name(self) -> str:
        return f"ML-{self.cfg.model_type}"

    @property
    def min_bars_required(self) -> int:
        return self.cfg.feature_window + 5

    def load_model(self, model_path: str = None) -> bool:
        path = model_path or Path(self.cfg.model_path) / "model.pkl"
        self._loaded = self._predictor.load(str(path))
        return self._loaded

    def generate_signal(self, bars: pd.DataFrame) -> Signal:
        if not self._loaded:
            return Signal("hold", 0.0, "ML model not loaded — run ml/trainer.py first")

        if len(bars) < self.min_bars_required:
            return Signal("hold", 0.0, "insufficient data for ML features")

        action, confidence, reason = self._predictor.predict(bars)
        return Signal(action=action, confidence=confidence, reason=reason)

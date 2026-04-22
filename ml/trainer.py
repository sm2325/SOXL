"""
Train a model on strategist trade data to mimic their decision-making.

Usage:
    python -m ml.trainer --trades data/strategist_trades.csv --symbol SOXL
"""
import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from .data_ingestion import load_strategist_trades
from .feature_engineering import build_training_features
from config import MLConfig


_MODELS = {
    "random_forest": RandomForestClassifier(
        n_estimators=200, max_depth=6, min_samples_leaf=5,
        class_weight="balanced", random_state=42,
    ),
    "xgboost": None,  # loaded lazily to avoid hard dep at import
    "gradient_boosting": GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42,
    ),
}


def _get_model(model_type: str):
    if model_type == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            use_label_encoder=False, eval_metric="mlogloss", random_state=42,
        )
    model = _MODELS.get(model_type)
    if model is None:
        raise ValueError(f"Unknown model type: {model_type}. Choose: {list(_MODELS.keys())}")
    return model


class ModelTrainer:
    def __init__(self, cfg: MLConfig = MLConfig()):
        self.cfg = cfg
        self.pipeline: Pipeline = None
        self.feature_names: list = None

    def train(self, trades_path: str, symbol: str = "SOXL") -> dict:
        """
        Full training pipeline:
        1. Load strategist trades
        2. Download corresponding market data
        3. Build features
        4. Train + cross-validate
        5. Save model
        """
        trades = load_strategist_trades(trades_path)
        symbol_trades = trades[trades["symbol"] == symbol].copy()

        if len(symbol_trades) < self.cfg.min_training_samples:
            raise ValueError(
                f"Only {len(symbol_trades)} trades for {symbol}, "
                f"need at least {self.cfg.min_training_samples}"
            )

        print(f"\n[Trainer] Downloading market data for {symbol}...")
        start = symbol_trades["timestamp"].min().strftime("%Y-%m-%d")
        end = symbol_trades["timestamp"].max().strftime("%Y-%m-%d")
        market_data = yf.download(symbol, start=start, end=end, interval="5m", progress=False)
        market_data.columns = [c.lower() for c in market_data.columns]
        market_data.index = pd.to_datetime(market_data.index, utc=True)

        print("[Trainer] Building features...")
        X, y = build_training_features(
            symbol_trades, market_data, window=self.cfg.feature_window
        )
        self.feature_names = list(X.columns)

        print(f"[Trainer] Dataset: {len(X)} samples | Labels: {y.value_counts().to_dict()}")

        base_model = _get_model(self.cfg.model_type)
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", base_model),
        ])

        scores = cross_val_score(self.pipeline, X, y, cv=5, scoring="accuracy")
        print(f"[Trainer] CV Accuracy: {scores.mean():.3f} ± {scores.std():.3f}")

        self.pipeline.fit(X, y)

        output_path = Path(self.cfg.model_path) / "model.pkl"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "pipeline": self.pipeline,
            "feature_names": self.feature_names,
            "model_type": self.cfg.model_type,
            "symbol": symbol,
            "cv_accuracy": scores.mean(),
        }, output_path)
        print(f"[Trainer] Model saved to {output_path}")

        return {
            "model_type": self.cfg.model_type,
            "n_samples": len(X),
            "cv_accuracy": scores.mean(),
            "cv_std": scores.std(),
            "feature_names": self.feature_names,
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trades", required=True, help="Path to strategist trade CSV/JSON")
    parser.add_argument("--symbol", default="SOXL")
    parser.add_argument("--model", default="random_forest", choices=list(_MODELS.keys()) + ["xgboost"])
    args = parser.parse_args()

    cfg = MLConfig(model_type=args.model)
    trainer = ModelTrainer(cfg)
    results = trainer.train(args.trades, symbol=args.symbol)
    print("\n=== Training Complete ===")
    for k, v in results.items():
        print(f"  {k}: {v}")

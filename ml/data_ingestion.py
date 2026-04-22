"""
Load and validate strategist trade data.

Expected CSV format (flexible column names):
    timestamp, symbol, action, quantity, price[, notes]

    timestamp : ISO 8601  e.g. "2026-01-15 10:30:00"
    symbol    : ticker    e.g. "SOXL"
    action    : buy | sell | short | cover
    quantity  : number of shares
    price     : execution price
    notes     : optional free-text

JSON format: list of objects with the same fields.
"""
from pathlib import Path
from typing import Union
import pandas as pd


_COLUMN_ALIASES = {
    "time": "timestamp", "date": "timestamp", "datetime": "timestamp",
    "ticker": "symbol", "stock": "symbol",
    "side": "action", "direction": "action", "trade": "action",
    "qty": "quantity", "shares": "quantity", "size": "quantity",
    "exec_price": "price", "fill_price": "price", "avg_price": "price",
}

_REQUIRED = {"timestamp", "symbol", "action", "quantity", "price"}
_ACTION_MAP = {
    "buy": "buy", "long": "buy", "b": "buy",
    "sell": "sell", "s": "sell",
    "short": "sell", "cover": "buy",
}


def load_strategist_trades(path: Union[str, Path]) -> pd.DataFrame:
    """
    Load strategist trade data from CSV or JSON.
    Returns a clean DataFrame with columns:
        [timestamp, symbol, action, quantity, price, notes]
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Trade data file not found: {path}")

    if path.suffix.lower() == ".json":
        df = pd.read_json(path)
    elif path.suffix.lower() in {".csv", ".tsv"}:
        sep = "\t" if path.suffix.lower() == ".tsv" else ","
        df = pd.read_csv(path, sep=sep)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}. Use .csv, .tsv, or .json")

    # normalize column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    df = df.rename(columns=_COLUMN_ALIASES)

    missing = _REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Got: {list(df.columns)}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["action"] = df["action"].str.strip().str.lower().map(_ACTION_MAP)
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    if "notes" not in df.columns:
        df["notes"] = ""

    df = df.dropna(subset=["timestamp", "action", "quantity", "price"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    print(f"[DataIngestion] Loaded {len(df)} trades from {path.name}")
    print(f"  Period: {df['timestamp'].min()} → {df['timestamp'].max()}")
    print(f"  Symbols: {df['symbol'].unique().tolist()}")
    print(f"  Actions: {df['action'].value_counts().to_dict()}")

    return df[["timestamp", "symbol", "action", "quantity", "price", "notes"]]


def make_sample_data(output_path: str = "data/sample_trades.csv") -> None:
    """Generate sample strategist trade data for testing."""
    import numpy as np
    from datetime import datetime, timedelta, timezone

    rng = np.random.default_rng(42)
    base = datetime(2026, 1, 15, 10, 0, tzinfo=timezone.utc)
    rows = []
    price = 25.0

    for i in range(80):
        price += rng.normal(0, 0.5)
        price = max(5.0, price)
        action = "buy" if rng.random() > 0.5 else "sell"
        rows.append({
            "timestamp": (base + timedelta(hours=i * 4)).isoformat(),
            "symbol": "SOXL",
            "action": action,
            "quantity": int(rng.integers(10, 100)),
            "price": round(price, 2),
            "notes": f"signal_{i}",
        })

    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"Sample data written to {output_path}")


if __name__ == "__main__":
    make_sample_data()

"""
Extract market features from OHLCV bars around a given timestamp.
These features are used both for training (on strategist trades) and inference.
"""
import numpy as np
import pandas as pd
import ta


def build_features(bars: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Compute a flat feature vector from the last `window` bars.

    Features:
        - Price vs EMAs (9, 21)
        - RSI(14)
        - Volume ratio vs 20-bar average
        - ATR(14) normalized
        - Bollinger band position
        - VWAP deviation
        - Recent momentum (5, 10, 20 bar returns)
        - Time of day (sin/cos encoded)
        - Day of week (sin/cos encoded)
    """
    if len(bars) < window + 1:
        return pd.Series(dtype=float)

    df = bars.tail(window + 30).copy()
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    ema9 = ta.trend.ema_indicator(close, window=9)
    ema21 = ta.trend.ema_indicator(close, window=21)
    rsi = ta.momentum.rsi(close, window=14)
    atr = ta.volatility.average_true_range(high, low, close, window=14)
    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)

    last_close = close.iloc[-1]
    last_high = high.iloc[-1]
    last_low = low.iloc[-1]

    # VWAP (rolling intraday approximation)
    typical_price = (high + low + close) / 3
    vwap = (typical_price * volume).rolling(window).sum() / volume.rolling(window).sum()

    last_ts = df.index[-1]
    if hasattr(last_ts, "hour"):
        hour = last_ts.hour + last_ts.minute / 60
        dow = last_ts.weekday()
    else:
        hour, dow = 12.0, 2

    features = {
        "ema9_dev": (last_close - ema9.iloc[-1]) / last_close,
        "ema21_dev": (last_close - ema21.iloc[-1]) / last_close,
        "ema_spread": (ema9.iloc[-1] - ema21.iloc[-1]) / last_close,
        "rsi": rsi.iloc[-1] / 100.0,
        "volume_ratio": volume.iloc[-1] / (volume.rolling(window).mean().iloc[-1] + 1e-9),
        "atr_pct": atr.iloc[-1] / last_close,
        "bb_position": (last_close - bb.bollinger_lband().iloc[-1]) / (
            bb.bollinger_hband().iloc[-1] - bb.bollinger_lband().iloc[-1] + 1e-9
        ),
        "vwap_dev": (last_close - vwap.iloc[-1]) / last_close,
        "ret_5": close.pct_change(5).iloc[-1],
        "ret_10": close.pct_change(10).iloc[-1],
        "ret_20": close.pct_change(20).iloc[-1],
        "high_low_range": (last_high - last_low) / last_close,
        "time_sin": np.sin(2 * np.pi * hour / 24),
        "time_cos": np.cos(2 * np.pi * hour / 24),
        "dow_sin": np.sin(2 * np.pi * dow / 5),
        "dow_cos": np.cos(2 * np.pi * dow / 5),
    }

    return pd.Series(features)


def build_training_features(
    trades: pd.DataFrame,
    market_data: pd.DataFrame,
    window: int = 20,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    For each strategist trade, look up market context and extract features.

    Args:
        trades:      DataFrame from data_ingestion.load_strategist_trades()
        market_data: Full OHLCV bar DataFrame (indexed by UTC timestamp)
        window:      Lookback bars for feature computation

    Returns:
        (X, y) where X is feature matrix and y is label (1=buy, -1=sell, 0=hold)
    """
    label_map = {"buy": 1, "sell": -1}
    rows, labels = [], []

    for _, trade in trades.iterrows():
        ts = trade["timestamp"]

        # get all bars up to (and including) this trade's timestamp
        past_bars = market_data[market_data.index <= ts]

        if len(past_bars) < window + 1:
            continue

        feats = build_features(past_bars, window=window)
        if feats.empty or feats.isna().any():
            continue

        rows.append(feats)
        labels.append(label_map.get(trade["action"], 0))

    if not rows:
        raise ValueError("No valid feature rows could be built — check data alignment")

    X = pd.DataFrame(rows).reset_index(drop=True)
    y = pd.Series(labels, name="label")
    return X, y

import pandas as pd
import ta

from .base import BaseStrategy, Signal
from config import StrategyConfig


class EmaRsiStrategy(BaseStrategy):
    """
    Entry:  9-EMA crosses above 21-EMA AND RSI between [rsi_entry_min, rsi_entry_max]
    Exit:   9-EMA crosses below 21-EMA  OR  RSI > rsi_exit
    Stop:   Hard stop at entry_price * (1 - stop_loss_pct)
    """

    def __init__(self, cfg: StrategyConfig = StrategyConfig()):
        self.cfg = cfg

    @property
    def name(self) -> str:
        return "EMA-RSI"

    @property
    def min_bars_required(self) -> int:
        return self.cfg.slow_ema + self.cfg.rsi_period + 5

    def generate_signal(self, bars: pd.DataFrame) -> Signal:
        if len(bars) < self.min_bars_required:
            return Signal("hold", 0.0, "insufficient data")

        close = bars["close"]

        fast_ema = ta.trend.ema_indicator(close, window=self.cfg.fast_ema)
        slow_ema = ta.trend.ema_indicator(close, window=self.cfg.slow_ema)
        rsi = ta.momentum.rsi(close, window=self.cfg.rsi_period)

        curr_fast, prev_fast = fast_ema.iloc[-1], fast_ema.iloc[-2]
        curr_slow, prev_slow = slow_ema.iloc[-1], slow_ema.iloc[-2]
        curr_rsi = rsi.iloc[-1]

        bullish_cross = prev_fast <= prev_slow and curr_fast > curr_slow
        bearish_cross = prev_fast >= prev_slow and curr_fast < curr_slow

        rsi_ok_entry = self.cfg.rsi_entry_min <= curr_rsi <= self.cfg.rsi_entry_max
        rsi_overbought = curr_rsi > self.cfg.rsi_exit

        if bullish_cross and rsi_ok_entry:
            stop = bars["close"].iloc[-1] * (1 - self.cfg.stop_loss_pct)
            return Signal(
                action="buy",
                confidence=self._confidence(curr_fast, curr_slow, curr_rsi),
                reason=f"EMA cross ↑ | RSI={curr_rsi:.1f}",
                stop_price=round(stop, 2),
            )

        if bearish_cross or rsi_overbought:
            reason = "EMA cross ↓" if bearish_cross else f"RSI overbought ({curr_rsi:.1f})"
            return Signal(action="sell", confidence=0.8, reason=reason)

        return Signal(action="hold", confidence=0.5, reason=f"RSI={curr_rsi:.1f} | no cross")

    def _confidence(self, fast: float, slow: float, rsi: float) -> float:
        spread = (fast - slow) / slow
        rsi_score = 1.0 - abs(rsi - 58) / 30   # peak confidence at RSI ~58
        return round(min(1.0, max(0.5, 0.5 + spread * 10 + rsi_score * 0.2)), 3)

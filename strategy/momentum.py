"""
Dual momentum strategy for SOXL.

Entry logic (all must be true):
  1. 20-day return > threshold  (short-term momentum)
  2. 60-day return > 0          (medium-term trend positive)
  3. SOXL 20-day return > SPY 20-day return  (relative strength)
  4. RSI(10) < rsi_exit         (not overbought)

Exit logic (any triggers exit):
  - 10-day return turns negative  (short momentum reversal)
  - RSI(10) > rsi_exit            (overbought)
  - Hard stop loss (handled by backtest engine via signal.stop_price)

SPY comparison acts as a regime filter: if semis are lagging the broad
market, stay out even if SOXL looks positive in isolation.
"""
import pandas as pd
import yfinance as yf
import ta

from .base import BaseStrategy, Signal
from config import StrategyConfig


class MomentumStrategy(BaseStrategy):
    def __init__(
        self,
        cfg: StrategyConfig = StrategyConfig(),
        short_window: int = 10,             # 10-day momentum (5-year winner)
        medium_window: int = 60,
        momentum_threshold: float = 0.01,   # 1% threshold (5-year winner)
        rsi_period: int = 10,
        rsi_exit: float = 82.0,
        use_relative: bool = True,           # compare vs SPY
    ):
        self.cfg = cfg
        self.short_window = short_window
        self.medium_window = medium_window
        self.momentum_threshold = momentum_threshold
        self.rsi_period = rsi_period
        self.rsi_exit = rsi_exit
        self.use_relative = use_relative
        self._spy_cache: pd.Series = None
        self._spy_cache_date = None

    @property
    def name(self) -> str:
        spy = "+SPY" if self.use_relative else ""
        return f"Momentum({self.short_window}/{self.medium_window},{self.momentum_threshold*100:.0f}%{spy})"

    @property
    def min_bars_required(self) -> int:
        return self.medium_window + self.rsi_period + 2

    def _get_spy_returns(self, as_of: pd.Timestamp) -> pd.Series:
        """Fetch SPY daily closes up to as_of date (cached by date)."""
        date_key = as_of.date()
        if self._spy_cache_date == date_key and self._spy_cache is not None:
            return self._spy_cache

        try:
            spy = yf.download(
                "SPY", period="120d", interval="1d", progress=False
            )
            if isinstance(spy.columns, pd.MultiIndex):
                spy.columns = spy.columns.get_level_values(0)
            spy.index = pd.to_datetime(spy.index, utc=True)
            self._spy_cache = spy["Close"].dropna()
            self._spy_cache_date = date_key
        except Exception:
            self._spy_cache = None

        return self._spy_cache

    def generate_signal(self, bars: pd.DataFrame) -> Signal:
        if len(bars) < self.min_bars_required:
            return Signal("hold", 0.0, "insufficient data")

        close = bars["close"]
        rsi = ta.momentum.rsi(close, window=self.rsi_period)

        curr_rsi = rsi.iloc[-1]
        ret_short = (close.iloc[-1] - close.iloc[-self.short_window]) / close.iloc[-self.short_window]
        ret_medium = (close.iloc[-1] - close.iloc[-self.medium_window]) / close.iloc[-self.medium_window]
        ret_10d = (close.iloc[-1] - close.iloc[-10]) / close.iloc[-10]

        # --- exit conditions ---
        if curr_rsi > self.rsi_exit:
            return Signal("sell", 0.85, f"RSI overbought ({curr_rsi:.1f})")

        if ret_10d < -0.03:   # 10-day return < -3%: short momentum reversed
            return Signal("sell", 0.80, f"10d momentum reversed ({ret_10d*100:+.1f}%)")

        # --- relative strength vs SPY ---
        spy_ok = True
        spy_context = ""
        if self.use_relative:
            ts = bars.index[-1]
            spy_series = self._get_spy_returns(ts)
            if spy_series is not None and len(spy_series) >= self.short_window:
                spy_ret = (spy_series.iloc[-1] - spy_series.iloc[-self.short_window]) / spy_series.iloc[-self.short_window]
                spy_ok = ret_short > spy_ret
                spy_context = f" | SOXL {ret_short*100:+.1f}% vs SPY {spy_ret*100:+.1f}%"

        # --- entry conditions ---
        short_ok = ret_short > self.momentum_threshold
        medium_ok = ret_medium > 0

        if short_ok and medium_ok and spy_ok and curr_rsi < self.rsi_exit:
            stop = close.iloc[-1] * (1 - self.cfg.stop_loss_pct)
            confidence = self._confidence(ret_short, ret_medium, curr_rsi)
            return Signal(
                action="buy",
                confidence=confidence,
                reason=f"20d={ret_short*100:+.1f}% 60d={ret_medium*100:+.1f}% RSI={curr_rsi:.0f}{spy_context}",
                stop_price=round(stop, 4),
            )

        return Signal(
            "hold", 0.4,
            f"20d={ret_short*100:+.1f}% 60d={ret_medium*100:+.1f}% RSI={curr_rsi:.0f}{spy_context}"
        )

    def _confidence(self, ret_short: float, ret_medium: float, rsi: float) -> float:
        score = 0.5
        score += min(0.25, ret_short * 2)
        score += min(0.15, ret_medium * 0.5)
        score -= max(0.0, (rsi - 60) / 100)
        return round(min(1.0, max(0.5, score)), 3)

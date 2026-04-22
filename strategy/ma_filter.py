"""
MA Cross Trend Filter Strategy for SOXL.

Research basis: A simple 22-day / 200-day MA cross applied to SOXL could
turn $10k → ~$2.1M since inception by avoiding catastrophic drawdowns
(SOXL has had 91% peak-to-trough declines in bear markets).

Logic:
  Entry  : close > MA22 AND close > MA200 AND SPY above its MA50 (macro filter)
  Exit   : close < MA22  OR  close < MA200
  Stop   : ATR-based trailing stop (1.5x ATR14) — adapts to SOXL volatility

Why ATR stops over fixed %:
  SOXL daily ATR regularly exceeds 5-8%. A fixed 2.5% stop gets hit on noise.
  1.5x ATR gives the trade room to breathe while still capping catastrophic loss.
"""
import pandas as pd
import ta
import yfinance as yf

from .base import BaseStrategy, Signal
from config import StrategyConfig


class MAFilterStrategy(BaseStrategy):
    def __init__(
        self,
        cfg: StrategyConfig = StrategyConfig(),
        fast_ma: int = 22,
        slow_ma: int = 200,
        atr_period: int = 14,
        atr_stop_mult: float = 1.5,
        spy_ma: int = 50,
        use_spy_filter: bool = True,
    ):
        self.cfg = cfg
        self.fast_ma = fast_ma
        self.slow_ma = slow_ma
        self.atr_period = atr_period
        self.atr_stop_mult = atr_stop_mult
        self.spy_ma = spy_ma
        self.use_spy_filter = use_spy_filter
        self._spy_cache: pd.Series = None
        self._spy_cache_date = None

    @property
    def name(self) -> str:
        return f"MA-Filter({self.fast_ma}/{self.slow_ma}+ATR{self.atr_stop_mult}x)"

    @property
    def min_bars_required(self) -> int:
        return self.slow_ma + self.atr_period + 2

    def _spy_above_ma(self, as_of: pd.Timestamp) -> bool:
        if not self.use_spy_filter:
            return True
        date_key = as_of.date()
        if self._spy_cache_date == date_key and self._spy_cache is not None:
            return self._spy_cache
        try:
            spy = yf.download("SPY", period="120d", interval="1d", progress=False)
            if isinstance(spy.columns, pd.MultiIndex):
                spy.columns = spy.columns.get_level_values(0)
            spy.index = pd.to_datetime(spy.index, utc=True)
            close = spy["Close"].dropna()
            ma = close.rolling(self.spy_ma).mean()
            result = bool(close.iloc[-1] > ma.iloc[-1])
            self._spy_cache = result
            self._spy_cache_date = date_key
            return result
        except Exception:
            return True  # fail open — don't block trades on data error

    def generate_signal(self, bars: pd.DataFrame) -> Signal:
        if len(bars) < self.min_bars_required:
            return Signal("hold", 0.0, "insufficient data")

        close = bars["close"]
        high  = bars["high"]
        low   = bars["low"]

        ma22  = ta.trend.sma_indicator(close, window=self.fast_ma)
        ma200 = ta.trend.sma_indicator(close, window=self.slow_ma)
        atr   = ta.volatility.average_true_range(high, low, close, window=self.atr_period)

        c     = close.iloc[-1]
        m22   = ma22.iloc[-1]
        m200  = ma200.iloc[-1]
        atr_v = atr.iloc[-1]

        above_fast = c > m22
        above_slow = c > m200
        spy_ok     = self._spy_above_ma(bars.index[-1])

        stop = round(c - self.atr_stop_mult * atr_v, 4)

        # ── Exit: price fell below either MA ────────────────────────────────
        if not above_fast or not above_slow:
            reason = (
                f"close ${c:.2f} < MA{self.fast_ma} ${m22:.2f}" if not above_fast
                else f"close ${c:.2f} < MA{self.slow_ma} ${m200:.2f}"
            )
            return Signal("sell", 0.9, reason)

        # ── Entry: above both MAs + SPY regime filter ────────────────────────
        if above_fast and above_slow and spy_ok:
            spread_pct = (m22 - m200) / m200 * 100
            confidence = min(1.0, 0.6 + spread_pct / 100)
            return Signal(
                "buy", round(confidence, 3),
                f"MA{self.fast_ma}>${m22:.2f} MA{self.slow_ma}>${m200:.2f} ATR={atr_v:.2f}",
                stop_price=stop,
            )

        # ── Above MAs but SPY filter blocking ───────────────────────────────
        return Signal(
            "hold", 0.4,
            f"MA ok but SPY below MA{self.spy_ma} — regime filter active"
        )

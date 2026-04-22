"""
RSI-2 Mean Reversion Swing Strategy for SOXL.

Research basis: Short-period RSI (2-day) on volatile instruments produces
extreme oversold readings that snap back hard. Documented 70-80%+ win rates
in backtests on ETFs. Key constraint: only buy dips WITHIN a bull trend
(price above 200-day MA) to avoid catching falling knives in bear markets.

Logic:
  Regime : close > MA200  (don't buy dips in a downtrend)
  Entry  : RSI(2) < 10  AND  RSI(10) < 45  (confirm oversold on both timeframes)
  Exit   : RSI(2) > 70  OR  +8% profit target  OR  −5% hard stop
  Hold   : typically 2–5 days

Why RSI(2) and not RSI(14)?
  On a 3x leveraged ETF, RSI(14) rarely reaches extreme levels because
  volatility is high. RSI(2) is much more sensitive and fires on 1-2 day
  selloffs that are statistical mean-reversion opportunities.
"""
import pandas as pd
import ta

from .base import BaseStrategy, Signal
from config import StrategyConfig


class RSI2SwingStrategy(BaseStrategy):
    def __init__(
        self,
        cfg: StrategyConfig = StrategyConfig(),
        rsi2_entry: float = 10.0,       # RSI(2) buy threshold
        rsi10_entry: float = 45.0,      # RSI(10) confirmation
        rsi2_exit: float = 70.0,        # RSI(2) exit threshold
        profit_target: float = 0.08,    # 8% profit target
        stop_loss: float = 0.05,        # 5% hard stop
        trend_ma: int = 200,            # only trade above this MA
        atr_period: int = 14,
    ):
        self.cfg = cfg
        self.rsi2_entry = rsi2_entry
        self.rsi10_entry = rsi10_entry
        self.rsi2_exit = rsi2_exit
        self.profit_target = profit_target
        self.stop_loss = stop_loss
        self.trend_ma = trend_ma
        self.atr_period = atr_period

    @property
    def name(self) -> str:
        return f"RSI2-Swing(entry<{self.rsi2_entry}/exit>{self.rsi2_exit}+MA{self.trend_ma})"

    @property
    def min_bars_required(self) -> int:
        return self.trend_ma + 10

    def generate_signal(self, bars: pd.DataFrame) -> Signal:
        if len(bars) < self.min_bars_required:
            return Signal("hold", 0.0, "insufficient data")

        close = bars["close"]
        high  = bars["high"]
        low   = bars["low"]

        rsi2  = ta.momentum.rsi(close, window=2)
        rsi10 = ta.momentum.rsi(close, window=10)
        ma200 = ta.trend.sma_indicator(close, window=self.trend_ma)
        atr   = ta.volatility.average_true_range(high, low, close, window=self.atr_period)

        c       = close.iloc[-1]
        r2      = rsi2.iloc[-1]
        r10     = rsi10.iloc[-1]
        m200    = ma200.iloc[-1]
        atr_v   = atr.iloc[-1]

        in_uptrend = c > m200

        # ── Exit: RSI recovered to overbought ───────────────────────────────
        if r2 > self.rsi2_exit:
            return Signal("sell", 0.85, f"RSI(2) overbought ({r2:.1f} > {self.rsi2_exit})")

        # ── Regime filter: below 200-day MA → don't buy dips ────────────────
        if not in_uptrend:
            return Signal(
                "hold", 0.2,
                f"Below MA{self.trend_ma} (${c:.2f} < ${m200:.2f}) — no dip buying in downtrend"
            )

        # ── Entry: extreme oversold within uptrend ───────────────────────────
        if r2 < self.rsi2_entry and r10 < self.rsi10_entry:
            stop = round(c * (1 - self.stop_loss), 4)
            target = round(c * (1 + self.profit_target), 4)
            # confidence scales inversely with RSI2 depth
            confidence = round(min(0.95, 0.65 + (self.rsi2_entry - r2) / 20), 3)
            return Signal(
                "buy", confidence,
                f"RSI(2)={r2:.1f} RSI(10)={r10:.1f} above MA200 | tgt=${target:.2f} stop=${stop:.2f}",
                stop_price=stop,
            )

        return Signal(
            "hold", 0.3,
            f"RSI(2)={r2:.1f} RSI(10)={r10:.1f} | waiting for oversold dip"
        )

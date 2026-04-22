"""
TradingView Buying Power strategy for SOXL / UVXY rotation.

Uses TradingView's multi-timeframe technical consensus score as a proxy for
options-market buying power. The score is computed as:

    score = BUY_count / (BUY + SELL + NEUTRAL) * 99

where the counts come from tradingview-ta's indicator consensus at the
specified timeframe. This closely matches the 超短 (ultra-short) buying
power columns in the historical signal CSV.

Regime logic (checked every 15 min during market hours):
  • score_5min > ENTRY_THRESH              → hold SOXL (all-in)
  • score_5min < EXIT_THRESH               → exit SOXL → cash
  • score_5min < UVXY_THRESH               → rotate cash → UVXY
  • in UVXY and score_5min > EXIT_THRESH   → exit UVXY → cash / SOXL

Signal.target_symbol carries which asset to buy; 'hold' keeps current.
"""
import logging
from dataclasses import dataclass
from typing import Optional
import pandas as pd

from tradingview_ta import TA_Handler, Interval as TaInterval

from .base import BaseStrategy, Signal

log = logging.getLogger("buying_power")

_TA_INTERVALS = {
    "5Min":  TaInterval.INTERVAL_5_MINUTES,
    "15Min": TaInterval.INTERVAL_15_MINUTES,
    "1Hour": TaInterval.INTERVAL_1_HOUR,
    "1Day":  TaInterval.INTERVAL_1_DAY,
}

SOXL_EXCHANGE = "AMEX"
UVXY_EXCHANGE = "CBOE"


def _tv_score(symbol: str, exchange: str, timeframe: str) -> Optional[float]:
    """Returns 0-99 buying power score from TradingView indicator consensus."""
    interval = _TA_INTERVALS.get(timeframe, TaInterval.INTERVAL_5_MINUTES)
    try:
        handler = TA_Handler(
            symbol=symbol,
            exchange=exchange,
            screener="america",
            interval=interval,
        )
        s = handler.get_analysis().summary
        total = s["BUY"] + s["SELL"] + s["NEUTRAL"]
        if total == 0:
            return None
        return round(s["BUY"] / total * 99, 1)
    except Exception as e:
        log.warning(f"TV score failed for {symbol}/{timeframe}: {e}")
        return None


class BuyingPowerStrategy(BaseStrategy):
    """
    Full-capital SOXL / UVXY rotation driven by TradingView buying power scores.

    Parameters
    ----------
    entry_thresh : float
        SOXL score above which we go all-in on SOXL. Default 65.
    exit_thresh : float
        SOXL score below which we exit SOXL (go to cash or UVXY). Default 30.
    uvxy_thresh : float
        SOXL score below which we actively rotate to UVXY. Default 25.
    confirm_timeframe : str
        Secondary TradingView timeframe used as confirmation filter. Default '15Min'.
    confirm_min : float
        Secondary timeframe score must exceed this to confirm SOXL entry. Default 45.
    """

    def __init__(
        self,
        entry_thresh: float = 65.0,
        exit_thresh: float = 30.0,
        uvxy_thresh: float = 25.0,
        confirm_timeframe: str = "15Min",
        confirm_min: float = 45.0,
    ):
        self.entry_thresh = entry_thresh
        self.exit_thresh = exit_thresh
        self.uvxy_thresh = uvxy_thresh
        self.confirm_timeframe = confirm_timeframe
        self.confirm_min = confirm_min
        self._last_scores: dict = {}

    @property
    def name(self) -> str:
        return f"BuyingPower(entry={self.entry_thresh}/exit={self.exit_thresh}/uvxy={self.uvxy_thresh})"

    @property
    def min_bars_required(self) -> int:
        return 1

    def get_scores(self) -> dict:
        """Fetch current SOXL and UVXY scores from TradingView. Cached per call."""
        soxl_fast = _tv_score("SOXL", SOXL_EXCHANGE, "5Min")
        soxl_confirm = _tv_score("SOXL", SOXL_EXCHANGE, self.confirm_timeframe)
        uvxy_fast = _tv_score("UVXY", UVXY_EXCHANGE, "5Min")

        scores = {
            "soxl_5min": soxl_fast,
            "soxl_confirm": soxl_confirm,
            "uvxy_5min": uvxy_fast,
        }
        self._last_scores = scores
        log.info(
            f"TV scores — SOXL(5min)={soxl_fast}  SOXL({self.confirm_timeframe})={soxl_confirm}  UVXY(5min)={uvxy_fast}"
        )
        return scores

    def generate_signal(self, bars: pd.DataFrame) -> Signal:
        """
        bars is unused — signal comes entirely from TradingView live analysis.
        Kept for interface compatibility.
        """
        scores = self.get_scores()
        return self._scores_to_signal(scores)

    def generate_signal_from_scores(self, scores: dict) -> Signal:
        """Generate signal from pre-fetched score dict (for testing / backtest)."""
        return self._scores_to_signal(scores)

    def _scores_to_signal(self, scores: dict) -> Signal:
        s = scores.get("soxl_5min")
        s_confirm = scores.get("soxl_confirm")
        uvxy = scores.get("uvxy_5min")

        if s is None:
            return Signal("hold", 0.5, "TV score unavailable", target_symbol=None)

        confirm_ok = (s_confirm is None) or (s_confirm >= self.confirm_min)
        reason_parts = [f"SOXL_5min={s}"]
        if s_confirm is not None:
            reason_parts.append(f"SOXL_{self.confirm_timeframe}={s_confirm}")
        if uvxy is not None:
            reason_parts.append(f"UVXY_5min={uvxy}")
        reason = "  ".join(reason_parts)

        # SOXL entry: fast score bullish AND confirmation not bearish
        if s >= self.entry_thresh and confirm_ok:
            conf = min(1.0, 0.6 + (s - self.entry_thresh) / 100)
            return Signal("buy", round(conf, 3), reason, target_symbol="SOXL")

        # UVXY rotation: extreme bearish
        if s <= self.uvxy_thresh and uvxy is not None:
            conf = min(1.0, 0.6 + (self.uvxy_thresh - s) / 100)
            return Signal("buy", round(conf, 3), reason, target_symbol="UVXY")

        # Exit zone
        if s <= self.exit_thresh:
            return Signal("sell", 0.8, reason, target_symbol=None)

        # Between thresholds: hold current
        return Signal("hold", 0.5, reason, target_symbol=None)

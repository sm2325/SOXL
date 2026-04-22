"""
Position sizing and risk management.
"""
import math
import yfinance as yf


def get_vix() -> float:
    """Fetch latest VIX level via yfinance."""
    try:
        vix = yf.Ticker("^VIX")
        hist = vix.history(period="1d", interval="1m")
        return float(hist["Close"].iloc[-1])
    except Exception:
        return 0.0


def calc_position_size(
    equity: float,
    price: float,
    max_position_pct: float = 0.10,
    atr: float = None,
    atr_risk_pct: float = 0.01,
) -> int:
    """
    Returns number of whole shares to buy.

    Uses the simpler of:
      - Max portfolio pct cap  (default: 10% of equity)
      - ATR-based sizing       (risk 1% of equity per ATR unit, if ATR provided)
    """
    max_by_pct = math.floor((equity * max_position_pct) / price)

    if atr and atr > 0:
        risk_dollars = equity * atr_risk_pct
        max_by_atr = math.floor(risk_dollars / atr)
        return max(1, min(max_by_pct, max_by_atr))

    return max(1, max_by_pct)


def should_stop(
    entry_price: float,
    current_price: float,
    stop_loss_pct: float = 0.025,
    explicit_stop: float = None,
) -> bool:
    """True if position should be closed due to stop loss."""
    if explicit_stop and current_price <= explicit_stop:
        return True
    pct_loss = (current_price - entry_price) / entry_price
    return pct_loss <= -stop_loss_pct


def vix_filter_ok(vix_threshold: float = 30.0) -> tuple[bool, float]:
    """Returns (trading_ok, vix_value)."""
    vix = get_vix()
    if vix <= 0:
        return True, vix   # data unavailable — don't block trading
    return vix <= vix_threshold, vix

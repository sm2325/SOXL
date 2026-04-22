"""
TradingView market data and technical analysis feed.

Bar data comes directly from TradingView's WebSocket via tvDatafeed.
Technical analysis signals come from TradingView via tradingview-ta.

Usage:
    feed = TVFeed()
    df   = feed.get_bars("SOXL", "AMEX", n_bars=500)
    df7  = feed.get_bars("SOXL", "AMEX", timeframe="5Min", n_bars=500)
    ta   = feed.get_analysis("SOXL")
    reco = feed.get_recommendation("SOXL")   # 'STRONG_BUY' | 'BUY' | 'NEUTRAL' | 'SELL'
"""
import logging
import os
import pandas as pd
from tvDatafeed import TvDatafeed, Interval as TvInterval
from tradingview_ta import TA_Handler, Interval as TaInterval

log = logging.getLogger("tv_feed")

# Map timeframe string → tvDatafeed Interval
TV_BAR_INTERVAL = {
    "1Min":  TvInterval.in_1_minute,
    "3Min":  TvInterval.in_3_minute,
    "5Min":  TvInterval.in_5_minute,
    "15Min": TvInterval.in_15_minute,
    "30Min": TvInterval.in_30_minute,
    "45Min": TvInterval.in_45_minute,
    "1Hour": TvInterval.in_1_hour,
    "2Hour": TvInterval.in_2_hour,
    "3Hour": TvInterval.in_3_hour,
    "4Hour": TvInterval.in_4_hour,
    "1Day":  TvInterval.in_daily,
    "1Week": TvInterval.in_weekly,
    "1Month":TvInterval.in_monthly,
}

# Map timeframe string → tradingview-ta Interval
TV_TA_INTERVAL = {
    "1Min":  TaInterval.INTERVAL_1_MINUTE,
    "5Min":  TaInterval.INTERVAL_5_MINUTES,
    "15Min": TaInterval.INTERVAL_15_MINUTES,
    "30Min": TaInterval.INTERVAL_30_MINUTES,
    "1Hour": TaInterval.INTERVAL_1_HOUR,
    "2Hour": TaInterval.INTERVAL_2_HOURS,
    "4Hour": TaInterval.INTERVAL_4_HOURS,
    "1Day":  TaInterval.INTERVAL_1_DAY,
    "1Week": TaInterval.INTERVAL_1_WEEK,
    "1Month":TaInterval.INTERVAL_1_MONTH,
}


class TVFeed:
    def __init__(self):
        username = os.getenv("TV_USERNAME", "")
        password = os.getenv("TV_PASSWORD", "")
        if username and password:
            self._tv = TvDatafeed(username, password)
            log.info(f"TradingView connected as {username}")
        else:
            self._tv = TvDatafeed()
            log.info("TradingView connected (anonymous)")

    # ── Bar data directly from TradingView WebSocket ─────────────────────────

    def get_bars(
        self,
        symbol: str,
        exchange: str = "AMEX",
        timeframe: str = "1Day",
        n_bars: int = 500,
    ) -> pd.DataFrame:
        """
        Returns OHLCV DataFrame indexed by UTC datetime, sourced directly
        from TradingView's data feed.

        exchange examples: AMEX, NASDAQ, NYSE, BINANCE
        timeframe: 1Min 3Min 5Min 15Min 30Min 45Min 1Hour 2Hour 3Hour 4Hour 1Day 1Week 1Month
        n_bars: number of bars to fetch (max ~5000 for intraday, unlimited for daily)
        """
        interval = TV_BAR_INTERVAL.get(timeframe, TvInterval.in_daily)
        df = self._tv.get_hist(symbol=symbol, exchange=exchange, interval=interval, n_bars=n_bars)

        if df is None or df.empty:
            log.warning(f"No data returned for {symbol}:{exchange} {timeframe}")
            return pd.DataFrame()

        df.index = pd.to_datetime(df.index, utc=True)
        df = df[["open", "high", "low", "close", "volume"]].sort_index()
        log.info(f"TradingView: {len(df)} {timeframe} bars for {symbol} ({df.index[0].date()} → {df.index[-1].date()})")
        return df

    def get_latest_price(self, symbol: str, exchange: str = "AMEX") -> float:
        df = self.get_bars(symbol, exchange, "1Min", n_bars=1)
        if df.empty:
            raise ValueError(f"Could not fetch price for {symbol}")
        return float(df["close"].iloc[-1])

    def get_multi_bars(
        self,
        symbols: list,
        exchange: str = "AMEX",
        timeframe: str = "1Day",
        n_bars: int = 500,
    ) -> dict:
        """Fetch multiple symbols. Returns {symbol: DataFrame}."""
        return {sym: self.get_bars(sym, exchange, timeframe, n_bars) for sym in symbols}

    # ── TradingView technical analysis signals ────────────────────────────────

    def get_analysis(
        self,
        symbol: str,
        exchange: str = "AMEX",
        timeframe: str = "1Day",
        screener: str = "america",
    ):
        """
        Returns tradingview-ta Analysis object:
          .summary      → {'RECOMMENDATION': 'STRONG_BUY'|'BUY'|'NEUTRAL'|'SELL'|'STRONG_SELL', 'BUY': n, ...}
          .indicators   → raw values: RSI, MACD, EMA20, ATR, ADX, BB, etc.
          .oscillators  → oscillator recommendations
          .moving_averages → MA recommendations
        """
        ta_interval = TV_TA_INTERVAL.get(timeframe, TaInterval.INTERVAL_1_DAY)
        handler = TA_Handler(
            symbol=symbol,
            exchange=exchange,
            screener=screener,
            interval=ta_interval,
        )
        analysis = handler.get_analysis()
        log.info(f"TV analysis {symbol} ({timeframe}): {analysis.summary}")
        return analysis

    def get_recommendation(
        self,
        symbol: str,
        exchange: str = "AMEX",
        timeframe: str = "1Day",
    ) -> str:
        """Returns 'STRONG_BUY', 'BUY', 'NEUTRAL', 'SELL', or 'STRONG_SELL'."""
        return self.get_analysis(symbol, exchange, timeframe).summary.get("RECOMMENDATION", "NEUTRAL")

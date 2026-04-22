"""
Real-time market data via Alpaca WebSocket.
Streams quotes, trades, and 1-min bars for any symbol.

Usage:
    stream = AlpacaStream(CONFIG.alpaca)
    stream.subscribe(["SOXL", "SPY"], on_bar=my_callback)
    stream.run()          # blocks; run in a thread for async use
"""
import logging
import threading
from typing import Callable, List, Optional

from alpaca.data.live import StockDataStream
from alpaca.data.models import Bar, Quote, Trade

from config import AlpacaConfig

log = logging.getLogger("alpaca_stream")


class AlpacaStream:
    def __init__(self, cfg: AlpacaConfig):
        self._cfg = cfg
        self._client: Optional[StockDataStream] = None
        self._thread: Optional[threading.Thread] = None

    def subscribe(
        self,
        symbols: List[str],
        on_bar: Optional[Callable[[Bar], None]] = None,
        on_quote: Optional[Callable[[Quote], None]] = None,
        on_trade: Optional[Callable[[Trade], None]] = None,
    ):
        self._client = StockDataStream(self._cfg.api_key, self._cfg.secret_key)

        if on_bar:
            self._client.subscribe_bars(on_bar, *symbols)
        if on_quote:
            self._client.subscribe_quotes(on_quote, *symbols)
        if on_trade:
            self._client.subscribe_trades(on_trade, *symbols)

        log.info(f"Subscribed to {symbols} | bars={bool(on_bar)} quotes={bool(on_quote)} trades={bool(on_trade)}")

    def run(self):
        """Block and process incoming messages."""
        if self._client is None:
            raise RuntimeError("Call subscribe() before run()")
        self._client.run()

    def run_in_background(self):
        """Start stream in a daemon thread."""
        self._thread = threading.Thread(target=self.run, daemon=True)
        self._thread.start()
        log.info("Alpaca stream running in background thread")

    def stop(self):
        if self._client:
            self._client.stop()
            log.info("Alpaca stream stopped")

from typing import Optional
from datetime import datetime, timezone
import pandas as pd

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from .base import BaseBroker, Order, Position, AccountInfo
from config import AlpacaConfig


_TIMEFRAME_MAP = {
    "1Min": TimeFrame(1, TimeFrameUnit.Minute),
    "5Min": TimeFrame(5, TimeFrameUnit.Minute),
    "15Min": TimeFrame(15, TimeFrameUnit.Minute),
    "1Hour": TimeFrame(1, TimeFrameUnit.Hour),
    "1Day": TimeFrame(1, TimeFrameUnit.Day),
}


class AlpacaBroker(BaseBroker):
    def __init__(self, cfg: AlpacaConfig):
        self._trading = TradingClient(cfg.api_key, cfg.secret_key, paper=cfg.paper)
        self._data = StockHistoricalDataClient(cfg.api_key, cfg.secret_key)

    def get_account(self) -> AccountInfo:
        acct = self._trading.get_account()
        return AccountInfo(
            equity=float(acct.equity),
            cash=float(acct.cash),
            buying_power=float(acct.buying_power),
        )

    def get_position(self, symbol: str) -> Optional[Position]:
        try:
            pos = self._trading.get_open_position(symbol)
            return Position(
                symbol=symbol,
                qty=float(pos.qty),
                avg_entry_price=float(pos.avg_entry_price),
                current_price=float(pos.current_price),
            )
        except Exception:
            return None

    def place_order(self, order: Order) -> str:
        side = OrderSide.BUY if order.side == "buy" else OrderSide.SELL

        if order.order_type == "market":
            req = MarketOrderRequest(
                symbol=order.symbol,
                qty=order.qty,
                side=side,
                time_in_force=TimeInForce.DAY,
            )
        else:
            req = LimitOrderRequest(
                symbol=order.symbol,
                qty=order.qty,
                side=side,
                time_in_force=TimeInForce.DAY,
                limit_price=order.limit_price,
            )

        resp = self._trading.submit_order(req)
        return str(resp.id)

    def cancel_order(self, order_id: str) -> bool:
        try:
            self._trading.cancel_order_by_id(order_id)
            return True
        except Exception:
            return False

    def get_bars(
        self,
        symbol: str,
        timeframe: str,
        start: str,
        end: str,
        limit: int = 1000,
    ) -> pd.DataFrame:
        tf = _TIMEFRAME_MAP.get(timeframe, TimeFrame(5, TimeFrameUnit.Minute))
        req = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=tf,
            start=datetime.fromisoformat(start),
            end=datetime.fromisoformat(end),
            limit=limit,
        )
        bars = self._data.get_stock_bars(req)
        df = bars.df

        if isinstance(df.index, pd.MultiIndex):
            df = df.xs(symbol, level="symbol")

        df.index = pd.to_datetime(df.index, utc=True)
        df = df[["open", "high", "low", "close", "volume"]].sort_index()
        return df

    def get_latest_price(self, symbol: str) -> float:
        from alpaca.data.requests import StockLatestQuoteRequest
        req = StockLatestQuoteRequest(symbol_or_symbols=symbol)
        quote = self._data.get_stock_latest_quote(req)
        return float(quote[symbol].ask_price)

    def is_market_open(self) -> bool:
        clock = self._trading.get_clock()
        return clock.is_open

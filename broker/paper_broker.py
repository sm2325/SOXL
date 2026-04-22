"""
Local paper broker — no API account needed.
Uses yfinance for market data. Tracks positions and P&L in memory.
Good for backtesting and offline development.
"""
from typing import Optional, Dict
from datetime import datetime, timezone
import uuid
import yfinance as yf
import pandas as pd

from .base import BaseBroker, Order, Position, AccountInfo


class LocalPaperBroker(BaseBroker):
    def __init__(self, starting_cash: float = 100_000.0):
        self._cash = starting_cash
        self._starting_cash = starting_cash
        self._positions: Dict[str, dict] = {}
        self._orders: Dict[str, Order] = {}
        self._trade_log: list = []

    def get_account(self) -> AccountInfo:
        equity = self._cash + sum(
            p["qty"] * self.get_latest_price(sym) for sym, p in self._positions.items()
        )
        return AccountInfo(equity=equity, cash=self._cash, buying_power=self._cash)

    def get_position(self, symbol: str) -> Optional[Position]:
        if symbol not in self._positions:
            return None
        p = self._positions[symbol]
        return Position(
            symbol=symbol,
            qty=p["qty"],
            avg_entry_price=p["avg_price"],
            current_price=self.get_latest_price(symbol),
        )

    def place_order(self, order: Order) -> str:
        price = self.get_latest_price(order.symbol)
        order_id = str(uuid.uuid4())[:8]

        if order.side == "buy":
            cost = price * order.qty
            if cost > self._cash:
                raise ValueError(f"Insufficient cash: need {cost:.2f}, have {self._cash:.2f}")
            self._cash -= cost

            if order.symbol in self._positions:
                pos = self._positions[order.symbol]
                total_qty = pos["qty"] + order.qty
                pos["avg_price"] = (pos["avg_price"] * pos["qty"] + price * order.qty) / total_qty
                pos["qty"] = total_qty
            else:
                self._positions[order.symbol] = {"qty": order.qty, "avg_price": price}

        elif order.side == "sell":
            if order.symbol not in self._positions:
                raise ValueError(f"No position in {order.symbol} to sell")
            pos = self._positions[order.symbol]
            proceeds = price * order.qty
            self._cash += proceeds
            pos["qty"] -= order.qty
            if pos["qty"] <= 0:
                del self._positions[order.symbol]

        self._trade_log.append({
            "order_id": order_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": order.symbol,
            "side": order.side,
            "qty": order.qty,
            "price": price,
        })

        print(f"[Paper] {order.side.upper()} {order.qty} {order.symbol} @ ${price:.2f}")
        return order_id

    def cancel_order(self, order_id: str) -> bool:
        return True

    def get_bars(
        self,
        symbol: str,
        timeframe: str,
        start: str,
        end: str,
        limit: int = 1000,
    ) -> pd.DataFrame:
        interval_map = {
            "1Min": "1m", "5Min": "5m", "15Min": "15m",
            "1Hour": "1h", "1Day": "1d",
        }
        interval = interval_map.get(timeframe, "5m")
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start, end=end, interval=interval)
        df.columns = [c.lower() for c in df.columns]
        df = df[["open", "high", "low", "close", "volume"]].dropna()
        df.index = pd.to_datetime(df.index, utc=True)
        return df.tail(limit)

    def get_latest_price(self, symbol: str) -> float:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="1d", interval="1m")
        return float(hist["Close"].iloc[-1])

    def is_market_open(self) -> bool:
        from datetime import time
        import pytz
        now = datetime.now(pytz.timezone("America/New_York"))
        return (
            now.weekday() < 5
            and time(9, 30) <= now.time() <= time(16, 0)
        )

    def get_trade_log(self) -> pd.DataFrame:
        return pd.DataFrame(self._trade_log)

    def summary(self) -> dict:
        acct = self.get_account()
        return {
            "starting_cash": self._starting_cash,
            "current_equity": acct.equity,
            "total_return_pct": (acct.equity - self._starting_cash) / self._starting_cash * 100,
            "total_trades": len(self._trade_log),
        }

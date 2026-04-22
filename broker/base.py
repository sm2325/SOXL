from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import pandas as pd


@dataclass
class Order:
    symbol: str
    qty: float
    side: str                          # 'buy' | 'sell'
    order_type: str = "market"
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None


@dataclass
class Position:
    symbol: str
    qty: float
    avg_entry_price: float
    current_price: float

    @property
    def unrealized_pnl_pct(self) -> float:
        return (self.current_price - self.avg_entry_price) / self.avg_entry_price


@dataclass
class AccountInfo:
    equity: float
    cash: float
    buying_power: float


class BaseBroker(ABC):
    """
    Broker-agnostic interface. Implement this to support any broker
    (Alpaca, IBKR, Schwab, Tradier, etc.) without changing strategy code.
    """

    @abstractmethod
    def get_account(self) -> AccountInfo:
        ...

    @abstractmethod
    def get_position(self, symbol: str) -> Optional[Position]:
        ...

    @abstractmethod
    def place_order(self, order: Order) -> str:
        """Returns order_id."""
        ...

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        ...

    @abstractmethod
    def get_bars(
        self,
        symbol: str,
        timeframe: str,
        start: str,
        end: str,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """
        Returns DataFrame with columns: [open, high, low, close, volume]
        indexed by UTC timestamp.
        """
        ...

    @abstractmethod
    def get_latest_price(self, symbol: str) -> float:
        ...

    @abstractmethod
    def is_market_open(self) -> bool:
        ...

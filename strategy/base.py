from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import pandas as pd


@dataclass
class Signal:
    action: str                       # 'buy' | 'sell' | 'hold'
    confidence: float                 # 0.0 – 1.0
    reason: str                       # human-readable explanation
    stop_price: Optional[float] = None
    target_symbol: Optional[str] = None  # asset to buy (None = use default symbol)


class BaseStrategy(ABC):
    """
    All strategies implement this interface.
    Strategies are stateless — they receive a bar DataFrame and return a Signal.
    """

    @abstractmethod
    def generate_signal(self, bars: pd.DataFrame) -> Signal:
        """
        Args:
            bars: OHLCV DataFrame indexed by timestamp (most recent last).
        Returns:
            Signal with action, confidence, and reason.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    def min_bars_required(self) -> int:
        return 50
